import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from combat_iq import prediction
from combat_iq.api.app import app
from combat_iq.config import fighters_path
from combat_iq.data import _read_fighters, load_fighters
from combat_iq.features import FEATURE_COLUMNS, InvalidMatchup, build_features

BASELINES = json.loads((Path(__file__).parent / "fixtures/predictions.json").read_text())


@pytest.mark.parametrize("case", BASELINES)
def test_saved_model_predictions_match_original(case):
    winner = case["red"] if case["label"] else case["blue"]
    expected = f"{winner} wins"
    assert prediction.predict(case["red"], case["blue"]) == {
        "model_pick": winner,
        "model_confidence": case["confidence"],
        "predicted_outcome": "red_win" if case["label"] else "non_red_win",
        "winner": winner,
        "win_probability": case["confidence"],
        "fight_outcome": expected,
        "confidence_rate": case["confidence"],
    }


def test_feature_order_values_and_dtype_match_original():
    raw = pd.read_csv(fighters_path())
    red, blue = BASELINES[0]["red"], BASELINES[0]["blue"]
    # Reconstruct the old algorithm as a regression oracle.
    red_row = raw.loc[raw.fighter == red].add_prefix("R_")
    blue_row = raw.loc[raw.fighter == blue].add_prefix("B_")
    original = pd.DataFrame(list(red_row.iloc[0]) + list(blue_row.iloc[0])).T
    original.columns = list(red_row.columns) + list(blue_row.columns)
    original = original.loc[:, FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(build_features(load_fighters(), red, blue), original)


@pytest.mark.parametrize(
    "red,blue",
    [("", "Mark Kerr"), (" ", "Mark Kerr"), ("Mark Kerr", "Mark Kerr"), ("missing", "Mark Kerr")],
)
def test_invalid_fighters(red, blue):
    with pytest.raises(InvalidMatchup):
        build_features(load_fighters(), red, blue)


def test_paths_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    case = BASELINES[0]
    assert prediction.predict(case["red"], case["blue"])["confidence_rate"] == case["confidence"]


def test_fighter_cache_is_reused_and_isolated():
    _read_fighters.cache_clear()
    first = load_fighters()
    name = first.index[0]
    expected = first.loc[name, "wins"]
    first.loc[name, "wins"] = -100
    assert load_fighters().loc[name, "wins"] == expected
    assert _read_fighters.cache_info().hits == 1


def test_cache_refreshes_when_data_changes(tmp_path):
    path = tmp_path / "fighters.csv"
    data = load_fighters().reset_index().iloc[:2]
    data.to_csv(path, index=False)
    load_fighters(path)
    data.loc[0, "wins"] = 123456
    data.to_csv(path, index=False)
    assert load_fighters(path).iloc[0]["wins"] == 123456


def test_duplicate_fighters_are_rejected(tmp_path):
    path = tmp_path / "duplicates.csv"
    row = load_fighters().reset_index().iloc[:1]
    pd.concat([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_fighters(path)


def test_model_is_loaded_once():
    prediction._read_model.cache_clear()
    assert prediction.load_model() is prediction.load_model()
    assert prediction._read_model.cache_info().hits == 1


def test_api_and_legacy_entry_point():
    from combat_iq.api.api_file import app as legacy_app

    assert legacy_app is app
    client = TestClient(app)
    assert client.get("/").json() == {"setup": "I'm on it dudes !"}
    case = BASELINES[0]
    response = client.get(
        "/predict", params={"red_fighter": case["red"], "blue_fighter": case["blue"]}
    )
    assert response.status_code == 200
    assert response.json() == prediction.predict(case["red"], case["blue"])
    assert client.get("/predict").status_code == 422
    assert (
        client.get(
            "/predict", params={"red_fighter": "missing", "blue_fighter": case["blue"]}
        ).status_code
        == 422
    )


def test_missing_artifact_returns_service_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("COMBAT_IQ_MODEL_PATH", str(tmp_path / "missing.pkl"))
    response = TestClient(app).get(
        "/predict",
        params={"red_fighter": BASELINES[0]["red"], "blue_fighter": BASELINES[0]["blue"]},
    )
    assert response.status_code == 503
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("label,winner,probability", [(1, "Red", 0.35), (0, "Blue", 0.65)])
def test_probability_matches_predicted_class_not_column_order_or_maximum(
    monkeypatch, label, winner, probability
):
    class Model:
        classes_ = np.array([1, 0])

        def predict(self, features):
            return np.array([label])

        def predict_proba(self, features):
            return np.array([[0.35, 0.65]])

    monkeypatch.setattr(prediction, "load_model", lambda: Model())
    monkeypatch.setattr(prediction, "load_fighters", lambda: None)
    monkeypatch.setattr(prediction, "build_features", lambda *args: None)
    result = prediction.predict("Red", "Blue")
    assert result["model_pick"] == winner
    assert result["model_confidence"] == probability
    assert result["predicted_outcome"] == ("red_win" if label == 1 else "non_red_win")
    assert result["winner"] == winner
    assert result["fight_outcome"] == f"{winner} wins"
    assert result["win_probability"] == probability
    assert result["confidence_rate"] == probability
