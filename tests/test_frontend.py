from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py"


def test_blue_winner_and_stale_result():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.selectbox[0].value == "Max Griffin"
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["result"]["prediction"]
    assert result["winner"] == "Mark Schultz"
    assert result["model_confidence"] == 0.619
    rendered = " ".join(str(element.proto) for element in app.get("html"))
    assert "model confidence" in rendered
    assert "estimated win probability" not in rendered
    assert "Predicted winner" not in rendered
    assert any("includes a draw" in caption.value for caption in app.caption)
    app.selectbox[0].select("Gustavo Lopez").run()
    assert "YOUR NEXT MATCHUP STARTS HERE" in " ".join(
        str(element.proto) for element in app.get("html")
    )


def test_red_winner_and_invalid_matchup():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.selectbox[0].select("Gustavo Lopez")
    app.selectbox[1].select("Mark Kerr").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["result"]["prediction"]["model_pick"] == "Gustavo Lopez"
    assert any("not been calibrated" in caption.value for caption in app.caption)
    app.selectbox[1].select("Gustavo Lopez").run()
    assert app.button[0].disabled
    assert app.warning


def test_missing_model_shows_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setenv("COMBAT_IQ_MODEL_PATH", str(tmp_path / "missing.pkl"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert (
        app.error[0].value == "The prediction is temporarily unavailable. Please try again later."
    )
