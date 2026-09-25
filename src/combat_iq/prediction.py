"""Prediction service shared by the API and Python callers."""

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from combat_iq.config import model_path
from combat_iq.data import load_fighters
from combat_iq.features import build_features


class Prediction(TypedDict):
    model_pick: str
    model_confidence: float
    predicted_outcome: Literal["red_win", "non_red_win"]
    winner: str
    win_probability: float
    fight_outcome: str
    confidence_rate: float


@lru_cache(maxsize=4)
def _read_model(path: Path, mtime_ns: int, size: int):
    # Only load trusted local artifacts: pickle can execute code when loaded.
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_model():
    path = model_path()
    stat = path.stat()
    return _read_model(path, stat.st_mtime_ns, stat.st_size)


def predict(red_fighter: str, blue_fighter: str) -> Prediction:
    features = build_features(load_fighters(), red_fighter, blue_fighter)
    model = load_model()
    label = np.asarray(model.predict(features)).reshape(-1)[0]
    if label not in (0, 1):
        raise ValueError("The model must predict binary labels 0 or 1.")
    winner = red_fighter if label == 1 else blue_fighter
    classes = np.asarray(model.classes_).reshape(-1)
    class_index = np.flatnonzero(classes == label)
    if class_index.size != 1:
        raise ValueError("The predicted label must match exactly one model class.")
    probabilities = np.asarray(model.predict_proba(features))[0]
    probability = round(float(probabilities[class_index[0]]), 3)
    return {
        "model_pick": winner,
        "model_confidence": probability,
        "predicted_outcome": "red_win" if label == 1 else "non_red_win",
        # Legacy aliases; these do not establish a calibrated chance of winning.
        "winner": winner,
        "win_probability": probability,
        # Retain these fields for existing clients, including the Streamlit example.
        "fight_outcome": f"{winner} wins",
        "confidence_rate": probability,
    }
