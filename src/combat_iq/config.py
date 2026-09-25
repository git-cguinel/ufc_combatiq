"""Resolve local artifacts independently of the current working directory."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def artifact_path(variable: str, relative_path: str) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / relative_path


def fighters_path() -> Path:
    return artifact_path("COMBAT_IQ_FIGHTERS_PATH", "data/processed/fighters.csv")


def model_path() -> Path:
    return artifact_path("COMBAT_IQ_MODEL_PATH", "models/of_model3_acc079468.pkl")
