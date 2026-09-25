"""Explicit training workflow; never runs when importing the API."""

import logging
import pickle
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from combat_iq.config import PROJECT_ROOT, training_path
from combat_iq.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def train_pipeline(
    data: pd.DataFrame | None = None,
    *,
    output_path: Path | None = None,
    iterations: int = 2500,
    cv_folds: int = 5,
) -> Pipeline:
    """Train a new binary model using the same feature order as inference.

    This fixes the old training routine; it does not reproduce or overwrite
    the historical fitted pipeline. Non-Red outcomes retain label 0.
    Set cv_folds=0 to skip cross-validation during a smoke test.
    """
    data = pd.read_csv(training_path()) if data is None else data
    features = data.loc[:, FEATURE_COLUMNS].copy()
    for column in ("R_fighter", "B_fighter"):
        features[column] = features[column].fillna("Unknown").astype(str)
    target = data["Winner"].eq("Red").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, train_size=0.8, random_state=42, stratify=target
    )
    # CatBoost handles numeric missing values and zeros directly. Selecting
    # categorical columns by name avoids indices changing in a transformer.
    pipeline = Pipeline(
        [
            (
                "classifier",
                CatBoostClassifier(
                    iterations=iterations,
                    depth=5,
                    learning_rate=0.04,
                    cat_features=["R_fighter", "B_fighter"],
                    eval_metric="AUC",
                    random_seed=42,
                    silent=True,
                    allow_writing_files=False,
                ),
            )
        ]
    )
    if cv_folds:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        score = cross_val_score(pipeline, x_train, y_train, scoring="accuracy", cv=cv).mean()
        logger.info("Cross-validation accuracy: %.4f", score)
    pipeline.fit(x_train, y_train)
    logger.info("Test accuracy: %.4f", accuracy_score(y_test, pipeline.predict(x_test)))
    path = output_path or PROJECT_ROOT / "models/trained_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(pipeline, stream)
    return pipeline
