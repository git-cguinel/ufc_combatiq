"""HTTP interface; business logic lives in the prediction service."""

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from combat_iq import prediction
from combat_iq.features import InvalidMatchup

logger = logging.getLogger(__name__)
app = FastAPI(title="UFC Combat IQ")


class PredictionResponse(BaseModel):
    model_pick: str = Field(description="Fighter shown as the model pick; non-red maps to blue.")
    model_confidence: float = Field(
        ge=0, le=1, description="Uncalibrated score for predicted_outcome, not a win probability."
    )
    predicted_outcome: Literal["red_win", "non_red_win"] = Field(
        description="Original target: red wins or non-red outcome (including draws)."
    )
    winner: str = Field(deprecated=True, description="Legacy alias for model_pick.")
    win_probability: float = Field(
        deprecated=True,
        description="Legacy alias for model_confidence; misleading historical name.",
    )
    fight_outcome: str = Field(
        deprecated=True,
        description="Legacy display text; consult predicted_outcome for exact meaning.",
    )
    confidence_rate: float = Field(
        deprecated=True, description="Legacy alias for model_confidence."
    )


@app.get("/")
def root():
    return {"setup": "I'm on it dudes !"}


@app.get("/predict", response_model=PredictionResponse)
def predict(red_fighter: str = Query(min_length=1), blue_fighter: str = Query(min_length=1)):
    try:
        return prediction.predict(red_fighter, blue_fighter)
    except InvalidMatchup as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.exception("A required prediction artifact is missing")
        raise HTTPException(
            status_code=503, detail="Prediction artifacts are unavailable."
        ) from exc
