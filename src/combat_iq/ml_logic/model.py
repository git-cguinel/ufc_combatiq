"""Compatibility aliases, including the original train_pipline spelling."""

from combat_iq.prediction import predict
from combat_iq.training import train_pipeline

train_pipline = train_pipeline
__all__ = ["predict", "train_pipeline", "train_pipline"]
