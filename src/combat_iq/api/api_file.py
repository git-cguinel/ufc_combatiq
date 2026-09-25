"""Compatibility entry point for existing Uvicorn commands."""

from combat_iq.api.app import app, predict, root

__all__ = ["app", "predict", "root"]
