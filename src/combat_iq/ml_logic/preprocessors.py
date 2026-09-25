"""Compatibility wrapper for historical notebooks."""

from combat_iq.data import load_fighters
from combat_iq.features import build_features


def preprocessed_df(red_fighter: str, blue_fighter: str):
    return build_features(load_fighters(), red_fighter, blue_fighter)
