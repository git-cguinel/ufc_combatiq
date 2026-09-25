"""Legacy full-table reader; inference uses combat_iq.data instead."""

import pandas as pd

from combat_iq.config import fighters_path


def get_data():
    return pd.read_csv(fighters_path())
