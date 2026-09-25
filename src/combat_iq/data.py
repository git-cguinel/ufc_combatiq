"""Validated, cached access to fighter statistics."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from combat_iq.config import fighters_path
from combat_iq.features import STAT_COLUMNS


@lru_cache(maxsize=4)
def _read_fighters(path: Path, mtime_ns: int, size: int) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=["fighter", *STAT_COLUMNS])
    if data["fighter"].isna().any() or data["fighter"].duplicated().any():
        raise ValueError("Fighter data must contain unique, nonempty names.")
    return data.set_index("fighter")


def load_fighters(path: Path | None = None) -> pd.DataFrame:
    """Return a copy so callers cannot mutate the process-wide cache."""
    path = (path or fighters_path()).resolve()
    stat = path.stat()
    return _read_fighters(path, stat.st_mtime_ns, stat.st_size).copy()
