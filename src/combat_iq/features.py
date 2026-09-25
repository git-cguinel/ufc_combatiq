"""The ordered input contract shared by training and prediction."""

import pandas as pd

STAT_COLUMNS = (
    "total_rounds_fought",
    "total_title_bouts",
    "current_win_streak",
    "current_lose_streak",
    "longest_win_streak",
    "wins",
    "losses",
    "draw",
    "win_by_Decision_Majority",
    "win_by_Decision_Split",
    "win_by_Decision_Unanimous",
    "win_by_KO/TKO",
    "win_by_Submission",
    "win_by_TKO_Doctor_Stoppage",
)
FEATURE_COLUMNS = ("R_fighter", "B_fighter") + tuple(
    f"{corner}_{stat}" for corner in ("B", "R") for stat in STAT_COLUMNS
)


class InvalidMatchup(ValueError):
    """The requested fighters do not form a valid matchup."""


def build_features(fighters: pd.DataFrame, red: str, blue: str) -> pd.DataFrame:
    """Build one model input from an indexed, validated fighter table."""
    if not red.strip() or not blue.strip():
        raise InvalidMatchup("Fighter names must not be blank.")
    if red == blue:
        raise InvalidMatchup("Select two different fighters.")
    for name in (red, blue):
        if name not in fighters.index:
            raise InvalidMatchup(f"Unknown fighter: {name}")
    row = {"R_fighter": red, "B_fighter": blue}
    for corner, name in (("B", blue), ("R", red)):
        row.update({f"{corner}_{stat}": fighters.at[name, stat] for stat in STAT_COLUMNS})
    # The historical pipeline selects columns by dtype. Object dtype is part
    # of its input contract; changing it alters the persisted model's behavior.
    return pd.DataFrame([row], columns=FEATURE_COLUMNS, dtype=object)
