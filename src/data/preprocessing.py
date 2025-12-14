import numpy as np
import pandas as pd

from data.load_data import load_dataframe

df = load_dataframe()

df

def mean_actual_productivity_by_team(df: pd.DataFrame) -> pd.Series:
    """
    Compute the mean of actual_productivity for each team.
    """
    required_cols = {"team", "actual_productivity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return (
        df.groupby("team", dropna=False)["actual_productivity"]
        .mean()
        .sort_index()
    )

mean_actual_productivity_by_team(df)