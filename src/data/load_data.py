"""Utilities for loading the UCI stock dataset (id=597).

This module exposes a single function that returns the dataset as a
``pandas.DataFrame`` so it can be imported and reused across notebooks or
other scripts.
"""

from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo

stocks = fetch_ucirepo(id=597)


print(stocks.variables)

def load_stocks_dataframe() -> pd.DataFrame:
    """Return the UCI stock dataset (id=597) as a ``pandas.DataFrame``.

    The dataset is fetched with ``ucimlrepo.fetch_ucirepo`` and the available
    feature and target columns are combined into a single DataFrame, making it
    convenient to import and reuse across notebooks.
    """

    stocks = fetch_ucirepo(id=597)

    # ``stocks.data.features`` and ``stocks.data.targets`` are already DataFrames
    # (or a Series for single targets). Combine them when targets are present so
    # callers can work with a single table.
    features = stocks.data.features
    targets = stocks.data.targets

    if targets is None or targets.empty:
        return features.copy()

    targets_df = targets if isinstance(targets, pd.DataFrame) else targets.to_frame()
    return pd.concat([features, targets_df], axis=1)


if __name__ == "__main__":
    df = load_stocks_dataframe()
    print(df.head())