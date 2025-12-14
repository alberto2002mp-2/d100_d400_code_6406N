"""Utilities for loading the UCI stock dataset (id=597).

This module exposes a single function that returns the dataset as a
``pandas.DataFrame`` so it can be imported and reused across notebooks or
other scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

__all__ = [
    "load_stocks_dataframe",
    "load_stock_dataframe",
    "load_local_productivity_dataframe",
    "load_dataframe",
]


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


def load_stock_dataframe() -> pd.DataFrame:
    """Alias for ``load_stocks_dataframe`` to match the expected import name."""

    return load_stocks_dataframe()


def load_local_productivity_dataframe() -> pd.DataFrame:
    """Read the bundled garments worker productivity CSV relative to the repo root."""

    csv_path = Path(__file__).resolve().parents[2] / "garments_worker_productivity.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find bundled CSV at {csv_path}")

    return pd.read_csv(csv_path)


def load_dataframe() -> pd.DataFrame:
    """Load the stock dataset, falling back to the local CSV if fetching fails."""

    try:
        return load_stocks_dataframe()
    except Exception:
        return load_local_productivity_dataframe()


if __name__ == "__main__":
    df = load_dataframe()
    print(df.head())
