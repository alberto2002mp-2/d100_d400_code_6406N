"""Integration-like test for QuantileClipper using the cleaned parquet dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from package.feature_engineering.quantile_clipper import QuantileClipper


def _load_numeric_from_parquet(parquet_path: Path, target: str = "actual_productivity") -> np.ndarray:
    """Load numeric features from the cleaned parquet file as a NumPy array."""
    if not parquet_path.is_file():
        pytest.skip(f"Cleaned parquet not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if target not in df.columns:
        pytest.skip(f"Target column '{target}' missing in parquet data.")

    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    feature_cols = [c for c in numeric_cols if c != target]
    if not feature_cols:
        pytest.skip("No numeric feature columns found in parquet data.")

    return df[feature_cols].to_numpy()


def test_quantile_clipper_on_parquet_data():
    parquet_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "clean_data"
        / "processed"
        / "df_clean.parquet"
    )

    X = _load_numeric_from_parquet(parquet_path)
    clipper = QuantileClipper(lower_quantile=0.05, upper_quantile=0.95)

    X_original = X.copy()
    transformed = clipper.fit_transform(X)

    # Shape and immutability checks.
    assert transformed.shape == X.shape
    np.testing.assert_allclose(X, X_original)

    # Expected clipping bounds computed directly for verification.
    lower_bounds = np.quantile(X, 0.05, axis=0)
    upper_bounds = np.quantile(X, 0.95, axis=0)

    # Ensure each column is clipped within the expected bounds (allow small tolerance).
    assert np.all(transformed >= lower_bounds - 1e-9)
    assert np.all(transformed <= upper_bounds + 1e-9)

    # Values originally inside the quantile range should remain unchanged where applicable.
    within_bounds_mask = (X >= lower_bounds) & (X <= upper_bounds)
    np.testing.assert_allclose(transformed[within_bounds_mask], X[within_bounds_mask])
