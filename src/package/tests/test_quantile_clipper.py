"""Tests for the QuantileClipper transformer."""

import numpy as np
import pytest

from package.feature_engineering.quantile_clipper import QuantileClipper


@pytest.mark.parametrize(
    "X, lower_q, upper_q, expected",
    [
        # Simple increasing values; expect clipping to 25th and 75th percentiles.
        (
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            0.25,
            0.75,
            np.array(
                [
                    [1.75, 2.75, 3.75],
                    [3.25, 4.25, 5.25],
                ]
            ),
        ),
        # Mix of negatives, zeros, and positives with wider quantile range.
        (
            np.array([[-10.0, 0.0], [0.0, 10.0], [10.0, -5.0]]),
            0.1,
            0.9,
            np.array(
                [
                    [-8.0, -1.0],
                    [0.0, 7.0],
                    [8.0, -1.0],
                ]
            ),
        ),
        # Constant columns should remain unchanged after clipping.
        (
            np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]),
            0.2,
            0.8,
            np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]),
        ),
        # Very small array (single sample) should pass through unchanged.
        (
            np.array([[42.0, -3.0]]),
            0.05,
            0.95,
            np.array([[42.0, -3.0]]),
        ),
    ],
)
def test_quantile_clipper_transformations(X, lower_q, upper_q, expected):
    clipper = QuantileClipper(lower_quantile=lower_q, upper_quantile=upper_q)

    # Preserve copy to ensure input is not mutated.
    X_original = X.copy()

    transformed = clipper.fit_transform(X)

    # Shape is preserved.
    assert transformed.shape == X.shape
    # No mutation of input data.
    np.testing.assert_allclose(X, X_original)
    # Values are clipped as expected.
    np.testing.assert_allclose(transformed, expected)

test_quantile_clipper_transformations()