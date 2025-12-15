"""Custom quantile-based clipping transformer for numeric arrays.

The transformer trims extreme values by capping each column at its
empirically estimated lower and upper quantiles. It is designed to
help linear models cope with outliers while retaining interpretability.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to column-wise quantile thresholds.

    Parameters
    ----------
    lower_quantile : float, default=0.01
        Lower quantile used to cap small values. Must satisfy 0 <= lower_quantile < upper_quantile <= 1.
    upper_quantile : float, default=0.99
        Upper quantile used to cap large values. Must satisfy lower_quantile < upper_quantile <= 1.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "QuantileClipper":
        """Estimate per-column clipping thresholds from the input array.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional numeric array (n_samples, n_features).
        y : ignored, optional
            Present for API consistency by convention.

        Returns
        -------
        QuantileClipper
            Fitted instance with learned bounds.

        Raises
        ------
        ValueError
            If quantile parameters are invalid or input is not two-dimensional.
        """
        if not (0.0 <= self.lower_quantile < self.upper_quantile <= 1.0):
            raise ValueError(
                "Quantiles must satisfy 0 <= lower_quantile < upper_quantile <= 1. "
                f"Got lower_quantile={self.lower_quantile}, upper_quantile={self.upper_quantile}."
            )

        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError(f"Input X must be 2D (n_samples, n_features); got shape {X_array.shape}.")

        # Compute column-wise quantiles; robust to constant columns and small sample sizes.
        self.lower_bounds_ = np.quantile(X_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X_array, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip each column of X to the learned quantile thresholds.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional numeric array to transform.

        Returns
        -------
        np.ndarray
            Array with values clipped to the fitted lower and upper bounds.

        Raises
        ------
        ValueError
            If the transformer is not fitted or input shape is incompatible.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("QuantileClipper is not fitted yet. Call 'fit' before 'transform'.")

        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError(f"Input X must be 2D (n_samples, n_features); got shape {X_array.shape}.")
        if X_array.shape[1] != self.lower_bounds_.shape[0]:
            raise ValueError(
                "Input has different number of features than during fit. "
                f"Expected {self.lower_bounds_.shape[0]}, got {X_array.shape[1]}."
            )

        # np.clip creates a new array, preserving dtype and shape.
        clipped = np.clip(X_array, self.lower_bounds_, self.upper_bounds_)
        return clipped

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> np.ndarray:
        """Fit to data, then transform it. Included for type hint clarity."""
        return super().fit_transform(X, y=y, **fit_params)
