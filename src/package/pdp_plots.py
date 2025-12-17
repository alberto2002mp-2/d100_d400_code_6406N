from __future__ import annotations

"""Partial dependence plots for the trained LightGBM pipeline."""

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from package.feature_importance import (
    get_feature_names,
    lgbm_importance,
    load_artifacts,
)
from package.models.model_training import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def _raw_feature_name(name: str) -> str:
    """Strip ColumnTransformer prefixes like 'numeric__' to recover original column name."""
    return name.split("__", 1)[1] if "__" in name else name


def top_lgbm_features(top_n: int = 5, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Tuple[list[str], pd.DataFrame]:
    """Return the top N raw feature names from LightGBM importances."""
    artifacts = load_artifacts(output_dir)
    lgbm_model = artifacts["lgbm_model"]
    X_test = artifacts["test_data"]["X_test"]

    feature_names = get_feature_names(lgbm_model.named_steps["preprocess"], X_test.columns)
    importances = lgbm_importance(lgbm_model, feature_names)

    # Map back to raw feature names and keep order while removing duplicates.
    raw_features: List[str] = []
    for feat in importances["feature"]:
        raw = _raw_feature_name(feat)
        if raw not in raw_features:
            raw_features.append(raw)
        if len(raw_features) >= top_n:
            break

    return raw_features, importances


def plot_lgbm_pdp(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 5,
    kind: str = "average",
) -> Tuple[plt.Figure, Path, list[str]]:
    """Create and save PDPs for the top LightGBM features."""
    artifacts = load_artifacts(output_dir)
    lgbm_model = artifacts["lgbm_model"]
    X_test = artifacts["test_data"]["X_test"]
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    X_test = X_test.copy()  # avoid SettingWithCopy in downstream steps
    if not hasattr(lgbm_model, "feature_names_in_"):
        lgbm_model.feature_names_in_ = np.array(X_test.columns, dtype=object)

    top_features, importances = top_lgbm_features(top_n=top_n, output_dir=output_dir)
    missing = [feat for feat in top_features if feat not in X_test.columns]
    if missing:
        raise ValueError(f"Features missing from X_test: {missing}")

    logger.info("Top %s features for PDP: %s", len(top_features), top_features)

    disp = PartialDependenceDisplay.from_estimator(
        lgbm_model,
        X_test,
        features=top_features,
        kind=kind,
    )
    fig = disp.figure_

    pdp_dir = Path(output_dir) / "pdp"
    pdp_dir.mkdir(parents=True, exist_ok=True)
    plot_path = pdp_dir / "lgbm_pdp.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    return fig, plot_path, top_features


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    fig, path, features = plot_lgbm_pdp()
    print(f"Saved PDP for features {features} -> {path}")
    plt.show(block=False)


if __name__ == "__main__":
    main()
