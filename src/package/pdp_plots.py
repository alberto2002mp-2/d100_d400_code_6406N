from __future__ import annotations

"""Partial dependence plots for the trained LightGBM pipeline."""

import logging
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

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
    """Create and save PDPs for the top LightGBM features, overlaying ElasticNet."""
    artifacts = load_artifacts(output_dir)
    lgbm_model = artifacts["lgbm_model"]
    glm_model = artifacts["glm_model"]
    X_test = artifacts["test_data"]["X_test"]
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    X_test = X_test.copy()
    X_test.columns = X_test.columns.astype(str)

    # Ensure feature name attributes exist to silence validation warnings.
    cols = np.array(X_test.columns, dtype=object)
    def _set_feature_names(obj: Any, col_array: np.ndarray) -> None:
        try:
            setattr(obj, "feature_names_in_", col_array)
            return
        except Exception:
            pass
        try:
            # Pipeline stores internally as _feature_names_in for validation.
            setattr(obj, "_feature_names_in", col_array)
        except Exception:
            logger.debug("Could not set feature names on %s", obj.__class__.__name__)

    _set_feature_names(lgbm_model, cols)
    _set_feature_names(glm_model, cols)
    if hasattr(lgbm_model, "named_steps") and "model" in lgbm_model.named_steps:
        model_step = lgbm_model.named_steps["model"]
        _set_feature_names(model_step, cols)
        if hasattr(model_step, "feature_name_"):
            try:
                model_step.feature_name_ = list(X_test.columns)
            except Exception:
                logger.debug("Could not set feature_name_ on LGBM model step.")
    if hasattr(glm_model, "named_steps") and "model" in glm_model.named_steps:
        _set_feature_names(glm_model.named_steps["model"], cols)

    top_features, importances = top_lgbm_features(top_n=top_n, output_dir=output_dir)
    missing = [feat for feat in top_features if feat not in X_test.columns]
    if missing:
        raise ValueError(f"Features missing from X_test: {missing}")

    logger.info("Top %s features for PDP: %s", len(top_features), top_features)

    n_features = len(top_features)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        # Compute PDP values explicitly for reliable overlay.
        lgbm_pd = partial_dependence(lgbm_model, X_test, features=[feature], kind=kind)
        glm_pd = partial_dependence(glm_model, X_test, features=[feature], kind=kind)

        # Handle sklearn version differences: values vs grid_values.
        def _grid(bunch):
            return bunch.get("values") or bunch.get("grid_values")

        lgbm_grid = _grid(lgbm_pd)[0]
        glm_grid = _grid(glm_pd)[0]

        lgbm_avg = np.ravel(lgbm_pd["average"][0])
        glm_avg = np.ravel(glm_pd["average"][0])

        # If grids differ, interpolate GLM onto the LGBM grid.
        if not np.array_equal(lgbm_grid, glm_grid):
            glm_interp = np.interp(lgbm_grid, glm_grid, glm_avg)
            glm_x = lgbm_grid
            glm_y = glm_interp
        else:
            glm_x = glm_grid
            glm_y = glm_avg

        ax.plot(lgbm_grid, lgbm_avg, color="blue", label="LGBM", linewidth=2)
        ax.plot(glm_x, glm_y, color="orange", label="GLM", linewidth=2)

        ax.set_title(f"PDP: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial Dependence")
        ax.legend()

    # Hide any unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    pdp_dir = Path(output_dir) / "pdp"
    pdp_dir.mkdir(parents=True, exist_ok=True)
    plot_path = pdp_dir / "pdp_lgbm_glm.png"
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
