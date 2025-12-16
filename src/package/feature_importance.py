from __future__ import annotations

"""Feature importance reporting for saved regression models."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd

from package.models.model_training import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
)

logger = logging.getLogger(__name__)


def load_artifacts(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """Load trained models and saved test data."""
    output_path = Path(output_dir)
    glm_model = joblib.load(output_path / "glm_model.joblib")
    lgbm_model = joblib.load(output_path / "lgbm_model.joblib")
    test_data = joblib.load(output_path / "test_data.joblib")
    if not isinstance(test_data, dict) or "X_test" not in test_data:
        raise ValueError("test_data.joblib must contain X_test with feature columns.")
    return {"glm_model": glm_model, "lgbm_model": lgbm_model, "test_data": test_data}


def get_feature_names(preprocess: Any, input_features: Iterable[str]) -> list[str]:
    """Return feature names from a ColumnTransformer or fall back to the input names."""
    try:
        names = preprocess.get_feature_names_out(input_features)  # type: ignore[attr-defined]
    except Exception:
        try:
            names = preprocess.get_feature_names_out()  # type: ignore[attr-defined]
        except Exception:
            names = list(input_features)
    return list(names)


def elasticnet_importance(glm_pipeline: Any, feature_names: list[str]) -> pd.DataFrame:
    """Compute ElasticNet coefficients with names and absolute magnitude ranking."""
    model = glm_pipeline.named_steps["model"]
    coefs = np.asarray(model.coef_).ravel()
    if len(coefs) != len(feature_names):
        logger.warning(
            "Coefficient length (%s) and feature length (%s) mismatch; truncating to shortest.",
            len(coefs),
            len(feature_names),
        )
    limit = min(len(coefs), len(feature_names))
    df = pd.DataFrame(
        {
            "feature": feature_names[:limit],
            "coefficient": coefs[:limit],
        }
    )
    df["importance"] = df["coefficient"].abs()
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def lgbm_importance(lgbm_pipeline: Any, feature_names: list[str]) -> pd.DataFrame:
    """Compute LightGBM feature importances aligned to preprocessed features."""
    model = lgbm_pipeline.named_steps["model"]
    importances = np.asarray(model.feature_importances_).ravel()
    if len(importances) != len(feature_names):
        logger.warning(
            "Importance length (%s) and feature length (%s) mismatch; truncating to shortest.",
            len(importances),
            len(feature_names),
        )
    limit = min(len(importances), len(feature_names))
    df = pd.DataFrame(
        {
            "feature": feature_names[:limit],
            "importance": importances[:limit],
        }
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def summarize_differences(glm_df: pd.DataFrame, lgbm_df: pd.DataFrame, top_n: int = 10) -> str:
    """Report top feature sets and differences between models."""
    glm_top = set(glm_df.head(top_n)["feature"])
    lgbm_top = set(lgbm_df.head(top_n)["feature"])
    only_glm = glm_top - lgbm_top
    only_lgbm = lgbm_top - glm_top
    both = glm_top & lgbm_top

    lines = [
        f"Top {top_n} overlap: {sorted(both)}",
        f"Only ElasticNet: {sorted(only_glm)}",
        f"Only LightGBM: {sorted(only_lgbm)}",
    ]
    return "\n".join(lines)


def main(output_dir: str | Path = DEFAULT_OUTPUT_DIR, top_n: int = 10) -> None:
    logging.basicConfig(level=logging.INFO)
    artifacts = load_artifacts(output_dir)
    glm_pipeline = artifacts["glm_model"]
    lgbm_pipeline = artifacts["lgbm_model"]
    X_test = artifacts["test_data"]["X_test"]
    feature_names = get_feature_names(glm_pipeline.named_steps["preprocess"], X_test.columns)

    glm_df = elasticnet_importance(glm_pipeline, feature_names)
    lgbm_df = lgbm_importance(lgbm_pipeline, feature_names)

    print(f"\nTop {top_n} ElasticNet coefficients (abs sorted):")
    print(glm_df.head(top_n).to_string(index=False))

    print(f"\nTop {top_n} LightGBM importances:")
    print(lgbm_df.head(top_n).to_string(index=False))

    print("\nDifferences:")
    print(summarize_differences(glm_df, lgbm_df, top_n=top_n))


if __name__ == "__main__":
    main()
