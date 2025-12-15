"""Training module for garment productivity regression models.

This module loads the cleaned productivity dataset, prepares numeric features,
trains baseline and tuned regression models, and writes artifacts for later use.
It is designed to be imported from notebooks or executed as a script.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEED = 42
TARGET_COLUMN = "actual_productivity"
DEFAULT_PARQUET_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "data"
    / "clean_data"
    / "processed"
    / "df_clean.parquet"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "artifacts"


logger = logging.getLogger(__name__)


def set_random_seeds(seed: int) -> None:
    """Set seeds for python, numpy, and os-level determinism where possible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_df_clean(parquet_path: str | Path = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    """Load cleaned Parquet dataset.

    Args:
        parquet_path: Location of the cleaned parquet file.

    Returns:
        DataFrame containing the cleaned garment productivity data.

    Raises:
        FileNotFoundError: If the parquet file is missing.
        ValueError: If the loaded object is not a DataFrame.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at: {path}")

    df = pd.read_parquet(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded object is not a pandas DataFrame.")
    return df

def _get_numeric_features(df: pd.DataFrame, target: str) -> list[str]:
    """Return numeric feature names excluding the target column."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    features = [col for col in numeric_cols if col != target]
    if not features:
        raise ValueError("No numeric predictor columns available after excluding target.")
    return features


def split_random_numeric(
    df: pd.DataFrame, target: str = TARGET_COLUMN, test_size: float = 0.2, seed: int = DEFAULT_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split numeric predictors and target into train and test sets.

    Args:
        df: Input DataFrame containing features and target.
        target: Name of the target column.
        test_size: Fraction of data to allocate to test split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    features = _get_numeric_features(df, target)
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def create_numeric_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric-only preprocessing.

    Always applies median imputation; optionally adds scaling.

    Args:
        scale: If True, apply StandardScaler after imputation.

    Returns:
        ColumnTransformer configured for numeric features.
    """
    numeric_selector = make_column_selector(dtype_include=np.number)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric_selector)],
        remainder="drop",
    )


def evaluate_regression(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute regression metrics RMSE, MAE, and R2."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _run_baseline_and_tune(
    model_name: str,
    base_pipeline: Pipeline,
    param_dist: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int,
    seed: int,
) -> Tuple[Pipeline, Dict[str, Dict[str, float]]]:
    """Fit baseline, tune with RandomizedSearchCV, and report metrics."""
    logger.info("Training baseline %s model", model_name)
    baseline_model = clone(base_pipeline)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = evaluate_regression(y_test, baseline_pred)

    logger.info("Tuning %s model with RandomizedSearchCV", model_name)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=clone(base_pipeline),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_pred = tuned_model.predict(X_test)
    tuned_metrics = evaluate_regression(y_test, tuned_pred)

    metrics = {"baseline": baseline_metrics, "tuned": tuned_metrics}
    return tuned_model, metrics


def train_and_tune(
    model_name: str,
    pipeline: Pipeline,
    param_dist: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 20,
    seed: int = DEFAULT_SEED,
) -> Tuple[Pipeline, Dict[str, Dict[str, float]]]:
    """Wrapper to train baseline and tuned versions of a regression pipeline."""
    return _run_baseline_and_tune(
        model_name=model_name,
        base_pipeline=pipeline,
        param_dist=param_dist,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=n_iter,
        seed=seed,
    )


def run_training(
    parquet_path: str | Path = DEFAULT_PARQUET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> None:
    """Train GLM-style and LightGBM regressors and persist artifacts."""
    set_random_seeds(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_df_clean(parquet_path)
    X_train, X_test, y_train, y_test = split_random_numeric(
        df, target=TARGET_COLUMN, test_size=0.2, seed=seed
    )

    # GLM-style Ridge model with scaling.
    glm_preprocessor = create_numeric_preprocessor(scale=True)
    glm_pipeline = Pipeline(steps=[("preprocess", glm_preprocessor), ("model", Ridge())])
    glm_param_dist = {"model__alpha": loguniform(1e-4, 100)}

    tuned_glm, glm_metrics = train_and_tune(
        model_name="ridge",
        pipeline=glm_pipeline,
        param_dist=glm_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=20,
        seed=seed,
    )

    # LightGBM model without scaling.
    lgbm_preprocessor = create_numeric_preprocessor(scale=False)
    lgbm_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                lgbm_preprocessor,
            ),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    random_state=seed,
                ),
            ),
        ]
    )
    lgbm_param_dist = {
        "model__n_estimators": randint(200, 1500),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(10, 80),
        "model__min_child_samples": randint(5, 80),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    }

    tuned_lgbm, lgbm_metrics = train_and_tune(
        model_name="lgbm",
        pipeline=lgbm_pipeline,
        param_dist=lgbm_param_dist,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=20,
        seed=seed,
    )

    joblib.dump(tuned_glm, output_path / "glm_model.joblib")
    joblib.dump(tuned_lgbm, output_path / "lgbm_model.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, output_path / "test_data.joblib")

    metrics_payload = {"ridge": glm_metrics, "lgbm": lgbm_metrics}
    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Training complete. Artifacts saved to %s", output_path)


def load_training_outputs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """Load persisted models, test data, and metrics."""
    output_path = Path(output_dir)
    glm_path = output_path / "glm_model.joblib"
    lgbm_path = output_path / "lgbm_model.joblib"
    test_path = output_path / "test_data.joblib"
    metrics_path = output_path / "metrics.json"

    for path in [glm_path, lgbm_path, test_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Expected artifact missing: {path}")

    glm_model = joblib.load(glm_path)
    lgbm_model = joblib.load(lgbm_path)
    test_data = joblib.load(test_path)

    metrics: Dict[str, Any] | None = None
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "glm_model": glm_model,
        "lgbm_model": lgbm_model,
        "test_data": test_data,
        "metrics": metrics,
    }


def main() -> None:
    """Execute full training when run as a script."""
    logging.basicConfig(level=logging.INFO)
    try:
        run_training()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
