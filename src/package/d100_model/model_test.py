import pandas as pd
df1 = pd.read_parquet("package/data/processed/df_clean.parquet")

from typing import Tuple
from sklearn.model_selection import train_test_split


def split_numeric_data_random(
    df: pd.DataFrame,
    target: str = "actual_productivity",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Randomly split the dataframe into train/test sets using ONLY numeric features
    (int64 and float64) to predict the target variable.

    Returns:
    X_train, X_test, y_train, y_test
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if target not in numeric_df.columns:
        raise ValueError("Target column must be numeric to be used for modelling.")

    # Split features and target
    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_numeric_data_random(df1)

print(X_train.shape, X_test.shape)
