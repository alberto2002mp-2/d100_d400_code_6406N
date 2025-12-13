import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns


def distributions_graphs(df: pd.DataFrame) -> None:
    """
    Plots histograms for numeric columns and bar charts for categorical columns
    using Plotly.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_binary_numeric = [c for c in numeric_cols if df[c].nunique() > 2]

    for col in non_binary_numeric:
        fig = px.histogram(
            df,
            x=col,
            nbins=50,
            marginal="box",
            title=f"Distribution of {col}",
        )
        fig.show()

    categorical_cols = df.select_dtypes(include=["bool", "object", "category"]).columns

    for col in categorical_cols:
        value_counts = df[col].value_counts(dropna=False).reset_index()
        value_counts.columns = [col, "Count"]

        fig = px.bar(
            value_counts,
            x=col,
            y="Count",
            title=f"Distribution of {col}",
        )
        fig.update_layout(xaxis_tickangle=45)
        fig.show()
