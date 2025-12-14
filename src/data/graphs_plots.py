import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from data.load_data import load_dataframe

df = load_dataframe()

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



def numeric_boxplot_graphs(df: pd.DataFrame) -> None:
    """
    Visualizes numeric columns using Plotly boxplots.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found to visualize.")
        return

    for col in numeric_cols:
        fig = px.box(
            df,
            y=col,
            title=f"Boxplot of {col}",
            points="outliers",
        )
        fig.show()

def target_distribution_graphs(df: pd.DataFrame, target: str = "actual_productivity") -> None:
    """
    Plots the distribution of the target variable using Plotly.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    value_counts = df[target].value_counts(dropna=False).reset_index()
    value_counts.columns = [target, "Count"]

    fig = px.bar(
        value_counts,
        x=target,
        y="Count",
        title=f"Distribution of {target}",
        labels={"x": target, "Count": "Count"},
    )
    fig.update_layout(xaxis_tickangle=45)
    fig.show()


def feature_correlations_graphs(
    df: pd.DataFrame, target: str = "actual_productivity", title_suffix: str = ""
) -> None:
    """
    Plots a horizontal bar chart of feature correlations with the target variable.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    # Compute correlations against the target from numeric columns only.
    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        raise TypeError(f"Target column '{target}' is not numeric.")

    correlations = numeric_df.corr(numeric_only=True)[target].drop(target).dropna()
    if correlations.empty:
        print("No numeric features to correlate with the target.")
        return

    correlations_sorted = correlations.sort_values()

    fig = go.Figure(
        go.Bar(
            x=correlations_sorted.values,
            y=correlations_sorted.index,
            orientation="h",
            text=[f"{v:.3f}" for v in correlations_sorted.values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Feature Correlations with {target} {title_suffix}",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Features",
        height=max(400, 25 * len(correlations_sorted)),
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=len(correlations_sorted) - 0.5,
                line=dict(color="black", width=1),
            )
        ],
    )
    fig.show()



def correlation_matrix_graph(df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """
    Plot a correlation matrix for numeric columns with red (-1) to green (+1) scaling.

    Diagonal cells are blanked out because they are always 1.0.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        print("Need at least two numeric columns to compute a correlation matrix.")
        return

    corr = numeric_df.corr(numeric_only=True)
    if corr.empty:
        print("No numeric correlations available.")
        return

    # Blank the diagonal so those cells remain empty in the heatmap.
    corr_no_diag = corr.copy()
    np.fill_diagonal(corr_no_diag.values, np.nan)

    text_values = corr_no_diag.round(2).astype(str)
    text_values = text_values.where(~np.eye(text_values.shape[0], dtype=bool), "")

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_no_diag.values,
            x=corr_no_diag.columns,
            y=corr_no_diag.index,
            colorscale=[[0, "#d73027"], [0.5, "#ffffff"], [1, "#1a9850"]],
            zmin=-1,
            zmax=1,
            colorbar_title="Correlation",
            text=text_values.values,
            texttemplate="%{text}",
            hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        xaxis={"side": "bottom"},
        height=max(500, 35 * len(corr_no_diag.columns)),
    )
    fig.show()



def categorical_stack_graphs(
    df: pd.DataFrame, target: str = "actual_productivity"
) -> None:
    """
    Plots 100% stacked bar charts for categorical features using Plotly.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target in cat_features:
        cat_features.remove(target)

    if not cat_features:
        print("No categorical features to plot against the target.")
        return

    for feature in cat_features:
        crosstab = pd.crosstab(df[feature], df[target], normalize="index")

        fig = go.Figure()
        for col in crosstab.columns:
            fig.add_bar(
                x=crosstab.index.astype(str),
                y=crosstab[col],
                name=str(col),
            )

        fig.update_layout(
            barmode="stack",
            title=f"{feature} Distribution by {target}",
            yaxis_title="Proportion",
            xaxis_tickangle=45,
        )
        fig.show()

