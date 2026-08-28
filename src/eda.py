"""
Exploratory Data Analysis (EDA) reproducing the visualisations described in
Sections 3.2 - 3.6 of Bhuria et al. (2025).
"""

from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Churn_Modelling.csv"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 3.2 - Univariate analysis
# ---------------------------------------------------------------------------
def one_dimensional_analysis(df: pd.DataFrame, fig_dir: Path = FIG_DIR) -> None:
    numeric_cols = [
        "RowNumber",
        "CustomerId",
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    # Figure 3 - histograms of every numeric column.
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_histograms.png", dpi=150)
    plt.close(fig)

    # Figure 4 - count plots for the categorical columns.
    for col, fname in [
        ("Surname", "fig04a_surname.png"),
        ("Geography", "fig04b_geography.png"),
        ("Gender", "fig04c_gender.png"),
    ]:
        top = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        top.plot(kind="barh", ax=ax, color="teal")
        ax.set_title(f"Count of {col}")
        ax.set_xlabel("Count")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3.3 - Bivariate analysis (box plots vs Exited)
# ---------------------------------------------------------------------------
def two_dimensional_analysis(df: pd.DataFrame, fig_dir: Path = FIG_DIR) -> None:
    cols = [
        "RowNumber",
        "CustomerId",
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x="Exited", y=col, ax=ax, palette=["#2ca02c", "#ff7f0e"])
        ax.set_title(f"{col} vs Exited")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_boxplots.png", dpi=150)
    plt.close(fig)

    # Figure 6 - count plots of categorical columns vs Exited
    for col, fname in [
        ("Surname", "fig06a_surname.png"),
        ("Geography", "fig06b_geography.png"),
        ("Gender", "fig06c_gender.png"),
    ]:
        top = df[col].value_counts().head(10).index
        sub = df[df[col].isin(top)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(
            data=sub, y=col, hue="Exited", palette=["#2ca02c", "#ff7f0e"], ax=ax
        )
        ax.set_title(f"{col} vs Exited")
        ax.legend(title="Exited", loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3.4 - Multivariate analysis
# ---------------------------------------------------------------------------
def multi_dimensional_analysis(df: pd.DataFrame, fig_dir: Path = FIG_DIR) -> None:
    pairs = [
        ("CreditScore", "Balance"),
        ("Age", "EstimatedSalary"),
        ("Tenure", "NumOfProducts"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (x, y) in zip(axes, pairs):
        for cls, color in zip([0, 1], ["#2ca02c", "#ff7f0e"]):
            sub = df[df["Exited"] == cls]
            ax.scatter(sub[x], sub[y], c=color, alpha=0.3, label=f"Exited={cls}", s=8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_scatter.png", dpi=150)
    plt.close(fig)

    # Correlation matrix heat map
    numeric = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix - numerical features")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig08_correlation.png", dpi=150)
    plt.close(fig)

    # Figure 9 - churn rate by category
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Age groups
    bins = [0, 30, 40, 50, 60, 70, 80, 90]
    labels = ["0-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"]
    age_buckets = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    churn_by_age = df.groupby(age_buckets)["Exited"].mean()
    axes[0, 0].bar(churn_by_age.index.astype(str), churn_by_age.values, color="teal")
    axes[0, 0].set_title("Churn rate by age group")
    axes[0, 0].set_ylabel("Churn rate")

    # (b) Geography
    geo = df.groupby("Geography")["Exited"].mean()
    axes[0, 1].bar(geo.index, geo.values, color="teal")
    axes[0, 1].set_title("Churn rate by Geography")
    axes[0, 1].set_ylabel("Churn rate")

    # (c) Gender
    gen = df.groupby("Gender")["Exited"].mean()
    axes[1, 0].bar(gen.index, gen.values, color="teal")
    axes[1, 0].set_title("Churn rate by Gender")
    axes[1, 0].set_ylabel("Churn rate")

    # (d) NumOfProducts
    prod = df.groupby("NumOfProducts")["Exited"].mean()
    axes[1, 1].bar(prod.index.astype(str), prod.values, color="teal")
    axes[1, 1].set_title("Churn rate by NumOfProducts")
    axes[1, 1].set_ylabel("Churn rate")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig09_churn_rates.png", dpi=150)
    plt.close(fig)

    # Figure 10 - box plots of Balance / EstimatedSalary vs Exited
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ["Balance", "EstimatedSalary"]):
        sns.boxplot(data=df, x="Exited", y=col, ax=ax, palette=["#2ca02c", "#ff7f0e"])
        ax.set_title(f"{col} by Exited")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_balance_salary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3.5/3.6 - Outlier detection + IQR removal
# ---------------------------------------------------------------------------
def outlier_analysis(df: pd.DataFrame, fig_dir: Path = FIG_DIR) -> None:
    cols = [
        "RowNumber",
        "CustomerId",
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    # Figure 11 - full box plots before IQR
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col])
        ax.set_title(f"Boxplot of {col}")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_boxplots_before.png", dpi=150)
    plt.close(fig)

    # Figure 12 - box plots of CreditScore, Age, NumOfProducts BEFORE / AFTER IQR
    for col in ["CreditScore", "Age", "NumOfProducts"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        clean = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].boxplot(df[col])
        axes[0].set_title(f"{col} - BEFORE IQR")
        axes[1].boxplot(clean[col])
        axes[1].set_title(f"{col} - AFTER IQR")
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig12_{col}_iqr.png", dpi=150)
        plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    one_dimensional_analysis(df)
    print("Saved fig03 + fig04 (univariate)")

    two_dimensional_analysis(df)
    print("Saved fig05 + fig06 (bivariate)")

    multi_dimensional_analysis(df)
    print("Saved fig07, fig08, fig09, fig10 (multivariate)")

    outlier_analysis(df)
    print("Saved fig11 + fig12 (outliers / IQR)")

    print("\nAll EDA figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
