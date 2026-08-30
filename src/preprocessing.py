"""
Data quality pipeline reproducing Sections 3.1, 3.5 and 3.6 of
Bhuria et al. (2025) - "Ensemble-based customer churn prediction in banking".

Steps performed in the same order described in the paper:

1. Section 3.1 - "missing data management": verify there are no NaN
   values; document any handling.
2. Section 3.1 - drop identifiers: RowNumber, CustomerId, Surname.
3. Section 3.1 - one-hot encoding for categorical variables (Geography)
   and binary encoding for Gender.
4. Section 3.1 - feature scaling on numerical features.
5. Section 3.5 - boxplots for every numerical column (Figure 11).
6. Section 3.6 - IQR outlier removal on CreditScore, Age and
   NumOfProducts (Figure 12) with the canonical factor=1.5.
7. Section 3.1 - 70/30 stratified split reproduces the paper's train/test
   sizes, then SMOTE is applied so that the minority class grows from
   ~30% of the train set to match the 8,000 samples claimed in the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Churn_Modelling.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ensure_dirs() -> None:
    """Make sure every output directory exists."""
    for path in (OUT_DIR, FIG_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(path=None) -> pd.DataFrame:
    """Backwards-compatible loader."""
    return pd.read_csv(DATA_PATH if path is None else path)


def load_raw() -> pd.DataFrame:
    """Load the original Kaggle CSV (paper Section 3.1 input)."""
    return pd.read_csv(DATA_PATH)

TARGET = "Exited"
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_COLS = ["Geography", "Gender"]
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# Outliers are removed from these three columns only (paper Section 3.6).
IQR_COLUMNS = ["CreditScore", "Age", "NumOfProducts"]


def load_raw() -> pd.DataFrame:
    """Load the original Kaggle CSV."""
    return pd.read_csv(DATA_PATH)


def report_missing(df: pd.DataFrame) -> dict:
    """Section 3.1: missing-data management.  Returns a report dict."""
    n_missing = int(df.isnull().sum().sum())
    return {"total_missing": n_missing, "by_column": df.isnull().sum().to_dict()}


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Section 3.1: drop RowNumber, CustomerId, Surname (high-cardinality)."""
    return df.drop(columns=DROP_COLS).copy()


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Section 3.1: one-hot encode Geography, binary encode Gender.

    The paper says "one-hot encoding for categorical variables". For the
    binary Gender column we use a single 0/1 column (Female = 1) which is
    functionally equivalent to one-hot + drop_first.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    # Geography dummy columns become Geography_France, Geography_Germany,
    # Geography_Spain. Cast to int for downstream scalers.
    for col in [c for c in df.columns if c.startswith("Geography_")]:
        df[col] = df[col].astype(int)
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    return df


def feature_scaling(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Section 3.1: feature scaling with StandardScaler fitted on train."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def iqr_remove_outliers(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """Section 3.6: IQR outlier removal.

    For each column in ``columns``, compute Q1, Q3, IQR.  A row is kept
    only if its value in every column falls inside [Q1 - factor*IQR,
    Q3 + factor*IQR].  This matches the paper's "any outliers in
    CreditScore / Age / NumOfProducts will be removed" wording.
    """
    df = df.copy()
    keep = pd.Series(True, index=df.index)
    bounds = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        bounds[col] = (lo, hi)
        keep &= df[col].between(lo, hi)
    df.attrs["iqr_bounds"] = bounds
    return df.loc[keep].copy()


def drop_top_rows(df: pd.DataFrame, n_drop: int) -> pd.DataFrame:
    """Drop the first ``n_drop`` rows.  Useful for matching the paper's
    CM-derived test size (the paper's CM sums to 1,970 = 0.20 * 9,850,
    implying 150 rows were dropped before the 80/20 split)."""
    if n_drop <= 0:
        return df
    return df.iloc[n_drop:].reset_index(drop=True).copy()


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.30,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """70/30 stratified train/test split.

    Set ``shuffle=False`` to take the last ``test_size`` fraction of rows
    as the test set without shuffling; this reproduces the kind of
    non-shuffled split that some papers accidentally perform.
    """
    from sklearn.model_selection import train_test_split

    if shuffle:
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    n = len(X)
    cut = int(round(n * (1.0 - test_size)))
    return X[:cut], X[cut:], y[:cut], y[cut:]


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_minority: Optional[int] = 8000,
    random_state: int = 42,
):
    """Section 3.1: SMOTE minority -> 8,000 samples (matches paper text).

    The paper says "By boosting the number of lost consumers, SMOTE will
    help to balance the dataset from 2,037 to 8,000. The dataset will
    have 7,963 non-churned and 8,000 churned consumers following SMote,
    so generating 15,963 rows total."

    Pass ``target_minority=None`` for plain balanced SMOTE on the train
    set only.
    """
    from imblearn.over_sampling import SMOTE

    if target_minority is not None:
        n_maj = int(np.sum(y_train == 0))
        sm = SMOTE(
            sampling_strategy={0: n_maj, 1: target_minority},
            random_state=random_state,
        )
    else:
        sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------
def run_full_pipeline(*, mode: str = "smote_train") -> dict:
    """Run the paper's pipeline end-to-end and return the artefacts.

    Parameters
    ----------
    mode : str
        "smote_train"   - correct methodology (SMOTE only on training set).
                          Achieves the paper's no-SMOTE accuracy but cannot
                          reach the 0.90 SMOTE claim.
        "smote_full"    - apply SMOTE on the full dataset before splitting.
                          This is the pipeline that reproduces the paper's
                          reported 0.90 accuracy (but introduces data
                          leakage).
        "no_smote"      - no oversampling at all.
    """
    df_raw = load_raw()
    missing_report = report_missing(df_raw)

    df_id = drop_identifiers(df_raw)
    df_enc = encode_categoricals(df_id)

    # Section 3.6: IQR removal BEFORE splitting (matches paper flowchart).
    df_clean = iqr_remove_outliers(df_enc, IQR_COLUMNS, factor=1.5)

    X = df_clean.drop(columns=[TARGET]).values
    y = df_clean[TARGET].values

    if mode == "smote_full":
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_s, y_s = apply_smote(X, y, target_minority=8000)
        X_train, X_test, y_train, y_test = split_data(X_s, y_s)
        X_train, X_test = feature_scaling(X_train, X_test)
    elif mode == "smote_train":
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train, X_test = feature_scaling(X_train, X_test)
        X_train, y_train = apply_smote(X_train, y_train, target_minority=8000)
    elif mode == "no_smote":
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train, X_test = feature_scaling(X_train, X_test)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "df_raw_shape": df_raw.shape,
        "df_clean_shape": df_clean.shape,
        "iqr_bounds": df_clean.attrs.get("iqr_bounds"),
        "missing_report": missing_report,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_balance": dict(zip(*np.unique(y_train, return_counts=True))),
        "y_test_balance": dict(zip(*np.unique(y_test, return_counts=True))),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Boxplots for EDA (Figure 11 and Figure 12)
# ---------------------------------------------------------------------------
def plot_boxplots_all(df: pd.DataFrame, out_path: Path) -> None:
    """Figure 11: box plot for every numeric column before outlier removal."""
    import matplotlib.pyplot as plt

    cols = [
        "RowNumber", "CustomerId", "CreditScore", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for ax, col in zip(axes.flatten(), cols):
        ax.boxplot(df[col])
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Fig. 11 - Box plots for every numeric feature (before IQR)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplots_before_after_iqr(df_raw: pd.DataFrame, df_clean: pd.DataFrame, out_dir: Path) -> None:
    """Figure 12: CreditScore/Age/NumOfProducts box plots before/after IQR."""
    import matplotlib.pyplot as plt

    for col in IQR_COLUMNS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].boxplot(df_raw[col])
        axes[0].set_title(f"{col} - BEFORE IQR")
        axes[1].boxplot(df_clean[col])
        axes[1].set_title(f"{col} - AFTER IQR")
        fig.suptitle(f"Fig. 12 - IQR removal: {col}")
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = col.replace(" ", "_")
        fig.savefig(out_dir / f"fig12_{safe}_iqr.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    df = load_raw()
    print(f"Raw shape: {df.shape}")
    print(f"Missing values: {report_missing(df)['total_missing']}")

    df_id = drop_identifiers(df)
    df_enc = encode_categoricals(df_id)
    print(f"After drop identifiers + encoding: {df_enc.shape}")
    print("Columns:", list(df_enc.columns))

    df_clean = iqr_remove_outliers(df_enc, IQR_COLUMNS, factor=1.5)
    print(f"\nAfter IQR on {IQR_COLUMNS}: {df_clean.shape}")
    print("IQR bounds:", df_clean.attrs["iqr_bounds"])

    for mode in ("no_smote", "smote_train", "smote_full"):
        info = run_full_pipeline(mode=mode)
        print(f"\n--- mode={mode} ---")
        for k, v in info.items():
            if not k.startswith("X_") and "Shape" not in k and "X_train" not in k and "X_test" not in k:
                print(f"  {k}: {v}")
