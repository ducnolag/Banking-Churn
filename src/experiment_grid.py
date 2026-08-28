"""
Bank Churn Voting Classifier - Reproduction Experiments.

Tries multiple preprocessing and modeling variations to find the
configuration that best matches the published metrics in
Bhuria et al. (2025) - Discover Sustainability 6:28.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Churn_Modelling.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Targets reported in the paper
# ---------------------------------------------------------------------------
@dataclass
class PaperTarget:
    accuracy: float
    cm: np.ndarray | None  # [[TN, FP], [FN, TP]] if known


PAPER_NO_SMOTE = PaperTarget(
    accuracy=0.87,  # abstract; Section 4.2 says 88%
    cm=np.array([[1546, 20], [214, 190]]),
)
PAPER_SMOTE = PaperTarget(
    accuracy=0.90,
    cm=np.array([[1453, 124], [48, 345]]),
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def iqr_remove(
    df: pd.DataFrame,
    cols: List[str],
    factor: float = 1.5,
    mode: str = "union",  # 'union' or 'intersection'
) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        col_mask = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
        if mode == "union":
            mask = mask | col_mask
        else:
            mask = mask & col_mask
    return df.loc[~mask].copy()


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    return df


def make_split(
    df: pd.DataFrame,
    *,
    iqr_factor: float | None = 1.5,
    iqr_cols: List[str] | None = None,
    test_size: float = 0.20,
    scale: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if iqr_cols is None:
        iqr_cols = ["CreditScore", "Age", "NumOfProducts"]
    work = df
    if iqr_factor is not None:
        work = iqr_remove(work, iqr_cols, factor=iqr_factor)
    work = encode_features(work)

    X = work.drop(columns=["Exited"]).values
    y = work["Exited"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def base_models() -> Dict[str, object]:
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, criterion="entropy", max_depth=8
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, max_depth=12
        ),
        "SVC": SVC(kernel="rbf", probability=True, random_state=42, C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="minkowski"),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def voting(estimators, voting: str = "hard"):
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_one(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    smote: bool,
    voting_mode: str = "hard",
    smote_target: tuple | None = None,
) -> Dict[str, Dict]:
    if smote:
        if smote_target is not None:
            # SMOTE the minority to a specific count (e.g., (8000,) for 8k churners).
            n_min = smote_target[0]
            n_maj = int(np.sum(y_train == 0))
            ratio = {0: n_maj, 1: n_min}
            sm = SMOTE(sampling_strategy=ratio, random_state=42)
        else:
            sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(
            f"  [SMOTE] train shape={X_train.shape} balance={np.bincount(y_train.astype(int)).tolist()}"
        )

    base = base_models()
    estimators = list(base.items())

    results = {}
    for name, model in base.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            **metrics(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred).tolist(),
        }

    vc = voting(estimators, voting=voting_mode)
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    label = f"Voting ({voting_mode} vote)"
    if smote:
        label += " + SMOTE"
    results[label] = {
        **metrics(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred).tolist(),
    }
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_to_paper(pred_cm: list, paper_cm: np.ndarray, label: str) -> str:
    pred_cm = np.array(pred_cm)
    return (
        f"{label:<35s}  pred CM={pred_cm.flatten().tolist()}  "
        f"paper CM={paper_cm.flatten().tolist()}  "
        f"pred_acc={pred_cm.diagonal().sum()/pred_cm.sum():.4f}"
    )


# ---------------------------------------------------------------------------
# Main experiment grid
# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {df.shape}")
    print(f"Class balance: {df['Exited'].value_counts().to_dict()}")

    summary_rows = []

    configs = [
        # (name, iqr_factor, test_size)
        ("IQR1.5_split30", 1.5, 0.30),
        ("IQR1.5_split20", 1.5, 0.20),
        ("IQR2.0_split30", 2.0, 0.30),
        ("IQR2.0_split20", 2.0, 0.20),
        ("IQR2.5_split30", 2.5, 0.30),
        ("IQR2.5_split20", 2.5, 0.20),
        ("NO_IQR_split20", None, 0.20),
        ("NO_IQR_split30", None, 0.30),
    ]

    for cfg_name, factor, ts in configs:
        print(f"\n{'='*70}\n  Config: {cfg_name}  (iqr={factor}, test_size={ts})\n{'='*70}")
        X_train, X_test, y_train, y_test = make_split(
            df, iqr_factor=factor, test_size=ts, scale=True
        )
        print(
            f"  Train: {X_train.shape}  Test: {X_test.shape}  "
            f"Test balance: {dict(zip(*np.unique(y_test, return_counts=True)))}"
        )

        for smote in (False, True):
            label = "WITH SMOTE" if smote else "no SMOTE"
            print(f"\n  --- {label} ---")
            res = run_one(X_train, X_test, y_train, y_test, smote=smote)
            for name, m in res.items():
                paper_target = PAPER_SMOTE if smote else PAPER_NO_SMOTE
                row = {
                    "config": cfg_name,
                    "smote": smote,
                    "model": name,
                    "acc": round(m["accuracy"], 4),
                    "prec": round(m["precision"], 4),
                    "rec": round(m["recall"], 4),
                    "f1": round(m["f1"], 4),
                    "cm": m["cm"],
                    "paper_acc": paper_target.accuracy,
                }
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "experiment_grid.csv", index=False)
    print(f"\nSaved experiment grid -> {OUT_DIR / 'experiment_grid.csv'}")

    # Find the best match for the SMOTE voting classifier.
    print("\n\n===== Best matches to paper =====")
    smote_voting = df_summary[df_summary["model"].str.contains("Voting") & df_summary["smote"]]
    smote_voting["delta"] = (smote_voting["acc"] - 0.90).abs()
    print("\nTop 5 - SMOTE Voting closest to paper 0.90:")
    print(smote_voting.nsmallest(5, "delta")[
        ["config", "smote", "model", "acc", "prec", "rec", "f1", "cm"]
    ].to_string(index=False))

    no_smote_voting = df_summary[
        df_summary["model"].str.contains("Voting") & (~df_summary["smote"])
    ]
    no_smote_voting["delta"] = (no_smote_voting["acc"] - 0.87).abs()
    print("\nTop 5 - no-SMOTE Voting closest to paper 0.87:")
    print(no_smote_voting.nsmallest(5, "delta")[
        ["config", "smote", "model", "acc", "prec", "rec", "f1", "cm"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
