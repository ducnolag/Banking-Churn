"""
Investigate one more hypothesis: SMOTE applied to the full dataset before
split (the paper's claim of "7,963 non-churn + 8,000 churn = 15,963 rows"
only makes sense if SMOTE is applied to the whole dataset - data leakage,
but it's a known pattern in this paper).
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def encode(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"]).copy()
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    return df


def main():
    df = pd.read_csv(ROOT / "data" / "Churn_Modelling.csv")
    df_enc = encode(df)
    X = df_enc.drop(columns=["Exited"]).values
    y = df_enc["Exited"].values

    base = {
        "DT": DecisionTreeClassifier(random_state=42, criterion="entropy"),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVC": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGB": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            eval_metric="logloss", random_state=42,
        ),
    }
    estimators = list(base.items())

    # Approach: SMOTE the full dataset then split
    print("\nApproach A: SMOTE full dataset, then 70/30 split")
    sm = SMOTE(random_state=42)
    Xs, ys = sm.fit_resample(X, y)
    print(f"  After SMOTE: {Xs.shape} balance={np.bincount(ys.astype(int)).tolist()}")
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, ys, test_size=0.30, stratify=ys, random_state=42
    )
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")
    for name, m in base.items(): m.fit(Xtr, ytr)
    vc = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
    vc.fit(Xtr, ytr)
    yp = vc.predict(Xte)
    print(f"  Voting Classifier: acc={accuracy_score(yte, yp):.4f}")
    print("  CM:", confusion_matrix(yte, yp).tolist())
    print(classification_report(yte, yp, target_names=["Stayed","Churned"]))

    # Approach B: SMOTE the full dataset then split, with paper's claim
    # (minority 2,037 -> 8,000; non-churn stays at 7,963)
    print("\nApproach B: SMOTE minority to 8000, then 70/30 split")
    sm = SMOTE(sampling_strategy={1: 8000}, random_state=42)
    Xs, ys = sm.fit_resample(X, y)
    print(f"  After SMOTE: {Xs.shape} balance={np.bincount(ys.astype(int)).tolist()}")
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, ys, test_size=0.30, stratify=ys, random_state=42
    )
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")
    for name, m in base.items(): m.fit(Xtr, ytr)
    vc = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
    vc.fit(Xtr, ytr)
    yp = vc.predict(Xte)
    print(f"  Voting Classifier: acc={accuracy_score(yte, yp):.4f}")
    print("  CM:", confusion_matrix(yte, yp).tolist())
    print(classification_report(yte, yp, target_names=["Stayed","Churned"]))

    # Approach C: scale raw X first, then SMOTE
    print("\nApproach C: scale raw X first, SMOTE, then split")
    sc = StandardScaler().fit(X)
    X_scaled = sc.transform(X)
    sm = SMOTE(random_state=42)
    Xs, ys = sm.fit_resample(X_scaled, y)
    print(f"  After SMOTE: {Xs.shape} balance={np.bincount(ys.astype(int)).tolist()}")
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, ys, test_size=0.30, stratify=ys, random_state=42
    )
    for name, m in base.items(): m.fit(Xtr, ytr)
    vc = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
    vc.fit(Xtr, ytr)
    yp = vc.predict(Xte)
    print(f"  Voting Classifier: acc={accuracy_score(yte, yp):.4f}")
    print("  CM:", confusion_matrix(yte, yp).tolist())
    print(classification_report(yte, yp, target_names=["Stayed","Churned"]))


if __name__ == "__main__":
    main()
