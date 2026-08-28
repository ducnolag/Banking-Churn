"""
Sweep test_size when using the SMOTE-before-split (leakage) approach
to see if we can find a configuration where test_size ~ 1,970 (matching
the paper's stated CM denominator).
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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

    # The paper's CM denominators sum to 1,970. If after-SMOTE total
    # is 15,926, then test_size=1970/15926 = 0.1237.
    # Maybe the paper used different split.
    for test_size in [0.1237, 0.20, 0.25, 0.30]:
        sm = SMOTE(random_state=42)
        Xs, ys = sm.fit_resample(X, y)
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, ys, test_size=test_size, stratify=ys, random_state=42
        )
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        for name, m in base.items(): m.fit(Xtr, ytr)
        vc = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
        vc.fit(Xtr, ytr)
        yp_vc = vc.predict(Xte)

        # Also test individual models and soft vote
        vc_soft = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        vc_soft.fit(Xtr, ytr)
        yp_vc_soft = vc_soft.predict(Xte)

        print(f"\ntest_size={test_size} -> test rows={len(yte)}")
        print(f"  Voting hard : acc={accuracy_score(yte, yp_vc):.4f}  CM={confusion_matrix(yte, yp_vc).tolist()}")
        print(f"  Voting soft : acc={accuracy_score(yte, yp_vc_soft):.4f}  CM={confusion_matrix(yte, yp_vc_soft).tolist()}")

    # Also try: 80/20 split, then SMOTE only on training (proper)
    # vs SMOTE-on-full then 80/20
    print("\n\n--- Proper 80/20 split with SMOTE on train only ---")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    sm = SMOTE(random_state=42)
    Xtr_sm, ytr_sm = sm.fit_resample(Xtr, ytr)
    for name, m in base.items(): m.fit(Xtr_sm, ytr_sm)
    vc = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
    vc.fit(Xtr_sm, ytr_sm)
    yp = vc.predict(Xte)
    print(f"Voting hard (proper SMOTE on train only): acc={accuracy_score(yte, yp):.4f}  CM={confusion_matrix(yte, yp).tolist()}")


if __name__ == "__main__":
    main()
