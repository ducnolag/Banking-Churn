"""
Voting Classifier with SMOTE - test soft vs hard voting,
plus SMOTE minority oversampling to specific counts the paper claims
(minority 2,037 -> 8,000).
"""

from __future__ import annotations
import warnings
from pathlib import Path

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
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def encode(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"]).copy()
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    return df


def main():
    df = pd.read_csv(ROOT / "data" / "Churn_Modelling.csv")

    # Variant 1: NO IQR, 80/20 split (matches paper test size best)
    # Variant 2: mild IQR (factor 3) to keep ~10000 rows
    rows = []

    for cfg_name, do_iqr, factor, ts in [
        ("NO_IQR_split20", False, 0, 0.20),
        ("NO_IQR_split30", False, 0, 0.30),
        ("IQR3_split20", True, 3.0, 0.20),
    ]:
        work = df.copy()
        if do_iqr:
            iqr_cols = ["CreditScore", "Age", "NumOfProducts"]
            mask = pd.Series(False, index=work.index)
            for col in iqr_cols:
                q1, q3 = work[col].quantile(0.25), work[col].quantile(0.75)
                iqr = q3 - q1
                mask |= (work[col] < q1 - factor*iqr) | (work[col] > q3 + factor*iqr)
            work = work.loc[~mask]

        work = encode(work)
        X = work.drop(columns=["Exited"]).values
        y = work["Exited"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=42
        )
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(f"\n=== {cfg_name} ===  Train {X_train.shape} Test {X_test.shape}")

        # Try several SMOTE target sizes and voting modes
        for smote_label, smote_kwargs in [
            ("SMOTE_default", {}),
            ("SMOTE_8000", dict(sampling_strategy={1: 8000})),
            ("SMOTE_balance", dict(sampling_strategy="auto")),  # equal classes
        ]:
            for voting_mode in ["hard", "soft"]:
                try:
                    if smote_kwargs:
                        sm = SMOTE(random_state=42, **smote_kwargs)
                    else:
                        sm = SMOTE(random_state=42)
                    Xs, ys = sm.fit_resample(X_train, y_train)
                except ValueError as e:
                    print(f"  Skip {smote_label} on {cfg_name}: {e}")
                    continue

                base = {
                    "DT": DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=8),
                    "RF": RandomForestClassifier(n_estimators=300, random_state=42, max_depth=12),
                    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=7),
                    "XGB": XGBClassifier(
                        n_estimators=300, learning_rate=0.05, max_depth=4,
                        eval_metric="logloss", random_state=42
                    ),
                }
                estimators = list(base.items())
                for name, m in base.items():
                    m.fit(Xs, ys)
                vc = VotingClassifier(estimators=estimators, voting=voting_mode, n_jobs=-1)
                vc.fit(Xs, ys)
                y_pred = vc.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                tag = f"Voting {voting_mode} {smote_label}"
                row = {
                    "config": cfg_name,
                    "model": tag,
                    "acc": round(acc, 4),
                    "prec": round(prec, 4),
                    "rec": round(rec, 4),
                    "f1": round(f1, 4),
                    "cm": cm.tolist(),
                    "train_size": int(Xs.shape[0]),
                }
                rows.append(row)
                print(f"  {tag:<35s} acc={acc:.4f} prec={prec:.4f} "
                      f"rec={rec:.4f} f1={f1:.4f}  CM={cm.flatten().tolist()}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "voting_smote_grid.csv", index=False)
    print("\nSaved:", OUT / "voting_smote_grid.csv")

    # Find closest match to paper SMOTE (0.90)
    df_out["delta"] = (df_out["acc"] - 0.90).abs()
    print("\nTop 10 closest to paper SMOTE 0.90:")
    print(df_out.nsmallest(10, "delta").to_string(index=False))


if __name__ == "__main__":
    main()
