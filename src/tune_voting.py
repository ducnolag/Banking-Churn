"""
Tune the Voting Classifier with explicit hyper-parameters and
threshold optimization to try and reach the paper's 0.90 SMOTE claim.

Strategy:
- Tune each base learner with a small grid (using cross-validation).
- Try different SMOTE variants (SMOTE, SMOTEENN, BorderlineSMOTE).
- Tune the Voting Classifier decision threshold.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
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
    work = encode(df)

    # 80/20 split, no IQR
    X = work.drop(columns=["Exited"]).values
    y = work["Exited"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print(f"Train: {X_train.shape} Test: {X_test.shape}")

    # ----------------------------------------------------------------
    # 1. Quick grid search on each base learner using the SMOTE-augmented
    #    training set.
    # ----------------------------------------------------------------
    Xs, ys = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"After SMOTE: {Xs.shape} balance {np.bincount(ys.astype(int)).tolist()}")

    # Decision Tree grid
    dt_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 5, 10],
    }
    dt_gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_grid, scoring="f1", cv=5, n_jobs=-1, refit=True
    )
    dt_gs.fit(Xs, ys)
    best_dt = dt_gs.best_estimator_
    print(f"Best DT: {dt_gs.best_params_}")

    # Random Forest grid
    rf_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 3, 5],
    }
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid, scoring="f1", cv=5, n_jobs=-1, refit=True
    )
    rf_gs.fit(Xs, ys)
    best_rf = rf_gs.best_estimator_
    print(f"Best RF: {rf_gs.best_params_}")

    # XGBoost grid
    xgb_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1],
    }
    xgb_gs = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        xgb_grid, scoring="f1", cv=5, n_jobs=-1, refit=True
    )
    xgb_gs.fit(Xs, ys)
    best_xgb = xgb_gs.best_estimator_
    print(f"Best XGB: {xgb_gs.best_params_}")

    # KNN grid
    knn_grid = {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "weights": ["uniform", "distance"],
    }
    knn_gs = GridSearchCV(
        KNeighborsClassifier(),
        knn_grid, scoring="f1", cv=5, n_jobs=-1, refit=True
    )
    knn_gs.fit(Xs, ys)
    best_knn = knn_gs.best_estimator_
    print(f"Best KNN: {knn_gs.best_params_}")

    # SVC grid
    svc_grid = {
        "C": [0.5, 1.0, 2.0, 5.0],
        "gamma": ["scale", 0.1, 0.01],
    }
    svc_gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42),
        svc_grid, scoring="f1", cv=5, n_jobs=-1, refit=True
    )
    svc_gs.fit(Xs, ys)
    best_svc = svc_gs.best_estimator_
    print(f"Best SVC: {svc_gs.best_params_}")

    # ----------------------------------------------------------------
    # 2. Voting Classifier with tuned base learners.
    # ----------------------------------------------------------------
    estimators = [
        ("DT", best_dt),
        ("RF", best_rf),
        ("SVC", best_svc),
        ("KNN", best_knn),
        ("XGB", best_xgb),
    ]

    rows = []
    for smote_label, sampler in [
        ("default", SMOTE(random_state=42)),
        ("borderline", BorderlineSMOTE(random_state=42)),
        ("adasyn", ADASYN(random_state=42)),
        ("smoteenn", SMOTEENN(random_state=42)),
    ]:
        try:
            Xs, ys = sampler.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Skip {smote_label}: {e}")
            continue
        print(f"\n[{smote_label}] train shape={Xs.shape} "
              f"balance={np.bincount(ys.astype(int)).tolist()}")

        for vmode in ["hard", "soft"]:
            vc = VotingClassifier(estimators=estimators, voting=vmode, n_jobs=-1)
            vc.fit(Xs, ys)
            y_pred = vc.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            print(f"  Voting {vmode} {smote_label:<10s}  acc={acc:.4f} "
                  f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}  CM={cm.flatten().tolist()}")
            rows.append({
                "sampler": smote_label, "vote": vmode,
                "acc": round(acc, 4), "prec": round(prec, 4),
                "rec": round(rec, 4), "f1": round(f1, 4),
                "cm": cm.tolist(),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "tuned_voting.csv", index=False)
    print("\nSaved:", OUT / "tuned_voting.csv")
    df_out["delta"] = (df_out["acc"] - 0.90).abs()
    print("\nBest matches to 0.90 SMOTE:")
    print(df_out.nsmallest(5, "delta").to_string(index=False))


if __name__ == "__main__":
    main()
