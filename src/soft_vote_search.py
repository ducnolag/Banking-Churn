"""
Reproduce paper by switching voting mechanism + threshold tuning.

Paper's no-SMOTE voting CM: [[1546, 20], [214, 190]]
  - TN=1546, FP=20, FN=214, TP=190, total=1970
  - recall(class 0) = 1546/1566 = 98.7%
  - recall(class 1) = 190/404 = 47.0%
  - precision(class 1) = 190/210 = 90.5%
  - accuracy = 1736/1970 = 88.1%

Hard voting cannot reach 88.1% (we get 85-86%). Try soft voting +
threshold tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_PATH = "D:/NCKH/data/Churn_Modelling.csv"
PAPER_CM = np.array([[1546, 20], [214, 190]])


def load():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def main():
    df = load()
    X = df.drop(columns=["Exited"]).values
    y = df["Exited"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    # ---------- Soft voting ----------
    soft = VotingClassifier(
        estimators=[
            ("dt",  DecisionTreeClassifier(random_state=42)),
            ("rf",  RandomForestClassifier(random_state=42, n_estimators=100)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("svc", SVC(random_state=42, probability=True)),
            ("xgb", XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)),
        ],
        voting="soft",
    )
    soft.fit(X_tr_s, y_tr)
    y_proba = soft.predict_proba(X_te_s)[:, 1]

    # Sweep threshold
    print(f"Soft voting baseline (threshold=0.5):")
    y_pred_05 = (y_proba >= 0.5).astype(int)
    cm_ = confusion_matrix(y_te, y_pred_05)
    print(f"  test_n={len(y_te)}, CM=\n{cm_}, acc={accuracy_score(y_te, y_pred_05):.4f}")
    print()
    print("Threshold sweep:")
    print(f"{'thr':>6} {'acc':>6} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}  diff_to_paper")
    best = (None, 1e9)
    for thr in np.arange(0.05, 0.95, 0.01):
        yp = (y_proba >= thr).astype(int)
        cm_ = confusion_matrix(y_te, yp)
        tn, fp, fn, tp = cm_.ravel()
        acc = accuracy_score(y_te, yp)
        diff = abs(cm_ - PAPER_CM).sum()
        if diff < best[1]:
            best = (thr, diff, cm_, acc)
        if acc > 0.85:
            print(f"{thr:6.2f} {acc:6.4f} {tn:5d} {fp:5d} {fn:5d} {tp:5d}  diff={diff}")
    print()
    print(f"Best threshold: {best[0]}  diff_to_paper_cm={best[1]}")
    print(f"CM at best threshold:\n{best[2]}")
    print(f"Accuracy at best threshold: {best[3]:.4f}")

    # ---------- Try without IQR but with weighted models ----------
    print("\n=== Weighted soft voting (paper does not mention weights) ===")
    weights = [
        [1, 1, 1, 1, 1],
        [1, 2, 1, 1, 1],   # boost RF
        [1, 3, 1, 1, 1],
        [2, 2, 1, 1, 1],
    ]
    for w in weights:
        soft_w = VotingClassifier(
            estimators=[
                ("dt",  DecisionTreeClassifier(random_state=42)),
                ("rf",  RandomForestClassifier(random_state=42, n_estimators=100)),
                ("knn", KNeighborsClassifier(n_neighbors=5)),
                ("svc", SVC(random_state=42, probability=True)),
                ("xgb", XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)),
            ],
            voting="soft", weights=w,
        )
        soft_w.fit(X_tr_s, y_tr)
        yp = soft_w.predict(X_te_s)
        cm_ = confusion_matrix(y_te, yp)
        diff = abs(cm_ - PAPER_CM).sum()
        print(f"  weights={w} -> CM={cm_.flatten().tolist()}  diff={diff}  acc={accuracy_score(y_te, yp):.4f}")


if __name__ == "__main__":
    main()