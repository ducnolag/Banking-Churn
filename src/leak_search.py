"""
Try to reproduce the paper's exact 1970-row test set.

Paper's published no-SMOTE voting CM = [[1546, 20], [214, 190]]
sums to 1970 with 79.5% non-churn / 20.5% churn (matches raw distribution).

Hypotheses to test:
  H1: paper used 65/35 split -> 6500/3500 -> drop ~1530 rows for some reason -> 1970
  H2: paper did 70/30 + 34.3% random drop from test
  H3: paper did not split at all; CM values come from running 5-fold CV
      averaged (1970 ~ 5 x 394, doesn't fit 10000)
  H4: paper applied SMOTE to test set too after removing its class balance
      (1577 non-churn + 393 churn = 1970 matches 30% stratified of raw)
  H5: paper split 70/30, then truncated test set by removing all rows where
      CreditScore < Q1 or > Q3 (i.e. a partial IQR on test only)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

DATA_PATH = "D:/NCKH/data/Churn_Modelling.csv"
PAPER_NO_SMOTE_CM = np.array([[1546, 20], [214, 190]])
PAPER_SMOTE_CM = np.array([[1453, 124], [48, 345]])


def load():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def fit_predict(X_tr, y_tr, X_te):
    ens = VotingClassifier(
        estimators=[
            ("dt",  DecisionTreeClassifier(random_state=42)),
            ("rf",  RandomForestClassifier(random_state=42, n_estimators=100)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("svc", SVC(random_state=42)),
            ("xgb", XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)),
        ],
        voting="hard",
    )
    ens.fit(X_tr, y_tr)
    return ens.predict(X_te)


# ---------------------------------------------------------------------------
def h1_smote_test_set(df):
    """H4: SMOTE on full dataset, stratified 70/30 split.

    This actually reproduces the 80/20 ratio in test if we apply SMOTE
    ONLY to the training side but the test side remains raw 80/20 stratified.
    But paper CM sums to 1970 which is just 30% of 6567.
    Try: drop IQR first, split 70/30 -> test has 2871 rows.
    Then SMOTE test to balance it -> 2304 + 567 = 2871 -> 567*4.84 ≈ 2745
    """
    pass


def h2_smote_then_resplit(df):
    """Apply SMOTE to full df first -> ~15000 rows -> split 70/30.
    Test balance in that case is 50/50, paper test balance is 80/20 -> NO."""
    pass


def h3_just_check_50_50():
    """The 1970 row test might come from random sampling."""
    pass


# ---------------------------------------------------------------------------
# Actually: let's brute force search over a few parameters
# ---------------------------------------------------------------------------
def grid_search():
    df = load()
    X_full = df.drop(columns=["Exited"]).values
    y_full = df["Exited"].values

    print("Brute-force search for a pipeline that yields test size ~1970")
    print("with class balance ~80/20 (paper Section 4.2).")
    print()

    # Try different IQR settings
    iqr_options = {
        "no IQR": lambda d: d,
        "IQR CScore only": lambda d: _iqr(d, ["CreditScore"]),
        "IQR Age only":     lambda d: _iqr(d, ["Age"]),
        "IQR CS+Age":       lambda d: _iqr(d, ["CreditScore", "Age"]),
        "IQR CS+Age+NumP":  lambda d: _iqr(d, ["CreditScore", "Age", "NumOfProducts"]),
    }
    split_options = [0.20, 0.25, 0.30, 0.35, 0.40]
    smote_options = ["none", "full", "train"]

    for iqr_name, iqr_fn in iqr_options.items():
        for test_size in split_options:
            df2 = iqr_fn(df)
            X = df2.drop(columns=["Exited"]).values
            y = df2["Exited"].values
            for sm in smote_options:
                if sm == "none":
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                elif sm == "full":
                    n_maj = int((y == 0).sum())
                    Xs, ys = SMOTE(sampling_strategy={0: n_maj, 1: 8000}, random_state=42).fit_resample(X, y)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        Xs, ys, test_size=test_size, stratify=ys, random_state=42
                    )
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                elif sm == "train":
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
                    n_maj = int((y_tr == 0).sum())
                    X_tr, y_tr = SMOTE(sampling_strategy={0: n_maj, 1: 8000}, random_state=42).fit_resample(X_tr, y_tr)
                try:
                    y_pred = fit_predict(X_tr, y_tr, X_te)
                    cm_ = confusion_matrix(y_te, y_pred)
                    acc = accuracy_score(y_te, y_pred)
                    test_size_n = len(y_te)
                    test_balance = np.bincount(y_te)
                    pct_non_churn = test_balance[0] / test_size_n
                    if abs(test_size_n - 1970) < 50 or abs(pct_non_churn - 0.795) < 0.02:
                        print(f"  {iqr_name:20s} split={test_size:.2f} smote={sm:5s} "
                              f"-> test_n={test_size_n} balance={test_balance} "
                              f"non-churn%={pct_non_churn:.3f} acc={acc:.4f}")
                        print(f"    CM=\n{cm_}")
                except Exception as e:
                    pass


def _iqr(df, cols, factor=1.5):
    keep = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr_v = q3 - q1
        lo, hi = q1 - factor * iqr_v, q3 + factor * iqr_v
        keep &= df[col].between(lo, hi)
    return df.loc[keep].copy()


if __name__ == "__main__":
    grid_search()