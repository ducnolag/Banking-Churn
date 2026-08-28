"""
Re-investigation: Bhuria et al. (2025) has internal inconsistencies.

From Section 4.2 / 4.3 confusion matrices the test set size is always 1,970
rows with ~80% non-churn / ~20% churn, i.e. the 30% split of the *raw*
10,000-row dataset (3,000 rows). The published voting-classifier test CM
contains 2,393 entries summed (1,546 + 20 + 214 + 190 = 1,970) -> 1,970,
NOT 3,000.

That means the 30% hold-out is taken BEFORE any IQR removal (3,000 -> 1,970
implies 1,030 rows were dropped).  A 30% split on 10,000 = 3,000; after
dropping 1,030 outliers in the test set alone we get 1,970.

So the pipeline is most likely:

  raw data (10,000)
      -> drop IDs, one-hot encode (Section 3.1)
      -> 70/30 split  -> train 7,000 / test 3,000
      -> IQR outlier removal on train (CreditScore/Age/NumOfProducts)
      -> feature scaling (StandardScaler fit on train)
      -> SMOTE on train only (2,037 -> 8,000)
      -> evaluate on the 3,000-row test set that *might* also have had
         SMOTE applied (giving 1,970 test rows post SMOTE-downsample? no,
         1,970 = original 3,000 - outliers dropped from test).

The 1,970 test rows can also be reproduced by taking the raw 10,000,
dropping 1,030 outliers from the whole dataset, then splitting 70/30
(8,970 -> 6,279 train / 2,691 test). That doesn't match either.

The only clean reproduction is:
  - SMOTE applied to the ENTIRE pre-split dataset
  - then 70/30 split on the balanced set

because that gives 8,970 + 8,000 SMOTE = 15,677 -> 70/30 = 10,973/4,704
and the test set still has roughly 50/50 class balance. The paper's CM
shows 80/20 in test, so that's NOT what they did.

Most likely the paper's published numbers are simply wrong or fabricated
for the SMOTE case.  For the no-SMOTE case the CM (1546, 20, 214, 190)
sums to 1,970 with 80% non-churn and ~88% accuracy -- that IS reproducible
if we take 70/30 stratified split on the *raw* dataset (10,000 -> 7,000/3,000)
then drop ~1,030 outliers from the test set or don't IQR at all and just
report on a 1,970-row subsample.

This script tries every plausible permutation and prints the test-set
sizes / class balances so we can see which one matches the paper.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH = Path("D:/NCKH/data/Churn_Modelling.csv")

# Paper's published no-SMOTE test confusion matrix
PAPER_NO_SMOTE_CM = np.array([[1546, 20], [214, 190]])  # rows = true, cols = pred
PAPER_SMOTE_CM    = np.array([[1453, 124], [48, 345]])


def load():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df["Gender"] = (df["Gender"] == "Female").astype(int)
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def iqr(df, cols, factor=1.5):
    keep = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr_v = q3 - q1
        lo, hi = q1 - factor * iqr_v, q3 + factor * iqr_v
        keep &= df[col].between(lo, hi)
    return df.loc[keep].copy()


def fit_predict_voting(X_tr, y_tr, X_te):
    """Train a hard-vote ensemble, return test CM and accuracy."""
    rf  = RandomForestClassifier(random_state=42, n_estimators=100)
    dt  = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svc = SVC(random_state=42)
    xgb = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False, verbosity=0)
    ens = VotingClassifier(
        estimators=[("dt", dt), ("rf", rf), ("knn", knn), ("svc", svc), ("xgb", xgb)],
        voting="hard",
    )
    ens.fit(X_tr, y_tr)
    y_pred = ens.predict(X_te)
    from sklearn.metrics import confusion_matrix, accuracy_score
    return confusion_matrix([1] if False else None, y_pred) if False else None, None


def cm(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    return confusion_matrix(y_true, y_pred), accuracy_score(y_true, y_pred)


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------
def variant_a_no_iqr(df):
    """No IQR.  Raw 10k -> 70/30 split -> scale -> train."""
    X = df.drop(columns=["Exited"]).values
    y = df["Exited"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, y_tr, X_te, y_te


def variant_b_iqr_then_split(df):
    """IQR on full dataset, then split."""
    df2 = iqr(df, ["CreditScore", "Age", "NumOfProducts"])
    X = df2.drop(columns=["Exited"]).values
    y = df2["Exited"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, y_tr, X_te, y_te


def variant_c_split_then_iqr(df):
    """70/30 split first, then IQR on train only."""
    X = df.drop(columns=["Exited"]).values
    y = df["Exited"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    train_df = pd.DataFrame(X_tr, columns=df.columns.drop("Exited"))
    train_df["Exited"] = y_tr
    train_df = iqr(train_df, ["CreditScore", "Age", "NumOfProducts"])
    X_tr = train_df.drop(columns=["Exited"]).values
    y_tr = train_df["Exited"].values
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, y_tr, X_te, y_te


def variant_smote_full_then_split(df):
    """SMOTE on the full dataset (after IQR), then 70/30 split."""
    df2 = iqr(df, ["CreditScore", "Age", "NumOfProducts"])
    X = df2.drop(columns=["Exited"]).values
    y = df2["Exited"].values
    n_maj = int((y == 0).sum())
    sm = SMOTE(sampling_strategy={0: n_maj, 1: 8000}, random_state=42)
    X, y = sm.fit_resample(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, y_tr, X_te, y_te


def variant_smote_train_only(df):
    """IQR on full, then 70/30 split, then SMOTE on train."""
    df2 = iqr(df, ["CreditScore", "Age", "NumOfProducts"])
    X = df2.drop(columns=["Exited"]).values
    y = df2["Exited"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    n_maj = int((y_tr == 0).sum())
    sm = SMOTE(sampling_strategy={0: n_maj, 1: 8000}, random_state=42)
    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    return X_tr, y_tr, X_te, y_te


def run():
    df = load()
    print(f"Raw shape after drop IDs + encode: {df.shape}")

    pipelines = {
        "A. no IQR, no SMOTE":           variant_a_no_iqr,
        "B. IQR then split, no SMOTE":   variant_b_iqr_then_split,
        "C. split then IQR train, no SMOTE": variant_c_split_then_iqr,
        "D. SMOTE on full then split":   variant_smote_full_then_split,
        "E. SMOTE on train only":        variant_smote_train_only,
    }
    for name, fn in pipelines.items():
        try:
            X_tr, y_tr, X_te, y_te = fn(df)
        except Exception as e:
            print(f"\n{name}  -> FAILED: {e}")
            continue
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
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
        y_pred = ens.predict(X_te)
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        cm_ = confusion_matrix(y_te, y_pred)
        acc = accuracy_score(y_te, y_pred)
        print(f"\n{name}")
        print(f"  Train shape: {X_tr.shape}  Test shape: {X_te.shape}")
        print(f"  Test class balance: {np.bincount(y_te)}")
        print(f"  Confusion matrix:\n    {cm_}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {precision_score(y_te, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_te, y_pred, zero_division=0):.4f}")
        print(f"  F1:        {f1_score(y_te, y_pred, zero_division=0):.4f}")


if __name__ == "__main__":
    run()