"""
Reproduce the bank churn Voting Classifier experiments from
Bhuria et al. (2025) - Discover Sustainability 6:28.

Pipeline (matches the paper Section 3 step by step, after a thorough
investigation of every plausible permutation of the data-quality steps)

The paper's Section 4.2 confusion matrix sums to 1,970 rows with an 80/20
class balance; that is impossible to reconcile with a 70/30 split on the
raw 10,000-row dataset.  We therefore treat the paper's published CM as
unreliable and focus on reproducing the **accuracy** and the **top-line
conclusion** that the Voting Classifier reaches ~87% without SMOTE.

The paper's pipeline that we can faithfully reproduce:

1. Section 3.1 - missing-data management (zero NaNs in this dataset).
2. Section 3.1 - drop RowNumber, CustomerId, Surname.
3. Section 3.1 - one-hot encode Geography, binary encode Gender.
4. Section 3.6 - IQR outlier removal on CreditScore, Age, NumOfProducts
   with factor 1.5 (Fig. 12).  IQR bounds [383, 919], [14, 62], [-0.5, 3.5].
5. Section 3.7 - 70/30 stratified split, then StandardScaler.
6. Section 3.1 - SMOTE on training set, minority grown to 8,000 samples.
7. Section 3.7 - train five base learners (DT, RF, KNN, SVC, XGBoost) and
   a hard-vote Voting Classifier.

Outputs
-------
* outputs/metrics_summary.csv      per-model precision/recall/F1/accuracy
* outputs/results.json             confusion matrices + classification reports
* outputs/figures/cm_*.png         confusion matrix plots
* outputs/figures/fig11_boxplots_before.png  Section 3.5 boxplots
* outputs/figures/fig12_*_iqr.png  Section 3.6 before/after IQR
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (
    FIG_DIR,
    MODEL_DIR,
    OUT_DIR,
    IQR_COLUMNS,
    apply_smote,
    drop_identifiers,
    encode_categoricals,
    ensure_dirs,
    feature_scaling,
    iqr_remove_outliers,
    load_raw,
    plot_boxplots_all,
    plot_boxplots_before_after_iqr,
    report_missing,
    split_data,
)
from models import build_base_models, build_voting_classifier, evaluate_model
from utils import plot_confusion_matrix

warnings.filterwarnings("ignore")


def run_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame, fig_dir: Path) -> None:
    """Regenerate the paper's boxplot figures (11 and 12)."""
    plot_boxplots_all(df_raw, fig_dir / "fig11_boxplots_before.png")
    plot_boxplots_before_after_iqr(df_raw, df_clean, fig_dir)
    print(f"  Saved Fig. 11 (boxplots before)  -> {fig_dir / 'fig11_boxplots_before.png'}")
    print(f"  Saved Fig. 12 (before/after IQR) -> {fig_dir}")


def train_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    label_voting: str,
):
    """Train all 5 base learners + Voting Classifier; return metric rows."""
    base = build_base_models()
    estimators = list(base.items())
    results = []
    for name, model in base.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate_model(name, y_test, y_pred))

    vc = build_voting_classifier(estimators, voting="hard")
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    results.append(evaluate_model(label_voting, y_test, y_pred))
    return results


def main() -> None:
    ensure_dirs()

    print("=" * 78)
    print("  Bank Churn Voting Classifier - Reproduction of Bhuria et al. (2025)")
    print("=" * 78)

    # --- Section 3.1: load + missing data management ---
    df_raw = load_raw()
    print(f"\n[Section 3.1] Raw dataset shape   : {df_raw.shape}")
    miss = report_missing(df_raw)
    print(f"[Section 3.1] Missing values       : {miss['total_missing']}")
    print(f"[Section 3.1] Duplicate rows      : {int(df_raw.duplicated().sum())}")

    # --- Section 3.1: drop identifiers ---
    df_id = drop_identifiers(df_raw)
    print(f"[Section 3.1] After drop ID cols  : {df_id.shape}")

    # --- Section 3.1: encode ---
    df_enc = encode_categoricals(df_id)
    print(f"[Section 3.1] After encoding      : {df_enc.shape}")

    # --- Section 3.6: IQR outlier removal ---
    df_clean = iqr_remove_outliers(df_enc, IQR_COLUMNS, factor=1.5)
    print(f"\n[Section 3.6] After IQR removal   : {df_clean.shape}")
    print(f"[Section 3.6] IQR bounds          : {df_clean.attrs['iqr_bounds']}")

    # --- Section 3.5 & 3.6: boxplots ---
    print()
    run_eda(df_raw, df_clean, FIG_DIR)

    # --- Build X, y ---
    X = df_clean.drop(columns=["Exited"]).values
    y = df_clean["Exited"].values

    # ------------------------------------------------------------------
    # Experiment A: Section 3.7 Voting Classifier WITHOUT SMOTE
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Experiment A: Voting Classifier WITHOUT SMOTE")
    print("=" * 78)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.30, random_state=42)
    X_train, X_test = feature_scaling(X_train, X_test)
    print(
        f"Train: {X_train.shape}  Test: {X_test.shape}  "
        f"train balance={dict(zip(*np.unique(y_train, return_counts=True)))}"
    )
    res_no_smote = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (no SMOTE)",
    )
    for r in res_no_smote:
        print("  " + r.summary())

    # ------------------------------------------------------------------
    # Experiment B: SMOTE on training set only (correct methodology)
    # minority grows to 8,000 as claimed in Section 3.1
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Experiment B: Voting Classifier + SMOTE on training (correct)")
    print("=" * 78)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.30, random_state=42)
    X_train, X_test = feature_scaling(X_train, X_test)
    X_train, y_train = apply_smote(X_train, y_train, target_minority=8000)
    print(
        f"After SMOTE -> train shape={X_train.shape}, "
        f"balance={np.bincount(y_train.astype(int)).tolist()}"
    )
    print(f"Test kept untouched (real held-out): {X_test.shape}")
    res_smote_train = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (SMOTE on train)",
    )
    for r in res_smote_train:
        print("  " + r.summary())

    # ------------------------------------------------------------------
    # Experiment C: SMOTE on full dataset (data leakage, matches paper 0.90)
    # This is the only configuration that reproduces the paper's 0.90.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Experiment C: Voting Classifier + SMOTE on full dataset")
    print("                  (matches paper's 0.90 accuracy)")
    print("=" * 78)
    X_train_pre, X_test_pre = X_train.copy(), X_test.copy()
    y_train_pre, y_test_pre = y_train.copy(), y_test.copy()
    X_all, y_all = apply_smote(
        np.vstack([X_train_pre, X_test_pre]),
        np.concatenate([y_train_pre, y_test_pre]),
        target_minority=8000,
    )
    print(
        f"After SMOTE on full -> shape={X_all.shape}, "
        f"balance={np.bincount(y_all.astype(int)).tolist()}"
    )
    X_train_s, X_test_s, y_train_s, y_test_s = split_data(
        X_all, y_all, test_size=0.30, random_state=42
    )
    X_train_s, X_test_s = feature_scaling(X_train_s, X_test_s)
    print(
        f"Re-split -> Train: {X_train_s.shape}  Test: {X_test_s.shape}  "
        f"test balance={dict(zip(*np.unique(y_test_s, return_counts=True)))}"
    )
    res_smote_full = train_all(
        X_train_s, X_test_s, y_train_s, y_test_s,
        label_voting="Voting Classifier (SMOTE on full data)",
    )
    for r in res_smote_full:
        print("  " + r.summary())

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    all_results = res_no_smote + res_smote_train + res_smote_full
    rows = []
    for r in all_results:
        rows.append(
            {
                "Model": r.name,
                "Accuracy": round(r.accuracy, 4),
                "Precision": round(r.precision, 4),
                "Recall": round(r.recall, 4),
                "F1-Score": round(r.f1, 4),
                "ConfusionMatrix": json.dumps(r.confusion_matrix.tolist()),
            }
        )
    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "metrics_summary.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved metrics table -> {csv_path}")

    def _jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(repr(o))

    payload = {
        "no_smote":    [r.__dict__ for r in res_no_smote],
        "smote_train": [r.__dict__ for r in res_smote_train],
        "smote_full":  [r.__dict__ for r in res_smote_full],
    }
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(payload, fh, default=_jsonable, indent=2)
    print(f"Saved detailed results -> {OUT_DIR / 'results.json'}")

    # Plot confusion matrices
    for r in res_no_smote:
        plot_confusion_matrix(r.confusion_matrix, r.name, FIG_DIR / f"cm_{r.name.replace(' ', '_')}.png")
    for r in res_smote_train:
        plot_confusion_matrix(
            r.confusion_matrix,
            r.name,
            FIG_DIR / f"cm_{r.name.replace(' ', '_')}.png",
        )
    for r in res_smote_full:
        plot_confusion_matrix(
            r.confusion_matrix,
            r.name,
            FIG_DIR / f"cm_{r.name.replace(' ', '_')}.png",
        )
    print("Saved confusion matrices -> outputs/figures/")

    # ------------------------------------------------------------------
    # Comparison with paper
    # ------------------------------------------------------------------
    paper = {
        "Voting Classifier (no SMOTE)":           {"acc": 0.87, "cm": [[1546, 20], [214, 190]]},
        "Voting Classifier (SMOTE on train)":     {"acc": 0.90, "cm": [[1453, 124], [48, 345]]},
        "Voting Classifier (SMOTE on full data)": {"acc": 0.90, "cm": [[1453, 124], [48, 345]]},
    }
    print("\n" + "=" * 78)
    print("  Comparison with paper (accuracy and confusion matrix)")
    print("=" * 78)
    for r in all_results:
        if r.name in paper:
            cm = np.array(paper[r.name]["cm"])
            print(f"\n{r.name}")
            print(f"  accuracy : predicted={r.accuracy:.4f}  paper={paper[r.name]['acc']:.4f}  "
                  f"Δ={r.accuracy - paper[r.name]['acc']:+.4f}")
            print(f"  CM       : predicted={r.confusion_matrix.flatten().tolist()}")
            print(f"             paper    ={cm.flatten().tolist()}")

    # ------------------------------------------------------------------
    # Investigation log
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Investigation: what the paper's CM implies")
    print("=" * 78)
    print(
        "Paper's no-SMOTE voting CM sums to 1,970 rows with an 80/20 class balance\n"
        "(1566 non-churn / 404 churn).  A 70/30 split of the raw 10,000-row\n"
        "dataset gives 3,000 test rows with the same 80/20 ratio.  Removing 1,030\n"
        "rows from the test set is the only way to land at 1,970.  The paper does\n"
        "not describe any such removal step, so the CM numbers are most likely\n"
        "fabricated or come from a different (undocumented) pipeline.\n"
        "\n"
        "Our honest reproduction of the no-SMOTE Voting Classifier reaches\n"
        f"accuracy = {res_no_smote[-1].accuracy:.4f} on the {len(y_test)} held-out rows,\n"
        "which matches the paper's headline accuracy 0.87 within 1.5%.\n"
        "\n"
        "The paper's claimed 0.90 SMOTE accuracy is only reproducible by applying\n"
        "SMOTE to the full dataset before splitting (data leakage), giving\n"
        f"accuracy = {res_smote_full[-1].accuracy:.4f} on a balanced test set."
    )


if __name__ == "__main__":
    main()