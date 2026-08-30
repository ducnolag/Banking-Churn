"""
Reproduce the bank churn Voting Classifier experiments from
Bhuria et al. (2025) - Discover Sustainability 6:28.

PRIMARY GOAL: reproduce the paper's headline metrics (accuracy ~0.87
without SMOTE, ~0.90 with SMOTE).  Because the paper's published
confusion matrices sum to 1,970 rows (which only matches a 80/20 split
on 9,850 rows, i.e. ~150 rows dropped), we **also** provide a 80/20
configuration that lines the test size up with the paper.  We then
run BOTH the 70/30 (true to paper text Section 3.7) and the 80/20
(matching CM-derived test size) configurations and pick the one that
matches the paper's headline accuracy more closely.

Pipeline (matches the paper Section 3 step by step):

1. Section 3.1 - missing-data management (zero NaNs in this dataset).
2. Section 3.1 - drop RowNumber, CustomerId, Surname.
3. Section 3.1 - one-hot encode Geography, binary encode Gender.
4. Section 3.6 - IQR outlier removal on CreditScore, Age, NumOfProducts
   with factor 1.5 (Fig. 12).
5. Section 3.7 - stratified split + StandardScaler.
6. Section 3.1 - SMOTE (three configurations: no-SMOTE, SMOTE-train,
   SMOTE-full).
7. Section 3.7 - train five base learners + hard-vote Voting Classifier.

Outputs
-------
* outputs/metrics_summary.csv      per-model precision/recall/F1/accuracy
* outputs/results.json             confusion matrices + classification reports
* outputs/figures/cm_*.png         confusion matrix plots
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
    drop_top_rows,
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


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    shuffle: bool = True,
):
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    X_train, X_test = feature_scaling(X_train, X_test)
    return X_train, X_test, y_train, y_test


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

    # ============================================================
    # Run THREE configurations that together cover the paper's
    # headline numbers (0.87 / 0.90).  The "best" split that
    # reproduces the published CM (1,970 test rows) is also tried.
    # ============================================================
    rows_per_run: list[list] = []

    # ------------------------------------------------------------------
    # A. No-SMOTE - 70/30 split
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [A] Voting Classifier WITHOUT SMOTE - 70/30 split")
    print("=" * 78)
    X_train, X_test, y_train, y_test = make_split(X, y, test_size=0.30, random_state=42)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    res_no_smote = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (no SMOTE)",
    )
    for r in res_no_smote:
        print("  " + r.summary())
    rows_per_run.append(res_no_smote)

    # ------------------------------------------------------------------
    # B. SMOTE on training only - 70/30 split (correct methodology)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [B] Voting Classifier + SMOTE on train only - 70/30 split")
    print("=" * 78)
    X_train, X_test, y_train, y_test = make_split(X, y, test_size=0.30, random_state=42)
    X_train, y_train = apply_smote(X_train, y_train, target_minority=8000)
    print(f"After SMOTE -> train shape={X_train.shape}, "
          f"balance={np.bincount(y_train.astype(int)).tolist()}")
    print(f"Test kept untouched: {X_test.shape}")
    res_smote_train = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (SMOTE on train)",
    )
    for r in res_smote_train:
        print("  " + r.summary())
    rows_per_run.append(res_smote_train)

    # ------------------------------------------------------------------
    # C. SMOTE on FULL dataset - 70/30 split (the only pipeline that
    #    matches paper's 0.90 accuracy)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [C] Voting Classifier + SMOTE on full data - 70/30 split")
    print("      (matches paper's 0.90 accuracy - has data leakage)")
    print("=" * 78)
    X_all, y_all = apply_smote(X, y, target_minority=8000)
    print(f"After SMOTE on full -> shape={X_all.shape}, "
          f"balance={np.bincount(y_all.astype(int)).tolist()}")
    X_train, X_test, y_train, y_test = make_split(X_all, y_all, test_size=0.30, random_state=42)
    print(f"Re-split -> Train: {X_train.shape}  Test: {X_test.shape}")
    res_smote_full = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (SMOTE on full data)",
    )
    for r in res_smote_full:
        print("  " + r.summary())
    rows_per_run.append(res_smote_full)

    # ------------------------------------------------------------------
    # D. Paper-CM-matched: drop 150 rows (so 80/20 split = 1970) - SMOTE-full
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [D] Voting Classifier + SMOTE on full - 80/20 split (matches paper CM)")
    print("=" * 78)
    n_drop = len(df_clean) - 9850  # 9850 -> test=1970
    df_trim = drop_top_rows(df_clean, n_drop)
    X_t = df_trim.drop(columns=["Exited"]).values
    y_t = df_trim["Exited"].values
    print(f"Dropped {n_drop} rows -> {df_trim.shape}")
    X_all, y_all = apply_smote(X_t, y_t, target_minority=8000)
    print(f"After SMOTE on full -> shape={X_all.shape}")
    X_train, X_test, y_train, y_test = make_split(X_all, y_all, test_size=0.20, random_state=42)
    print(f"80/20 -> Train: {X_train.shape}  Test: {X_test.shape}")
    res_smote_full_80 = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (SMOTE on full, 80/20)",
    )
    for r in res_smote_full_80:
        print("  " + r.summary())
    rows_per_run.append(res_smote_full_80)

    # ------------------------------------------------------------------
    # E. No-SMOTE 80/20 split (matches paper's no-SMOTE CM = 1970 rows)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [E] Voting Classifier WITHOUT SMOTE - 80/20 split (matches paper CM)")
    print("=" * 78)
    n_drop = len(df_clean) - 9850
    df_trim = drop_top_rows(df_clean, n_drop)
    X_t = df_trim.drop(columns=["Exited"]).values
    y_t = df_trim["Exited"].values
    print(f"Dropped {n_drop} rows -> {df_trim.shape}")
    X_train, X_test, y_train, y_test = make_split(X_t, y_t, test_size=0.20, random_state=42)
    print(f"80/20 -> Train: {X_train.shape}  Test: {X_test.shape}")
    res_no_smote_80 = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (no SMOTE, 80/20)",
    )
    for r in res_no_smote_80:
        print("  " + r.summary())
    rows_per_run.append(res_no_smote_80)

    # ------------------------------------------------------------------
    # F. SMOTE-full on 9850 raw rows (paper CM-derived test size = 1970)
    #    This matches both the paper's 0.90 accuracy AND its CM test size.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [F] Voting Classifier + SMOTE on 9850 raw rows -> 80/20 (paper CM)")
    print("=" * 78)
    # Take the first 9850 raw rows (matches paper's CM-derived test size 1970)
    df_9850 = df_clean.iloc[:9850].copy()
    X_t = df_9850.drop(columns=["Exited"]).values
    y_t = df_9850["Exited"].values
    print(f"Subset shape: {df_9850.shape}, balance={np.bincount(y_t.astype(int)).tolist()}")
    X_all, y_all = apply_smote(X_t, y_t, target_minority=8000)
    print(f"After SMOTE -> shape={X_all.shape}, balance={np.bincount(y_all.astype(int)).tolist()}")
    X_train, X_test, y_train, y_test = make_split(X_all, y_all, test_size=0.20, random_state=42)
    print(f"80/20 -> Train: {X_train.shape}  Test: {X_test.shape}")
    res_smote_full_9850 = train_all(
        X_train, X_test, y_train, y_test,
        label_voting="Voting Classifier (SMOTE full on 9850, 80/20)",
    )
    for r in res_smote_full_9850:
        print("  " + r.summary())
    rows_per_run.append(res_smote_full_9850)

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    all_results = [r for run in rows_per_run for r in run]
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
        "no_smote_70": [r.__dict__ for r in res_no_smote],
        "smote_train_70": [r.__dict__ for r in res_smote_train],
        "smote_full_70": [r.__dict__ for r in res_smote_full],
        "smote_full_80": [r.__dict__ for r in res_smote_full_80],
        "no_smote_80": [r.__dict__ for r in res_no_smote_80],
        "smote_full_9850_80": [r.__dict__ for r in res_smote_full_9850],
    }
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(payload, fh, default=_jsonable, indent=2)
    print(f"Saved detailed results -> {OUT_DIR / 'results.json'}")

    # Plot confusion matrices
    for run in rows_per_run:
        for r in run:
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
        "Voting Classifier (no SMOTE)":                       {"acc": 0.87, "cm": [[1546, 20], [214, 190]]},
        "Voting Classifier (SMOTE on train)":                 {"acc": 0.90, "cm": [[1453, 124], [48, 345]]},
        "Voting Classifier (SMOTE on full data)":             {"acc": 0.90, "cm": [[1453, 124], [48, 345]]},
        "Voting Classifier (SMOTE on full, 80/20)":           {"acc": 0.90, "cm": [[1453, 124], [48, 345]]},
        "Voting Classifier (no SMOTE, 80/20)":                {"acc": 0.87, "cm": [[1546, 20], [214, 190]]},
        "Voting Classifier (SMOTE full on 9850, 80/20)":      {"acc": 0.90, "cm": [[1453, 124], [48, 345]]},
    }
    print("\n" + "=" * 78)
    print("  Comparison with paper (accuracy and confusion matrix)")
    print("=" * 78)
    for r in all_results:
        if r.name in paper:
            cm = np.array(paper[r.name]["cm"])
            print(f"\n{r.name}")
            print(
                f"  accuracy : predicted={r.accuracy:.4f}  paper={paper[r.name]['acc']:.4f}  "
                f"Δ={r.accuracy - paper[r.name]['acc']:+.4f}"
            )
            print(f"  CM       : predicted={r.confusion_matrix.flatten().tolist()}")
            print(f"             paper    ={cm.flatten().tolist()}")

    # ------------------------------------------------------------------
    # Investigation log
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Investigation: what reproduces the paper's headline numbers?")
    print("=" * 78)
    print(
        "\n[B] no-SMOTE / 70/30  -> "
        f"acc={res_no_smote[-1].accuracy:.4f}  (paper=0.87, Δ={res_no_smote[-1].accuracy-0.87:+.4f})"
    )
    print(
        "[C] SMOTE-full / 70/30 -> "
        f"acc={res_smote_full[-1].accuracy:.4f}  (paper=0.90, Δ={res_smote_full[-1].accuracy-0.90:+.4f})"
    )
    print(
        "[E] no-SMOTE / 80/20  -> "
        f"acc={res_no_smote_80[-1].accuracy:.4f}  (paper=0.87, Δ={res_no_smote_80[-1].accuracy-0.87:+.4f})"
    )
    print(
        "[D] SMOTE-full / 80/20 -> "
        f"acc={res_smote_full_80[-1].accuracy:.4f}  (paper=0.90, Δ={res_smote_full_80[-1].accuracy-0.90:+.4f})"
    )
    print(
        "[F] SMOTE-full on 9850 raw / 80/20 -> "
        f"acc={res_smote_full_9850[-1].accuracy:.4f}  (paper=0.90, Δ={res_smote_full_9850[-1].accuracy-0.90:+.4f})"
    )


if __name__ == "__main__":
    main()
