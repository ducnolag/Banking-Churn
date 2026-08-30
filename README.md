# Bank Churn Voting Classifier - Reproduction

Reproduction of the experiments described in:

> Bhuria R., Gupta S., Kaur U., Bharany S., Rehman A. U., Hussen S., Tejani G. G., Jangir P. (2025).
> **"Ensemble-based customer churn prediction in banking: a voting classifier approach
> for improved client retention using demographic and behavioral data."**
> *Discover Sustainability* 6, 28.
> https://link.springer.com/article/10.1007/s43621-025-00807-8

## Dataset

The Kaggle **"Bank Customer Churn Prediction Dataset"** (saurabhbadole), the
exact 10,000-row Churn_Modelling.csv referenced in the paper's Data Availability
statement. Place it in `data/Churn_Modelling.csv`.

## What the paper actually does (data-quality pipeline)

| Section | What the paper does | Where in this repo |
|---|---|---|
| 3.1 | "Missing data management" - verify no NaN values | `preprocessing.report_missing` |
| 3.1 | Drop RowNumber, CustomerId, Surname | `preprocessing.drop_identifiers` |
| 3.1 | One-hot encode categorical variables | `preprocessing.encode_categoricals` |
| 3.1 | Feature scaling | `preprocessing.feature_scaling` (StandardScaler) |
| 3.5 | Box plots for every numeric column (Fig. 11) | `preprocessing.plot_boxplots_all` -> `outputs/figures/fig11_boxplots_before.png` |
| 3.6 | IQR outlier removal on CreditScore, Age, NumOfProducts (Fig. 12) | `preprocessing.iqr_remove_outliers` -> `outputs/figures/fig12_*.png` |
| 3.1 | SMOTE balances 7,963 -> 8,000 churn (= 15,963 rows total) | `preprocessing.apply_smote` |
| 3.7 | Hard-vote Voting Classifier over DT / RF / KNN / SVC / XGBoost | `models.build_voting_classifier` |

## Layout

```
d:\NCKH\
├── data/
│   └── Churn_Modelling.csv                 (10,000 x 14)
├── src/
│   ├── main.py                             # End-to-end pipeline (6 experiments)
│   ├── preprocessing.py                    # All Section 3.1, 3.5, 3.6 functions
│   ├── models.py                           # 5 base learners + ensemble
│   ├── utils.py                            # Confusion-matrix plotting helper
│   ├── eda.py                              # All paper figures (Fig. 3 - 12)
│   ├── experiment_grid.py                  # Reproduction search grid
│   ├── voting_smote_search.py              # SMOTE sampler comparison
│   ├── tune_voting.py                      # Hyper-parameter tuning
│   ├── smote_leak_test.py                  # SMOTE-before-split variants
│   └── smote_leak_search.py                # Test-size sweep with leakage
├── outputs/
│   ├── figures/                            # Paper figures + confusion matrices
│   ├── metrics_summary.csv                 # All model metrics
│   ├── results.json                        # Confusion matrices + classification reports
│   ├── experiment_grid.csv
│   ├── voting_smote_grid.csv
│   └── tuned_voting.csv
└── README.md
```

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run the full data-quality + modelling pipeline (6 experiments)
python src/main.py

# 3. regenerate the paper's EDA figures
python src/eda.py
```

## Reproduction results

Six configurations are evaluated.  All of them apply the Section 3.1 - 3.6
data-quality pipeline **identically** (drop IDs -> one-hot encode -> scale ->
IQR outlier removal on CreditScore / Age / NumOfProducts at factor 1.5, bounds
[383, 919], [14, 62], [-0.5, 3.5]).

| # | Configuration | Predicted | Paper | Δ |
|---|---|---:|---:|---:|
| A | Voting, no SMOTE - 70/30           | 0.8579 | 0.87 | −0.0121 |
| B | Voting, SMOTE on train only - 70/30 | 0.7935 | 0.90 | −0.1065 |
| C | Voting, SMOTE on full data - 70/30  | 0.9086 | 0.90 | +0.0086 |
| D | Voting, SMOTE on full - 80/20       | 0.9085 | 0.90 | +0.0085 |
| E | Voting, no SMOTE - 80/20            | 0.8615 | 0.87 | −0.0085 |
| F | Voting, SMOTE on 9850 raw rows - 80/20 | 0.9085 | 0.90 | +0.0085 |

**Best matches (giống paper nhất)**:
- No-SMOTE headline 0.87 → **Config E (80/20) cho acc = 0.8615**, Δ = −0.0085.
- SMOTE headline 0.90 → **Config C/D/F cho acc = 0.9085–0.9086**, Δ = +0.0085.

Config C/D/F all reproduce the paper's 0.90 accuracy within 0.01. They apply
SMOTE to the full dataset *before* the train/test split, which is a
data-leakage issue acknowledged in the paper text (Section 3.1 says SMOTE is
applied to the training set only, but the published post-SMOTE size of
15,963 = 7,963 non-churn + 8,000 churn only makes sense if SMOTE is applied
before splitting).  Both pipelines are provided so the reader can compare.

## Notes on reproducibility

* The paper has multiple internal inconsistencies - e.g. the SMOTE
  confusion matrix sums to 1,970 test rows that only arise from a specific
  data-leakage pipeline, and the text claims a precision/recall/F1 of 0.90
  for both classes which cannot simultaneously hold for the CM they publish.
* Our best honest reproduction of the **no-SMOTE** Voting Classifier reaches
  0.8615 accuracy (Config E), matching the paper's headline number 0.87
  within 0.85%.
* Config B (SMOTE on train only - correct methodology) only reaches 0.79
  because the paper's stated target minority (8,000) over-samples the train
  set and the held-out real test distribution is imbalanced, hurting accuracy.
* All five base learners and the Voting Classifier are evaluated for each
  configuration; their confusion matrices are saved to
  `outputs/figures/cm_*.png`.

## How to pick which number to cite

| Your priority | Use |
|---|---|
| **Honest reproduction** (no data leakage) | Config A (no SMOTE) or Config E (no SMOTE 80/20) |
| **Match paper headline 0.87 / 0.90** | Config E / Config F (or D / C) |
| **Reproduce paper's CM test size (1970)** | Config E (1914) is closest |
| **Methodologically correct SMOTE pipeline** | Config B — but accuracy drops to ~0.79 |
