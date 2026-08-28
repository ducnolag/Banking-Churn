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
│   ├── main.py                             # End-to-end pipeline (3 experiments)
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

# 2. run the full data-quality + modelling pipeline
python src/main.py

# 3. regenerate the paper's EDA figures
python src/eda.py
```

## Reproduction results

Three configurations are evaluated.  All of them apply the Section 3.1 - 3.6
data-quality pipeline **identically** (drop IDs -> one-hot encode -> scale ->
IQR outlier removal on CreditScore / Age / NumOfProducts at factor 1.5, bounds
[383, 919], [14, 62], [-0.5, 3.5]). They differ only in **when** SMOTE is
applied.

| Configuration                         | Predicted | Paper |
| ------------------------------------- | ---------:| -----:|
| Voting Classifier, no SMOTE           | 0.8558    | 0.87  |
| Voting Classifier, SMOTE on train only | 0.8206   | 0.90  |
| Voting Classifier, SMOTE on full data | **0.9101**| 0.90  |

The third configuration reproduces the paper's 0.90 metric within 0.01.  It
applies SMOTE to the full dataset *before* the train/test split, which is a
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
  0.856 accuracy, matching the paper's headline number 0.87 within 1.4%.
* All five base learners and the Voting Classifier are evaluated; their
  confusion matrices are saved to `outputs/figures/`.
