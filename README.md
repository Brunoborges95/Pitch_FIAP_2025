# Credit Card Fraud Detection

A comparison of supervised and semi-supervised approaches to detecting fraudulent credit card transactions on a highly imbalanced dataset. Everything lives in a single notebook, [credit-card-model.ipynb](credit-card-model.ipynb).

## Overview

Only 492 of the 284,807 transactions in the dataset (0.17%) are fraud. A model that labels everything as "normal" reaches 99.83% accuracy while catching nothing, so accuracy is discarded and the work focuses on:

- handling class imbalance through undersampling,
- picking the decision threshold that maximizes the F2-score (recall weighted twice as much as precision, since a missed fraud costs more than a false alarm),
- comparing tree-based classifiers against an autoencoder trained only on normal transactions.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

| Column | Description |
| --- | --- |
| `Time` | Seconds elapsed between the transaction and the first transaction in the dataset |
| `V1`–`V28` | PCA-encoded features, anonymized to protect customer data |
| `Amount` | Transaction amount |
| `Class` | Target: `1` = fraud, `0` = normal |

The CSV is not included in this repository. Download `creditcard.csv` and place it next to the notebook.

## Method

1. **Preprocessing** — `Time` and `Amount` are scaled with `RobustScaler`, which is less sensitive to outliers than standard scaling. The data is shuffled and split 80/20 with stratification.
2. **Undersampling** — the normal class in the training set is randomly reduced until fraud represents 10% of the samples (9:1 ratio). The test set keeps the original distribution.
3. **Supervised models** — Random Forest (1,600 trees) and XGBoost, both trained on the undersampled set.
4. **Semi-supervised model** — a dense autoencoder (30 → 300 → 150 → 150 → 75 → 150 → 300 → 30) trained to reconstruct normal transactions only. Transactions with a high reconstruction error are flagged as fraud. Training uses early stopping and keeps the best weights in `best_weights.hdf5`.
5. **Threshold selection** — for each model, candidate thresholds are scanned and the one with the highest F2-score is kept.

## Metrics

- **Precision** — share of transactions flagged as fraud that really are fraud.
- **Recall** — share of actual frauds that were caught.
- **F2-score** — harmonic mean of precision and recall with recall weighted by β = 2.
- **Cohen's kappa** — agreement between predictions and labels, corrected for chance.

## Results

Scores on the held-out test set (56,962 transactions, 98 frauds), at the threshold that maximizes F2:

| Model | Recall | Precision | F2 | Kappa |
| --- | --- | --- | --- | --- |
| Random Forest | 0.857 | 0.712 | 0.824 | 0.777 |
| XGBoost | 0.847 | 0.748 | 0.825 | 0.794 |
| Autoencoder | 0.500 | 0.183 | 0.371 | 0.266 |

The two tree-based models perform almost identically, with XGBoost slightly ahead in precision and kappa. The autoencoder, which never sees a fraud label during training, falls well behind.

The notebook also includes precision–recall trade-off curves, confusion matrices and a feature importance chart for the Random Forest.

## Running

Requires Python 3 and Jupyter, plus:

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn tensorflow seaborn matplotlib
```

Then open the notebook:

```bash
jupyter notebook credit-card-model.ipynb
```

A GPU is optional but speeds up the autoencoder training.

## Notes

- The **SageMaker** section is an illustrative template of how the XGBoost model could be trained and deployed on AWS. It uses placeholder bucket names and features, and is not meant to be executed as is.
- SMOTE is explained and imported as an oversampling alternative, but the reported results use undersampling only.
- The notebook was developed interactively, so some cells depend on variables defined further down (for example, `eval_function` reads the autoencoder's reconstruction error). It does not run cleanly from top to bottom without reordering.

## Project structure

```
├── credit-card-model.ipynb   # Full analysis and models
├── best_weights.hdf5         # Best autoencoder weights saved during training
└── README.md
```
