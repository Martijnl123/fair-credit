# Fairness vs calibration in credit default models

What does a fairness constraint cost a lender that needs calibrated default probabilities?
This repo applies **Calibrated Equalized Odds** (Pleiss, Raghavan, Wu, Kleinberg & Weinberger,
"On Fairness and Calibration", NeurIPS 2017) to three credit default datasets and measures
fairness, predictive performance and the lender's view (approval and default rates per group)
before and after mitigation.

Built as the final assignment for Data Wrangling (INFOMDW).

## Method in one paragraph

Train a classifier that outputs calibrated scores. For each group, compute a generalized
error rate (average score given to true negatives, or average of 1 − score for true positives).
The group with the lower error is moved up to match the other: for a random share of that group,
the model's score is replaced by the group's base rate. A constant base rate is itself calibrated,
so calibration survives while one error rate is equalised. The post-processor is fitted on a
validation set and applied to the test set.

## Datasets

| Name | Source | Rows | Label (1 = favourable) | Protected attribute |
|---|---|---|---|---|
| `german` | UCI Statlog German Credit | 1,000 | good credit risk | age (> 25 privileged) |
| `taiwan` | UCI Default of Credit Card Clients | 30,000 | no default next month | sex (male privileged) |
| `homecredit` | Kaggle Home Credit Default Risk, `application_train.csv` | ~307,000 | loan repaid | sex (male privileged) |
| `synthetic` | generated in code | 4,000 | toy | toy |

Download the raw files into `data/raw/` (see `data/raw/README.md`). They are not committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                            # runs on synthetic data, no downloads needed
```

## Running

```bash
python run.py --dataset german     --model lr  --constraint fnr --seeds 5
python run.py --dataset taiwan     --model gbm --constraint fnr --seeds 5
python run.py --dataset homecredit --model lr  --constraint weighted --seeds 5
```

| Flag | Values | Meaning |
|---|---|---|
| `--dataset` | `german`, `taiwan`, `homecredit`, `synthetic` | which dataset |
| `--model` | `lr` (logistic regression), `gbm` (gradient boosting with isotonic calibration) | base classifier |
| `--constraint` | `fnr`, `fpr`, `weighted` | which generalized error rate to equalise |
| `--seeds` | integer, default 5 | number of random stratified splits |
| `--out` | folder, default `results` | where outputs go |

Each run writes to `results/`:
`*_labels.csv` (fairness of the original labels), `*_runs.csv` (every seed),
`*_summary.csv` (mean and std per stage) and `*_reliability.png` (calibration per group, seed 0).

## Pipeline

1. Load data; encode label as 1 = favourable and protected attribute as 1 = privileged.
2. Compute DP_Diff, DIR and consistency on the original labels.
3. Stratified split on label × group: 60% train, 20% validation, 20% test.
4. Train the base classifier on train (baseline).
5. Fit Calibrated Equalized Odds on validation scores, apply it to test scores.
6. On test: DP_Diff, DIR, consistency, accuracy, balanced accuracy, AUC, TPR/FPR gaps,
   approval rate and default rate among approved per group, for baseline and mitigated.
7. Repeat over seeds and report mean and standard deviation.

## Metric definitions

- **DP_Diff** = P(ŷ = 1 | unprivileged) − P(ŷ = 1 | privileged); fair at 0.
- **DIR** = P(ŷ = 1 | unprivileged) / P(ŷ = 1 | privileged); fair at 1, acceptable 0.8 to 1.25.
- **Consistency** = 1 − mean |ŷᵢ − mean ŷ of i's 5 nearest neighbours|, on standardised features.
  On data above 20,000 rows a fixed random subsample is used (same seed before and after).

## Results

_To be filled in._

## Limitations

_To be filled in._ Expected points: the method targets equalized odds, not demographic parity,
so DP_Diff and DIR may not improve; random replacement lowers consistency and means identical
applicants can get different decisions; small test sets (German: 200 rows) make DIR unstable.
