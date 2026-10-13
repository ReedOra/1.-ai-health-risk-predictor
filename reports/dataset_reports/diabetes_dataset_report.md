# Diabetes dataset report

- Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Records loaded: 768
- Features: 8
- Target: `Outcome` — 1 = diabetes onset within 5 years (dataset-defined, NOT a clinical diagnosis)
- Final usable records: 768

## Missing values after cleaning
- Glucose: 5 (0.7%)
- BloodPressure: 35 (4.6%)
- SkinThickness: 227 (29.6%)
- Insulin: 374 (48.7%)
- BMI: 11 (1.4%)

## Class distribution
- class 0: 500 (65.1%)
- class 1: 268 (34.9%)

## Cleaning decisions
- Glucose: 5 zero values treated as missing (zero is physiologically impossible).
- BloodPressure: 35 zero values treated as missing (zero is physiologically impossible).
- SkinThickness: 227 zero values treated as missing (zero is physiologically impossible).
- Insulin: 374 zero values treated as missing (zero is physiologically impossible).
- BMI: 11 zero values treated as missing (zero is physiologically impossible).
- Removed 0 exact duplicate rows.
- No outlier removal and no resampling applied; missing values are imputed inside the cross-validated pipeline (train-only fit).

## Known limitations
- Small public dataset; not representative of the general population.
- The target is a dataset-defined label, not a confirmed clinical diagnosis.
