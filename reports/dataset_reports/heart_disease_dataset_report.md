# Heart disease dataset report

- Source: https://archive.ics.uci.edu/dataset/45/heart+disease
- Records loaded: 303
- Features: 13
- Target: `num` — num 1-4 collapsed to 1 = angiographic narrowing present; 0 = absent (dataset-defined, NOT a clinical diagnosis)
- Final usable records: 303

## Missing values after cleaning
- ca: 4 (1.3%)
- thal: 2 (0.7%)

## Class distribution
- class 0: 164 (54.1%)
- class 1: 139 (45.9%)

## Cleaning decisions
- Removed 0 exact duplicate rows.
- Target collapsed to binary: original labels 1-4 (139 rows) -> 1; 0 stays 0.
- No outlier removal and no resampling applied; missing values are imputed inside the cross-validated pipeline (train-only fit).

## Known limitations
- Small public dataset; not representative of the general population.
- The target is a dataset-defined label, not a confirmed clinical diagnosis.
