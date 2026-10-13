# Datasets
A **dataset** is a table: each row is a person, each column a **feature** (e.g. Glucose); the **target** is the label to predict.
Diabetes and heart disease use different measurements, so they get separate datasets and models; they are never merged.

| Task | Source | Target | Notes |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes Database (NIDDK; Kaggle mirror https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | Outcome: 1 = diabetes onset within 5 years | Adult women of Pima heritage; zeros in some columns mean "not measured" |
| Heart disease | UCI Heart Disease, Cleveland subset https://archive.ics.uci.edu/dataset/45/heart+disease | `num` 0 = no narrowing; 1–4 collapsed to 1 | Few hundred patients, old data; needs clinical tests |

Actual row counts, missing values and class balance are written to `reports/dataset_reports/` when you train; none are quoted here because they must come from the files you load.
Limitations: small, old, narrow populations; labels are dataset-defined; several inputs need lab or clinical tests an ordinary person may not have (the app will not guess them).
