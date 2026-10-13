# Heart disease model comparison (selected: **Logistic Regression**)
Selection criterion: mean cross-validated F1-score of the positive class (ties within 0.005 -> higher CV recall).

| Model | CV Acc | CV Prec | CV Rec | CV F1 (±std) | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression ✔** | 0.847 | 0.848 | 0.819 | 0.831 ± 0.022 | 0.869 | 0.812 | 0.929 | 0.867 | 0.958 |
| Decision Tree | 0.748 | 0.722 | 0.756 | 0.735 ± 0.054 | 0.770 | 0.719 | 0.821 | 0.767 | 0.877 |
| Random Forest | 0.814 | 0.809 | 0.783 | 0.792 ± 0.047 | 0.885 | 0.839 | 0.929 | 0.881 | 0.946 |
| Extra Trees | 0.834 | 0.827 | 0.819 | 0.818 ± 0.036 | 0.852 | 0.828 | 0.857 | 0.842 | 0.932 |
| Hist Gradient Boosting | 0.802 | 0.808 | 0.747 | 0.772 ± 0.051 | 0.918 | 0.897 | 0.929 | 0.912 | 0.963 |
