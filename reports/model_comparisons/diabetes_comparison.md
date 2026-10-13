# Diabetes model comparison (selected: **Extra Trees**)
Selection criterion: mean cross-validated F1-score of the positive class (ties within 0.005 -> higher CV recall).

| Model | CV Acc | CV Prec | CV Rec | CV F1 (±std) | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.762 | 0.649 | 0.701 | 0.672 ± 0.013 | 0.734 | 0.603 | 0.704 | 0.650 | 0.813 |
| Decision Tree | 0.690 | 0.547 | 0.720 | 0.618 ± 0.048 | 0.714 | 0.561 | 0.852 | 0.676 | 0.789 |
| Random Forest | 0.783 | 0.695 | 0.682 | 0.687 ± 0.016 | 0.760 | 0.655 | 0.667 | 0.661 | 0.824 |
| **Extra Trees ✔** | 0.770 | 0.652 | 0.738 | 0.692 ± 0.021 | 0.721 | 0.582 | 0.722 | 0.645 | 0.823 |
| Hist Gradient Boosting | 0.762 | 0.690 | 0.593 | 0.636 ± 0.023 | 0.760 | 0.681 | 0.593 | 0.634 | 0.815 |
