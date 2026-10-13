# Model evaluation
TP/TN/FP/FN = true/false positives/negatives (positive = class 1).
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 · Precision · Recall / (Precision + Recall)
- Specificity = TN / (TN + FP)
Zero denominators are reported as "n/a", never invented. ROC-AUC uses predicted probabilities, only if both classes are present.

**Why not accuracy alone?** If 35 of 100 people are positive, a model that always says "negative" is 65% accurate yet misses everyone. **False negatives** (missed cases) are especially worrying in health; **false positives** cause worry and extra tests. We select by positive-class F1 on cross-validation, which balances both, and show recall and specificity separately.

**Why train/test split?** Testing on data the model has seen flatters it. We hold out 20% first, choose the model with cross-validation on the other 80%, and look at the test set once. Preprocessing is inside the pipeline, so nothing learned from validation/test data leaks in.

**How the models work (short).** Logistic Regression weighs features into a probability. A Decision Tree asks a chain of yes/no questions. Random Forest and Extra Trees average many randomised trees. Gradient Boosting builds trees one after another, each correcting the last. None is assumed best; the CV comparison decides.
Scores are uncalibrated; results on small public datasets do not establish clinical validity.
