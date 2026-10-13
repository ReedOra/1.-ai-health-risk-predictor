"""Metric calculation and figures. Zero denominators return None, never an invented value."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def _ratio(num, den):
    return float(num / den) if den else None


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Binary metrics with class 1 as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    roc, roc_note = None, None
    if y_score is None:
        roc_note = "no probability scores supplied"
    elif len(np.unique(y_true)) < 2:
        roc_note = "only one class present in y_true"
    else:
        roc = float(roc_auc_score(y_true, y_score))
    return {"n": int(len(y_true)), "support_positive": tp + fn, "support_negative": tn + fp,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": _ratio(tp + tn, len(y_true)), "precision": precision, "recall": recall,
            "f1": f1, "specificity": _ratio(tn, tn + fp), "roc_auc": roc, "roc_auc_note": roc_note,
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def save_confusion_matrix(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_roc_curve(y_true, y_score, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_comparison_chart(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["model"] for r in rows]
    cv = [r["cv_f1_mean"] for r in rows]
    err = [r["cv_f1_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(names, cv, xerr=err, color="#2a7f9e")
    ax.set_xlabel("Cross-validated F1 (positive class), mean ± std"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_class_distribution(y, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.bar([str(i) for i in counts.index], counts.values, color=["#7fb77e", "#d9534f"])
    ax.set_xlabel("Class (1 = positive)"); ax.set_ylabel("Records"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
