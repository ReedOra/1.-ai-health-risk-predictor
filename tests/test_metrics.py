from src.model_evaluation import compute_metrics


def test_known_confusion_matrix():
    m = compute_metrics([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0], [.9, .8, .4, .1, .2, .1, .6, .3])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert abs(m["accuracy"] - 6 / 8) < 1e-9 and abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9 and abs(m["specificity"] - 4 / 5) < 1e-9
    assert m["support_positive"] == 3 and m["support_negative"] == 5 and m["roc_auc"] is not None


def test_zero_denominators_are_none_not_invented():
    m = compute_metrics([0, 0, 0], [0, 0, 0])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["roc_auc"] is None and m["roc_auc_note"]
