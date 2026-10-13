"""Train, select (CV only), then evaluate once on the held-out test set and save artifacts."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as S  # noqa: E402
from src import data_loader as dl, model_evaluation as ev, preprocessing as pp  # noqa: E402

SELECTION_CRITERION = "mean cross-validated F1-score of the positive class (ties within 0.005 -> higher CV recall)"


def run_task(spec, df, models_dir, reports_dir, seed=S.RANDOM_SEED, cv_folds=S.CV_FOLDS, log=()):
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    for d in (models_dir, reports_dir / "metrics", reports_dir / "figures", reports_dir / "dataset_reports",
              reports_dir / "model_comparisons"):
        d.mkdir(parents=True, exist_ok=True)
    X, y = dl.split_xy(df, spec)
    # 1) split BEFORE any model selection
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=S.TEST_SIZE, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pipelines = {n: pp.build_pipeline(spec, n, m) for n, m in pp.candidate_models(seed).items()}

    # 2) cross-validation on TRAINING data only
    rows = []
    for name, pipe in pipelines.items():
        r = cross_validate(clone(pipe), X_tr, y_tr, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        rows.append({"model": name, **{f"cv_{k}_mean": float(np.mean(r[f"test_{k}"])) for k in
                                       ["accuracy", "precision", "recall", "f1", "roc_auc"]},
                     "cv_f1_std": float(np.std(r["test_f1"]))})
    best = max(rows, key=lambda r: (round(r["cv_f1_mean"] / 0.005), r["cv_recall_mean"]))
    selected = best["model"]  # decided BEFORE the test set is touched

    # 3) one-time held-out evaluation (reporting only; never fed back into selection)
    for r in rows:
        pipe = clone(pipelines[r["model"]]).fit(X_tr, y_tr)
        score = pipe.predict_proba(X_te)[:, 1]
        pred = (score >= S.DECISION_THRESHOLD).astype(int)
        r["test"] = ev.compute_metrics(y_te, pred, score)
        r["selected"] = r["model"] == selected
        if r["selected"]:
            final_pipe, final_score = pipe, score

    final = next(r for r in rows if r["selected"])["test"]
    figs = reports_dir / "figures"
    ev.save_confusion_matrix(final["confusion_matrix"], figs / f"{spec['name']}_confusion_matrix.png",
                             f"{spec['title']}: {selected} (test)")
    ev.save_comparison_chart(rows, figs / f"{spec['name']}_model_comparison.png", f"{spec['title']}: CV F1 by model")
    ev.save_class_distribution(y, figs / f"{spec['name']}_class_distribution.png", f"{spec['title']}: class distribution")
    if final["roc_auc"] is not None:
        ev.save_roc_curve(y_te, final_score, figs / f"{spec['name']}_roc_curve.png", f"{spec['title']}: ROC (test)")

    meta = {"task": spec["name"], "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__, "seed": seed, "test_size": S.TEST_SIZE, "cv_folds": cv_folds,
            "threshold": S.DECISION_THRESHOLD, "selected_model": selected, "selection_criterion": SELECTION_CRITERION,
            "features": spec["numeric"] + spec["categorical"], "target": spec["target"],
            "positive_meaning": spec["positive_meaning"], "source_url": spec["source_url"],
            "n_records_used": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "cleaning_log": list(log), "model_params": {k: str(v) for k, v in final_pipe.named_steps["model"].get_params().items()}}
    joblib.dump(final_pipe, models_dir / "model.joblib")
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    results = {"meta": meta, "candidates": rows, "final_test": final}
    (reports_dir / "metrics" / f"{spec['name']}_metrics.json").write_text(json.dumps(results, indent=2))
    (reports_dir / "model_comparisons" / f"{spec['name']}_comparison.md").write_text(comparison_markdown(spec, results))
    return results


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


def comparison_markdown(spec, res):
    out = [f"# {spec['title']} model comparison (selected: **{res['meta']['selected_model']}**)",
           f"Selection criterion: {res['meta']['selection_criterion']}.\n",
           "| Model | CV Acc | CV Prec | CV Rec | CV F1 (±std) | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["candidates"]:
        t = r["test"]
        out.append(f"| {'**' + r['model'] + ' ✔**' if r['selected'] else r['model']} | {_f(r['cv_accuracy_mean'])} | "
                   f"{_f(r['cv_precision_mean'])} | {_f(r['cv_recall_mean'])} | {_f(r['cv_f1_mean'])} ± {_f(r['cv_f1_std'])} | "
                   f"{_f(t['accuracy'])} | {_f(t['precision'])} | {_f(t['recall'])} | {_f(t['f1'])} | {_f(t['roc_auc'])} |")
    return "\n".join(out) + "\n"


def run_from_disk(task_name):
    """CLI path: real dataset -> clean -> train -> save. Raises DatasetError with instructions if data is missing."""
    spec = S.TASKS[task_name]
    dl.download_if_possible(spec)
    raw = dl.load_raw(spec)
    cleaned, log = dl.clean(raw, spec)
    (S.REPORTS_DIR / "dataset_reports").mkdir(parents=True, exist_ok=True)
    (S.REPORTS_DIR / "dataset_reports" / f"{task_name}_dataset_report.md").write_text(dl.dataset_report(raw, cleaned, log, spec))
    return run_task(spec, cleaned, S.MODELS_DIR / task_name, S.REPORTS_DIR, log=log)


if __name__ == "__main__":
    names = sys.argv[1:] or list(S.TASKS)
    for n in names:
        try:
            res = run_from_disk(n)
            print(f"[{n}] selected {res['meta']['selected_model']}; test F1 = {_f(res['final_test']['f1'])}")
        except dl.DatasetError as exc:
            print(f"[{n}] NOT TRAINED: {exc}")
