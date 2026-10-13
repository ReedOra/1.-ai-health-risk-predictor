import json
import tempfile
from pathlib import Path

from config import settings as S
from src import training as tr
from tests.helpers import synthetic_df

_CACHE = {}


def _train(task):
    if task not in _CACHE:
        d = tempfile.mkdtemp()
        res = tr.run_task(S.TASKS[task], synthetic_df(S.TASKS[task]), Path(d) / "m", Path(d) / "r", cv_folds=3)
        _CACHE[task] = (Path(d), res)
    return _CACHE[task]


def test_training_saves_artifacts_and_selects_without_test_data():
    for task in S.TASKS:
        d, res = _train(task)
        assert (d / "m" / "model.joblib").exists() and (d / "m" / "metadata.json").exists()
        assert (d / "r" / "metrics" / f"{task}_metrics.json").exists()
        assert (d / "r" / "figures" / f"{task}_confusion_matrix.png").exists()
        # selection rule must depend only on CV columns
        best_cv = max(res["candidates"], key=lambda r: (round(r["cv_f1_mean"] / 0.005), r["cv_recall_mean"]))
        assert best_cv["model"] == res["meta"]["selected_model"]
        assert res["meta"]["n_train"] + res["meta"]["n_test"] == res["meta"]["n_records_used"]
        assert json.loads((d / "m" / "metadata.json").read_text())["features"] == S.TASKS[task]["numeric"] + S.TASKS[task]["categorical"]


def test_all_five_candidates_compared():
    assert len(_train("diabetes")[1]["candidates"]) == 5
