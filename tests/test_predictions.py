from config import settings as S
from src import prediction as pr
from tests.test_models import _train


def _valid(task):
    return {k: (v[0] + v[1]) / 2 if k not in ("cp", "restecg", "slope", "thal", "sex", "fbs", "exang", "ca") else v[0]
            for k, v in S.TASKS[task]["inputs"].items()}


def test_roundtrip_prediction_valid_output():
    for task in S.TASKS:
        d, _ = _train(task)
        model, meta = pr.load_model(task, d / "m")
        out = pr.predict(task, _valid(task), model, meta)
        assert 0.0 <= out["score"] <= 1.0 and out["label"] in (0, 1)


def test_out_of_range_and_non_numeric_rejected():
    v = _valid("diabetes"); v["Glucose"] = 9999; v["BMI"] = "abc"
    try:
        pr.validate_inputs("diabetes", v); assert False
    except pr.InputValidationError as e:
        assert len(e.problems) == 2


def test_missing_required_rejected_optional_allowed():
    v = _valid("diabetes"); v["Insulin"] = None
    assert pr.validate_inputs("diabetes", v)["Insulin"] != pr.validate_inputs("diabetes", v)["Insulin"]  # NaN
    v["Glucose"] = None
    try:
        pr.validate_inputs("diabetes", v); assert False
    except pr.InputValidationError:
        pass


def test_missing_model_files_message():
    try:
        pr.load_model("diabetes", "/nonexistent"); assert False
    except pr.ModelNotTrainedError as e:
        assert "python -m src.train_diabetes" in str(e)
