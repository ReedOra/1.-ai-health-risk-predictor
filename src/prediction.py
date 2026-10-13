"""Load a saved pipeline, validate user input, return a prediction. Models are loaded once by the caller."""
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from config import settings as S


class ModelNotTrainedError(Exception):
    pass


class InputValidationError(ValueError):
    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(problems))


def load_model(task, models_dir=None):
    d = Path(models_dir or S.MODELS_DIR / task)
    mp, meta = d / "model.joblib", d / "metadata.json"
    if not mp.exists() or not meta.exists():
        raise ModelNotTrainedError(f"No trained {S.TASKS[task]['title']} model found. Run: python -m src.train_{task}")
    return joblib.load(mp), json.loads(meta.read_text())  # only load files this project produced


def validate_inputs(task, values: dict) -> dict:
    spec, problems, clean = S.TASKS[task], [], {}
    for name, (lo, hi, required, label, _) in spec["inputs"].items():
        v = values.get(name)
        if v is None or (isinstance(v, str) and not v.strip()):
            if required:
                problems.append(f"{label} is required.")
            clean[name] = float("nan")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            problems.append(f"{label} must be a number."); continue
        if math.isnan(v) or math.isinf(v) or not (lo <= v <= hi):
            problems.append(f"{label} must be between {lo} and {hi}."); continue
        clean[name] = v
    extra = set(values) - set(spec["inputs"])
    if extra:
        problems.append(f"Unexpected fields: {sorted(extra)}")
    if problems:
        raise InputValidationError(problems)
    return clean


def predict(task, values, model=None, meta=None):
    model, meta = (model, meta) if model is not None else load_model(task)
    clean = validate_inputs(task, values)
    row = pd.DataFrame([clean])[meta["features"]]  # enforce training feature order
    score = float(model.predict_proba(row)[0, 1])
    return {"score": score, "label": int(score >= meta["threshold"]), "threshold": meta["threshold"],
            "model": meta["selected_model"],
            "imputed": [k for k, v in clean.items() if math.isnan(v)],
            "note": ("Uncalibrated model score for the dataset-defined positive class. "
                     "It is NOT a clinical probability and NOT a diagnosis.")}
