"""Dataset acquisition, schema validation and cleaning (no model code here)."""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetError(Exception):
    """Raised when a dataset is missing, malformed or fails validation."""


def download_if_possible(spec: dict, timeout: int = 20) -> Path:
    """Download the raw file when the spec has an official URL. Never fakes success."""
    path = Path(spec["raw_file"])
    if path.exists():
        return path
    if not spec.get("download_url"):
        raise DatasetError(f"Please download the {spec['title']} dataset manually. {spec['source_note']} "
                           f"Source: {spec['source_url']}  ->  save as {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(spec["download_url"], path)  # fixed https URL from settings
    except Exception as exc:
        raise DatasetError(f"Could not download {spec['download_url']} ({exc}). Download it in a "
                           f"browser and save it as {path}.") from exc
    return path


def _looks_headerless(path: Path) -> bool:
    first = pd.read_csv(path, header=None, nrows=1, na_values="?")
    return all(pd.api.types.is_number(v) for v in first.iloc[0].tolist())


def load_raw(spec: dict, path=None) -> pd.DataFrame:
    """Load the raw CSV and validate its schema against the task spec."""
    path = Path(path or spec["raw_file"])
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}. {spec['source_note']} Source: {spec['source_url']}")
    try:
        if _looks_headerless(path):
            df = pd.read_csv(path, header=None, names=spec["columns"], na_values="?")
        else:
            df = pd.read_csv(path, na_values="?")
    except Exception as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc
    return validate_schema(df, spec)


def validate_schema(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in spec["columns"] if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing required columns: {missing}")
    df = df[spec["columns"]]
    for col in spec["columns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[spec["target"]].isna().any():
        raise DatasetError("Target column contains missing or non-numeric values.")
    if len(df) < 50:
        raise DatasetError(f"Only {len(df)} rows; too few to train and evaluate reliably.")
    return df


def clean(df: pd.DataFrame, spec: dict):
    """Apply documented cleaning. Returns (clean_df, log). Imputation happens later,
    inside the sklearn pipeline, so that it is fitted on training data only."""
    log, out = [], df.copy()
    for col in spec["zero_is_missing"]:
        n = int((out[col] == 0).sum())
        out.loc[out[col] == 0, col] = np.nan
        log.append(f"{col}: {n} zero values treated as missing (zero is physiologically impossible).")
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    log.append(f"Removed {before - len(out)} exact duplicate rows.")
    if spec["name"] == "heart_disease":
        n_pos = int((out[spec["target"]] > 0).sum())
        out[spec["target"]] = (out[spec["target"]] > 0).astype(int)
        log.append(f"Target collapsed to binary: original labels 1-4 ({n_pos} rows) -> 1; 0 stays 0.")
    bad = set(out[spec["target"]].unique()) - {0, 1}
    if bad:
        raise DatasetError(f"Unexpected target values after cleaning: {bad}")
    log.append("No outlier removal and no resampling applied; missing values are imputed inside the "
               "cross-validated pipeline (train-only fit).")
    return out, log


def split_xy(df: pd.DataFrame, spec: dict):
    return df[spec["numeric"] + spec["categorical"]], df[spec["target"]].astype(int)


def dataset_report(raw: pd.DataFrame, cleaned: pd.DataFrame, log: list, spec: dict) -> str:
    """Markdown report built only from the data actually loaded."""
    miss = cleaned.isna().sum()
    miss_txt = "\n".join(f"- {c}: {int(n)} ({n / len(cleaned):.1%})" for c, n in miss.items() if n) or "- none"
    dist = cleaned[spec["target"]].value_counts().sort_index()
    dist_txt = "\n".join(f"- class {k}: {v} ({v / len(cleaned):.1%})" for k, v in dist.items())
    return (f"# {spec['title']} dataset report\n\n- Source: {spec['source_url']}\n"
            f"- Records loaded: {len(raw)}\n- Features: {len(spec['numeric']) + len(spec['categorical'])}\n"
            f"- Target: `{spec['target']}` — {spec['positive_meaning']}\n"
            f"- Final usable records: {len(cleaned)}\n\n## Missing values after cleaning\n{miss_txt}\n\n"
            f"## Class distribution\n{dist_txt}\n\n## Cleaning decisions\n" + "\n".join(f"- {l}" for l in log) +
            "\n\n## Known limitations\n- Small public dataset; not representative of the general population.\n"
            "- The target is a dataset-defined label, not a confirmed clinical diagnosis.\n")
