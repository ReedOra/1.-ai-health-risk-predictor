import tempfile
from pathlib import Path

import pandas as pd

from config import settings as S
from src import data_loader as dl
from tests.helpers import synthetic_df


def test_missing_file_gives_actionable_error():
    try:
        dl.load_raw(S.DIABETES, "does_not_exist.csv"); assert False
    except dl.DatasetError as e:
        assert "not found" in str(e)


def test_schema_validation_rejects_missing_columns():
    try:
        dl.validate_schema(pd.DataFrame({"a": range(100)}), S.DIABETES); assert False
    except dl.DatasetError as e:
        assert "Missing required columns" in str(e)


def test_corrupted_file_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.csv"; p.write_text("x,y\n1,2\n")
        try:
            dl.load_raw(S.DIABETES, p); assert False
        except dl.DatasetError:
            pass


def test_headerless_heart_file_and_binary_target():
    df = synthetic_df(S.HEART); df["num"] = df["num"] * 3  # labels 0 and 3, like the multi-class original
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "h.data"; df.to_csv(p, header=False, index=False)
        cleaned, _ = dl.clean(dl.load_raw(S.HEART, p), S.HEART)
    assert set(cleaned["num"].unique()) <= {0, 1}


def test_zero_handling_keeps_valid_pregnancies():
    df = synthetic_df(S.DIABETES); df.loc[:4, "Glucose"] = 0; df.loc[:9, "Pregnancies"] = 0
    cleaned, _ = dl.clean(df, S.DIABETES)
    assert cleaned["Glucose"].isna().sum() >= 5 and (cleaned["Pregnancies"] == 0).sum() >= 10
