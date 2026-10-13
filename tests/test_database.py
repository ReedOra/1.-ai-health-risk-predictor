import tempfile
from pathlib import Path

from src import database as db


def _p():
    return Path(tempfile.mkdtemp()) / "t.db"


def test_save_list_clear_predictions():
    p = _p()
    db.save_prediction("diabetes", "Random Forest", 0.42, 1, p)
    rows = db.list_predictions(p)
    assert len(rows) == 1 and rows[0][1] == "diabetes" and rows[0][4] == 1
    db.clear_predictions(p)
    assert db.list_predictions(p) == []


def test_sql_injection_text_is_stored_as_data():
    p = _p()
    db.set_contact("x'); DROP TABLE settings;--", "+911234567890", p)
    assert db.get_contact(p)[0].startswith("x')")
    assert db.get_contact(p)[1] == "+911234567890"


def test_contact_roundtrip_and_delete():
    p = _p()
    assert db.get_contact(p) is None
    db.set_contact("A", "+911111111111", p); db.set_contact("B", "+922222222222", p)
    assert db.get_contact(p) == ("B", "+922222222222")
    db.delete_contact(p)
    assert db.get_contact(p) is None


def test_no_health_inputs_columns():
    import sqlite3
    p = _p(); db.save_prediction("heart_disease", "m", 0.1, 0, p)
    cols = [r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(predictions)")]
    assert cols == ["id", "created_at", "task", "model", "score", "flagged"]
