"""Local SQLite storage. Only task, time, model score and flag are saved -- never the health inputs.
All queries are parameterised. A local file is NOT a multi-user or authenticated store."""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from config import settings as S

DEFAULT_DB = Path(os.environ.get("HEALTH_APP_DB", S.ROOT / "data" / "app.db"))


def _conn(db_path=None):
    path = Path(db_path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,"
                 " task TEXT NOT NULL, model TEXT NOT NULL, score REAL NOT NULL, flagged INTEGER NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return conn


def save_prediction(task, model, score, flagged, db_path=None):
    with closing(_conn(db_path)) as c, c:
        c.execute("INSERT INTO predictions (created_at, task, model, score, flagged) VALUES (?, ?, ?, ?, ?)",
                  (datetime.now(timezone.utc).isoformat(timespec="seconds"), task, model, float(score), int(flagged)))


def list_predictions(db_path=None):
    with closing(_conn(db_path)) as c:
        return c.execute("SELECT created_at, task, model, score, flagged FROM predictions ORDER BY id DESC").fetchall()


def clear_predictions(db_path=None):
    with closing(_conn(db_path)) as c, c:
        c.execute("DELETE FROM predictions")


def set_contact(name, phone, db_path=None):
    with closing(_conn(db_path)) as c, c:
        for k, v in (("contact_name", name), ("contact_phone", phone)):
            c.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (k, v))


def get_contact(db_path=None):
    with closing(_conn(db_path)) as c:
        d = dict(c.execute("SELECT key, value FROM settings WHERE key IN ('contact_name', 'contact_phone')").fetchall())
    return (d["contact_name"], d["contact_phone"]) if "contact_phone" in d else None


def delete_contact(db_path=None):
    with closing(_conn(db_path)) as c, c:
        c.execute("DELETE FROM settings WHERE key IN ('contact_name', 'contact_phone')")
