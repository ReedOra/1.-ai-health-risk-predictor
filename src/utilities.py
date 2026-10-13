"""Small helpers: disclaimer text, saved-report loading, phone validation."""
import json
import re
from pathlib import Path

from config import settings as S

DISCLAIMER = ("This application provides preliminary machine-learning estimates and general health information "
              "for educational purposes. It is not a medical diagnosis, a substitute for laboratory testing, "
              "professional medical advice, or emergency services. Consult a qualified healthcare professional "
              "for individual medical concerns.")

EMERGENCY_NOTE = ("If you have severe or sudden symptoms (for example chest pain, trouble breathing, fainting, "
                  "or signs of a stroke), call your local emergency number now (112 in India) or go to the nearest "
                  "hospital. Do not wait for, or rely on, this app.")


def load_metrics(task, reports_dir=None):
    """Return the saved evaluation results, or None if training has not been run."""
    p = Path(reports_dir or S.REPORTS_DIR) / "metrics" / f"{task}_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def normalize_phone(raw):
    """Return a cleaned number like +919876543210, or None if it does not look like a phone number."""
    s = re.sub(r"[\s\-().]", "", raw or "")
    return s if re.fullmatch(r"\+?[0-9]{7,15}", s) else None


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"
