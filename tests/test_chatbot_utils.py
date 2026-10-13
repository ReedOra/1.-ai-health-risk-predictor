import json
import tempfile
from pathlib import Path

from src import chatbot as cb, utilities as ut, voice_input as vi


def test_fallback_without_credentials(monkeypatch=None):
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    assert cb.get_api_key(None) is None
    out = cb.ask("What is diabetes?", use_api=False, api_key=None)
    assert out["mode"] == "faq-fallback" and "insulin" in out["text"].lower()


def test_urgent_and_medication_rules_answered_locally():
    assert "emergency" in cb.ask("I have chest pain", use_api=True, api_key="k", sender=lambda *a: 1 / 0)["text"].lower()
    assert "can't recommend" in cb.ask("what dose of metformin", use_api=True, api_key="k")["text"]
    assert "Tele-MANAS" in cb.ask("I want to end my life")["text"]


def test_api_success_and_failure_paths():
    ok = cb.ask("hello", api_key="k", use_api=True,
                sender=lambda k, p: {"content": [{"type": "text", "text": "Hi there"}]})
    assert ok["mode"] == "api" and ok["text"] == "Hi there"
    bad = cb.ask("what is blood pressure", api_key="k", use_api=True, sender=lambda k, p: 1 / 0)
    assert bad["mode"] == "faq-fallback" and "could not be reached" in bad["text"]


def test_api_not_used_without_consent():
    called = []
    cb.ask("hello", api_key="k", use_api=False, sender=lambda *a: called.append(1))
    assert not called


def test_phone_normalisation():
    assert ut.normalize_phone("+91 98765-43210") == "+919876543210"
    assert ut.normalize_phone("abc") is None and ut.normalize_phone("123") is None


def test_load_metrics_missing_and_corrupt():
    d = Path(tempfile.mkdtemp()); (d / "metrics").mkdir()
    assert ut.load_metrics("diabetes", d) is None
    (d / "metrics" / "diabetes_metrics.json").write_text("{not json")
    assert ut.load_metrics("diabetes", d) is None
    (d / "metrics" / "diabetes_metrics.json").write_text(json.dumps({"ok": 1}))
    assert ut.load_metrics("diabetes", d) == {"ok": 1}


def test_voice_fails_gracefully():
    text, err = vi.transcribe_wav(b"not a wav")
    assert text is None and err


def test_app_compiles():
    import ast
    ast.parse(Path(__file__).resolve().parent.parent.joinpath("app.py").read_text())
