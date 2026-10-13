"""Medical-information assistant.
Mode 1 (optional): Anthropic Messages API, key read from the ANTHROPIC_API_KEY environment variable or Streamlit
secrets; only used after the user gives consent, since the typed question leaves the computer.
Mode 2 (always available): a curated FAQ matcher. It is keyword matching, NOT an AI model, and is labelled as such."""
import json
import os
import re
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You are a general health-education assistant inside a student prototype. Use simple language and short answers. "
    "Never give a diagnosis, never prescribe or suggest medicines or doses, never claim to have examined the user, "
    "and never treat a model risk score as a confirmed disease. Do not invent facts, statistics or citations; say when "
    "you are unsure. Say when a question needs a qualified healthcare professional. For severe or urgent symptoms, tell "
    "the user to contact local emergency services immediately.")

URGENT = re.compile(r"chest pain|can'?t breathe|cannot breathe|short(ness)? of breath|faint|unconscious|stroke|"
                    r"slurred speech|face droop|severe bleeding|heart attack", re.I)
SELF_HARM = re.compile(r"suicid|kill myself|end my life|self[- ]harm|hurt myself", re.I)
MEDICATION = re.compile(r"\b(dose|dosage|how many (mg|tablets|pills)|\d+\s?mg|which (medicine|drug|tablet)|prescribe)\b", re.I)

URGENT_REPLY = ("Those symptoms can be serious. Please call your local emergency number now (112 in India) or go to the "
                "nearest hospital. Do not wait for an app or a risk score.")
SELF_HARM_REPLY = ("I'm really sorry you're feeling this way. You deserve support right now. Please contact your local "
                   "emergency number (112 in India) or a crisis line such as Tele-MANAS (14416 in India), or reach out to "
                   "someone you trust to be with you.")
MED_REPLY = ("I can't recommend medicines or doses. A doctor or pharmacist who knows your history should do that. "
             "I'm happy to explain general information about a condition instead.")

FAQ = [
    (("diabetes", "blood sugar", "glucose", "sugar"),
     "Diabetes is a condition where blood sugar stays too high because the body makes too little insulin or cannot use it well. "
     "Diagnosis needs laboratory blood tests ordered by a clinician. Common general advice includes regular physical activity, "
     "a balanced diet, and routine screening if you have risk factors.",
     "https://www.who.int/news-room/fact-sheets/detail/diabetes"),
    (("heart disease", "heart attack", "cardio", "cholesterol", "heart"),
     "Heart (cardiovascular) disease covers conditions affecting the heart and blood vessels. Risk is shaped by factors such as "
     "blood pressure, cholesterol, smoking, diabetes, age and family history. Only a clinician can assess your heart health.",
     "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"),
    (("blood pressure", "hypertension", "bp"),
     "High blood pressure often has no symptoms, so it is found by measuring it. Regular checks are the way to know. "
     "Ask a clinician what your readings mean for you.",
     "https://www.who.int/news-room/fact-sheets/detail/hypertension"),
    (("exercise", "physical activity", "workout", "walk"),
     "Public-health guidance generally encourages regular moderate physical activity for adults, adjusted to your ability. "
     "If you have a health condition or have been inactive, check with a clinician before starting something strenuous.",
     "https://www.who.int/news-room/fact-sheets/detail/physical-activity"),
    (("diet", "food", "eat", "nutrition", "weight"),
     "A generally healthy diet includes plenty of vegetables, fruit, whole grains and legumes, with limited sugar, salt and "
     "unhealthy fats. A dietitian can tailor advice to your needs.",
     "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"),
    (("score", "result", "prediction", "risk", "probability", "mean"),
     "The score is the model's output for a dataset-defined label, measured against a small public dataset. It is not calibrated, "
     "not a clinical probability and not a diagnosis. A low score does not rule disease out, and a high score does not confirm it. "
     "Discuss your situation and any tests with a qualified professional.", None),
    (("accuracy", "precision", "recall", "f1", "specificity", "confusion"),
     "Recall is the share of real positive cases the model catches (misses are false negatives). Precision is the share of its "
     "positive flags that were right (mistakes are false positives). Specificity is the share of negatives correctly cleared. "
     "Accuracy alone can hide a lot of misses, so look at all of them on the Model Performance page.", None),
    (("how", "work", "app", "use"),
     "Enter the measurements on a prediction page, press the predict button, and read the score with its limitations. "
     "The models were trained on public datasets and compared on a held-out test set; see Model Performance.", None),
    (("sleep", "stress"),
     "Regular sleep and ways to manage stress support general health. If sleep or mood problems persist, speak to a healthcare professional.", None),
]


def classify_safety(text):
    if SELF_HARM.search(text):
        return SELF_HARM_REPLY
    if URGENT.search(text):
        return URGENT_REPLY
    if MEDICATION.search(text):
        return MED_REPLY
    return None


def faq_answer(question):
    """Keyword-based FAQ answer. Returns (text, source_url_or_None)."""
    safe = classify_safety(question)
    if safe:
        return safe, None
    q = question.lower()
    best, best_hits = None, 0
    for keys, text, src in FAQ:
        hits = sum(1 for k in keys if k in q)
        if hits > best_hits:
            best, best_hits = (text, src), hits
    if best:
        return best
    return ("I don't have a prepared answer for that. I can explain diabetes, heart health, blood pressure, exercise, diet, "
            "or what your result and the performance metrics mean. For personal medical questions please ask a qualified "
            "healthcare professional.", None)


def get_api_key(secrets=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and secrets is not None:
        try:
            key = secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
    return key or None


def _default_sender(key, payload):
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), method="POST",
                                 headers={"content-type": "application/json", "x-api-key": key,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # fixed https URL
        return json.loads(resp.read())


def ask(question, history=(), api_key=None, use_api=False, context="", sender=_default_sender):
    """Return dict(text, mode, source). mode is 'api' or 'faq-fallback'. Never raises."""
    safe = classify_safety(question)
    if safe:  # safety rules answered locally, regardless of mode
        return {"text": safe, "mode": "safety-rule", "source": None}
    if use_api and api_key:
        msgs = [{"role": m["role"], "content": m["content"]} for m in history][-8:]
        msgs.append({"role": "user", "content": question})
        system = SYSTEM_PROMPT + (f"\nContext supplied by the user: {context}" if context else "")
        try:
            data = sender(api_key, {"model": API_MODEL, "max_tokens": 600, "system": system, "messages": msgs})
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
            if text:
                return {"text": text, "mode": "api", "source": None}
        except Exception:
            pass  # fall through to the FAQ with an honest label
        text, src = faq_answer(question)
        return {"text": "(The AI service could not be reached, so this is a basic FAQ answer.)\n\n" + text,
                "mode": "faq-fallback", "source": src}
    text, src = faq_answer(question)
    return {"text": text, "mode": "faq-fallback", "source": src}
