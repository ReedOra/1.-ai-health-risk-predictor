"""AI Health Risk Predictor and Intelligent Medical Assistant -- Streamlit app.
Run with:  streamlit run app.py
All model numbers shown here are read from reports/metrics/*.json written by the training scripts."""
import pandas as pd
import streamlit as st

from config import settings as S
from src import chatbot, database as db, prediction as pr, utilities as ut, voice_input as vi

st.set_page_config(page_title="AI Health Risk Predictor", page_icon="🩺", layout="wide")

PAGES = ["Home", "Diabetes Risk", "Heart Disease Risk", "AI Medical Assistant", "Model Performance",
         "Prediction History", "Trusted Contact", "About & Limitations"]

CHOICES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
STEPS = {"BMI": 0.1, "DiabetesPedigreeFunction": 0.01, "oldpeak": 0.1}

GUIDANCE = {
    "diabetes": ["Diabetes can only be confirmed with laboratory tests ordered by a clinician (for example fasting glucose or HbA1c).",
                 "Established general advice: regular physical activity, a balanced diet, and routine screening if you have risk factors.",
                 "A low score does not rule diabetes out. Please talk to a healthcare professional if you have symptoms such as "
                 "excessive thirst, frequent urination, unexplained weight loss or blurred vision."],
    "heart_disease": ["Heart conditions are assessed by clinicians using examination, ECG and other tests; this score cannot replace that.",
                      "Established general advice: do not smoke, stay physically active as advised for you, and have blood pressure and "
                      "cholesterol checked regularly.",
                      "A low score does not rule heart disease out. Seek care promptly for chest pain, breathlessness or fainting."],
}


# ---------------------------------------------------------------- helpers
def go(page):
    st.session_state["nav"] = page


@st.cache_resource(show_spinner=False)
def _load_cached(task, mtime):
    return pr.load_model(task)


def get_model(task):
    p = S.MODELS_DIR / task / "model.joblib"
    if not p.exists():
        return None
    try:
        return _load_cached(task, p.stat().st_mtime)
    except Exception:
        return None


def disclaimer():
    st.warning(ut.DISCLAIMER)


def secrets_get():
    try:
        return st.secrets
    except Exception:
        return None


# ---------------------------------------------------------------- pages
def page_home():
    st.title("🩺 AI Health Risk Predictor and Intelligent Medical Assistant")
    st.caption("Smart Living · Student prototype · Educational use only")
    disclaimer()
    st.write("This tool uses two machine-learning models, trained on public research datasets, to give a **preliminary** "
             "estimate for diabetes and heart disease, and offers a simple assistant for general health information.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Risk estimates")
        st.write("Separate models for diabetes and heart disease, each compared against other algorithms on held-out data.")
        st.button("Diabetes assessment", on_click=go, args=("Diabetes Risk",), use_container_width=True)
        st.button("Heart disease assessment", on_click=go, args=("Heart Disease Risk",), use_container_width=True)
    with c2:
        st.subheader("Assistant & voice")
        st.write("Ask general health questions by typing or (if set up) speaking. Works offline in a basic FAQ mode.")
        st.button("Open the assistant", on_click=go, args=("AI Medical Assistant",), use_container_width=True)
    with c3:
        st.subheader("Transparency")
        st.write("See the real evaluation results, the confusion matrices and the limits of the data.")
        st.button("Model performance", on_click=go, args=("Model Performance",), use_container_width=True)
        st.button("Trusted contact", on_click=go, args=("Trusted Contact",), use_container_width=True)
    st.subheader("How to use")
    st.markdown("1. Read the disclaimer.  2. Open a risk page and enter your measurements (leave nothing guessed).  "
                "3. Read the score **and** its limitations.  4. Ask the assistant if something is unclear.  "
                "5. Speak to a healthcare professional about any concern.")
    missing = [S.TASKS[t]["title"] for t in S.TASKS if get_model(t) is None]
    if missing:
        st.info("Models not trained yet for: " + ", ".join(missing) + ". See README (training commands). "
                "Prediction pages stay disabled until training succeeds.")
    st.error(ut.EMERGENCY_NOTE)


def input_form(task):
    spec = S.TASKS[task]
    values = {}
    with st.form(f"form_{task}"):
        cols = st.columns(2)
        for i, (name, (lo, hi, required, label, help_)) in enumerate(spec["inputs"].items()):
            with cols[i % 2]:
                text = label + ("" if required else " (optional)")
                if name in CHOICES:
                    values[name] = st.selectbox(text, options=list(CHOICES[name]), index=None, key=f"{task}_{name}",
                                                format_func=lambda k, n=name: CHOICES[n][k], help=help_ or None)
                else:
                    values[name] = st.number_input(text, min_value=float(lo), max_value=float(hi), value=None,
                                                   step=STEPS.get(name, 1.0), key=f"{task}_{name}",
                                                   help=(help_ + f" Allowed range {lo}–{hi}.").strip())
        save = st.checkbox("Save a short summary (time, score, flag; not your inputs) to local history", value=False,
                           key=f"save_{task}")
        submitted = st.form_submit_button("Predict", type="primary")
    return values, save, submitted


def page_risk(task):
    spec = S.TASKS[task]
    st.title(f"{spec['title']} risk assessment")
    disclaimer()
    loaded = get_model(task)
    if loaded is None:
        st.warning(f"The {spec['title']} model has not been trained yet. In PowerShell run: "
                   f"`python -m src.train_{task}` (see README for getting the dataset first).")
        return
    model, meta = loaded
    st.caption(f"Model: **{meta['selected_model']}** · trained on {meta['n_records_used']} records from the "
               f"{spec['title']} dataset · label meaning: {meta['positive_meaning']}")
    if task == "heart_disease":
        st.info("This model needs clinical test results (ECG, stress test, blood tests, imaging). If you do not have them, "
                "the app cannot give an estimate — it will not guess them for you.")
    values, save, submitted = input_form(task)
    if submitted:
        try:
            result = pr.predict(task, values, model, meta)
        except pr.InputValidationError as e:
            for p in e.problems:
                st.error(p)
            return
        except Exception:
            st.error("Something went wrong while predicting. Please check your inputs and try again.")
            return
        st.session_state["last_result"] = {"task": task, "score": result["score"], "flag": result["label"]}
        if save:
            db.save_prediction(task, result["model"], result["score"], result["label"])
        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Model score (uncalibrated)", f"{result['score'] * 100:.0f}%")
        c2.metric(f"Flag at {result['threshold']:.2f} threshold", "Above threshold" if result["label"] else "Below threshold")
        (st.warning if result["label"] else st.info)(
            "The model flagged this input as similar to dataset records labelled positive. This is not a diagnosis."
            if result["label"] else
            "The model did not flag this input. This does **not** mean you are free of disease.")
        st.caption(result["note"] + " The flag threshold is a fixed prototype setting and not a clinical cut-off.")
        if result["imputed"]:
            st.caption("Not provided, so the training-data median was used for: " + ", ".join(result["imputed"]))
        st.markdown("**General guidance**")
        for line in GUIDANCE[task]:
            st.markdown(f"- {line}")
        st.markdown("- This score is no reason to delay or skip professional care.")
        st.error(ut.EMERGENCY_NOTE)
        st.button("Ask the assistant about this result", on_click=go, args=("AI Medical Assistant",))


def page_assistant():
    st.title("AI medical assistant")
    disclaimer()
    key = chatbot.get_api_key(secrets_get())
    use_api = False
    if key:
        use_api = st.checkbox("Use the online AI service for answers. Your typed questions (and the result summary below, if "
                              "ticked) will be sent to a third party.", value=False)
        st.caption("Mode: " + ("**AI service** (online)" if use_api else "**Basic FAQ** (offline keyword matching, not an AI model)"))
    else:
        st.info("Mode: **Basic FAQ** — keyword matching over a small curated list, not an AI model. To enable the online "
               "assistant set ANTHROPIC_API_KEY (see README).")
    share = st.checkbox("Share only my latest result summary (disease, score, flag; no measurements) with the assistant",
                        value=False, disabled=not use_api) if "last_result" in st.session_state else False
    context = ""
    if share and use_api:
        r = st.session_state["last_result"]
        context = f"{r['task']} model score {r['score']:.2f}, flag={r['flag']}"

    st.session_state.setdefault("chat", [])
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("source"):
                st.caption(f"General source: {m['source']}")

    question = None
    with st.expander("🎤 Voice input"):
        if not hasattr(st, "audio_input"):
            st.info("Your Streamlit version has no microphone widget. Upgrade with `pip install -U streamlit`, or type below.")
        elif not vi.recognizer_available():
            st.info("Voice needs the optional package: `pip install SpeechRecognition`. Until then, please type your question.")
        else:
            ok = st.checkbox("I understand my recording is sent to Google's online speech service (internet required).")
            audio = st.audio_input("Record your question")
            if audio is not None and ok and st.button("Transcribe"):
                text, err = vi.transcribe_wav(audio.getvalue())
                st.session_state["voice_text"] = text or ""
                if err:
                    st.error(err)
            if st.session_state.get("voice_text"):
                edited = st.text_input("Check the transcript", value=st.session_state["voice_text"])
                if st.button("Send transcript"):
                    question = edited
                    st.session_state["voice_text"] = ""
    typed = st.chat_input("Ask a general health question…")
    question = typed or question
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking…"):
            ans = chatbot.ask(question, st.session_state["chat"], api_key=key, use_api=use_api, context=context)
        with st.chat_message("assistant"):
            st.write(ans["text"])
            st.caption({"api": "AI service", "faq-fallback": "Basic FAQ (not AI)", "safety-rule": "Safety rule"}[ans["mode"]])
            if ans["source"]:
                st.caption(f"General source: {ans['source']}")
        st.session_state["chat"] += [{"role": "user", "content": question},
                                     {"role": "assistant", "content": ans["text"], "source": ans["source"]}]
    if st.session_state["chat"] and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()


def page_performance():
    st.title("Model performance")
    st.write("Every number on this page is read from the files produced by the training scripts "
             "(`reports/metrics/*.json`). Nothing is typed in by hand.")
    tabs = st.tabs([S.TASKS[t]["title"] for t in S.TASKS])
    for tab, task in zip(tabs, S.TASKS):
        with tab:
            res = ut.load_metrics(task)
            if res is None:
                st.info(f"No results yet. Train the model first: `python -m src.train_{task}`")
                continue
            meta, final = res["meta"], res["final_test"]
            st.success(f"Selected model: **{meta['selected_model']}** (chosen on cross-validation only)")
            st.caption(f"Selection criterion: {meta['selection_criterion']}. Trained {meta['trained_at_utc']} UTC · "
                       f"{meta['n_records_used']} records ({meta['n_train']} train / {meta['n_test']} test) · "
                       f"classes {meta['class_counts']} · positive = {meta['positive_meaning']}")
            cols = st.columns(6)
            for c, (lab, k) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                          ("F1-score", "f1"), ("Specificity", "specificity"), ("ROC-AUC", "roc_auc")]):
                c.metric(lab, ut.fmt(final[k], pct=k != "roc_auc"))
            if final["roc_auc"] is None:
                st.caption(f"ROC-AUC not available: {final['roc_auc_note']}")
            st.caption(f"Held-out test set: {final['n']} people · positives {final['support_positive']} · "
                       f"negatives {final['support_negative']}")
            cm = pd.DataFrame(final["confusion_matrix"], index=["Actually negative", "Actually positive"],
                              columns=["Predicted negative", "Predicted positive"])
            a, b = st.columns(2)
            a.markdown("**Confusion matrix (test set)**")
            a.dataframe(cm)
            a.caption(f"False negatives (missed positives): {final['fn']} · False positives (false alarms): {final['fp']}")
            fig = S.REPORTS_DIR / "figures" / f"{task}_confusion_matrix.png"
            if fig.exists():
                b.image(str(fig))
            rows = [{"Model": r["model"] + (" ✔" if r["selected"] else ""),
                     "CV Accuracy": r["cv_accuracy_mean"], "CV Precision": r["cv_precision_mean"],
                     "CV Recall": r["cv_recall_mean"], "CV F1": r["cv_f1_mean"], "CV F1 std": r["cv_f1_std"],
                     "Test Accuracy": r["test"]["accuracy"], "Test Precision": r["test"]["precision"],
                     "Test Recall": r["test"]["recall"], "Test F1": r["test"]["f1"],
                     "Test ROC-AUC": r["test"]["roc_auc"]} for r in res["candidates"]]
            st.markdown("**Candidate comparison** (CV = 5-fold on training data; Test = held-out, reported once)")
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
            figs = [S.REPORTS_DIR / "figures" / f"{task}_{n}.png" for n in ("roc_curve", "model_comparison", "class_distribution")]
            shown = [f for f in figs if f.exists()]
            if shown:
                for col, f in zip(st.columns(len(shown)), shown):
                    col.image(str(f))
            st.warning("Small public dataset: these results do not show clinical validity or that the model works for the "
                       "wider population. The positive label is dataset-defined, not a confirmed diagnosis.")
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy**: share of all predictions that were right. Can look high even if many sick people are missed.\n"
            "- **Precision**: of those flagged positive, how many truly were. Low precision = many false alarms.\n"
            "- **Recall (sensitivity)**: of the truly positive, how many were caught. Low recall = many **false negatives** (missed cases).\n"
            "- **F1-score**: balance of precision and recall.\n- **Specificity**: of the truly negative, how many were correctly cleared.\n"
            "- **ROC-AUC**: how well scores rank positives above negatives (0.5 = chance, 1 = perfect).\n"
            "- A model's test accuracy describes the whole test set, **not** the certainty of your own result.")


def page_history():
    st.title("Prediction history")
    st.caption(f"Stored only on this computer in `{db.DEFAULT_DB}`. Only time, disease, model, score and flag are saved — never your measurements. "
               "A local file has no login or user separation, so use this app only on a device you control.")
    rows = db.list_predictions()
    if not rows:
        st.info("No saved predictions yet. Tick the save option on a prediction page to keep a summary.")
        return
    df = pd.DataFrame(rows, columns=["Time (UTC)", "Task", "Model", "Score", "Flagged"])
    df["Flagged"] = df["Flagged"].map({0: "No", 1: "Yes"})
    st.dataframe(df, hide_index=True)
    if st.checkbox("I want to permanently delete all saved history"):
        if st.button("Delete history", type="primary"):
            db.clear_predictions()
            st.success("History deleted.")
            st.rerun()


def page_contact():
    st.title("Trusted contact")
    st.info("This helps you reach someone you trust. It never contacts anyone automatically, and it is **not** a substitute for "
            "emergency services. " + ut.EMERGENCY_NOTE)
    contact = db.get_contact()
    with st.form("contact_form"):
        name = st.text_input("Contact name (nickname is fine)", value=contact[0] if contact else "")
        phone = st.text_input("Phone number with country code, e.g. +919876543210", value=contact[1] if contact else "")
        if st.form_submit_button("Save contact"):
            clean = ut.normalize_phone(phone)
            if not name.strip() or clean is None:
                st.error("Please enter a name and a valid phone number (7–15 digits, optional leading +).")
            else:
                db.set_contact(name.strip(), clean)
                st.success("Contact saved on this computer.")
                st.rerun()
    if contact:
        st.subheader(f"Contact {contact[0]}")
        if st.checkbox(f"Yes, I want to contact {contact[0]} ({contact[1]}) now"):
            st.link_button("Open my phone dialer", f"tel:{contact[1]}")
            st.caption("This only opens your device's dialer if it supports phone links (mostly phones). The app cannot "
                       "tell whether a call was made or answered, and no message is sent.")
        if st.button("Delete saved contact"):
            db.delete_contact()
            st.rerun()


def page_about():
    st.title("About & limitations")
    disclaimer()
    st.markdown(
        "**Purpose**: a student prototype showing responsible use of machine learning for preliminary health-risk estimates.\n\n"
        "**Technologies**: Python, pandas, scikit-learn, matplotlib, joblib, SQLite, Streamlit; optional SpeechRecognition and an "
        "optional online language-model API.\n\n**Datasets**: Pima Indians Diabetes Database (NIDDK; adult women of Pima heritage) and "
        "the UCI Heart Disease Cleveland subset (about 300 patients, collected decades ago). Both are small, old and not representative "
        "of the general population. Their labels are dataset-defined, not clinical diagnoses.\n\n"
        "**Evaluation**: stratified 80/20 split, 5-fold cross-validation on the training part to choose the model, one final "
        "evaluation on the held-out test set. Scores are uncalibrated.\n\n"
        "**Privacy**: no accounts, no analytics. Predictions are kept only if you tick the save box and contain no measurements. "
        "The online assistant and voice transcription send data to third parties only after you opt in. No legal compliance "
        "(HIPAA, GDPR, etc.) is claimed.\n\n"
        "**Feature status**: predictions, performance page, history and contact storage are built-in and need no external service. "
        "The AI assistant needs an API key; voice needs a microphone, the optional package, and internet. Without them the app uses "
        "the FAQ mode and typed input.")


PAGE_FUNCS = {"Home": page_home, "Diabetes Risk": lambda: page_risk("diabetes"),
              "Heart Disease Risk": lambda: page_risk("heart_disease"), "AI Medical Assistant": page_assistant,
              "Model Performance": page_performance, "Prediction History": page_history,
              "Trusted Contact": page_contact, "About & Limitations": page_about}

st.sidebar.title("🩺 Health Risk Predictor")
st.session_state.setdefault("nav", "Home")
page = st.sidebar.radio("Navigate", PAGES, key="nav")
st.sidebar.caption("Preliminary estimates for education only. Not medical advice or emergency services.")
PAGE_FUNCS[page]()
