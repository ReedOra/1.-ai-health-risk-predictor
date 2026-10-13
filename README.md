# AI Health Risk Predictor and Intelligent Medical Assistant

> This application provides preliminary machine-learning estimates and general health information for educational purposes. It is not a medical diagnosis, a substitute for laboratory testing, professional medical advice, or emergency services. Consult a qualified healthcare professional for individual medical concerns.

A Streamlit prototype (theme: Smart Living) that estimates diabetes and heart-disease risk with two separately trained scikit-learn models, explains the real evaluation results, and offers a health-information assistant, optional voice input, optional local history and a trusted-contact helper.

## Status (read this first)
| Item | Status |
|---|---|
| ML pipeline code (loading, cleaning, 5-model comparison, CV selection, held-out evaluation, saving) | Implemented; **tested on synthetic data only** (software tests) |
| **Models trained on the real datasets** | **Not yet. You must run the training commands below.** No real metrics ship with this ZIP |
| Streamlit app (8 pages) | Implemented; code compiles; **not launched in the build environment** (Streamlit unavailable there). Please smoke-test it yourself |
| SQLite history, trusted contact | Implemented; database logic unit-tested |
| FAQ assistant | Implemented; unit-tested |
| Online AI assistant | Implemented; needs your API key; **not tested against the live service** |
| Voice input | Implemented; needs microphone + `SpeechRecognition` + internet; **not tested** |
| Phone-dialer link | Implemented; **not tested**; the app cannot tell if a call happened |

## Windows PowerShell setup
Install Python 3.11 or 3.12 from python.org (tick "Add python.exe to PATH"). Then:

```powershell
cd path\to\AI-Health-Risk-Predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Get the datasets
- **Heart disease**: downloaded automatically by the training script from the UCI repository
  (`https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`, page: https://archive.ics.uci.edu/dataset/45/heart+disease). If it fails, save that file by hand as `data\raw\processed.cleveland.data`.
- **Diabetes**: download `diabetes.csv` from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database (free Kaggle login; original source: NIDDK, Smith et al. 1988) and save it as `data\raw\diabetes.csv`. Check the dataset page for licence terms.

### 2. Train both models (creates models, metrics, charts and dataset reports)
```powershell
python -m src.train_diabetes
python -m src.train_heart_disease
```
Outputs: `models\<task>\model.joblib` + `metadata.json`, `reports\metrics\*.json`, `reports\model_comparisons\*.md`, `reports\dataset_reports\*.md`, `reports\figures\*.png`. If a dataset is missing the script prints exactly what to do and trains nothing.

### 3. Run the tests
```powershell
python -m pytest -q          # or: python run_tests.py
```

### 4. Launch the app
```powershell
streamlit run app.py
```
Open http://localhost:8501 if the browser does not open by itself.

### Optional: AI assistant
```powershell
$env:ANTHROPIC_API_KEY = "your-key"     # this PowerShell window only
streamlit run app.py
```
Or copy `.streamlit\secrets.toml.example` to `.streamlit\secrets.toml` and fill it in. Never commit keys. In the app you must also tick the consent box, because your questions are sent to a third party. Without a key the assistant uses the offline FAQ mode (keyword matching, labelled as not AI).

### Optional: voice input
```powershell
pip install SpeechRecognition
```
Needs a browser microphone permission and internet (audio is sent to Google's free web speech service after you tick a consent box). Requires Streamlit ≥ 1.40 for the microphone widget. If unavailable, type your question.

## Method
- Two independent tasks, each with its own dataset, features and model. Positive class = 1 (dataset-defined label, not a diagnosis).
- Cleaning: for diabetes, zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are treated as missing (physiologically impossible); Pregnancies = 0 is kept. For heart disease, labels 1–4 collapse to 1. Duplicates removed. No outlier removal, no resampling.
- Stratified 80/20 split (seed 42) **before** anything else. Imputation, scaling (only Logistic Regression) and one-hot encoding live inside the pipeline, so they are fitted on training folds only.
- Candidates: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Hist Gradient Boosting (fixed conservative settings, no tuning).
- Selection: highest mean 5-fold CV F1 of the positive class on training data (ties within 0.005 → higher CV recall). The test set is scored once afterwards and never influences the choice.
- Metrics: accuracy, precision, recall, F1, specificity, ROC-AUC, confusion matrix, supports. Formulas in `docs/model_evaluation.md`.

## Project layout
`app.py` (UI) · `config/settings.py` · `src/` (data_loader, preprocessing, training, model_evaluation, prediction, chatbot, voice_input, database, utilities) · `tests/` · `docs/` · `reports/` · `models/`

## Privacy
No accounts or analytics. History stores only time, disease, model name, score and flag, and only if you tick the box (`data\app.db`; delete from the History page). The trusted contact is stored locally too. No legal compliance is claimed. Use on a device you control.

## Troubleshooting
- `streamlit` not recognised → activate the venv first.
- "model not trained" → run the training commands above.
- Heart download fails → save the file manually (see step 1).
- Voice widget missing → `pip install -U streamlit`.

## Future improvements
Calibration analysis, hyperparameter search with nested CV, user accounts, richer datasets with common lifestyle features, SHAP explanations.
