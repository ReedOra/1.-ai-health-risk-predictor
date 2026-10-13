"""Central configuration: paths, random seed and the two task specifications.

Every feature list below comes from the public dataset documentation; the
loaders validate real files against these schemas before training.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5  # fixed, documented; never tuned on the test set

# --- Diabetes: Pima Indians Diabetes Database (NIDDK, Smith et al. 1988) ---
DIABETES = {
    "name": "diabetes",
    "title": "Diabetes",
    "raw_file": RAW_DIR / "diabetes.csv",
    "source_url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
    "source_note": ("Kaggle requires a free login, so download diabetes.csv manually "
                    "and place it in data/raw/. Original source: National Institute of "
                    "Diabetes and Digestive and Kidney Diseases (Smith et al., 1988)."),
    "download_url": None,
    "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"],
    "target": "Outcome",
    "positive_meaning": "1 = diabetes onset within 5 years (dataset-defined, NOT a clinical diagnosis)",
    "numeric": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                "BMI", "DiabetesPedigreeFunction", "Age"],
    "categorical": [],
    # Zero is physiologically impossible for these columns -> treated as missing.
    # Pregnancies == 0 is valid and is NOT touched.
    "zero_is_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "scaled_models": ["Logistic Regression"],
    # (min, max, required, label, help)
    "inputs": {
        "Pregnancies": (0, 20, True, "Number of pregnancies", "Enter 0 if none or not applicable."),
        "Glucose": (40, 300, True, "Plasma glucose (mg/dL)", "2-hour oral glucose tolerance test value in the original data."),
        "BloodPressure": (30, 200, True, "Diastolic blood pressure (mm Hg)", "The lower number of a BP reading."),
        "SkinThickness": (5, 100, False, "Triceps skin-fold thickness (mm)", "Optional; needs calipers. If blank, the training median is used."),
        "Insulin": (10, 900, False, "2-hour serum insulin (mu U/ml)", "Optional; needs a lab test. If blank, the training median is used."),
        "BMI": (10, 70, True, "Body-mass index (kg/m²)", "Weight in kg divided by height in metres squared."),
        "DiabetesPedigreeFunction": (0.05, 2.5, True, "Diabetes pedigree function", "A dataset-specific score of family history of diabetes."),
        "Age": (18, 100, True, "Age (years)", "The dataset covers adult women only."),
    },
}

# --- Heart disease: UCI Heart Disease, Cleveland subset ---
HEART = {
    "name": "heart_disease",
    "title": "Heart disease",
    "raw_file": RAW_DIR / "processed.cleveland.data",
    "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
    "source_note": "UCI Machine Learning Repository, Heart Disease (Cleveland Clinic Foundation subset). Check the page for current license terms.",
    "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "columns": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope", "ca", "thal", "num"],
    "target": "num",
    "positive_meaning": "num 1-4 collapsed to 1 = angiographic narrowing present; 0 = absent (dataset-defined, NOT a clinical diagnosis)",
    "numeric": ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"],
    "categorical": ["cp", "restecg", "slope", "thal"],
    "zero_is_missing": [],
    "scaled_models": ["Logistic Regression"],
    "inputs": {
        "age": (18, 100, True, "Age (years)", ""),
        "sex": (0, 1, True, "Sex (0 = female, 1 = male)", "As recorded in the dataset."),
        "cp": (1, 4, True, "Chest pain type", "1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic."),
        "trestbps": (80, 220, True, "Resting blood pressure (mm Hg)", "Systolic, measured at rest."),
        "chol": (100, 600, True, "Serum cholesterol (mg/dL)", "Needs a blood test."),
        "fbs": (0, 1, True, "Fasting blood sugar > 120 mg/dL (1 = yes)", "Needs a blood test."),
        "restecg": (0, 2, True, "Resting ECG result", "0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy. Needs an ECG."),
        "thalach": (60, 220, True, "Maximum heart rate achieved (bpm)", "From an exercise stress test."),
        "exang": (0, 1, True, "Exercise-induced angina (1 = yes)", "Chest pain during the stress test."),
        "oldpeak": (0.0, 7.0, True, "ST depression after exercise", "ECG measurement from a stress test."),
        "slope": (1, 3, True, "Slope of peak-exercise ST segment", "1 upsloping, 2 flat, 3 downsloping."),
        "ca": (0, 3, True, "Major vessels coloured by fluoroscopy", "Needs a specialist imaging test."),
        "thal": (3, 7, True, "Thalassemia test code", "3 normal, 6 fixed defect, 7 reversible defect. Needs a thallium stress test."),
    },
}

TASKS = {"diabetes": DIABETES, "heart_disease": HEART}
