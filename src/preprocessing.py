"""Preprocessing + candidate models. All learned steps live inside one sklearn Pipeline,
so cross-validation fits imputers/scalers/encoders on training folds only."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(numeric, categorical, scale: bool):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:  # only distance/gradient-based models need scaling; trees do not
        num_steps.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(num_steps), list(numeric))]
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(categorical)))
    return ColumnTransformer(parts)


def candidate_models(seed):
    """Fixed, conservative hyperparameters. No tuning was done, so none is claimed."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                            random_state=seed, n_jobs=1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                 random_state=seed),
    }


def build_pipeline(spec, name, model):
    return Pipeline([("prep", build_preprocessor(spec["numeric"], spec["categorical"], name in spec["scaled_models"])),
                     ("model", model)])
