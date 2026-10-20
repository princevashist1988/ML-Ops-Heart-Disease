"""Central configuration: paths, schema and constants shared by every stage.

Keeping the schema in one place means the data cleaner, the feature pipeline,
the API validator and the tests can never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("HEARTRISK_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

CLEAN_DATA_PATH = PROCESSED_DIR / "heart_cleveland_clean.csv"
EXTERNAL_DATA_PATH = PROCESSED_DIR / "heart_hungarian_clean.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
MODEL_CARD_PATH = Path(os.getenv("MODEL_CARD_PATH", MODELS_DIR / "model_card.json"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-risk")
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# The 14 attributes used by every published experiment on this dataset (UCI docs).
RAW_COLUMNS = [
    "age",  # years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure, mm Hg
    "chol",  # serum cholesterol, mg/dl
    "fbs",  # fasting blood sugar > 120 mg/dl
    "restecg",  # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise-induced angina
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",  # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",  # number of major vessels coloured by fluoroscopy (0-3)
    "thal",  # 3 normal, 6 fixed defect, 7 reversible defect
    "num",  # diagnosis: 0 = <50% narrowing, 1-4 = >50% narrowing
]

TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Engineered by heartrisk.features.ClinicalFeatureEngineer
ENGINEERED_FEATURES = ["hr_pct_predicted", "hypertensive"]

# Valid domain of every feature. Values outside are physiologically implausible
# (e.g. cholesterol recorded as 0 in the Hungarian file) and are set to missing.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (100, 650),
    "thalach": (60, 220),
    "oldpeak": (-3, 7),  # negative = ST elevation, clinically valid
    "ca": (0, 3),
}
CATEGORY_DOMAINS: dict[str, set[int]] = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {1, 2, 3, 4},
    "restecg": {0, 1, 2},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}

SEED = int(os.getenv("HEARTRISK_SEED", "42"))
