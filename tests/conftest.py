"""Shared fixtures. Tests never touch the network and never depend on a previous
training run: a small model is trained into a temp dir once per session."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from heartrisk import config, data
from heartrisk.models import make_model

HIGH_RISK = {
    "age": 67,
    "sex": 1,
    "cp": 4,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 3,
}
LOW_RISK = {
    "age": 41,
    "sex": 0,
    "cp": 2,
    "trestbps": 130,
    "chol": 204,
    "fbs": 0,
    "restecg": 2,
    "thalach": 172,
    "exang": 0,
    "oldpeak": 1.4,
    "slope": 1,
    "ca": 0,
    "thal": 3,
}


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return data.load_clean(config.CLEAN_DATA_PATH)


@pytest.fixture(scope="session")
def model_artifacts(tmp_path_factory, clean_df):
    """Fit a logistic-regression pipeline and write model + card like train.py does."""
    out = tmp_path_factory.mktemp("model")
    pipe = make_model("logistic_regression", {"C": 0.3, "solver": "saga", "max_iter": 5000})
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    model_path, card_path = out / "model.joblib", out / "model_card.json"
    joblib.dump(pipe, model_path)
    card = {
        "model_type": "logistic_regression",
        "decision_threshold": 0.45,
        "registry": {"name": "heart-disease-classifier", "version": 99},
        "training_profile": {"age": {"mean": 54.4, "std": 9.0, "missing_rate": 0.0}},
        "test_metrics_tuned_threshold": {"roc_auc": 0.9},
    }
    card_path.write_text(json.dumps(card))
    return {"model": model_path, "card": card_path, "pipeline": pipe}


@pytest.fixture()
def client(model_artifacts, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_artifacts["model"]))
    monkeypatch.setenv("MODEL_CARD_PATH", str(model_artifacts["card"]))
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    from heartrisk.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def high_risk():
    return dict(HIGH_RISK)


@pytest.fixture()
def low_risk():
    return dict(LOW_RISK)
