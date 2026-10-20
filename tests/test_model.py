"""Model construction, metrics, selection logic and an end-to-end quick training run."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from heartrisk import config, evaluate, train
from heartrisk.models import ESTIMATORS, make_model


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_every_model_family_fits_and_outputs_probabilities(name, clean_df):
    params = {"n_estimators": 30} if name != "logistic_regression" else {"max_iter": 2000}
    pipe = make_model(name, params).fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    proba = pipe.predict_proba(clean_df[config.FEATURES].head(20))[:, 1]
    assert proba.shape == (20,) and ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError):
        make_model("svm_rbf")


def test_baseline_pipeline_clearly_beats_chance(clean_df):
    pipe = make_model("logistic_regression", {"C": 0.3, "solver": "saga", "max_iter": 5000})
    auc = cross_val_score(
        pipe,
        clean_df[config.FEATURES],
        clean_df[config.TARGET],
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
    )
    assert auc.mean() > 0.85


def test_training_is_deterministic_for_a_fixed_seed(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    a = make_model("random_forest", {"n_estimators": 25}, seed=7).fit(X, y).predict_proba(X)
    b = make_model("random_forest", {"n_estimators": 25}, seed=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_serialised_model_round_trip(model_artifacts, clean_df, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(model_artifacts["pipeline"], path)
    X = clean_df[config.FEATURES]
    np.testing.assert_allclose(
        joblib.load(path).predict_proba(X), model_artifacts["pipeline"].predict_proba(X)
    )


def test_compute_metrics_on_a_hand_checked_example():
    y = [0, 0, 1, 1, 1]
    p = [0.1, 0.6, 0.4, 0.8, 0.9]
    m = evaluate.compute_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(5 / 6)


def test_threshold_choice_respects_precision_floor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    proba = np.clip(y * 0.35 + rng.normal(0.35, 0.18, 400), 0, 1)
    t = evaluate.choose_threshold(y, proba, beta=2, min_precision=0.7)
    pred = proba >= t
    assert (y[pred] == 1).mean() >= 0.7
    assert evaluate.choose_threshold(y, proba, beta=2, min_precision=0.0) <= t


def test_champion_selection_applies_parsimony_rule():
    cfg = {
        "primary_metric": "roc_auc",
        "selection": {"tolerance": 0.01, "prefer": ["logistic_regression"]},
    }
    results = [
        {"name": "random_forest", "cv_roc_auc_mean": 0.905},
        {"name": "logistic_regression", "cv_roc_auc_mean": 0.899},
    ]
    assert train.select_champion([dict(r) for r in results], cfg)["name"] == "logistic_regression"
    results[1]["cv_roc_auc_mean"] = 0.85
    assert train.select_champion([dict(r) for r in results], cfg)["name"] == "random_forest"


def test_quality_gate_reports_each_failure():
    cfg = {"quality_gate": {"min_test_roc_auc": 0.8, "min_test_recall": 0.75}}
    assert train.quality_gate({"test": {"roc_auc": 0.9, "recall": 0.8}}, cfg) == []
    assert len(train.quality_gate({"test": {"roc_auc": 0.7, "recall": 0.5}}, cfg)) == 2


@pytest.mark.slow
def test_quick_training_run_end_to_end(tmp_path, monkeypatch):
    """Runs the real training entrypoint on a tiny grid into an isolated MLflow store."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "MODEL_CARD_PATH", tmp_path / "models" / "model_card.json")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    cfg = train.quick_config(train.load_config(config.CONFIG_DIR / "train.yaml"))
    out = train.run(cfg, register=True, experiment="pytest")
    card = json.loads((tmp_path / "models" / "model_card.json").read_text())
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "models" / "mlflow_model" / "MLmodel").exists()
    assert card["registry"]["alias"] == "champion"
    assert 0.2 <= card["decision_threshold"] <= 0.7
    assert out["metrics"]["test"]["roc_auc"] > 0.8

    import mlflow

    loaded = mlflow.pyfunc.load_model(str(tmp_path / "models" / "mlflow_model"))
    from heartrisk.data import load_clean

    X = load_clean()[config.FEATURES].head(5).astype(float)
    assert loaded.predict(X).shape[0] == 5
