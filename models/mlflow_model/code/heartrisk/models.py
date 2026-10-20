"""Candidate model zoo. Names map to estimators; grids come from configs/train.yaml."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heartrisk.features import build_pipeline

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def make_model(name: str, params: dict | None = None, seed: int = 42):
    """Return an unfitted end-to-end pipeline for the named model."""
    if name not in ESTIMATORS:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(ESTIMATORS)}")
    params = dict(params or {})
    cls = ESTIMATORS[name]
    if "random_state" in cls().get_params():
        params.setdefault("random_state", seed)
    return build_pipeline(cls(**params))
