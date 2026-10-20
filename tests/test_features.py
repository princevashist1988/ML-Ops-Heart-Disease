"""Feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartrisk import config
from heartrisk.features import (
    ClinicalFeatureEngineer,
    build_pipeline,
    build_preprocessor,
    to_float,
    to_frame,
)


def test_engineer_adds_age_adjusted_heart_rate(clean_df):
    out = ClinicalFeatureEngineer().fit_transform(clean_df[config.FEATURES])
    row = clean_df.iloc[0]
    assert out.loc[0, "hr_pct_predicted"] == pytest.approx(row["thalach"] / (220 - row["age"]))
    assert set(out["hypertensive"].dropna().unique()) <= {0.0, 1.0}
    assert (out["hypertensive"] == (clean_df["trestbps"] >= 140)).all()


def test_engineer_disabled_is_identity(clean_df):
    X = clean_df[config.FEATURES]
    out = ClinicalFeatureEngineer(enabled=False).fit_transform(X)
    pd.testing.assert_frame_equal(out, X)


def test_engineer_propagates_missing_values():
    X = to_frame({"age": 50, "thalach": None, "trestbps": None})
    out = ClinicalFeatureEngineer().fit_transform(X)
    assert np.isnan(out.loc[0, "hr_pct_predicted"]) and np.isnan(out.loc[0, "hypertensive"])


def test_preprocessor_output_is_complete_and_scaled(clean_df):
    X = to_float(ClinicalFeatureEngineer().fit_transform(clean_df[config.FEATURES]))
    pre = build_preprocessor()
    Z = pre.fit_transform(X)
    names = pre.get_feature_names_out()
    assert not np.isnan(Z).any(), "imputers must remove every NaN (ca/thal are missing in 6 rows)"
    # 7 numeric + 4 binary + one-hot(cp 4 + restecg 3 + slope 3 + thal 3)
    assert Z.shape == (len(clean_df), 7 + 4 + 13)
    num_idx = [i for i, n in enumerate(names) if n.startswith("num__")]
    assert np.allclose(Z[:, num_idx].mean(axis=0), 0, atol=1e-8)


def test_pipeline_tolerates_unseen_category(clean_df, model_artifacts):
    rec = clean_df[config.FEATURES].iloc[[0]].copy()
    rec["thal"] = pd.array([5], dtype="Int64")  # never seen in training
    proba = model_artifacts["pipeline"].predict_proba(rec)[:, 1]
    assert 0 <= proba[0] <= 1


def test_to_frame_orders_columns_and_fills_gaps():
    df = to_frame([{"thal": 7, "age": 50}])
    assert list(df.columns) == config.FEATURES
    assert df.loc[0, "age"] == 50 and pd.isna(df.loc[0, "ca"])


def test_feature_names_survive_the_pipeline(clean_df):
    from sklearn.linear_model import LogisticRegression

    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(
        clean_df[config.FEATURES], clean_df[config.TARGET]
    )
    names = list(pipe[:-1].get_feature_names_out())
    assert "num__hr_pct_predicted" in names and "bin__hypertensive" in names
