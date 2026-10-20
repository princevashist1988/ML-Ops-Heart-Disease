"""Feature engineering and preprocessing, packaged as scikit-learn transformers.

Everything that touches a feature lives inside one ``Pipeline`` object, so the
exact same transformations run during cross-validation, final training and
online inference. Imputers and scalers are fitted on training folds only,
which keeps the hold-out and CV estimates free of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartrisk import config


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds two clinically motivated features.

    * ``hr_pct_predicted`` - peak heart rate as a fraction of the age-predicted
      maximum (220 - age). Failure to reach ~85% (chronotropic incompetence) is
      an established marker of coronary disease, and the ratio removes the
      strong age confound in raw ``thalach``.
    * ``hypertensive`` - resting systolic pressure >= 140 mm Hg (stage-2 cut-off).

    ``enabled`` is a hyper-parameter so the grid search itself runs the
    ablation "with vs without engineered features".
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.enabled:
            return X
        age = pd.to_numeric(X["age"], errors="coerce").astype(float)
        thalach = pd.to_numeric(X["thalach"], errors="coerce").astype(float)
        bp = pd.to_numeric(X["trestbps"], errors="coerce").astype(float)
        X["hr_pct_predicted"] = thalach / (220.0 - age)
        X["hypertensive"] = np.where(bp.isna(), np.nan, (bp >= 140).astype(float))
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(
            names + (config.ENGINEERED_FEATURES if self.enabled else []), dtype=object
        )


# Column selectors are module-level functions so pickled pipelines stay loadable.
def select_numeric(X: pd.DataFrame) -> list[str]:
    return [c for c in config.NUMERIC_FEATURES + ["hr_pct_predicted"] if c in X.columns]


def select_binary(X: pd.DataFrame) -> list[str]:
    return [c for c in config.BINARY_FEATURES + ["hypertensive"] if c in X.columns]


def select_categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in config.CATEGORICAL_FEATURES if c in X.columns]


def to_float(X):
    """Cast nullable integer columns to float so imputers see real NaNs."""
    return X.astype(float)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, select_numeric),
            ("bin", binary, select_binary),
            ("cat", categorical, select_categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator, engineer: bool = True) -> Pipeline:
    """Full model pipeline: engineer -> cast -> impute/scale/encode -> classifier."""
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        [
            ("features", ClinicalFeatureEngineer(enabled=engineer)),
            ("cast", FunctionTransformer(to_float, feature_names_out="one-to-one")),
            ("preprocess", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def to_frame(records: list[dict] | dict | pd.DataFrame) -> pd.DataFrame:
    """Normalise API/CLI input into a dataframe with the training column order."""
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame([records] if isinstance(records, dict) else records)
    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df[config.FEATURES]
