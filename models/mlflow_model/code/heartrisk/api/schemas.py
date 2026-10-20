"""Request/response contracts. Bounds come from heartrisk.config so the API rejects
exactly the values the training cleaner would have treated as implausible."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heartrisk.config import NUMERIC_RANGES as R

EXAMPLE = {
    "age": 63,
    "sex": 1,
    "cp": 4,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 2,
    "thal": 7,
}


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    age: float = Field(..., ge=R["age"][0], le=R["age"][1], description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = Field(
        ..., ge=R["trestbps"][0], le=R["trestbps"][1], description="Resting BP (mm Hg)"
    )
    chol: float = Field(
        ..., ge=R["chol"][0], le=R["chol"][1], description="Serum cholesterol (mg/dl)"
    )
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG: 0 normal, 1 ST-T abnormal, 2 LVH"
    )
    thalach: float = Field(
        ..., ge=R["thalach"][0], le=R["thalach"][1], description="Max heart rate achieved"
    )
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(
        ..., ge=R["oldpeak"][0], le=R["oldpeak"][1], description="ST depression (mm)"
    )
    slope: Literal[1, 2, 3] = Field(..., description="Peak ST slope: 1 up, 2 flat, 3 down")
    ca: Literal[0, 1, 2, 3] | None = Field(
        None, description="Major vessels coloured (0-3); optional"
    )
    thal: Literal[3, 6, 7] | None = Field(
        None, description="3 normal, 6 fixed, 7 reversible defect; optional"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted")
    label: str
    probability: float = Field(..., description="Model probability of heart disease")
    confidence: float = Field(..., description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]
    threshold: float = Field(..., description="Decision threshold applied to probability")
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PatientFeatures] = Field(..., min_length=1, max_length=256)


class BatchItem(BaseModel):
    prediction: int
    label: str
    probability: float
    confidence: float
    risk_band: str


class BatchResponse(BaseModel):
    predictions: list[BatchItem]
    count: int
    threshold: float
    model_version: str
    request_id: str
