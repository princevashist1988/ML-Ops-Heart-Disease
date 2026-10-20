"""Inference: load the packaged pipeline + model card and score patient records.

python -m heartrisk.predict --input samples/patient_high_risk.json
python -m heartrisk.predict --input data/processed/heart_cleveland_clean.csv --output preds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from heartrisk import config
from heartrisk.features import to_frame

RISK_BANDS = [(0.25, "low"), (0.60, "moderate"), (1.01, "high")]


def risk_band(probability: float) -> str:
    for upper, name in RISK_BANDS:
        if probability < upper:
            return name
    return "high"


@dataclass
class Prediction:
    prediction: int
    label: str
    probability: float  # P(heart disease)
    confidence: float  # probability of the predicted class
    risk_band: str
    threshold: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ModelService:
    """Thin, thread-safe (read-only) wrapper around the trained sklearn pipeline."""

    def __init__(
        self,
        model_path: Path | str = config.MODEL_PATH,
        card_path: Path | str = config.MODEL_CARD_PATH,
        threshold: float | None = None,
    ):
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)
        card_path = Path(card_path)
        self.card: dict = json.loads(card_path.read_text()) if card_path.exists() else {}
        self.threshold = float(
            threshold if threshold is not None else self.card.get("decision_threshold", 0.5)
        )

    @property
    def version(self) -> str:
        reg = self.card.get("registry", {})
        if reg:
            return f"{reg.get('name')}/v{reg.get('version')}"
        return self.card.get("created_at", "unversioned")

    @property
    def model_type(self) -> str:
        return self.card.get("model_type", type(self.pipeline[-1]).__name__)

    def predict_proba(self, records) -> np.ndarray:
        return self.pipeline.predict_proba(to_frame(records))[:, 1]

    def predict(self, records) -> list[Prediction]:
        out = []
        for p in self.predict_proba(records):
            p = float(p)
            pred = int(p >= self.threshold)
            out.append(
                Prediction(
                    prediction=pred,
                    label="heart disease" if pred else "no heart disease",
                    probability=round(p, 4),
                    confidence=round(p if pred else 1 - p, 4),
                    risk_band=risk_band(p),
                    threshold=self.threshold,
                )
            )
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, type=Path, help="JSON object/list or CSV file")
    ap.add_argument("--output", type=Path, help="optional CSV output path")
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = ap.parse_args(argv)

    svc = ModelService(args.model)
    if args.input.suffix == ".csv":
        records = pd.read_csv(args.input)
    else:
        records = json.loads(args.input.read_text())
    preds = [p.to_dict() for p in svc.predict(records)]
    if args.output:
        pd.DataFrame(preds).to_csv(args.output, index=False)
        print(f"Wrote {len(preds)} predictions to {args.output}")
    else:
        print(json.dumps(preds if len(preds) > 1 else preds[0], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
