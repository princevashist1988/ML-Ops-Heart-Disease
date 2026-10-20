"""Inference wrapper and CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from heartrisk.predict import ModelService, main, risk_band


@pytest.mark.parametrize(
    "p, band",
    [
        (0.0, "low"),
        (0.249, "low"),
        (0.25, "moderate"),
        (0.599, "moderate"),
        (0.6, "high"),
        (1.0, "high"),
    ],
)
def test_risk_bands(p, band):
    assert risk_band(p) == band


def test_service_uses_card_threshold_unless_overridden(model_artifacts, high_risk):
    svc = ModelService(model_artifacts["model"], model_artifacts["card"])
    assert svc.threshold == 0.45 and svc.version == "heart-disease-classifier/v99"
    forced = ModelService(model_artifacts["model"], model_artifacts["card"], threshold=1.0)
    assert forced.predict(high_risk)[0].prediction == 0


def test_cli_scores_json_and_csv(model_artifacts, tmp_path, high_risk, low_risk, capsys, clean_df):
    js = tmp_path / "in.json"
    js.write_text(json.dumps(high_risk))
    assert main(["--input", str(js), "--model", str(model_artifacts["model"])]) == 0
    assert json.loads(capsys.readouterr().out)["label"] in ("heart disease", "no heart disease")

    csv_in, csv_out = tmp_path / "in.csv", tmp_path / "out.csv"
    clean_df.head(10).to_csv(csv_in, index=False)
    assert (
        main(
            [
                "--input",
                str(csv_in),
                "--output",
                str(csv_out),
                "--model",
                str(model_artifacts["model"]),
            ]
        )
        == 0
    )
    assert len(pd.read_csv(csv_out)) == 10
