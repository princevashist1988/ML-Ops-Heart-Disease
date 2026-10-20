"""Serving API contract, validation, observability and failure modes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert (
        ready.status_code == 200 and ready.json()["model_version"] == "heart-disease-classifier/v99"
    )


def test_predict_returns_prediction_and_confidence(client, high_risk):
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["confidence"] == pytest.approx(
        max(body["probability"], 1 - body["probability"]), abs=1e-4
    )
    assert body["threshold"] == 0.45  # taken from the model card
    assert body["prediction"] == int(body["probability"] >= 0.45)
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_model_ranks_high_risk_above_low_risk(client, high_risk, low_risk):
    hi = client.post("/predict", json=high_risk).json()
    lo = client.post("/predict", json=low_risk).json()
    assert hi["probability"] > lo["probability"]
    assert hi["prediction"] == 1 and lo["prediction"] == 0
    assert hi["risk_band"] == "high" and lo["risk_band"] == "low"


def test_optional_fluoroscopy_fields_may_be_missing(client, high_risk):
    high_risk.pop("ca")
    high_risk["thal"] = None
    assert client.post("/predict", json=high_risk).status_code == 200


@pytest.mark.parametrize(
    "patch, field",
    [({"age": 5}, "age"), ({"cp": 9}, "cp"), ({"chol": "high"}, "chol"), ({"thal": 5}, "thal")],
)
def test_invalid_payloads_are_rejected_with_field_details(client, high_risk, patch, field):
    r = client.post("/predict", json={**high_risk, **patch})
    assert r.status_code == 422
    assert any(e["field"].startswith(field) for e in r.json()["detail"])


def test_unknown_fields_and_missing_required_fields_are_rejected(client, high_risk):
    assert client.post("/predict", json={**high_risk, "patient_name": "x"}).status_code == 422
    high_risk.pop("age")
    assert client.post("/predict", json=high_risk).status_code == 422


def test_batch_prediction(client, high_risk, low_risk):
    r = client.post("/predict/batch", json={"instances": [high_risk, low_risk, high_risk]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and len(body["predictions"]) == 3
    assert body["predictions"][0] == body["predictions"][2]
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_request_id_is_propagated(client, high_risk):
    r = client.post("/predict", json=high_risk, headers={"X-Request-ID": "trace-123"})
    assert r.headers["X-Request-ID"] == "trace-123" and r.json()["request_id"] == "trace-123"


def test_metrics_endpoint_exposes_prometheus_counters(client, high_risk):
    client.post("/predict", json=high_risk)
    client.post("/predict", json={**high_risk, "age": 1})
    text = client.get("/metrics").text
    assert 'heartrisk_http_requests_total{endpoint="/predict",method="POST",status="200"}' in text
    assert "heartrisk_predictions_total" in text
    assert "heartrisk_prediction_probability_bucket" in text
    assert 'heartrisk_validation_errors_total{endpoint="/predict"}' in text
    assert 'heartrisk_model_info{model_type="logistic_regression"' in text


def test_stats_dashboard_and_model_info(client, high_risk):
    client.post("/predict", json=high_risk)
    stats = client.get("/stats").json()
    assert stats["predictions"] >= 1 and len(stats["probability_histogram"]) == 10
    assert "Heart Disease Risk API" in client.get("/dashboard").text
    docs = client.get("/docs").text
    assert "/static/swagger/swagger-ui-bundle.js" in docs and "cdn.jsdelivr" not in docs
    assert client.get("/static/swagger/swagger-ui.css").status_code == 200
    info = client.get("/model-info").json()
    assert info["decision_threshold"] == 0.45 and info["model_type"] == "logistic_regression"


def test_threshold_can_be_overridden_by_environment(model_artifacts, monkeypatch, high_risk):
    monkeypatch.setenv("MODEL_PATH", str(model_artifacts["model"]))
    monkeypatch.setenv("MODEL_CARD_PATH", str(model_artifacts["card"]))
    monkeypatch.setenv("DECISION_THRESHOLD", "1.0")
    from heartrisk.api.main import app

    with TestClient(app) as c:
        body = c.post("/predict", json=high_risk).json()
    assert body["threshold"] == 1.0 and body["prediction"] == 0


def test_missing_model_reports_not_ready(tmp_path, monkeypatch, high_risk):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    from heartrisk.api.main import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # liveness unaffected
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=high_risk).status_code == 503
