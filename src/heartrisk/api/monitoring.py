"""Observability: Prometheus metrics, structured JSON logs and a rolling in-memory window.

Design notes
* Endpoint labels use the route template, never the raw URL, to bound cardinality.
* Request logs carry a request id, latency and outcome but never the clinical
  inputs themselves; prediction logs carry only aggregate scores.
* Live input means are exported next to the training means from the model card,
  so a dashboard can show feature drift without storing patient data.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import UTC, datetime

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary

# ----------------------------------------------------------------------------- metrics
REQUESTS = Counter(
    "heartrisk_http_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
LATENCY = Histogram(
    "heartrisk_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)
IN_FLIGHT = Gauge("heartrisk_http_requests_in_flight", "Requests being served")
PREDICTIONS = Counter("heartrisk_predictions_total", "Predictions served", ["label", "risk_band"])
PROBABILITY = Histogram(
    "heartrisk_prediction_probability",
    "Distribution of predicted disease probability",
    buckets=tuple(np.round(np.arange(0.1, 1.01, 0.1), 1)),
)
INFERENCE = Histogram(
    "heartrisk_model_inference_seconds",
    "Model inference time per request",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
VALIDATION_ERRORS = Counter("heartrisk_validation_errors_total", "Rejected payloads", ["endpoint"])
MODEL_INFO = Gauge(
    "heartrisk_model_info",
    "Loaded model (value is always 1)",
    ["model_version", "model_type", "threshold"],
)
MODEL_LOADED = Gauge("heartrisk_model_loaded", "1 when a model is loaded and ready")
FEATURE_INPUT = Summary(
    "heartrisk_input_feature", "Live values of monitored input features", ["feature"]
)
FEATURE_TRAIN_MEAN = Gauge(
    "heartrisk_training_feature_mean", "Training-set mean per feature", ["feature"]
)
MONITORED_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def observe_inputs(records: list[dict]) -> None:
    for rec in records:
        for f in MONITORED_FEATURES:
            v = rec.get(f)
            if v is not None:
                FEATURE_INPUT.labels(feature=f).observe(float(v))


def set_model_info(version: str, model_type: str, threshold: float, training_profile: dict) -> None:
    MODEL_INFO.labels(
        model_version=version, model_type=model_type, threshold=f"{threshold:.2f}"
    ).set(1)
    MODEL_LOADED.set(1)
    for f, stats in (training_profile or {}).items():
        if f in MONITORED_FEATURES and "mean" in stats:
            FEATURE_TRAIN_MEAN.labels(feature=f).set(stats["mean"])


# ----------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("heartrisk.api")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(JsonFormatter())
        logger.addHandler(stream)
        log_file = os.getenv("LOG_FILE")
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
                fh = logging.FileHandler(log_file)
                fh.setFormatter(JsonFormatter())
                logger.addHandler(fh)
            except OSError as exc:  # never let a log volume take the service down
                logger.warning("file logging disabled: %s", exc)
    return logger


# ----------------------------------------------------------------------------- rolling window
class RecentActivity:
    """Bounded, thread-safe ring buffer of recent requests for /stats and /dashboard."""

    def __init__(self, maxlen: int = 2000):
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.started = time.time()
        self.total = 0

    def add(self, **event) -> None:
        event.setdefault("ts", time.time())
        with self._lock:
            self._events.append(event)
            self.total += 1

    def snapshot(self, window_s: int = 900) -> dict:
        now = time.time()
        with self._lock:
            events = [e for e in self._events if now - e["ts"] <= window_s]
        api = [
            e
            for e in events
            if e["path"] not in ("/metrics", "/stats", "/dashboard", "/health", "/ready")
        ]
        lat = np.array([e["latency_ms"] for e in api]) if api else np.array([0.0])
        client_err = sum(1 for e in api if 400 <= e["status"] < 500)
        server_err = sum(1 for e in api if e["status"] >= 500)
        probs = [p for e in api for p in e.get("probabilities", [])]
        labels = [y for e in api for y in e.get("predictions", [])]
        hist, _ = np.histogram(probs, bins=10, range=(0, 1)) if probs else (np.zeros(10, int), None)
        per_min = {}
        for e in api:
            m = int((now - e["ts"]) // 60)
            per_min[m] = per_min.get(m, 0) + 1
        return {
            "uptime_s": round(now - self.started, 1),
            "window_s": window_s,
            "requests": len(api),
            "requests_total_since_start": self.total,
            "client_error_rate": round(client_err / len(api), 4) if api else 0.0,
            "server_error_rate": round(server_err / len(api), 4) if api else 0.0,
            "latency_ms": {
                "p50": round(float(np.percentile(lat, 50)), 2),
                "p95": round(float(np.percentile(lat, 95)), 2),
                "max": round(float(lat.max()), 2),
            },
            "predictions": len(probs),
            "positive_rate": round(float(np.mean(labels)), 4) if labels else 0.0,
            "probability_histogram": hist.tolist(),
            "requests_per_minute": [per_min.get(i, 0) for i in range(window_s // 60 - 1, -1, -1)],
            "status_counts": {
                str(s): sum(1 for e in api if e["status"] == s)
                for s in sorted({e["status"] for e in api})
            },
            "recent": [
                {
                    k: e.get(k)
                    for k in (
                        "time",
                        "request_id",
                        "method",
                        "path",
                        "status",
                        "latency_ms",
                        "summary",
                    )
                }
                for e in list(reversed(api))[:12]
            ],
        }
