"""Generate the Grafana dashboard JSON (keeps panel definitions reviewable as code).

python scripts/build_grafana_dashboard.py > deploy/monitoring/grafana/dashboards/heart-risk-api.json
"""

from __future__ import annotations

import json

DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def _next_id() -> int:
    global _id
    _id += 1
    return _id


def target(expr: str, legend: str = "", instant: bool = False) -> dict:
    return {
        "datasource": DS,
        "expr": expr,
        "legendFormat": legend,
        "refId": chr(64 + _id % 26 + 1),
        "instant": instant,
        "range": not instant,
    }


def stat(title, expr, unit, x, y, w=4, h=4, legend="", thresholds=None, decimals=None):
    steps = thresholds or [{"color": "green", "value": None}]
    p = {
        "id": _next_id(),
        "type": "stat",
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, legend)],
        "fieldConfig": {
            "defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps}},
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "area",
            "textMode": "auto",
        },
    }
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    return p


def timeseries(title, targets, unit, x, y, w=12, h=8, stack=False, desc=""):
    return {
        "id": _next_id(),
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(e, lg) for e, lg in targets],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 2,
                    "fillOpacity": 15 if stack else 5,
                    "stacking": {"mode": "normal" if stack else "none"},
                    "showPoints": "never",
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {
                "displayMode": "table",
                "placement": "bottom",
                "calcs": ["lastNotNull", "max"],
            },
            "tooltip": {"mode": "multi"},
        },
    }


def bargauge(title, expr, x, y, w=12, h=8):
    return {
        "id": _next_id(),
        "type": "bargauge",
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{**target(expr, "{{le}}", instant=True), "format": "heatmap"}],
        "fieldConfig": {
            "defaults": {"unit": "short", "color": {"mode": "continuous-GrYlRd"}},
            "overrides": [],
        },
        "options": {
            "orientation": "vertical",
            "displayMode": "gradient",
            "showUnfilled": True,
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }


def row(title, y):
    return {
        "id": _next_id(),
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


API = 'endpoint=~"/predict.*"'
panels = [
    row("Service health", 0),
    stat(
        "Request rate",
        f"sum(rate(heartrisk_http_requests_total{{{API}}}[1m]))",
        "reqps",
        0,
        1,
        decimals=2,
    ),
    stat(
        "Server error rate (5xx)",
        f'sum(rate(heartrisk_http_requests_total{{{API},status=~"5.."}}[5m])) / '
        f"clamp_min(sum(rate(heartrisk_http_requests_total{{{API}}}[5m])), 1e-9)",
        "percentunit",
        4,
        1,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 0.02},
            {"color": "red", "value": 0.05},
        ],
        decimals=1,
    ),
    stat(
        "p95 latency /predict",
        'histogram_quantile(0.95, sum by (le) (rate(heartrisk_http_request_duration_seconds_bucket{endpoint="/predict"}[5m])))',
        "s",
        8,
        1,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 0.1},
            {"color": "red", "value": 0.25},
        ],
        decimals=3,
    ),
    stat(
        "Predictions / min",
        "sum(rate(heartrisk_predictions_total[1m])) * 60",
        "short",
        12,
        1,
        decimals=1,
    ),
    stat(
        "Disease-flag rate (15m)",
        'sum(increase(heartrisk_predictions_total{label="heart_disease"}[15m])) / '
        "clamp_min(sum(increase(heartrisk_predictions_total[15m])), 1e-9)",
        "percentunit",
        16,
        1,
        decimals=0,
    ),
    stat(
        "Ready pods",
        "sum(heartrisk_model_loaded)",
        "short",
        20,
        1,
        thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
    ),
    row("Traffic & latency", 5),
    timeseries(
        "Requests by endpoint and status",
        [
            (
                "sum by (endpoint, status) (rate(heartrisk_http_requests_total[1m]))",
                "{{endpoint}} {{status}}",
            )
        ],
        "reqps",
        0,
        6,
    ),
    timeseries(
        "Latency quantiles - /predict",
        [
            (
                f'histogram_quantile({q}, sum by (le) (rate(heartrisk_http_request_duration_seconds_bucket{{endpoint="/predict"}}[2m])))',
                f"p{int(q * 100)}",
            )
            for q in (0.5, 0.95, 0.99)
        ]
        + [
            (
                "histogram_quantile(0.95, sum by (le) (rate(heartrisk_model_inference_seconds_bucket[2m])))",
                "model inference p95",
            )
        ],
        "s",
        12,
        6,
    ),
    row("Model behaviour", 14),
    timeseries(
        "Predictions by outcome and risk band",
        [
            (
                "sum by (label, risk_band) (rate(heartrisk_predictions_total[1m])) * 60",
                "{{label}} / {{risk_band}}",
            )
        ],
        "short",
        0,
        15,
        stack=True,
        desc="Predictions per minute",
    ),
    bargauge(
        "Predicted probability distribution (cumulative buckets, since start)",
        "sum by (le) (heartrisk_prediction_probability_bucket)",
        12,
        15,
    ),
    row("Data drift & input quality", 23),
    timeseries(
        "Live input mean vs training mean",
        [
            (
                "sum by (feature) (rate(heartrisk_input_feature_sum[5m])) / "
                "sum by (feature) (rate(heartrisk_input_feature_count[5m]))",
                "live {{feature}}",
            ),
            ("max by (feature) (heartrisk_training_feature_mean)", "train {{feature}}"),
        ],
        "short",
        0,
        24,
        w=16,
        desc="A sustained gap between live and training means signals input drift (alert: HeartRiskInputDrift)",
    ),
    timeseries(
        "Rejected payloads (422)",
        [("sum by (endpoint) (rate(heartrisk_validation_errors_total[1m])) * 60", "{{endpoint}}")],
        "short",
        16,
        24,
        w=8,
    ),
]

dashboard = {
    "uid": "heart-risk-api",
    "title": "Heart Disease Risk API",
    "tags": ["mlops", "heart-risk", "fastapi"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": panels,
    "annotations": {"list": []},
    "templating": {"list": []},
}

if __name__ == "__main__":
    print(json.dumps(dashboard, indent=2))
