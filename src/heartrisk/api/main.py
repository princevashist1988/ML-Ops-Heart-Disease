"""Heart disease risk API.

    uvicorn heartrisk.api.main:app --host 0.0.0.0 --port 8000

Endpoints
    POST /predict          single patient -> prediction + confidence
    POST /predict/batch    up to 256 patients
    GET  /health           liveness (process is up)
    GET  /ready            readiness (model loaded) - used by Kubernetes
    GET  /model-info       model card summary
    GET  /metrics          Prometheus exposition format
    GET  /stats            rolling 15-minute JSON summary
    GET  /dashboard        lightweight live monitoring page (no external assets)
    GET  /docs             OpenAPI / Swagger UI
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heartrisk import __version__, config
from heartrisk.api import monitoring as mon
from heartrisk.api.schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse
from heartrisk.predict import ModelService

log = mon.configure_logging()
activity = mon.RecentActivity()
QUIET_PATHS = {"/metrics", "/health", "/ready", "/stats"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    try:
        threshold = os.getenv("DECISION_THRESHOLD")
        svc = ModelService(
            os.getenv("MODEL_PATH", config.MODEL_PATH),
            os.getenv("MODEL_CARD_PATH", config.MODEL_CARD_PATH),
            threshold=float(threshold) if threshold else None,
        )
        app.state.model = svc
        mon.set_model_info(
            svc.version, svc.model_type, svc.threshold, svc.card.get("training_profile", {})
        )
        log.info(
            "model loaded",
            extra={
                "fields": {
                    "event": "startup",
                    "model_version": svc.version,
                    "model_type": svc.model_type,
                    "threshold": svc.threshold,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001 - keep serving /health, report not-ready
        app.state.model_error = str(exc)
        mon.MODEL_LOADED.set(0)
        log.error("model failed to load", extra={"fields": {"event": "startup", "error": str(exc)}})
    yield
    log.info("shutdown", extra={"fields": {"event": "shutdown"}})


app = FastAPI(
    title="Heart Disease Risk API",
    version=__version__,
    description="Predicts the risk of heart disease from 13 clinical attributes (UCI Cleveland). "
    "Educational prototype - not a medical device.",
    lifespan=lifespan,
    docs_url=None,  # served below from vendored assets: no CDN needed (air-gapped friendly)
)
STATIC_DIR = resources.files("heartrisk.api").joinpath("static")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/docs", include_in_schema=False)
def swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - docs",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
        swagger_favicon_url="/static/swagger/favicon-32x32.png",
    )


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    mon.IN_FLIGHT.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        mon.IN_FLIGHT.dec()
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        mon.REQUESTS.labels(request.method, endpoint, str(status)).inc()
        mon.LATENCY.labels(request.method, endpoint).observe(elapsed)
        summary = getattr(request.state, "summary", None)
        activity.add(
            time=datetime.now().strftime("%H:%M:%S"),
            request_id=request_id,
            method=request.method,
            path=endpoint,
            status=status,
            latency_ms=round(elapsed * 1000, 2),
            summary=summary,
            probabilities=getattr(request.state, "probabilities", []),
            predictions=getattr(request.state, "predictions", []),
        )
        if endpoint not in QUIET_PATHS or status >= 400:
            log.info(
                "request",
                extra={
                    "fields": {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "endpoint": endpoint,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                        "user_agent": request.headers.get("user-agent"),
                        "prediction": summary,
                    }
                },
            )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    route = request.scope.get("route")
    mon.VALIDATION_ERRORS.labels(getattr(route, "path", request.url.path)).inc()
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()
    ]
    log.warning(
        "validation failed",
        extra={
            "fields": {
                "event": "validation_error",
                "request_id": getattr(request.state, "request_id", None),
                "errors": errors,
            }
        },
    )
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "request_id": getattr(request.state, "request_id", None)},
    )


def _model(request: Request) -> ModelService:
    svc = request.app.state.model
    if svc is None:
        raise RuntimeError(f"Model not loaded: {request.app.state.model_error}")
    return svc


@app.exception_handler(RuntimeError)
async def not_ready_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _score(request: Request, records: list[dict]):
    svc = _model(request)
    t0 = time.perf_counter()
    preds = svc.predict(records)
    mon.INFERENCE.observe(time.perf_counter() - t0)
    mon.observe_inputs(records)
    for p in preds:
        mon.PREDICTIONS.labels(p.label.replace(" ", "_"), p.risk_band).inc()
        mon.PROBABILITY.observe(p.probability)
    request.state.probabilities = [p.probability for p in preds]
    request.state.predictions = [p.prediction for p in preds]
    pos = sum(p.prediction for p in preds)
    request.state.summary = (
        f"{preds[0].label} (p={preds[0].probability:.2f})"
        if len(preds) == 1
        else f"{len(preds)} scored, {pos} positive"
    )
    return svc, preds


@app.get("/", tags=["service"])
def root():
    return {
        "service": "heart-disease-risk-api",
        "version": __version__,
        "endpoints": [
            "/predict",
            "/predict/batch",
            "/health",
            "/ready",
            "/model-info",
            "/metrics",
            "/stats",
            "/dashboard",
            "/docs",
        ],
    }


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["service"])
def ready(request: Request):
    svc = request.app.state.model
    if svc is None:
        return JSONResponse(
            status_code=503, content={"status": "not ready", "error": request.app.state.model_error}
        )
    return {"status": "ready", "model_version": svc.version}


@app.get("/model-info", tags=["model"])
def model_info(request: Request):
    svc = _model(request)
    card = svc.card
    return {
        "model_version": svc.version,
        "model_type": svc.model_type,
        "decision_threshold": svc.threshold,
        "threshold_policy": card.get("threshold_policy"),
        "created_at": card.get("created_at"),
        "features": card.get("features"),
        "test_metrics": card.get("test_metrics_tuned_threshold"),
        "cv_metrics": card.get("cv_metrics"),
        "intended_use": card.get("intended_use"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(patient: PatientFeatures, request: Request):
    svc, preds = _score(request, [patient.model_dump()])
    return PredictionResponse(
        **preds[0].to_dict(), model_version=svc.version, request_id=request.state.request_id
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["model"])
def predict_batch(batch: BatchRequest, request: Request):
    svc, preds = _score(request, [p.model_dump() for p in batch.instances])
    items = [{k: v for k, v in p.to_dict().items() if k != "threshold"} for p in preds]
    return BatchResponse(
        predictions=items,
        count=len(items),
        threshold=svc.threshold,
        model_version=svc.version,
        request_id=request.state.request_id,
    )


@app.get("/metrics", tags=["monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", tags=["monitoring"])
def stats(request: Request):
    snap = activity.snapshot()
    svc = request.app.state.model
    snap["model_version"] = svc.version if svc else None
    snap["threshold"] = svc.threshold if svc else None
    snap["pod"] = os.getenv("POD_NAME") or os.getenv("HOSTNAME")
    return snap


@app.get("/dashboard", response_class=HTMLResponse, tags=["monitoring"])
def dashboard():
    return (
        resources.files("heartrisk.api")
        .joinpath("static/dashboard.html")
        .read_text(encoding="utf-8")
    )
