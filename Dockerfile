# Heart disease risk API - multi-stage build, non-root, ~1 worker per container
# (scale with replicas, not workers, so Prometheus counters stay per-process exact).
#
#   docker build -t heart-risk-api:1.0.0 .
#   docker run --rm -p 8000:8000 heart-risk-api:1.0.0
#
# BASE_IMAGE is overridable for air-gapped builds, e.g. an internal mirror of python:3.12-slim.
ARG BASE_IMAGE=python:3.12-slim

# ---------------------------------------------------------------- builder: resolve deps once
FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-serve.txt

# ---------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime
ARG MODEL_VERSION=local
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="Heart disease risk classifier (UCI Cleveland) served with FastAPI" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      ai.heartrisk.model-version="${MODEL_VERSION}"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HEARTRISK_ROOT=/app \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_CARD_PATH=/app/models/model_card.json \
    LOG_LEVEL=INFO \
    PORT=8000

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
 && mkdir -p /var/log/heartrisk && chown app:app /var/log/heartrisk
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/model.joblib models/model_card.json ./models/

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python3 -c "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/ready', timeout=2).status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn heartrisk.api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log --proxy-headers"]
