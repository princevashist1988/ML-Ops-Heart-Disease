#!/usr/bin/env bash
# Local replica of .github/workflows/ci-cd.yml: lint -> test -> train -> docker -> smoke.
# Stops at the first failing stage with a non-zero exit code (same contract as CI).
#   scripts/ci_local.sh                      # full run
#   SKIP_DOCKER=1 scripts/ci_local.sh        # Python stages only
#   BASE_IMAGE=<mirror>/python:3.12-slim scripts/ci_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore
PY="${PY:-python}"
IMAGE="${IMAGE:-heart-risk-api:1.0.0}"
mkdir -p logs reports

stage() { printf '\n==== [%s] %s ====\n' "$(date +%H:%M:%S)" "$1"; }

stage "1/6 lint (ruff)"
ruff check src tests scripts
ruff format --check src tests scripts

stage "2/6 unit tests (pytest + coverage)"
$PY -m pytest --cov --cov-report=term --cov-report=xml:reports/coverage.xml \
    --junitxml=reports/junit.xml -q 2>&1 | tee logs/pytest.log
$PY scripts/ci_summary.py tests reports/junit.xml reports/coverage.xml

stage "3/6 data acquisition + EDA"
$PY -m heartrisk.data all > logs/data.log 2>&1 && tail -n 3 logs/data.log
$PY -m heartrisk.eda > logs/eda.log 2>&1 && echo "EDA figures -> reports/figures/eda"

stage "4/6 train + MLflow tracking + quality gate"
rm -rf mlflow.db mlartifacts
$PY -m heartrisk.train 2>&1 | grep -v "matplotlib.category" | tee logs/train.log
$PY scripts/ci_summary.py training reports/metrics.json reports/model_comparison.csv

if [ "${SKIP_DOCKER:-0}" = "1" ]; then echo "SKIP_DOCKER=1 - done"; exit 0; fi

stage "5/6 docker build"
docker build ${DOCKER_BUILD_FLAGS:-} ${BASE_IMAGE:+--build-arg BASE_IMAGE=$BASE_IMAGE} -t "$IMAGE" . 2>&1 \
  | tail -n 5 | tee logs/docker_build.log

stage "6/6 run container + smoke test"
docker rm -f heart-risk-ci >/dev/null 2>&1 || true
docker run -d --name heart-risk-ci -p 18000:8000 "$IMAGE" >/dev/null
trap 'docker logs heart-risk-ci > logs/container.log 2>&1; docker rm -f heart-risk-ci >/dev/null 2>&1' EXIT
scripts/smoke_test.sh http://localhost:18000 | tee logs/container_smoke_test.log

printf '\nCI pipeline passed.\n'
