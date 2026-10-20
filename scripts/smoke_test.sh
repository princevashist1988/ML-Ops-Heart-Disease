#!/usr/bin/env bash
# End-to-end smoke test against a running API (container, Kubernetes service or ingress).
#   scripts/smoke_test.sh [BASE_URL]          default http://localhost:8000
# Exits non-zero on the first failed check, so CI fails loudly.
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-http://localhost:8000}}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
RETRIES="${RETRIES:-30}"
if [ -t 1 ]; then C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_END=$'\033[0m'; else C_OK=""; C_ERR=""; C_END=""; fi
pass() { printf '  %sPASS%s %s\n' "$C_OK" "$C_END" "$1"; }
fail() { printf '  %sFAIL%s %s\n' "$C_ERR" "$C_END" "$1"; exit 1; }

echo "Smoke testing ${BASE_URL}"
for i in $(seq 1 "$RETRIES"); do
  if curl -fsS "${BASE_URL}/ready" >/dev/null 2>&1; then break; fi
  [ "$i" -eq "$RETRIES" ] && fail "service not ready after ${RETRIES}s"
  sleep 1
done
pass "/ready -> $(curl -fsS "${BASE_URL}/ready")"

[ "$(curl -fsS "${BASE_URL}/health" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')" = "ok" ] \
  && pass "/health" || fail "/health"

HI=$(curl -fsS -X POST "${BASE_URL}/predict" -H 'Content-Type: application/json' \
      -d @"${HERE}/samples/patient_high_risk.json")
echo "    high-risk -> ${HI}"
python3 - "$HI" <<'PY' && pass "/predict high-risk patient flagged with confidence" || fail "/predict high-risk"
import json, sys
r = json.loads(sys.argv[1])
assert r["prediction"] == 1 and 0.5 <= r["confidence"] <= 1 and r["risk_band"] == "high", r
PY

LO=$(curl -fsS -X POST "${BASE_URL}/predict" -H 'Content-Type: application/json' \
      -d @"${HERE}/samples/patient_low_risk.json")
echo "    low-risk  -> ${LO}"
python3 - "$LO" <<'PY' && pass "/predict low-risk patient cleared" || fail "/predict low-risk"
import json, sys
r = json.loads(sys.argv[1])
assert r["prediction"] == 0 and r["risk_band"] == "low", r
PY

N=$(curl -fsS -X POST "${BASE_URL}/predict/batch" -H 'Content-Type: application/json' \
      -d @"${HERE}/samples/batch.json" | python3 -c 'import sys,json;print(json.load(sys.stdin)["count"])')
[ "$N" = "3" ] && pass "/predict/batch scored 3 patients" || fail "/predict/batch"

CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/predict" \
        -H 'Content-Type: application/json' -d @"${HERE}/samples/invalid_patient.json")
[ "$CODE" = "422" ] && pass "invalid payload rejected with 422" || fail "invalid payload returned ${CODE}"

curl -fsS "${BASE_URL}/metrics" | grep -q 'heartrisk_predictions_total' \
  && pass "/metrics exposes Prometheus counters" || fail "/metrics"

echo "All smoke tests passed."
