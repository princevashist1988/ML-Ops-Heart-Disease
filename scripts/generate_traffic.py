"""Synthetic traffic generator for monitoring demos (stdlib only, runs anywhere).

Replays realistic patients sampled from the cleaned dataset, mixes in a share of
invalid payloads, and can simulate input drift (an older, hypertensive population).

    python scripts/generate_traffic.py --url http://localhost:8000 --minutes 3
    python scripts/generate_traffic.py --url http://heart-risk.local --rps 5 --drift
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "heart_cleveland_clean.csv"


def load_patients() -> list[dict]:
    rows = []
    with open(DATA) as fh:
        for r in csv.DictReader(fh):
            rec = {}
            for f in FEATURES:
                v = r[f]
                if v == "":
                    rec[f] = None
                elif f == "oldpeak":
                    rec[f] = float(v)
                else:
                    rec[f] = int(float(v))
            rows.append(rec)
    return rows


def drifted(p: dict) -> dict:
    q = dict(p)
    q["age"] = min(95, q["age"] + random.randint(8, 18))
    q["trestbps"] = min(215, q["trestbps"] + random.randint(10, 30))
    q["chol"] = min(600, q["chol"] + random.randint(20, 60))
    return q


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "heart-risk-traffic-generator"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--rps", type=float, default=4.0, help="average requests per second")
    ap.add_argument("--invalid-share", type=float, default=0.05)
    ap.add_argument("--batch-share", type=float, default=0.1)
    ap.add_argument("--drift", action="store_true", help="shift inputs to an older population")
    args = ap.parse_args()

    patients = load_patients()
    end = time.time() + args.minutes * 60
    counts: dict[int, int] = {}
    sent = 0
    while time.time() < end:
        p = random.choice(patients)
        if args.drift:
            p = drifted(p)
        roll = random.random()
        if roll < args.invalid_share:
            status = post(f"{args.url}/predict", {**p, "age": random.choice([5, 130]), "cp": 9})
        elif roll < args.invalid_share + args.batch_share:
            batch = [
                drifted(x) if args.drift else x
                for x in random.sample(patients, random.randint(2, 8))
            ]
            status = post(f"{args.url}/predict/batch", {"instances": batch})
        else:
            status = post(f"{args.url}/predict", p)
        counts[status] = counts.get(status, 0) + 1
        sent += 1
        if sent % 50 == 0:
            print(f"sent={sent} status_counts={dict(sorted(counts.items()))}", flush=True)
        time.sleep(random.expovariate(args.rps))
    print(f"done: sent={sent} status_counts={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
