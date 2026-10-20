"""Render Markdown summaries for the GitHub Actions job summary page.

python scripts/ci_summary.py tests reports/junit.xml reports/coverage.xml
python scripts/ci_summary.py training reports/metrics.json reports/model_comparison.csv
"""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def tests(junit: str, coverage: str | None = None) -> str:
    if not Path(junit).exists():
        return "### Tests\nNo JUnit report produced (tests did not run).\n"
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    a = suite.attrib
    total, fail, err, skip = (int(a.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    status = "PASSED" if fail + err == 0 else "FAILED"
    out = [
        f"### Unit tests: {status}",
        "",
        "| total | passed | failed | errors | skipped | time (s) |",
        "|---|---|---|---|---|---|",
        f"| {total} | {total - fail - err - skip} | {fail} | {err} | {skip} "
        f"| {float(a.get('time', 0)):.1f} |",
    ]
    if coverage and Path(coverage).exists():
        rate = float(ET.parse(coverage).getroot().attrib.get("line-rate", 0))
        out += ["", f"Line coverage: **{rate:.1%}**"]
    failed = [tc.attrib["name"] for tc in suite.iter("testcase") if tc.find("failure") is not None]
    if failed:
        out += ["", "Failed:"] + [f"- `{n}`" for n in failed]
    return "\n".join(out) + "\n"


def training(metrics_path: str, comparison_path: str) -> str:
    if not Path(metrics_path).exists():
        return "### Training\nNo metrics produced (training failed before packaging).\n"
    m = json.loads(Path(metrics_path).read_text())
    out = [
        "### Model comparison (repeated stratified 5-fold CV on the training split)",
        "",
        "| model | CV ROC-AUC | CV recall | test ROC-AUC |",
        "|---|---|---|---|",
    ]
    with open(comparison_path) as fh:
        for r in csv.DictReader(fh):
            auc, sd = float(r["cv_roc_auc_mean"]), float(r["cv_roc_auc_std"])
            out.append(
                f"| {r['name']} | {auc:.3f} ± {sd:.3f} "
                f"| {float(r['cv_recall_mean']):.3f} | {float(r['test_roc_auc']):.3f} |"
            )
    t = m["test"]
    out += [
        "",
        f"**Champion:** `{m['champion']}`  |  decision threshold **{m['threshold']:.2f}**",
        "",
        "| hold-out metric | value |",
        "|---|---|",
    ]
    out += [f"| {k} | {t[k]:.3f} |" for k in ("roc_auc", "accuracy", "precision", "recall", "f1")]
    ext = m.get("external_hungarian") or {}
    if ext:
        out += [
            "",
            f"External validation (Hungarian site): ROC-AUC {ext['roc_auc']:.3f}, "
            f"recall {ext['recall']:.3f}, precision {ext['precision']:.3f}",
        ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    kind, *paths = sys.argv[1:]
    print({"tests": tests, "training": training}[kind](*paths))
