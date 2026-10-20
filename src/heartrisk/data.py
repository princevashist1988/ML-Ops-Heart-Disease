"""Data acquisition, parsing, cleaning and validation for the UCI Heart Disease dataset.

Usage
-----
    python -m heartrisk.data download   # fetch raw files into data/raw (+ provenance.json)
    python -m heartrisk.data prepare    # clean and write data/processed/*.csv
    python -m heartrisk.data all        # both

Download strategy: the official UCI archive is tried first. If it is unreachable
(e.g. a firewalled network), byte-identical public copies of the UCI file are used.
The Cleveland file is verified against a pinned SHA-256 (three independent copies
agree on it), so a mirror can never silently change the training data. The
Hungarian file, used only for external validation, falls back to the Weka ARFF
distribution, converted back to the UCI numeric coding.
Every file is fingerprinted in data/raw/provenance.json so a run can be traced to
the exact bytes it trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from heartrisk import config

log = logging.getLogger("heartrisk.data")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_LEGACY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.{site}.data"
)
# Byte-identical copies of the UCI processed.cleveland.data file
CLEVELAND_MIRRORS = [
    "https://raw.githubusercontent.com/reinaldoq/processing-heart-disease-dataset/master/processed.cleveland.data",
    "https://raw.githubusercontent.com/amiralansary/Heart-Disease-Diagnosis/master/data/processed.cleveland.data",
]
CLEVELAND_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
# Weka ARFF distribution (nominal labels). Used only for the Hungarian external set:
# its Cleveland copy is known to drop one record and duplicate another.
ARFF_MIRROR_URL = "https://raw.githubusercontent.com/renatopp/arff-datasets/master/classification/heart.{code}.arff"
SITES = {"cleveland": "c", "hungarian": "h"}

# Weka ARFF nominal labels -> UCI numeric codes (from the UCI heart-disease.names file)
ARFF_CODEBOOK: dict[str, dict[str, float]] = {
    "sex": {"female": 0, "male": 1},
    "cp": {"typ_angina": 1, "atyp_angina": 2, "non_anginal": 3, "asympt": 4},
    "fbs": {"f": 0, "t": 1},
    "restecg": {"normal": 0, "st_t_wave_abnormality": 1, "left_vent_hyper": 2},
    "exang": {"no": 0, "yes": 1},
    "slope": {"up": 1, "flat": 2, "down": 3},
    "thal": {"normal": 3, "fixed_defect": 6, "reversable_defect": 7},
    "num": {"<50": 0, ">50_1": 1, ">50_2": 2, ">50_3": 3, ">50_4": 4},
}


class DataValidationError(ValueError):
    """Raised when a dataframe violates the expected schema."""


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    duplicates_removed: int
    out_of_range_nulled: dict[str, int]
    missing_after: dict[str, int]
    positive_rate: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- download
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(url: str, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _uci_candidates(site: str) -> list[tuple[str, str]]:
    """(kind, url) pairs in priority order for one site."""
    cands = [("uci-zip", UCI_ZIP_URL), ("uci", UCI_LEGACY_URL.format(site=site))]
    if site == "cleveland":
        cands += [("mirror", url) for url in CLEVELAND_MIRRORS]
    cands.append(("weka-arff", ARFF_MIRROR_URL.format(code=SITES[site])))
    return cands


def _fetch_site(site: str, raw_dir: Path, allowed: set[str]) -> dict:
    errors = []
    zip_cache: dict[str, bytes] = {}
    for kind, url in _uci_candidates(site):
        if kind.split("-")[0] not in allowed and kind not in allowed:
            continue
        try:
            if kind == "uci-zip":
                blob = zip_cache.get(url) or _get(url)
                zip_cache[url] = blob
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    content = zf.read(f"processed.{site}.data")
            else:
                content = _get(url)
            if site == "cleveland" and kind != "weka-arff":
                digest = _sha256(content)
                if digest != CLEVELAND_SHA256:
                    raise ValueError(f"checksum mismatch ({digest[:12]}...)")
            name = f"heart.{SITES[site]}.arff" if kind == "weka-arff" else f"processed.{site}.data"
            (raw_dir / name).write_bytes(content)
            return {"file": name, "source": kind, "url": url, "sha256": _sha256(content)}
        except Exception as exc:  # noqa: BLE001 - any failure falls through to the next source
            errors.append(f"{kind}: {exc}")
            log.warning("%s: %s failed (%s)", site, kind, str(exc)[:120])
    raise RuntimeError(f"All sources failed for {site}: " + " | ".join(errors))


def download(raw_dir: Path = config.RAW_DIR, source: str = "auto") -> dict:
    """Download the raw files. ``source`` is one of auto | uci | mirror."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        "auto": {"uci", "mirror", "weka-arff"},
        "uci": {"uci"},
        "mirror": {"mirror", "weka-arff"},
    }[source]
    files = {"cleveland": _fetch_site("cleveland", raw_dir, allowed)}
    try:
        files["hungarian"] = _fetch_site("hungarian", raw_dir, allowed)
    except RuntimeError as exc:  # external set is optional
        log.warning("External validation data unavailable: %s", exc)
    provenance = {
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "UCI Heart Disease (id 45), doi:10.24432/C52P4X",
        "files": files,
        "usage": "cleveland = training/test; hungarian = external (out-of-site) validation",
    }
    (raw_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log.info("Downloaded: %s", {k: v["source"] for k, v in files.items()})
    return provenance


# --------------------------------------------------------------------------- parsing
def read_uci_processed(path: Path | str) -> pd.DataFrame:
    """Read a UCI ``processed.<site>.data`` file ('?' marks missing values)."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def read_arff(path: Path | str) -> pd.DataFrame:
    """Read the Weka ARFF mirror and convert nominal labels back to UCI numeric codes."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    body = text.split("@data", 1)[1] if "@data" in text else text.split("@DATA", 1)[1]
    df = pd.read_csv(
        io.StringIO(body.strip()),
        header=None,
        names=config.RAW_COLUMNS,
        na_values="?",
        quotechar="'",
        skipinitialspace=True,
        comment="%",
    )
    for col, mapping in ARFF_CODEBOOK.items():
        df[col] = df[col].map(
            lambda v, m=mapping: m.get(str(v).strip(), np.nan) if pd.notna(v) else np.nan
        )
    return df


def load_raw(site: str = "cleveland", raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    uci = raw_dir / f"processed.{site}.data"
    arff = raw_dir / f"heart.{SITES[site]}.arff"
    if uci.exists():
        return read_uci_processed(uci)
    if arff.exists():
        return read_arff(arff)
    raise FileNotFoundError(
        f"No raw file for '{site}' in {raw_dir}. Run `python -m heartrisk.data download`."
    )


# --------------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Type-coerce, null out implausible values, binarise the target and de-duplicate.

    Missing values are deliberately *kept*: they are imputed inside the model
    pipeline, fitted on training folds only, so the same logic runs at inference.
    """
    rows_in = len(df)
    out = df.copy()
    for col in config.RAW_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    nulled: dict[str, int] = {}
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        bad = out[col].notna() & ~out[col].between(lo, hi)
        nulled[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    for col, domain in config.CATEGORY_DOMAINS.items():
        bad = out[col].notna() & ~out[col].isin(domain)
        nulled[col] = int(bad.sum())
        out.loc[bad, col] = np.nan

    out = out[out["num"].notna()]
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    dupes = before - len(out)

    for col in config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]:
        out[col] = out[col].astype("Int64")

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(out),
        duplicates_removed=dupes,
        out_of_range_nulled={k: v for k, v in nulled.items() if v},
        missing_after={k: int(v) for k, v in out.isna().sum().items() if v},
        positive_rate=round(float(out[config.TARGET].mean()), 4),
    )
    return out, report


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Schema contract used by training, tests and CI. Raises DataValidationError."""
    expected = config.FEATURES + ([config.TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if len(df) == 0:
        raise DataValidationError("Dataframe is empty")
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.between(lo, hi).all():
            raise DataValidationError(f"{col} has values outside [{lo}, {hi}]")
    for col, domain in config.CATEGORY_DOMAINS.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.isin(domain).all():
            raise DataValidationError(f"{col} has values outside {sorted(domain)}")
    if require_target:
        if set(df[config.TARGET].unique()) - {0, 1}:
            raise DataValidationError("target must be binary 0/1")
        if df[config.TARGET].nunique() < 2:
            raise DataValidationError("target has a single class")


def load_clean(path: Path = config.CLEAN_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]:
        df[col] = df[col].astype("Int64")
    validate(df)
    return df


def prepare(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    targets = {"cleveland": config.CLEAN_DATA_PATH, "hungarian": config.EXTERNAL_DATA_PATH}
    for site, target_path in targets.items():
        try:
            raw = load_raw(site, raw_dir)
        except FileNotFoundError:
            if site == "cleveland":
                raise
            log.warning("External site %s not available, skipping", site)
            continue
        cleaned, report = clean(raw)
        validate(cleaned)
        path = out_dir / target_path.name
        cleaned.to_csv(path, index=False)
        summary[site] = {"path": str(path.relative_to(config.PROJECT_ROOT)), **report.to_dict()}
        log.info("%s: %s", site, report.to_dict())
    (out_dir / "cleaning_report.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["download", "prepare", "all"])
    parser.add_argument("--source", choices=["auto", "uci", "mirror"], default="auto")
    args = parser.parse_args(argv)
    if args.command in ("download", "all"):
        prov = download(source=args.source)
        print(json.dumps(prov, indent=2))
    if args.command in ("prepare", "all"):
        print(json.dumps(prepare(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
