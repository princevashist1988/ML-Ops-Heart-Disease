"""Data acquisition, parsing, cleaning and validation."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from heartrisk import config, data

UCI_LINES = """63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
38.0,1.0,3.0,138.0,175.0,0.0,0.0,173.0,0.0,0.0,1.0,?,3.0,0
"""

ARFF = """% comment line
@relation cleveland-14-heart-disease
@attribute 'age' real
@data
63,male,typ_angina,145,233,t,left_vent_hyper,150,no,2.3,down,0,fixed_defect,'<50'
67,female,asympt,160,286,f,normal,108,yes,1.5,flat,?,reversable_defect,'>50_1'
"""


@pytest.fixture()
def uci_file(tmp_path):
    p = tmp_path / "processed.cleveland.data"
    p.write_text(UCI_LINES)
    return p


def test_read_uci_marks_question_marks_as_missing(uci_file):
    df = data.read_uci_processed(uci_file)
    assert list(df.columns) == config.RAW_COLUMNS
    assert len(df) == 4
    assert np.isnan(df.loc[3, "ca"])


def test_clean_binarises_target_and_keeps_missing_for_pipeline(uci_file):
    df, report = data.clean(data.read_uci_processed(uci_file))
    assert "num" not in df.columns
    assert df[config.TARGET].tolist() == [0, 1, 1, 0]  # severities 1-4 collapse to 1
    assert report.missing_after == {"ca": 1}  # imputed later, inside the pipeline
    assert str(df["ca"].dtype) == "Int64"


def test_clean_nulls_physiologically_impossible_values(uci_file):
    raw = data.read_uci_processed(uci_file)
    raw.loc[0, "chol"] = 0  # Hungarian-style "0 means unknown"
    raw.loc[1, "thal"] = 5  # outside {3, 6, 7}
    df, report = data.clean(raw)
    assert pd.isna(df.loc[0, "chol"]) and pd.isna(df.loc[1, "thal"])
    assert report.out_of_range_nulled == {"chol": 1, "thal": 1}


def test_clean_removes_exact_duplicates(uci_file):
    raw = data.read_uci_processed(uci_file)
    df, report = data.clean(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))
    assert report.duplicates_removed == 1 and len(df) == 4


def test_arff_mirror_is_converted_to_uci_codes(tmp_path):
    p = tmp_path / "heart.c.arff"
    p.write_text(ARFF)
    df = data.read_arff(p)
    first = df.iloc[0]
    assert (
        first["sex"],
        first["cp"],
        first["fbs"],
        first["restecg"],
        first["slope"],
        first["thal"],
    ) == (1, 1, 1, 2, 3, 6)
    second = df.iloc[1]
    assert (
        second["sex"] == 0 and second["cp"] == 4 and second["num"] == 1 and np.isnan(second["ca"])
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["thalach"]), "Missing columns"),
        (lambda d: d.assign(age=d["age"].where(d.index != 0, 150)), "age"),
        (lambda d: d.assign(cp=d["cp"].where(d.index != 0, 9)), "cp"),
        (lambda d: d.assign(target=0), "single class"),
        (lambda d: d.iloc[0:0], "empty"),
    ],
)
def test_validate_rejects_contract_violations(clean_df, mutate, message):
    with pytest.raises(data.DataValidationError, match=message):
        data.validate(mutate(clean_df.copy()))


def test_committed_clean_dataset_matches_known_uci_facts(clean_df):
    """303 Cleveland patients, 139 with disease, 4 missing ca and 2 missing thal."""
    assert clean_df.shape == (303, 14)
    assert int(clean_df[config.TARGET].sum()) == 139
    assert clean_df.isna().sum()[lambda s: s > 0].to_dict() == {"ca": 4, "thal": 2}
    data.validate(clean_df)


def test_download_rejects_tampered_mirror_and_falls_back(tmp_path, monkeypatch):
    """A mirror whose bytes do not match the pinned checksum must never be used."""

    def fake_get(url, timeout=30):
        if "archive.ics.uci.edu" in url:
            raise ConnectionError("blocked")
        if url.endswith(".arff"):
            return ARFF.encode()
        return b"tampered,content\n"

    monkeypatch.setattr(data, "_get", fake_get)
    prov = data.download(raw_dir=tmp_path)
    assert prov["files"]["cleveland"]["source"] == "weka-arff"
    assert not (tmp_path / "processed.cleveland.data").exists()
    assert (tmp_path / "provenance.json").exists()


def test_download_accepts_mirror_with_correct_checksum(tmp_path, monkeypatch):
    payload = UCI_LINES.encode()
    monkeypatch.setattr(data, "CLEVELAND_SHA256", hashlib.sha256(payload).hexdigest())

    def fake_get(url, timeout=30):
        if "archive.ics.uci.edu" in url:
            raise ConnectionError("blocked")
        return payload if url.endswith("processed.cleveland.data") else ARFF.encode()

    monkeypatch.setattr(data, "_get", fake_get)
    prov = data.download(raw_dir=tmp_path)
    assert prov["files"]["cleveland"]["source"] == "mirror"
    assert (tmp_path / "processed.cleveland.data").read_bytes() == payload
    assert data.load_raw("cleveland", tmp_path).shape == (4, 14)
