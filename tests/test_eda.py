"""EDA produces every figure and a coherent summary, even without raw files."""

from __future__ import annotations

import json

from heartrisk import config, data, eda


def test_eda_writes_figures_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(eda, "EDA_DIR", tmp_path / "figures" / "eda")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    def no_raw(*_a, **_k):
        raise FileNotFoundError

    monkeypatch.setattr(data, "load_raw", no_raw)
    summary = eda.run()
    assert len(summary["figures"]) == 8
    assert all((tmp_path / f).stat().st_size > 10_000 for f in summary["figures"])
    assert summary["class_counts"] == {"No disease": 164, "Disease": 139}
    # the engineered-feature motivation must hold in the data
    assert summary["below_85pct_hr_disease_rate"] > summary["above_85pct_hr_disease_rate"]
    assert json.loads((tmp_path / "eda_summary.json").read_text())["rows"] == 303


def test_outlier_table_covers_numeric_features(clean_df):
    table = eda.outlier_table(clean_df)
    assert list(table["feature"]) == config.NUMERIC_FEATURES
    assert (table["iqr_outliers"] >= 0).all()
