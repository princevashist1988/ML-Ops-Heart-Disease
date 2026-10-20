"""Exploratory data analysis: figures + a JSON summary.

    python -m heartrisk.eda        # writes reports/figures/eda/*.png and reports/eda_summary.json

The notebook notebooks/01_eda.ipynb calls these functions and adds commentary.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from heartrisk import config, data  # noqa: E402
from heartrisk.evaluate import ACCENT, CALM, GRID, INK, style_axes  # noqa: E402

log = logging.getLogger("heartrisk.eda")
EDA_DIR = config.FIGURES_DIR / "eda"
LABELS = {0: "No disease", 1: "Disease"}
PALETTE = {0: CALM, 1: ACCENT}
CATEGORY_NAMES = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abnormal", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
    "sex": {0: "female", 1: "male"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "exang": {0: "no", 1: "yes"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
}
NICE = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "oldpeak": "ST depression (mm)",
    "ca": "Vessels coloured (0-3)",
}


def _save(fig, name: str) -> Path:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> Path:
    fig, axes = plt.subplots(
        1, 2 if raw is not None else 1, figsize=(10 if raw is not None else 5, 3.8)
    )
    axes = np.atleast_1d(axes)
    counts = df[config.TARGET].value_counts().sort_index()
    bars = axes[0].bar(
        [LABELS[i] for i in counts.index], counts.values, color=[CALM, ACCENT], width=0.6
    )
    for b, c in zip(bars, counts.values, strict=True):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            c + 3,
            f"{c} ({c / counts.sum():.0%})",
            ha="center",
            fontsize=10,
            color=INK,
        )
    axes[0].set_ylabel("Patients")
    axes[0].set_ylim(0, counts.max() * 1.18)
    style_axes(axes[0], "Binary target: near-balanced (54/46)")
    if raw is not None:
        sev = raw["num"].value_counts().sort_index()
        axes[1].bar(
            sev.index.astype(int).astype(str),
            sev.values,
            color=[CALM] + [ACCENT] * (len(sev) - 1),
            width=0.6,
        )
        axes[1].set_xlabel("Original UCI 'num' (0 = <50% narrowing, 1-4 = severity)")
        axes[1].set_ylabel("Patients")
        style_axes(axes[1], "Source label collapsed to 0 vs 1-4")
    return _save(fig, "01_class_balance.png")


def plot_missing(df: pd.DataFrame, external: pd.DataFrame | None = None) -> Path:
    frames = {"Cleveland (train)": df}
    if external is not None:
        frames["Hungarian (external)"] = external
    miss = pd.DataFrame({k: v[config.FEATURES].isna().mean() * 100 for k, v in frames.items()})
    fig, ax = plt.subplots(figsize=(9, 3.8))
    miss.plot.bar(ax=ax, color=[ACCENT, "#64748b"][: len(frames)], width=0.75)
    ax.set_ylabel("% missing")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(miss.iloc[:, 0]):
        if v > 0:
            ax.text(i - 0.19, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8, color=ACCENT)
    style_axes(ax, "Missingness: 6 cells in Cleveland vs structural gaps in the external site")
    ax.legend(frameon=False)
    return _save(fig, "02_missing_values.png")


def plot_numeric_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES, strict=True):
        for t in (0, 1):
            vals = df.loc[df[config.TARGET] == t, col].dropna().astype(float)
            discrete = col == "ca"
            bins = np.arange(-0.5, 4.5, 1) if discrete else 18
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                color=PALETTE[t],
                label=LABELS[t],
                density=True,
                edgecolor="white",
                linewidth=0.5,
            )
            if not discrete:
                ax.axvline(vals.median(), color=PALETTE[t], ls="--", lw=1.4)
        ax.set_xlabel(NICE[col])
        style_axes(ax, col)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Numeric features by diagnosis (dashed = class median)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    return _save(fig, "03_numeric_histograms.png")


def plot_boxplots(df: pd.DataFrame) -> Path:
    long = df.melt(id_vars=config.TARGET, value_vars=config.NUMERIC_FEATURES[:-1])
    long["value"] = long["value"].astype(float)
    long["Diagnosis"] = long[config.TARGET].map(LABELS)
    g = sns.catplot(
        data=long,
        x="Diagnosis",
        y="value",
        col="variable",
        kind="box",
        sharey=False,
        height=3.2,
        aspect=0.75,
        palette=[CALM, ACCENT],
        hue="Diagnosis",
        legend=False,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    for ax in g.axes.ravel():
        style_axes(ax)
    g.figure.suptitle(
        "Spread and outliers (IQR whiskers)",
        x=0.01,
        ha="left",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    path = EDA_DIR / "04_boxplots.png"
    g.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return path


def plot_categorical_rates(df: pd.DataFrame) -> Path:
    cols = ["cp", "thal", "ca", "slope", "exang", "sex", "restecg", "fbs"]
    base = df[config.TARGET].mean()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for ax, col in zip(axes.ravel(), cols, strict=True):
        grp = df.groupby(df[col].astype("float"))[config.TARGET].agg(["mean", "size"]).dropna()
        names = [CATEGORY_NAMES[col].get(int(i), str(i)) for i in grp.index]
        colours = [ACCENT if m > base else CALM for m in grp["mean"]]
        ax.bar(names, grp["mean"], color=colours, width=0.65)
        ax.axhline(base, color="grey", ls="--", lw=1)
        for i, (m, n) in enumerate(zip(grp["mean"], grp["size"], strict=True)):
            ax.text(i, m + 0.02, f"{m:.0%}\nn={n}", ha="center", fontsize=7.5, color=INK)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", labelsize=7.5, rotation=20)
        style_axes(ax, col)
    axes[0, 0].set_ylabel("Disease rate")
    axes[1, 0].set_ylabel("Disease rate")
    fig.suptitle(
        f"Disease rate by category (dashed = overall {base:.0%})",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    return _save(fig, "05_categorical_disease_rates.png")


def plot_correlation(df: pd.DataFrame) -> Path:
    cols = config.FEATURES + [config.TARGET]
    corr = df[cols].astype(float).corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.6))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(175, 25, s=80, l=45, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title(
        "Correlation heatmap (Spearman, robust to ordinal codes)",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    return _save(fig, "06_correlation_heatmap.png")


def plot_heart_rate_reserve(df: pd.DataFrame) -> Path:
    """Motivates the engineered feature hr_pct_predicted."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for t in (0, 1):
        d = df[df[config.TARGET] == t]
        ax.scatter(
            d["age"],
            d["thalach"],
            s=22,
            alpha=0.7,
            color=PALETTE[t],
            label=LABELS[t],
            edgecolor="white",
            linewidth=0.4,
        )
    ages = np.linspace(df["age"].min() - 2, df["age"].max() + 2, 50)
    ax.plot(ages, 220 - ages, color=INK, lw=1.4, label="age-predicted max (220 - age)")
    ax.plot(ages, 0.85 * (220 - ages), color=INK, lw=1.2, ls="--", label="85% of predicted max")
    ax.set(xlabel="Age (years)", ylabel="Max heart rate achieved (bpm)")
    style_axes(ax, "Peak heart rate vs age: diseased patients cluster below the 85% line")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    return _save(fig, "07_heart_rate_vs_age.png")


def plot_mutual_information(df: pd.DataFrame) -> tuple[Path, pd.Series]:
    X = df[config.FEATURES].astype(float)
    X = X.fillna(X.median())
    discrete = [c not in ("age", "trestbps", "chol", "thalach", "oldpeak") for c in config.FEATURES]
    mi = pd.Series(
        mutual_info_classif(
            X, df[config.TARGET], discrete_features=discrete, random_state=config.SEED
        ),
        index=config.FEATURES,
    ).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(
        mi.index, mi.values, color=[ACCENT if v >= mi.median() else "#94a3b8" for v in mi.values]
    )
    ax.set_xlabel("Mutual information with diagnosis (nats)")
    style_axes(ax, "Univariate signal strength")
    return _save(fig, "08_mutual_information.png"), mi.sort_values(ascending=False)


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.NUMERIC_FEATURES:
        s = df[col].astype(float).dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append(
            {
                "feature": col,
                "min": s.min(),
                "q1": q1,
                "median": s.median(),
                "q3": q3,
                "max": s.max(),
                "iqr_outliers": n_out,
            }
        )
    return pd.DataFrame(rows)


def run() -> dict:
    sns.set_theme(style="white", rc={"axes.edgecolor": GRID})
    df = data.load_clean()
    raw = data.load_raw("cleveland")
    ext = pd.read_csv(config.EXTERNAL_DATA_PATH) if config.EXTERNAL_DATA_PATH.exists() else None
    figs = [
        plot_class_balance(df, raw),
        plot_missing(df, ext),
        plot_numeric_distributions(df),
        plot_boxplots(df),
        plot_categorical_rates(df),
        plot_correlation(df),
        plot_heart_rate_reserve(df),
    ]
    mi_path, mi = plot_mutual_information(df)
    figs.append(mi_path)
    hr = df["thalach"] / (220 - df["age"])
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "class_counts": {
            LABELS[k]: int(v) for k, v in df[config.TARGET].value_counts().sort_index().items()
        },
        "missing_cells": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicates": int(df.duplicated().sum()),
        "describe": df[config.NUMERIC_FEATURES].astype(float).describe().round(2).to_dict(),
        "outliers_iqr": outlier_table(df).set_index("feature")["iqr_outliers"].to_dict(),
        "mutual_information": mi.round(4).to_dict(),
        "spearman_with_target": df[config.FEATURES + [config.TARGET]]
        .astype(float)
        .corr(method="spearman")[config.TARGET]
        .drop(config.TARGET)
        .round(3)
        .sort_values()
        .to_dict(),
        "hr_pct_predicted_median_by_class": hr.groupby(df[config.TARGET])
        .median()
        .round(3)
        .to_dict(),
        "below_85pct_hr_disease_rate": round(float(df.loc[hr < 0.85, config.TARGET].mean()), 3),
        "above_85pct_hr_disease_rate": round(float(df.loc[hr >= 0.85, config.TARGET].mean()), 3),
        "figures": [str(p.relative_to(config.PROJECT_ROOT)) for p in figs],
    }
    (config.REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "describe"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
