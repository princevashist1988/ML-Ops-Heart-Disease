"""Metrics, operating-threshold selection and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# One palette for every figure in the project (report + MLflow artifacts)
INK = "#1f2933"
ACCENT = "#c2410c"  # disease / positive class
CALM = "#0f766e"  # no disease / negative class
GRID = "#d9dee3"
MODEL_COLOURS = {
    "logistic_regression": "#2563eb",
    "random_forest": "#0f766e",
    "gradient_boosting": "#c2410c",
}


def style_axes(ax, title: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_threshold(y_true, proba, beta: float = 2.0, min_precision: float = 0.6) -> float:
    """Pick the operating threshold that maximises F-beta on out-of-fold predictions.

    beta=2 weights recall 4x as much as precision: in screening, a missed case
    (false negative) costs far more than an extra follow-up test. A precision
    floor stops the optimiser from degenerating to "flag everyone".
    """
    y_true = np.asarray(y_true).astype(int)
    best_t, best_score = 0.5, -1.0
    for t in np.round(np.arange(0.20, 0.71, 0.01), 2):
        pred = (proba >= t).astype(int)
        if precision_score(y_true, pred, zero_division=0) < min_precision:
            continue
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if score > best_score + 1e-9:
            best_t, best_score = float(t), score
    return best_t


# --------------------------------------------------------------------------- plots
def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_pr(y_true, proba, path: Path, label: str = "model") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    style_axes(axes[0], f"ROC curve - {label}")
    axes[0].legend(loc="lower right", frameon=False)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(
        rec, prec, color=CALM, lw=2, label=f"AP = {average_precision_score(y_true, proba):.3f}"
    )
    axes[1].axhline(np.mean(y_true), ls="--", color="grey", lw=1, label="prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.05))
    style_axes(axes[1], "Precision-recall curve")
    axes[1].legend(loc="lower left", frameon=False)
    return _save(fig, path)


def plot_confusion(y_true, proba, threshold: float, path: Path, label: str = "model") -> Path:
    cm = confusion_matrix(
        np.asarray(y_true).astype(int), (np.asarray(proba) >= threshold).astype(int)
    )
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Oranges")
    names = ["No disease", "Disease"]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else INK,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"{label} @ threshold {threshold:.2f}", loc="left", fontsize=11, fontweight="bold")
    return _save(fig, path)


def plot_calibration(y_true, proba, path: Path, label: str = "model") -> Path:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color=ACCENT, lw=2, label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed disease rate")
    style_axes(ax, "Calibration (reliability)")
    ax.legend(frameon=False)
    return _save(fig, path)


def feature_importance(pipeline) -> pd.DataFrame:
    """Signed coefficients for linear models, impurity importance for trees."""
    names = pipeline[:-1].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = clf.coef_.ravel()
        kind = "coefficient"
    elif hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        kind = "importance"
    else:
        return pd.DataFrame(columns=["feature", "value", "kind"])
    df = pd.DataFrame({"feature": names, "value": values, "kind": kind})
    return df.reindex(df["value"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_feature_importance(pipeline, path: Path, label: str = "model", top: int = 15) -> Path:
    fi = feature_importance(pipeline).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    colours = [ACCENT if v > 0 else CALM for v in fi["value"]]
    ax.barh(fi["feature"], fi["value"], color=colours)
    kind = fi["kind"].iloc[0] if len(fi) else "importance"
    ax.set_xlabel(
        "Coefficient (standardised)" if kind == "coefficient" else "Mean impurity decrease"
    )
    style_axes(ax, f"Top features - {label}")
    return _save(fig, path)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """summary: rows = model, columns = cv_<metric>_mean / _std."""
    metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(summary)
    x = np.arange(len(metrics))
    for i, (name, row) in enumerate(summary.iterrows()):
        means = [row[f"cv_{m}_mean"] for m in metrics]
        stds = [row[f"cv_{m}_std"] for m in metrics]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=name,
            color=MODEL_COLOURS.get(name, "#64748b"),
        )
    ax.set_xticks(
        x + width * (len(summary) - 1) / 2, [m.upper().replace("_", "-") for m in metrics]
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Repeated stratified 5-fold CV (mean +/- sd)")
    style_axes(ax, "Model comparison on the training set")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, path)
