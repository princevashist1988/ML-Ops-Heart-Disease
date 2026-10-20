"""Train, tune, compare and package the heart-disease classifier with MLflow tracking.

    python -m heartrisk.train                       # full run from configs/train.yaml
    python -m heartrisk.train --quick               # tiny grid, for smoke tests / CI checks
    python -m heartrisk.train --no-register         # skip the MLflow model registry

MLflow layout (experiment "heart-disease-risk"):
    parent run per model family  -> best params, CV mean/sd, hold-out metrics, plots, model
      child run per grid point   -> params + CV metrics (every candidate is tracked)
    champion run                 -> chosen model, threshold, external validation, registry
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from heartrisk import __version__, config, data, evaluate
from heartrisk.models import make_model

log = logging.getLogger("heartrisk.train")
SCORING = ["roc_auc", "accuracy", "precision", "recall", "f1"]
# MLflow stores sklearn models with skops (no arbitrary-code pickle). Our own
# transformers must be explicitly allow-listed to be loadable.
SKOPS_TRUSTED = [
    "heartrisk.features.ClinicalFeatureEngineer",
    "heartrisk.features.select_numeric",
    "heartrisk.features.select_binary",
    "heartrisk.features.select_categorical",
    "heartrisk.features.to_float",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",  # tree ensembles; files are produced by this pipeline only
]


def mlflow_model_kwargs(model, example: pd.DataFrame) -> dict:
    return {
        "signature": infer_signature(example, model.predict_proba(example)),
        "input_example": example,
        "code_paths": [str(Path(__file__).parent)],
        "serialization_format": "skops",
        "skops_trusted_types": SKOPS_TRUSTED,
        "pyfunc_predict_fn": "predict_proba",
    }


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def quick_config(cfg: dict) -> dict:
    """Shrink grids and CV so a full pipeline run takes seconds."""
    cfg = json.loads(json.dumps(cfg))
    cfg["cv"] = {"n_splits": 3, "n_repeats": 1}
    for spec in cfg["models"].values():
        spec["grid"] = {k: [v[len(v) // 2]] for k, v in spec["grid"].items()}
        if "n_estimators" in spec.get("params", {}):
            spec["params"]["n_estimators"] = 50
    return cfg


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=config.PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def setup_mlflow(experiment: str) -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        config.MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(
            experiment, artifact_location=config.MLFLOW_ARTIFACT_ROOT.resolve().as_uri()
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def _flat_params(params: dict) -> dict:
    return {k.replace("clf__", "").replace("features__", "fe_"): str(v) for k, v in params.items()}


def tune_family(name, spec, X_tr, y_tr, X_te, y_te, cfg, out_dir: Path) -> dict:
    """Grid-search one model family inside a parent MLflow run."""
    seed = cfg["seed"]
    cv = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["n_splits"], n_repeats=cfg["cv"]["n_repeats"], random_state=seed
    )
    search = GridSearchCV(
        make_model(name, spec.get("params"), seed),
        param_grid=spec["grid"],
        scoring=SCORING,
        refit=cfg["primary_metric"],
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"stage": "tuning", "model_family": name, "git_sha": git_sha()})
        mlflow.log_params(
            {
                "model": name,
                "n_candidates": int(np.prod([len(v) for v in spec["grid"].values()])),
                "cv_splits": cfg["cv"]["n_splits"],
                "cv_repeats": cfg["cv"]["n_repeats"],
                **{f"fixed_{k}": v for k, v in (spec.get("params") or {}).items()},
            }
        )
        t0 = time.perf_counter()
        search.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - t0
        res = pd.DataFrame(search.cv_results_)

        # one nested run per candidate so every configuration is visible in the UI
        for i, row in res.iterrows():
            with mlflow.start_run(run_name=f"{name}-{i:02d}", nested=True):
                mlflow.set_tags({"stage": "candidate", "model_family": name})
                mlflow.log_params(_flat_params(row["params"]))
                mlflow.log_metrics(
                    {
                        **{f"cv_{m}_mean": row[f"mean_test_{m}"] for m in SCORING},
                        **{f"cv_{m}_std": row[f"std_test_{m}"] for m in SCORING},
                        "train_roc_auc_mean": row["mean_train_roc_auc"],
                        "mean_fit_time_s": row["mean_fit_time"],
                        "rank_roc_auc": int(row["rank_test_roc_auc"]),
                    }
                )

        best = search.best_estimator_
        bi = search.best_index_
        cv_summary = {f"cv_{m}_mean": float(res.loc[bi, f"mean_test_{m}"]) for m in SCORING}
        cv_summary |= {f"cv_{m}_std": float(res.loc[bi, f"std_test_{m}"]) for m in SCORING}
        cv_summary["cv_overfit_gap"] = float(
            res.loc[bi, "mean_train_roc_auc"] - res.loc[bi, "mean_test_roc_auc"]
        )
        proba_te = best.predict_proba(X_te)[:, 1]
        test = {
            f"test_{k}": float(v) for k, v in evaluate.compute_metrics(y_te, proba_te, 0.5).items()
        }

        mlflow.log_params({f"best_{k}": v for k, v in _flat_params(search.best_params_).items()})
        mlflow.log_metrics({**cv_summary, **test, "search_seconds": fit_seconds})

        fam_dir = out_dir / name
        fam_dir.mkdir(parents=True, exist_ok=True)
        res_cols = (
            ["params", "rank_test_roc_auc"]
            + [f"mean_test_{m}" for m in SCORING]
            + [f"std_test_{m}" for m in SCORING]
            + ["mean_train_roc_auc", "mean_fit_time"]
        )
        res[res_cols].sort_values("rank_test_roc_auc").to_csv(
            fam_dir / "cv_results.csv", index=False
        )
        evaluate.plot_roc_pr(y_te, proba_te, fam_dir / "roc_pr.png", name)
        evaluate.plot_confusion(y_te, proba_te, 0.5, fam_dir / "confusion_matrix.png", name)
        evaluate.plot_calibration(y_te, proba_te, fam_dir / "calibration.png", name)
        evaluate.plot_feature_importance(best, fam_dir / "feature_importance.png", name)
        evaluate.feature_importance(best).to_csv(fam_dir / "feature_importance.csv", index=False)
        mlflow.log_artifacts(str(fam_dir), artifact_path="evaluation")

        mlflow.sklearn.log_model(
            best, name="model", **mlflow_model_kwargs(best, X_tr.head(5).astype(float))
        )
        log.info(
            "%-20s CV ROC-AUC %.3f +/- %.3f | test ROC-AUC %.3f | %.1fs",
            name,
            cv_summary["cv_roc_auc_mean"],
            cv_summary["cv_roc_auc_std"],
            test["test_roc_auc"],
            fit_seconds,
        )
        return {
            "name": name,
            "run_id": parent.info.run_id,
            "estimator": best,
            "best_params": search.best_params_,
            **cv_summary,
            **test,
        }


def select_champion(results: list[dict], cfg: dict) -> dict:
    key = f"cv_{cfg['primary_metric']}_mean"
    ranked = sorted(results, key=lambda r: r[key], reverse=True)
    best = ranked[0]
    tol = cfg.get("selection", {}).get("tolerance", 0.0)
    for preferred in cfg.get("selection", {}).get("prefer", []):
        cand = next((r for r in results if r["name"] == preferred), None)
        if cand and best[key] - cand[key] <= tol:
            best = cand
            break
    best["selection_reason"] = (
        f"highest CV {cfg['primary_metric']}"
        if best is ranked[0]
        else f"within {tol} of best CV {cfg['primary_metric']} ({ranked[0]['name']}); "
        "preferred for interpretability"
    )
    return best


def run(cfg: dict, register: bool = True, experiment: str = config.MLFLOW_EXPERIMENT) -> dict:
    np.random.seed(cfg["seed"])
    df = data.load_clean()
    X, y = df[config.FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"]
    )
    log.info("Train %d rows / test %d rows, positive rate %.3f", len(X_tr), len(X_te), y.mean())
    setup_mlflow(experiment)
    out_dir = config.REPORTS_DIR / "evaluation"

    results = [
        tune_family(n, s, X_tr, y_tr, X_te, y_te, cfg, out_dir) for n, s in cfg["models"].items()
    ]
    summary = pd.DataFrame(results).set_index("name").drop(columns=["estimator"])
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    evaluate.plot_model_comparison(summary, config.FIGURES_DIR / "model_comparison.png")

    champ = select_champion(results, cfg)
    model = champ["estimator"]
    log.info("Champion: %s (%s)", champ["name"], champ["selection_reason"])

    # Operating threshold from out-of-fold predictions on the training set only
    oof = cross_val_predict(
        clone(model),
        X_tr,
        y_tr,
        method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]),
    )[:, 1]
    tcfg = cfg["threshold"]
    threshold = evaluate.choose_threshold(y_tr, oof, tcfg["beta"], tcfg["min_precision"])
    proba_te = model.predict_proba(X_te)[:, 1]
    test_default = evaluate.compute_metrics(y_te, proba_te, 0.5)
    test_tuned = evaluate.compute_metrics(y_te, proba_te, threshold)

    external = {}
    if config.EXTERNAL_DATA_PATH.exists():
        ext = pd.read_csv(config.EXTERNAL_DATA_PATH)
        ext_proba = model.predict_proba(ext[config.FEATURES])[:, 1]
        external = evaluate.compute_metrics(ext[config.TARGET], ext_proba, threshold)
        evaluate.plot_roc_pr(
            ext[config.TARGET],
            ext_proba,
            config.FIGURES_DIR / "external_hungarian_roc_pr.png",
            f"{champ['name']} on Hungarian",
        )

    fig_dir = config.FIGURES_DIR
    evaluate.plot_roc_pr(y_te, proba_te, fig_dir / "champion_roc_pr.png", champ["name"])
    evaluate.plot_confusion(
        y_te, proba_te, threshold, fig_dir / "champion_confusion_tuned.png", champ["name"]
    )
    evaluate.plot_confusion(
        y_te, proba_te, 0.5, fig_dir / "champion_confusion_default.png", champ["name"]
    )
    evaluate.plot_calibration(y_te, proba_te, fig_dir / "champion_calibration.png", champ["name"])
    evaluate.plot_feature_importance(
        model, fig_dir / "champion_feature_importance.png", champ["name"]
    )

    # ------------------------------------------------------------- package artefacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    model_kwargs = mlflow_model_kwargs(model, X_tr.head(3).astype(float))
    mlflow_dir = config.MODELS_DIR / "mlflow_model"
    if mlflow_dir.exists():
        import shutil

        shutil.rmtree(mlflow_dir)
    mlflow.sklearn.save_model(model, str(mlflow_dir), **model_kwargs)

    provenance_path = config.RAW_DIR / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    card = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "package_version": __version__,
        "model_type": champ["name"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "selection_reason": champ["selection_reason"],
        "hyperparameters": dict(champ["best_params"]),
        "decision_threshold": threshold,
        "threshold_policy": (
            f"max F{tcfg['beta']:g} on out-of-fold train predictions, "
            f"precision >= {tcfg['min_precision']}"
        ),
        "features": {
            "numeric": config.NUMERIC_FEATURES,
            "binary": config.BINARY_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
            "engineered": config.ENGINEERED_FEATURES
            if champ["best_params"].get("features__enabled", True)
            else [],
        },
        "training_data": {
            "rows_train": len(X_tr),
            "rows_test": len(X_te),
            "sha256": provenance.get("files", {}).get("cleveland", {}).get("sha256"),
            "source": provenance.get("files", {}).get("cleveland", {}).get("url"),
        },
        "training_profile": {
            c: {
                "mean": round(float(X_tr[c].astype(float).mean()), 3),
                "std": round(float(X_tr[c].astype(float).std()), 3),
                "missing_rate": round(float(X_tr[c].isna().mean()), 4),
            }
            for c in config.NUMERIC_FEATURES
        },
        "cv_metrics": {k: round(v, 4) for k, v in champ.items() if k.startswith("cv_")},
        "test_metrics_threshold_0_5": {k: round(v, 4) for k, v in test_default.items()},
        "test_metrics_tuned_threshold": {k: round(v, 4) for k, v in test_tuned.items()},
        "external_validation_hungarian": {k: round(v, 4) for k, v in external.items()},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
        },
        "intended_use": (
            "Educational decision-support prototype. Not a medical device; not for clinical use."
        ),
    }

    # ------------------------------------------------------------- champion run + registry
    with mlflow.start_run(run_name=f"champion-{champ['name']}") as run_:
        mlflow.set_tags(
            {
                "stage": "champion",
                "model_family": champ["name"],
                "git_sha": card["git_sha"],
                "selection_reason": champ["selection_reason"],
                "source_tuning_run": champ["run_id"],
            }
        )
        mlflow.log_params(
            {
                "model": champ["name"],
                "decision_threshold": threshold,
                **{f"best_{k}": v for k, v in _flat_params(champ["best_params"]).items()},
                "train_rows": len(X_tr),
                "test_rows": len(X_te),
                "data_sha256": str(card["training_data"]["sha256"])[:16],
            }
        )
        mlflow.log_metrics(
            {
                **{k: v for k, v in champ.items() if k.startswith("cv_")},
                **{f"test_{k}": v for k, v in test_tuned.items()},
                **{f"test_default_{k}": v for k, v in test_default.items()},
                **{f"ext_hungarian_{k}": v for k, v in external.items()},
            }
        )
        mlflow.log_dict(card, "model_card.json")
        mlflow.log_artifact(str(config.CONFIG_DIR / "train.yaml"), artifact_path="config")
        mlflow.log_artifacts(str(fig_dir), artifact_path="figures")
        if provenance:
            mlflow.log_dict(provenance, "data/provenance.json")
        mlflow.sklearn.log_model(
            model,
            name="model",
            **model_kwargs,
            registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
        )
        card["mlflow_run_id"] = run_.info.run_id

    if register:
        client = mlflow.MlflowClient()
        latest = max(
            client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'"),
            key=lambda v: int(v.version),
        )
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", latest.version)
        client.update_model_version(
            config.REGISTERED_MODEL_NAME,
            latest.version,
            description=f"{champ['name']} | test ROC-AUC {test_tuned['roc_auc']:.3f} | "
            f"threshold {threshold:.2f}",
        )
        card["registry"] = {
            "name": config.REGISTERED_MODEL_NAME,
            "version": int(latest.version),
            "alias": "champion",
        }

    config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, default=str))
    metrics = {
        "champion": champ["name"],
        "threshold": threshold,
        "test": test_tuned,
        "test_default_threshold": test_default,
        "external_hungarian": external,
        "cv": card["cv_metrics"],
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return {"card": card, "summary": summary, "metrics": metrics}


def quality_gate(metrics: dict, cfg: dict) -> list[str]:
    gate = cfg.get("quality_gate", {})
    failures = []
    if metrics["test"]["roc_auc"] < gate.get("min_test_roc_auc", 0):
        failures.append(
            f"test ROC-AUC {metrics['test']['roc_auc']:.3f} < {gate['min_test_roc_auc']}"
        )
    if metrics["test"]["recall"] < gate.get("min_test_recall", 0):
        failures.append(f"test recall {metrics['test']['recall']:.3f} < {gate['min_test_recall']}")
    return failures


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=config.CONFIG_DIR / "train.yaml")
    p.add_argument("--quick", action="store_true", help="tiny grid for smoke tests")
    p.add_argument("--no-register", action="store_true", help="skip MLflow model registry")
    p.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    p.add_argument("--skip-gate", action="store_true", help="do not fail on the quality gate")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg = quick_config(cfg)
    if not config.CLEAN_DATA_PATH.exists():
        log.info("Clean data missing - running data preparation first")
        if not (config.RAW_DIR / "processed.cleveland.data").exists():
            data.download()
        data.prepare()

    out = run(cfg, register=not args.no_register, experiment=args.experiment)
    m = out["metrics"]
    print(
        "\n"
        + out["summary"][
            [c for c in out["summary"].columns if c.startswith(("cv_roc", "test_roc", "test_rec"))]
        ]
        .round(4)
        .to_string()
    )
    print(
        f"\nChampion: {m['champion']}  threshold={m['threshold']:.2f}  "
        f"test ROC-AUC={m['test']['roc_auc']:.3f}  recall={m['test']['recall']:.3f}  "
        f"precision={m['test']['precision']:.3f}"
    )
    failures = quality_gate(m, cfg)
    if failures and not (args.skip_gate or args.quick):
        log.error("QUALITY GATE FAILED: %s", "; ".join(failures))
        return 1
    log.info("Quality gate passed" if not failures else "Quality gate skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
