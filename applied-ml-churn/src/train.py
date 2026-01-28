# src/train.py
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,  # PR-AUC
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,            # ROC-AUC
)

from src.features import build_feature_artifacts

LOGGER = logging.getLogger(__name__)

# ----------------------------
# IO helpers
# ----------------------------
def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suffix} (expected .parquet or .csv)")

def _resolve_split_paths(data_dir: Path, metadata_path: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Returns: (train_path, val_path)

    Priority:
      1) metadata.json saved by src/data.py if present
      2) default filenames in data_dir: train.(parquet|csv) and val.(parquet|csv)
    """
    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"

    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        train_path = Path(meta["saved_files"]["train"])
        val_path = Path(meta["saved_files"]["val"])
        return train_path, val_path

    # Fallback: look for train/val parquet then csv
    for ext in ("parquet", "csv"):
        train_path = data_dir / f"train.{ext}"
        val_path = data_dir / f"val.{ext}"
        if train_path.exists() and val_path.exists():
            return train_path, val_path

    raise FileNotFoundError(
        f"Could not find train/val splits in {data_dir}. "
        "Expected metadata.json or train/val parquet/csv files."
    )

def _load_feature_lists_from_metadata(data_dir: Path, metadata_path: Optional[Path]) -> Tuple[Optional[list], Optional[list], Optional[str]]:
    """
    Returns: (numeric_cols, categorical_cols, target_col) if metadata exists, else (None, None, None)
    """
    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"

    if not metadata_path.exists():
        return None, None, None

    meta = json.loads(metadata_path.read_text())
    return meta.get("numeric_cols"), meta.get("categorical_cols"), meta.get("target_col")

# ----------------------------
# Training + evaluation
# ----------------------------
def _evaluate_binary(y_true, y_prob, y_pred) -> Dict[str, float]:
    """
    Computes required metrics (PR-AUC, ROC-AUC) + a few helpful extras.
    """
    metrics: Dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return metrics

def train_and_evaluate(
    data_dir: Path,
    *,
    target_col: str = "churn",
    metadata_path: Optional[Path] = None,
    model_out: Path = Path("models/model.joblib"),
    config_out: Path = Path("models/config.json"),
    metrics_out: Path = Path("reports/metrics/val_metrics.json"),
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    drop_cols: Optional[list[str]] = None,
) -> None:
    # Resolve paths + load data
    train_path, val_path = _resolve_split_paths(data_dir, metadata_path=metadata_path)
    LOGGER.info("Loading train split: %s", train_path)
    LOGGER.info("Loading val split:   %s", val_path)

    train_df = _read_table(train_path)
    val_df = _read_table(val_path)

    # Prefer metadata feature lists if present (keeps things consistent with src/data.py)
    meta_num, meta_cat, meta_target = _load_feature_lists_from_metadata(data_dir, metadata_path)
    if meta_target is not None:
        target_col = meta_target

    if target_col not in train_df.columns or target_col not in val_df.columns:
        raise ValueError(
            f"Target column '{target_col}' must exist in train and val. "
            f"Train cols: {list(train_df.columns)}"
        )

    drop_cols = drop_cols or []
    # Common ID columns are usually not helpful features; keep opt-in drop via CLI
    # (You can pass: --drop_cols customerid)
    # NOTE: do not silently drop anything unless user requests it.

    # Split X/y
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].astype(int)

    # Build baseline model
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=None,
    )

    # Build preprocess+model pipeline
    # If numeric/categorical cols were saved in metadata, use them.
    artifacts = build_feature_artifacts(
        df=train_df.drop(columns=drop_cols) if drop_cols else train_df,
        target_col=target_col,
        numeric_cols=meta_num,
        categorical_cols=meta_cat,
        drop_cols=drop_cols,
        model=model,
    )

    pipeline = artifacts.pipeline

    # Fit
    LOGGER.info("Fitting pipeline (preprocess + LogisticRegression)...")
    pipeline.fit(
        X_train.drop(columns=drop_cols) if drop_cols else X_train,
        y_train,
    )

    # Evaluate on validation
    LOGGER.info("Evaluating on validation split...")
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_val.drop(columns=drop_cols) if drop_cols else X_val)[:, 1]
    else:
        raise RuntimeError("Pipeline model does not support predict_proba (required for PR-AUC/ROC-AUC).")

    y_pred = pipeline.predict(X_val.drop(columns=drop_cols) if drop_cols else X_val)

    val_metrics = _evaluate_binary(y_true=y_val, y_prob=y_prob, y_pred=y_pred)

    # Save artifacts
    model_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving trained pipeline to: %s", model_out)
    joblib.dump(pipeline, model_out)

    config: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "target_col": target_col,
        "drop_cols": drop_cols,
        "feature_columns": artifacts.feature_columns,
        "numeric_cols": meta_num,
        "categorical_cols": meta_cat,
        "model": {
            "type": "LogisticRegression",
            "params": {
                "C": C,
                "solver": solver,
                "max_iter": max_iter,
                "random_state": random_state,
            },
        },
        "metrics": {
            "primary": ["pr_auc", "roc_auc"],
            "reported": list(val_metrics.keys()),
        },
    }
    config_out.write_text(json.dumps(config, indent=2))
    LOGGER.info("Saved config to: %s", config_out)

    payload = {
        "split": "val",
        "target_col": target_col,
        "metrics": val_metrics,
    }
    metrics_out.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Saved validation metrics to: %s", metrics_out)

    LOGGER.info("Done. Val PR-AUC=%.4f | ROC-AUC=%.4f", val_metrics["pr_auc"], val_metrics["roc_auc"])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train churn model (baseline Logistic Regression).")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with processed splits.")
    parser.add_argument("--metadata", type=str, default=None, help="Optional path to metadata.json.")
    parser.add_argument("--target", type=str, default="churn", help="Target column name (if no metadata).")

    parser.add_argument("--model_out", type=str, default="models/model.joblib", help="Path to save trained pipeline.")
    parser.add_argument("--config_out", type=str, default="models/config.json", help="Path to save training config.")
    parser.add_argument("--metrics_out", type=str, default="reports/metrics/val_metrics.json", help="Path to save val metrics.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for Logistic Regression.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for Logistic Regression.")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Solver for Logistic Regression.")
    parser.add_argument(
        "--drop_cols",
        nargs="*",
        default=[],
        help="Optional columns to drop from features (e.g., customerid).",
    )

    args = parser.parse_args()

    train_and_evaluate(
        data_dir=Path(args.data_dir),
        target_col=args.target,
        metadata_path=Path(args.metadata) if args.metadata else None,
        model_out=Path(args.model_out),
        config_out=Path(args.config_out),
        metrics_out=Path(args.metrics_out),
        random_state=args.seed,
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        drop_cols=args.drop_cols,
    )


if __name__ == "__main__":
    main()

