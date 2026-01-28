# src/evaluate.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,  # PR-AUC
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,            # ROC-AUC
)

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


def _resolve_split_path(data_dir: Path, split: str, metadata_path: Optional[Path] = None) -> Path:
    """
    Returns the path to the requested split (train/val/test).
    Priority:
      1) metadata.json if present (saved by src/data.py)
      2) data_dir/{split}.parquet or data_dir/{split}.csv
    """
    split = split.lower().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"

    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        return Path(meta["saved_files"][split])

    for ext in ("parquet", "csv"):
        p = data_dir / f"{split}.{ext}"
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find split '{split}' in {data_dir}. "
        "Expected metadata.json or train/val/test parquet/csv files."
    )


def _load_target_from_metadata(data_dir: Path, metadata_path: Optional[Path]) -> Optional[str]:
    if metadata_path is None:
        metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    meta = json.loads(metadata_path.read_text())
    return meta.get("target_col")


# ----------------------------
# Plot helpers (matplotlib-only)
# ----------------------------
def _save_roc_curve(y_true, y_score, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="Model")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pr_curve(y_true, y_score, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="Model")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------
# Main evaluation routine
# ----------------------------
def evaluate(
    *,
    model_path: Path,
    data_dir: Path,
    split: str = "val",
    target_col: str = "churn",
    metadata_path: Optional[Path] = None,
    threshold: float = 0.5,
    figures_dir: Path = Path("reports/figures"),
    metrics_out: Optional[Path] = None,
) -> dict:
    """
    Compute:
      - ROC-AUC, PR-AUC (using predicted probabilities)
      - Confusion matrix at threshold (default 0.5)
      - Precision/Recall/F1 at threshold (default 0.5)
    Save:
      - ROC curve PNG
      - PR curve PNG
      - Confusion matrix PNG
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model pipeline
    LOGGER.info("Loading model pipeline: %s", model_path)
    pipeline = joblib.load(model_path)

    # Resolve data split path and load it
    split_path = _resolve_split_path(data_dir, split=split, metadata_path=metadata_path)
    LOGGER.info("Loading data split '%s': %s", split, split_path)
    df = _read_table(split_path)

    # Prefer target from metadata if present
    meta_target = _load_target_from_metadata(data_dir, metadata_path)
    if meta_target is not None:
        target_col = meta_target

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data split columns.")

    X = df.drop(columns=[target_col])
    y_true = df[target_col].astype(int).values

    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("Loaded pipeline does not support predict_proba (needed for ROC-AUC/PR-AUC).")

    y_score = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    # Metrics
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    results = {
        "split": split,
        "target_col": target_col,
        "threshold": threshold,
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_at_0_5": precision,
            "recall_at_0_5": recall,
            "f1_at_0_5": f1,
        },
        "confusion_matrix_labels": ["0 (No churn)", "1 (Churn)"],
        "confusion_matrix": cm,
    }

    # Save plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"roc_curve_{split}.png"
    pr_path = figures_dir / f"pr_curve_{split}.png"
    cm_path = figures_dir / f"confusion_matrix_{split}.png"

    _save_roc_curve(y_true, y_score, roc_path, title=f"ROC Curve ({split})")
    _save_pr_curve(y_true, y_score, pr_path, title=f"Precision–Recall Curve ({split})")
    _save_confusion_matrix(y_true, y_pred, cm_path, title=f"Confusion Matrix @ {threshold:.2f} ({split})")

    LOGGER.info("Saved ROC curve: %s", roc_path)
    LOGGER.info("Saved PR curve:  %s", pr_path)
    LOGGER.info("Saved CM plot:   %s", cm_path)

    # Optional metrics JSON output
    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(results, indent=2))
        LOGGER.info("Saved metrics JSON: %s", metrics_out)

    LOGGER.info("Done. %s ROC-AUC=%.4f | PR-AUC=%.4f", split, roc_auc, pr_auc)
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained churn model and save plots.")
    parser.add_argument("--model", type=str, default="models/model.joblib", help="Path to trained pipeline joblib.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with processed splits.")
    parser.add_argument("--metadata", type=str, default=None, help="Optional metadata.json path.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Which split to evaluate.")
    parser.add_argument("--target", type=str, default="churn", help="Target column name (if no metadata).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for confusion matrix & P/R/F1.")
    parser.add_argument("--figures_dir", type=str, default="reports/figures", help="Directory to save PNG figures.")
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Optional path to save metrics JSON (e.g., reports/metrics/val_eval.json).",
    )

    args = parser.parse_args()

    evaluate(
        model_path=Path(args.model),
        data_dir=Path(args.data_dir),
        split=args.split,
        target_col=args.target,
        metadata_path=Path(args.metadata) if args.metadata else None,
        threshold=float(args.threshold),
        figures_dir=Path(args.figures_dir),
        metrics_out=Path(args.metrics_out) if args.metrics_out else None,
    )


if __name__ == "__main__":
    main()
