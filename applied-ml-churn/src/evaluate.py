# src/evaluate.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
# Threshold sweep + selection
# ----------------------------
def build_threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      threshold, precision, recall, f1, positive_rate
    where positive_rate = fraction predicted positive at that threshold.
    """
    rows = []
    n = len(y_true)
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "positive_rate": float(y_pred.sum() / n),
            }
        )
    return pd.DataFrame(rows)

def select_operational_threshold(
    table: pd.DataFrame,
    mode: str,
    *,
    contact_pct: float = 0.15,
    min_recall: float = 0.70,
) -> tuple[float, str]:
    """
    Choose a threshold based on an operational goal.

    mode:
      - "top_pct": choose threshold whose positive_rate is closest to contact_pct
      - "min_recall": choose threshold with recall >= min_recall and max precision (ties: higher threshold)
      - "none": no selection (returns 0.5 with a generic message)
    """
    mode = mode.lower().strip()

    if mode == "none":
        return 0.5, "No operational selection requested; using default threshold 0.50."

    if mode == "top_pct":
        # closest positive rate to contact_pct; tie-breaker: higher threshold (usually improves precision)
        tmp = table.copy()
        tmp["diff"] = (tmp["positive_rate"] - contact_pct).abs()
        best = tmp.sort_values(["diff", "threshold"], ascending=[True, False]).iloc[0]
        t = float(best["threshold"])
        pr = float(best["positive_rate"])
        reason = (
            f"Outreach capacity is {contact_pct:.0%}; chose threshold {t:.2f} "
            f"which selects ~{pr:.0%} of customers (closest match in the sweep)."
        )
        return t, reason

    if mode == "min_recall":
        feasible = table[table["recall"] >= min_recall].copy()
        if len(feasible) > 0:
            best = feasible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
            t = float(best["threshold"])
            p = float(best["precision"])
            r = float(best["recall"])
            reason = (
                f"Goal is recall ≥ {min_recall:.2f}; chose threshold {t:.2f} "
                f"achieving recall={r:.2f} with the highest precision={p:.2f} among feasible thresholds."
            )
            return t, reason
        # fallback if impossible
        best = table.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]
        t = float(best["threshold"])
        r = float(best["recall"])
        reason = (
            f"Could not reach recall ≥ {min_recall:.2f} in the tested thresholds; "
            f"chose threshold {t:.2f} with the highest observed recall={r:.2f}."
        )
        return t, reason

    raise ValueError("mode must be one of: none, top_pct, min_recall")


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
    threshold: float = 0.5,  # still used for the "at 0.5" outputs
    figures_dir: Path = Path("reports/figures"),
    metrics_out: Optional[Path] = None,
    threshold_table_out: Optional[Path] = None,
    op_mode: str = "top_pct",
    contact_pct: float = 0.15,
    min_recall: float = 0.70,
) -> dict:
    """
    Computes:
      - ROC-AUC, PR-AUC (using predicted probabilities)
      - Confusion matrix at threshold=0.5 (and P/R/F1 at 0.5)
      - Precision/Recall table for thresholds 0.05..0.95 (step 0.05)
      - Operational threshold selection based on op_mode

    Saves plots in reports/figures:
      - ROC curve PNG
      - PR curve PNG
      - Confusion matrix PNG (for 0.5 and operational threshold)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    LOGGER.info("Loading model pipeline: %s", model_path)
    pipeline = joblib.load(model_path)

    split_path = _resolve_split_path(data_dir, split=split, metadata_path=metadata_path)
    LOGGER.info("Loading data split '%s': %s", split, split_path)
    df = _read_table(split_path)

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

    # AUC metrics
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    # Metrics at 0.5 (as requested)
    y_pred_05 = (y_score >= threshold).astype(int)
    precision_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    recall_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    cm_05 = confusion_matrix(y_true, y_pred_05, labels=[0, 1]).tolist()

    # Threshold sweep table
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)  # 0.05..0.95
    thr_table = build_threshold_table(y_true, y_score, thresholds)

    # "Small table" for readability (threshold, precision, recall)
    small_table = thr_table[["threshold", "precision", "recall"]].copy()

    # Pick operational threshold based on a goal
    op_threshold, op_reason = select_operational_threshold(
        thr_table,
        mode=op_mode,
        contact_pct=contact_pct,
        min_recall=min_recall,
    )
    y_pred_op = (y_score >= op_threshold).astype(int)
    precision_op = float(precision_score(y_true, y_pred_op, zero_division=0))
    recall_op = float(recall_score(y_true, y_pred_op, zero_division=0))
    f1_op = float(f1_score(y_true, y_pred_op, zero_division=0))
    pos_rate_op = float(y_pred_op.mean())
    cm_op = confusion_matrix(y_true, y_pred_op, labels=[0, 1]).tolist()

    # Save plots
    figures_dir.mkdir(parents=True, exist_ok=True)

    roc_path = figures_dir / f"roc_curve_{split}.png"
    pr_path = figures_dir / f"pr_curve_{split}.png"
    cm05_path = figures_dir / f"confusion_matrix_{split}_thr0_50.png"
    cmop_path = figures_dir / f"confusion_matrix_{split}_thr{op_threshold:.2f}.png"

    _save_roc_curve(y_true, y_score, roc_path, title=f"ROC Curve ({split})")
    _save_pr_curve(y_true, y_score, pr_path, title=f"Precision–Recall Curve ({split})")
    _save_confusion_matrix(y_true, y_pred_05, cm05_path, title=f"Confusion Matrix @ 0.50 ({split})")
    _save_confusion_matrix(y_true, y_pred_op, cmop_path, title=f"Confusion Matrix @ {op_threshold:.2f} ({split})")

    LOGGER.info("Saved ROC curve: %s", roc_path)
    LOGGER.info("Saved PR curve:  %s", pr_path)
    LOGGER.info("Saved CM (0.5):  %s", cm05_path)
    LOGGER.info("Saved CM (op):   %s", cmop_path)

    # Optional: save threshold table CSV
    if threshold_table_out is not None:
        threshold_table_out.parent.mkdir(parents=True, exist_ok=True)
        thr_table.to_csv(threshold_table_out, index=False)
        LOGGER.info("Saved threshold table CSV: %s", threshold_table_out)

    results = {
        "split": split,
        "target_col": target_col,
        "auc": {"roc_auc": roc_auc, "pr_auc": pr_auc},
        "threshold_0_5": {
            "threshold": float(threshold),
            "precision": precision_05,
            "recall": recall_05,
            "f1": f1_05,
            "confusion_matrix": cm_05,
        },
        "threshold_sweep": {
            "thresholds": thresholds.tolist(),
            "table": thr_table.to_dict(orient="records"),
            "small_table": small_table.to_dict(orient="records"),
        },
        "operational_threshold": {
            "mode": op_mode,
            "threshold": float(op_threshold),
            "reason": op_reason,
            "positive_rate": pos_rate_op,
            "precision": precision_op,
            "recall": recall_op,
            "f1": f1_op,
            "confusion_matrix": cm_op,
        },
        "plots": {
            "roc_curve_png": str(roc_path),
            "pr_curve_png": str(pr_path),
            "confusion_matrix_0_5_png": str(cm05_path),
            "confusion_matrix_operational_png": str(cmop_path),
        },
    }

    # Print the “small table” (nice for CLI usage)
    LOGGER.info("Threshold sweep table (threshold, precision, recall):\n%s", small_table.to_string(index=False))

    # Optional metrics JSON output
    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(results, indent=2))
        LOGGER.info("Saved metrics JSON: %s", metrics_out)

    LOGGER.info(
        "Done. %s ROC-AUC=%.4f | PR-AUC=%.4f | OpThr=%.2f (pos_rate=%.0f%%)",
        split,
        roc_auc,
        pr_auc,
        op_threshold,
        100 * pos_rate_op,
    )
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained churn model and save plots + threshold sweep.")
    parser.add_argument("--model", type=str, default="models/model.joblib", help="Path to trained pipeline joblib.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with processed splits.")
    parser.add_argument("--metadata", type=str, default=None, help="Optional metadata.json path.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Which split to evaluate.")
    parser.add_argument("--target", type=str, default="churn", help="Target column name (if no metadata).")

    # 0.5 threshold outputs
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the required 0.5-based metrics/CM.")

    # Operational threshold selection
    parser.add_argument(
        "--op_mode",
        type=str,
        default="top_pct",
        choices=["none", "top_pct", "min_recall"],
        help="How to choose an operational threshold.",
    )
    parser.add_argument("--contact_pct", type=float, default=0.15, help="For op_mode=top_pct: desired fraction selected.")
    parser.add_argument("--min_recall", type=float, default=0.70, help="For op_mode=min_recall: recall constraint.")

    # Outputs
    parser.add_argument("--figures_dir", type=str, default="reports/figures", help="Directory to save PNG figures.")
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="reports/metrics/val_eval.json",
        help="Path to save evaluation JSON.",
    )
    parser.add_argument(
        "--threshold_table_out",
        type=str,
        default="reports/metrics/threshold_table_val.csv",
        help="Path to save threshold sweep CSV.",
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
        threshold_table_out=Path(args.threshold_table_out) if args.threshold_table_out else None,
        op_mode=args.op_mode,
        contact_pct=float(args.contact_pct),
        min_recall=float(args.min_recall),
    )


if __name__ == "__main__":
    main()
