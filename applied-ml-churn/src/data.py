# src/data.py
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

LEAKAGE_PATTERNS = [
    # direct churn indicators
    r"\bchurn_reason\b",
    r"\breason\b.*\bchurn\b",
    r"\bchurn_date\b",
    r"\btermination_date\b",
    r"\bcancel_date\b",
    r"\bdisconnect_date\b",
    r"\bend_date\b",
    r"\bclosed_date\b",
    r"\baccount_close\b",
    r"\bstatus\b",  # sometimes "status" encodes churn directly; we handle carefully below
    r"\bis_churn\b",
    r"\bchurn_flag\b",
    r"\bleft\b",
    r"\bdeactivate\b",
    r"\bretention\b.*\boffer\b",  # post-churn interventions can leak
    r"\bwinback\b",
    r"\boutcome\b",
]

# Optional: allowlist non-leaky "status" columns if you have them later
SAFE_STATUS_ALLOWLIST = {
    # Example: "marital_status" is not churn leakage (but adjust for your dataset)
    "marital_status",
    "employment_status",
}

# ----------------------------
# Helpers
# ----------------------------
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase + underscores:
    - strip whitespace
    - lowercase
    - replace non-alphanumeric with underscore
    - collapse multiple underscores
    - strip leading/trailing underscores
    """
    def _clean(col: str) -> str:
        col = col.strip().lower()
        col = re.sub(r"[^a-z0-9]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        return col

    new_cols = [_clean(c) for c in df.columns]

    # Ensure uniqueness if collisions happen after standardization
    seen = {}
    unique_cols = []
    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            unique_cols.append(c)
        else:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")

    df = df.copy()
    df.columns = unique_cols
    return df


def convert_target_to_binary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert churn-like target column to 0/1.
    Accepts Yes/No (case-insensitive). Leaves 0/1 as-is if already numeric.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in columns: {list(df.columns)}")

    df = df.copy()
    s = df[target_col]

    # If numeric already (0/1), keep
    if pd.api.types.is_numeric_dtype(s):
        # Ensure it is 0/1
        df[target_col] = s.astype(int)
        return df

    # Otherwise map strings
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    )

    # If any unmapped values exist, fail fast with a helpful message
    if mapped.isna().any():
        bad_vals = sorted(set(s[mapped.isna()].astype(str).unique()))
        raise ValueError(
            f"Unrecognized target values in '{target_col}': {bad_vals}. "
            "Expected Yes/No (or 0/1)."
        )

    df[target_col] = mapped.astype(int)
    return df


def infer_numeric_and_categorical(
    df: pd.DataFrame, target_col: str, numeric_coerce_threshold: float = 0.70
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Identify numeric vs categorical columns.
    Also attempts to coerce object columns that are mostly numeric into numeric dtype.

    Returns:
      numeric_cols, categorical_cols, df (possibly with coerced numeric columns)
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        # For object columns, try coercing to numeric if most values are numeric-like
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            # Treat empty strings as missing first for the test
            s_str = s.astype(str).str.strip()
            s_str = s_str.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})

            coerced = pd.to_numeric(s_str, errors="coerce")

            non_missing = s_str.notna().sum()
            if non_missing == 0:
                # All missing -> treat as categorical
                categorical_cols.append(col)
                continue

            success_rate = coerced.notna().sum() / non_missing

            if success_rate >= numeric_coerce_threshold:
                df[col] = coerced
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            # Everything else (dates, etc.) treat as categorical for now
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, df


def simple_missing_value_handling(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Simple missing handling:
    - numeric: fill with median
    - categorical: fill with 'missing'
    """
    df = df.copy()

    for col in numeric_cols:
        if col in df.columns:
            median = df[col].median(skipna=True)
            df[col] = df[col].fillna(median)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("missing")

    return df

def find_leakage_columns(columns: list[str], target_col: str) -> list[str]:
    """
    Return columns that look like leakage based on name patterns.
    This is heuristic (name-based) but catches common mistakes.
    """
    leakage = []
    for c in columns:
        if c == target_col:
            continue

        # Special handling for generic terms like "status"
        if c in SAFE_STATUS_ALLOWLIST:
            continue

        for pat in LEAKAGE_PATTERNS:
            if re.search(pat, c):
                # Avoid dropping benign columns that happen to contain "end_date" etc. if you want:
                leakage.append(c)
                break

    return sorted(set(leakage))

def assert_no_leakage(
    df: pd.DataFrame,
    target_col: str,
    mode: str = "drop",  # "drop" or "raise"
) -> pd.DataFrame:
    """
    Simple leakage check:
    - Identify obvious leak columns by name
    - Either drop them or raise an error

    NOTE: This is a heuristic check. It does not guarantee the absence of leakage.
    """
    cols = list(df.columns)
    leakage_cols = find_leakage_columns(cols, target_col=target_col)

    if not leakage_cols:
        LOGGER.info("Leakage check: no suspicious columns found.")
        return df

    msg = f"Leakage check: found suspicious columns {leakage_cols}"
    if mode == "raise":
        raise ValueError(msg + " (set mode='drop' to auto-remove)")
    elif mode == "drop":
        LOGGER.warning(msg + " — dropping them.")
        return df.drop(columns=leakage_cols)
    else:
        raise ValueError("mode must be 'drop' or 'raise'")


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path
    metadata: Path


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by target:
    70/15/15 default via two-step splitting.
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    y = df[target_col]

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )

    # Split the remaining temp into val and test (proportional split)
    # temp = val + test, so val fraction within temp = val_size / (val_size + test_size)
    val_frac_of_temp = val_size / (val_size + test_size)

    y_temp = temp_df[target_col]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_frac_of_temp),
        stratify=y_temp,
        random_state=random_state,
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    file_format: str = "parquet",
) -> SplitPaths:
    """
    Save splits into out_dir as parquet (preferred) or csv.
    Falls back to csv if parquet is requested but not available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = file_format.strip().lower()
    if fmt not in {"parquet", "csv"}:
        raise ValueError("file_format must be 'parquet' or 'csv'")

    # Try parquet, fallback to csv if engine missing
    def _to_parquet_safe(df: pd.DataFrame, path: Path) -> bool:
        try:
            df.to_parquet(path, index=False)
            return True
        except Exception as e:
            LOGGER.warning("Parquet save failed (%s). Falling back to CSV.", e)
            return False

    if fmt == "parquet":
        train_path = out_dir / "train.parquet"
        val_path = out_dir / "val.parquet"
        test_path = out_dir / "test.parquet"

        ok = (
            _to_parquet_safe(train_df, train_path)
            and _to_parquet_safe(val_df, val_path)
            and _to_parquet_safe(test_df, test_path)
        )
        if ok:
            return SplitPaths(
                train=train_path,
                val=val_path,
                test=test_path,
                metadata=out_dir / "metadata.json",
            )

        # fallback to csv if parquet failed
        fmt = "csv"

    # CSV save
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return SplitPaths(
        train=train_path,
        val=val_path,
        test=test_path,
        metadata=out_dir / "metadata.json",
    )


# ----------------------------
# Pipeline
# ----------------------------
def build_processed_splits(
    input_csv: Path,
    out_dir: Path,
    target_col_raw: str = "Churn",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    file_format: str = "parquet",
) -> SplitPaths:
    """
    End-to-end:
    - load csv
    - standardize columns
    - convert target to 0/1
    - infer numeric vs categorical (+ coerce numeric-like)
    - missing handling
    - stratified split
    - save splits + metadata
    """
    LOGGER.info("Loading CSV from: %s", input_csv)
    df = pd.read_csv(input_csv)

    df = standardize_column_names(df)

    # After standardization, churn becomes "churn" (from "Churn")
    target_col = re.sub(r"[^a-z0-9]+", "_", target_col_raw.strip().lower()).strip("_")
    if target_col not in df.columns:
        # Try common alternative: if raw was already standardized differently
        raise ValueError(
            f"Target column '{target_col}' not found after standardization. "
            f"Available columns: {list(df.columns)}"
        )
    
    df = assert_no_leakage(df, target_col=target_col, mode="drop")

    df = convert_target_to_binary(df, target_col=target_col)

    numeric_cols, categorical_cols, df = infer_numeric_and_categorical(df, target_col=target_col)

    # Drop rows with missing target (should be rare; but keep pipeline safe)
    df = df.dropna(subset=[target_col])

    df = simple_missing_value_handling(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    LOGGER.info("Columns after leakage drop: %d columns", len(df.columns))

    train_df, val_df, test_df = stratified_train_val_test_split(
        df,
        target_col=target_col,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    paths = save_splits(train_df, val_df, test_df, out_dir=out_dir, file_format=file_format)

    # Save metadata (column lists + split sizes)
    metadata = {
        "input_csv": str(input_csv),
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "split": {"train": train_size, "val": val_size, "test": test_size},
        "random_state": random_state,
        "saved_files": {"train": str(paths.train), "val": str(paths.val), "test": str(paths.test)},
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Saved metadata to: %s", paths.metadata)

    return paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Prepare Telco churn dataset splits.")
    parser.add_argument("--input", type=str, default="data/raw/telco_churn.csv", help="Path to raw CSV.")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory for splits.")
    parser.add_argument("--target", type=str, default="Churn", help="Target column name (raw CSV).")
    parser.add_argument("--train", type=float, default=0.70, help="Train split fraction.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--test", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"], help="Save format.")
    args = parser.parse_args()

    build_processed_splits(
        input_csv=Path(args.input),
        out_dir=Path(args.out),
        target_col_raw=args.target,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
        random_state=args.seed,
        file_format=args.format,
    )


if __name__ == "__main__":
    main()
