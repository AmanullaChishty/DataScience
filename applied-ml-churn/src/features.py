# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass(frozen=True)
class FeatureArtifacts:
    """
    Bundles the main objects you typically want to keep around.
    - preprocessor: transforms raw X -> model-ready numeric matrix
    - feature_columns: list of original feature column names used
    - pipeline: sklearn Pipeline(preprocessor -> model) if model provided, else preprocessor-only pipeline
    """
    preprocessor: ColumnTransformer
    feature_columns: List[str]
    pipeline: Pipeline


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    *,
    onehot_min_frequency: Optional[Union[int, float]] = None,
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
      - numeric: median impute + standard scaler
      - categorical: most_frequent impute + one-hot encode

    Args:
      numeric_cols: numeric feature column names
      categorical_cols: categorical feature column names
      onehot_min_frequency:
        Optional; if set, will group infrequent categories (sklearn >= 1.1).
        - int: categories with count < int become "infrequent"
        - float: categories with freq < float become "infrequent"
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # handle_unknown='ignore' prevents crashes when val/test has unseen categories
    # sparse_output is newer; we set both safely via try/except below
    ohe_kwargs = dict(handle_unknown="ignore")
    if onehot_min_frequency is not None:
        ohe_kwargs["min_frequency"] = onehot_min_frequency

    try:
        ohe = OneHotEncoder(**ohe_kwargs, sparse_output=False)
    except TypeError:
        # For older sklearn versions where sparse_output isn't available
        ohe = OneHotEncoder(**ohe_kwargs, sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def build_model_pipeline(
    preprocessor: ColumnTransformer,
    model,
) -> Pipeline:
    """
    Combine preprocessor + model into a single sklearn Pipeline that can be fit/predict.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Convenience helper to determine feature columns.
    If numeric_cols/categorical_cols are provided, they are validated against df columns.

    Returns:
      feature_columns, numeric_cols, categorical_cols
    """
    drop_cols = drop_cols or []
    base_features = [c for c in df.columns if c not in [target_col, *drop_cols]]

    if numeric_cols is None or categorical_cols is None:
        # Infer types from dataframe dtypes (simple heuristic)
        inferred_numeric = []
        inferred_categorical = []
        for c in base_features:
            if pd.api.types.is_numeric_dtype(df[c]):
                inferred_numeric.append(c)
            else:
                inferred_categorical.append(c)
        numeric_cols = numeric_cols or inferred_numeric
        categorical_cols = categorical_cols or inferred_categorical

    numeric_cols = [c for c in numeric_cols if c in base_features]
    categorical_cols = [c for c in categorical_cols if c in base_features]

    # final feature set (preserve a stable order)
    feature_columns = [*numeric_cols, *categorical_cols]
    return feature_columns, numeric_cols, categorical_cols

def build_feature_artifacts(
    df: pd.DataFrame,
    target_col: str,
    *,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
    model=None,
    onehot_min_frequency: Optional[Union[int, float]] = None,
) -> FeatureArtifacts:
    """
    One-stop builder:
      - determines feature columns (optional)
      - builds preprocessor
      - builds pipeline ready to attach model

    If model is None, returns a Pipeline containing only the preprocessor step
    (useful for transforming data without fitting a model yet).
    """
    feature_columns, num_cols, cat_cols = get_feature_columns(
        df=df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
    )

    preprocessor = build_preprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        onehot_min_frequency=onehot_min_frequency,
    )

    if model is None:
        pipeline = Pipeline(steps=[("preprocess", preprocessor)])
    else:
        pipeline = build_model_pipeline(preprocessor=preprocessor, model=model)

    return FeatureArtifacts(
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        pipeline=pipeline,
    )
