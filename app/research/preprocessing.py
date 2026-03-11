"""
Sklearn preprocessing pipeline for ML features.

Handles:
- Missing value imputation (median for numerics, 0 for external/optional columns)
- Outlier clipping (winsorize at 1st/99th percentile)
- Scaling (StandardScaler for linear models, passthrough for tree models)
- Column selection to enforce exact feature ordering

Designed so the *fitted* pipeline can be serialized alongside the model
artifact, ensuring inference uses identical transforms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.research.feature_eng import (
    EXTERNAL_FEATURE_COLS,
    get_ml_feature_names,
)


class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] percentile bounds.

    Fit computes bounds from training data; transform clips to those bounds.
    Prevents extreme outliers from dominating linear models.
    """

    def __init__(self, lower: float = 1.0, upper: float = 99.0) -> None:
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: None = None) -> "WinsorizeTransformer":
        self.lower_bounds_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.lower_bounds_ is not None
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def build_preprocessing_pipeline(
    feature_names: list[str],
    scale: bool = True,
    winsorize: bool = True,
) -> Pipeline:
    """Build a scikit-learn pipeline for the given feature columns.

    Parameters
    ----------
    feature_names : ordered list of column names the model expects
    scale : whether to apply StandardScaler (True for linear, False for trees)
    winsorize : whether to clip outliers at 1st/99th percentile
    """
    steps: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if winsorize:
        steps.append(("winsorize", WinsorizeTransformer(lower=1.0, upper=99.0)))
    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def prepare_X(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    include_external: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Extract the feature matrix from a DataFrame.

    Returns (X, column_names) with consistent ordering.
    NaN / inf are left as-is for the pipeline to handle.
    """
    if feature_names is None:
        feature_names = get_ml_feature_names(include_external=include_external)

    for col in feature_names:
        if col not in df.columns:
            df = df.copy()
            df[col] = np.nan

    X = df[feature_names].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    return X, feature_names
