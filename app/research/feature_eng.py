"""
Extended feature engineering for the ML pipeline.

Takes a raw DataFrame of MarketFeatures snapshots and produces additional
derived columns suited for tabular ML.  All transformations are pure
functions on DataFrames — no statefulness, no look-ahead.

Feature categories:
1. Raw orderbook signals (spread, microprice, imbalance, depth)
2. Momentum / trend (1m, 5m, 15m price changes)
3. Volatility (1m price std)
4. Trade flow (net signed volume, trade count)
5. Relative / ratio features (spread_pct, depth_ratio, imb×mom interaction)
6. Time context (hour_of_day, minutes_to_resolution if available)
7. External enrichment columns (populated by optional providers; NaN if absent)

All features are computed only from *past and present* data — no leakage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Core feature columns produced by MarketFeatures snapshots ──────────

RAW_FEATURE_COLS = [
    "spread",
    "microprice",
    "orderbook_imbalance",
    "bid_depth_5c",
    "ask_depth_5c",
    "recent_trade_flow",
    "volatility_1m",
    "momentum_1m",
    "momentum_5m",
    "momentum_15m",
    "trade_count_1m",
    "mid_price",
    "best_bid",
    "best_ask",
]

# Columns added by this module
DERIVED_FEATURE_COLS = [
    "spread_pct",
    "depth_ratio",
    "depth_total",
    "imbalance_x_momentum",
    "flow_per_trade",
    "bid_ask_range",
    "micro_minus_mid",
    "momentum_accel",
    "vol_adj_momentum",
    "hour_of_day",
    "minute_of_hour",
]

# Optional columns from external providers — always present but may be NaN
EXTERNAL_FEATURE_COLS = [
    "sentiment_score",
    "news_intensity",
    "minutes_to_resolution",
]

ALL_ML_FEATURES = RAW_FEATURE_COLS + DERIVED_FEATURE_COLS + EXTERNAL_FEATURE_COLS


def engineer_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add all derived features to *df* in-place-safe (returns a copy).

    Assumes *df* already has the RAW_FEATURE_COLS (missing ones filled with 0).
    """
    df = df.copy()

    _ensure_columns(df, RAW_FEATURE_COLS + EXTERNAL_FEATURE_COLS)

    mid = df["mid_price"].replace(0, np.nan)

    df["spread_pct"] = df["spread"] / mid
    df["depth_ratio"] = np.where(
        df["ask_depth_5c"] > 0,
        df["bid_depth_5c"] / df["ask_depth_5c"],
        1.0,
    )
    df["depth_total"] = df["bid_depth_5c"] + df["ask_depth_5c"]
    df["imbalance_x_momentum"] = df["orderbook_imbalance"] * df["momentum_1m"]
    df["flow_per_trade"] = np.where(
        df["trade_count_1m"] > 0,
        df["recent_trade_flow"] / df["trade_count_1m"],
        0.0,
    )
    df["bid_ask_range"] = df["best_ask"] - df["best_bid"]
    df["micro_minus_mid"] = df["microprice"] - mid
    df["momentum_accel"] = df["momentum_1m"] - df["momentum_5m"]
    df["vol_adj_momentum"] = np.where(
        df["volatility_1m"] > 0,
        df["momentum_1m"] / df["volatility_1m"],
        0.0,
    )

    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        df["hour_of_day"] = ts.dt.hour
        df["minute_of_hour"] = ts.dt.minute
    else:
        df["hour_of_day"] = 0
        df["minute_of_hour"] = 0

    return df


def get_ml_feature_names(include_external: bool = False) -> list[str]:
    """Return the ordered list of feature column names used by the model."""
    base = RAW_FEATURE_COLS + DERIVED_FEATURE_COLS
    if include_external:
        base = base + EXTERNAL_FEATURE_COLS
    return base


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Add missing columns as 0 / NaN so downstream code never KeyErrors."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0 if c not in EXTERNAL_FEATURE_COLS else np.nan
