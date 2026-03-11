"""
Dataset construction and temporal splitting.

Builds ML-ready DataFrames from the SQLite features table and provides
walk-forward cross-validation splits that respect time ordering (no leakage).

Walk-forward scheme
-------------------
Given N rows sorted by time, the splitter produces K folds:

  Fold 1: train=[0, T₁)         val=[T₁, T₁+W)
  Fold 2: train=[0, T₂)         val=[T₂, T₂+W)
  ...
  Fold K: train=[0, T_K)        val=[T_K, end)

Where Tₖ = initial_train_size + k * step_size, W = val_size.
Training sets grow (expanding window) to maximise data usage while
never leaking future information into the past.

A final held-out test set (last ``test_frac`` of the data) is carved
out *before* walk-forward — it is never seen during model selection.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from app.research.feature_eng import engineer_features
from app.research.targets import add_all_targets


# ── Dataset loading ───────────────────────────────────────────────────────


def load_features_df(db_path: str) -> pd.DataFrame:
    """Load the features table from SQLite into a DataFrame."""
    db_file = db_path.replace("sqlite:///", "") if db_path.startswith("sqlite:///") else db_path
    try:
        conn = sqlite3.connect(db_file)
        raw = pd.read_sql("SELECT * FROM features ORDER BY id", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return raw

    records = []
    for _, row in raw.iterrows():
        data = json.loads(row["data_json"])
        data["_db_id"] = row["id"]
        data["_market_id"] = row["market_id"]
        data["_token_id"] = row["token_id"]
        records.append(data)

    return pd.DataFrame(records)


def build_dataset(
    db_path: str,
    horizon: int = 6,
    min_edge: float = 0.02,
    fee: float = 0.02,
    min_rows: int = 50,
) -> pd.DataFrame | None:
    """Full pipeline: load → engineer features → add targets → clean.

    Returns None if insufficient data.
    """
    df = load_features_df(db_path)
    if len(df) < min_rows:
        return None

    df = engineer_features(df)
    df = add_all_targets(df, horizon=horizon, min_edge=min_edge, fee=fee)

    df = df.dropna(subset=["target_direction"])
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def generate_synthetic_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-looking synthetic data for demo / testing.

    The data has mild autocorrelation in prices and features that loosely
    resemble real Polymarket orderbook dynamics.  NOT suitable for
    evaluating real profitability — only for pipeline validation.
    """
    rng = np.random.default_rng(seed)

    # Volatility calibrated so 6-step moves can exceed 1-2 cents.
    # Step std of 0.008 produces ≈2-cent moves in 6 steps ~40% of the time.
    step_std = 0.008
    mid = np.cumsum(rng.normal(0, step_std, n)) + 0.50
    mid = np.clip(mid, 0.05, 0.95)
    spread = rng.uniform(0.01, 0.08, n)
    imbalance = np.clip(np.cumsum(rng.normal(0, 0.05, n)), -0.95, 0.95)

    df = pd.DataFrame({
        "mid_price": mid,
        "best_bid": mid - spread / 2,
        "best_ask": mid + spread / 2,
        "spread": spread,
        "microprice": mid + imbalance * spread * 0.1,
        "orderbook_imbalance": imbalance,
        "bid_depth_5c": rng.exponential(25, n),
        "ask_depth_5c": rng.exponential(25, n),
        "recent_trade_flow": np.cumsum(rng.normal(0, 1.0, n)) * 0.1,
        "volatility_1m": np.abs(rng.normal(0.015, 0.008, n)),
        "momentum_1m": np.diff(mid, prepend=mid[0]),
        "momentum_5m": pd.Series(mid).diff(5).fillna(0).values,
        "momentum_15m": pd.Series(mid).diff(15).fillna(0).values,
        "trade_count_1m": rng.poisson(4, n),
        "last_trade_price": mid + rng.normal(0, 0.005, n),
        "seconds_since_last_update": rng.uniform(0.5, 5.0, n),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5s", tz="UTC"),
    })

    return df


# ── Temporal splitting ────────────────────────────────────────────────────


@dataclass
class SplitIndices:
    """Integer index ranges for one fold."""
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def train_test_split_temporal(
    n: int,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train+val (front) and test (back).

    The test set is never used during walk-forward model selection.
    """
    cutoff = int(n * (1 - test_frac))
    return np.arange(cutoff), np.arange(cutoff, n)


def walk_forward_splits(
    n_trainval: int,
    n_folds: int = 5,
    min_train_frac: float = 0.4,
    val_frac: float = 0.15,
) -> list[SplitIndices]:
    """Generate expanding-window walk-forward split indices.

    Parameters
    ----------
    n_trainval : total rows available (train+val, excluding test)
    n_folds    : number of folds
    min_train_frac : minimum fraction of n_trainval used as the first train set
    val_frac   : fraction of n_trainval used as each validation window
    """
    val_size = max(int(n_trainval * val_frac), 10)
    min_train = max(int(n_trainval * min_train_frac), 20)
    step = max((n_trainval - min_train - val_size) // max(n_folds - 1, 1), 1)

    folds: list[SplitIndices] = []
    for k in range(n_folds):
        train_end = min_train + k * step
        val_start = train_end
        val_end = min(val_start + val_size, n_trainval)
        if val_start >= n_trainval:
            break
        folds.append(SplitIndices(
            train_start=0,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))

    return folds
