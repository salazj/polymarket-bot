"""
Model evaluation: classification metrics, calibration analysis, and
profit-aware simulation.

The profit simulation makes these explicit assumptions:
- Entry at best_ask for BUY signals; exit at future mid_price.
- Fixed round-trip fee (default 2 cents).
- Fixed order size (1 unit).
- No partial fills, no slippage beyond the ask price.
- Only trades where model confidence > threshold are taken.
These are OPTIMISTIC assumptions; real trading will perform worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss,
)


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics for one model on one dataset."""
    accuracy: float = 0.0
    log_loss_val: float = 0.0
    roc_auc: float | None = None
    brier_score: float = 0.0
    precision_at_60: float = 0.0   # precision when threshold=0.6
    recall_at_60: float = 0.0
    confusion: list[list[int]] = field(default_factory=list)
    classification_report_text: str = ""
    calibration_bins: list[float] = field(default_factory=list)
    calibration_true: list[float] = field(default_factory=list)
    calibration_pred: list[float] = field(default_factory=list)
    profit_simulation: dict[str, Any] = field(default_factory=dict)


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationResult:
    """Compute all classification metrics."""
    result = EvaluationResult()

    result.accuracy = float(accuracy_score(y_true, y_pred))
    try:
        result.log_loss_val = float(log_loss(y_true, y_proba, labels=[0, 1]))
    except ValueError:
        result.log_loss_val = 999.0
    result.brier_score = float(brier_score_loss(y_true, y_proba[:, 1]))

    try:
        result.roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
    except ValueError:
        result.roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    result.confusion = cm
    result.classification_report_text = classification_report(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    proba_pos = y_proba[:, 1]
    pred_60 = (proba_pos >= 0.6).astype(int)
    mask_60 = pred_60 == 1
    if mask_60.any():
        result.precision_at_60 = float(np.mean(y_true[mask_60] == 1))
        result.recall_at_60 = float(np.sum((pred_60 == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1))
    else:
        result.precision_at_60 = 0.0
        result.recall_at_60 = 0.0

    n_bins = min(10, max(2, len(y_true) // 20))
    try:
        prob_true, prob_pred = calibration_curve(y_true, proba_pos, n_bins=n_bins)
        result.calibration_true = prob_true.tolist()
        result.calibration_pred = prob_pred.tolist()
    except ValueError:
        pass

    return result


def simulate_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    threshold: float = 0.6,
    fee: float = 0.02,
    order_size: float = 1.0,
) -> dict[str, Any]:
    """Simulate trading profit assuming we buy when model says "up" with
    confidence >= threshold.

    Parameters
    ----------
    y_true       : actual labels (1=price went up)
    y_proba      : model predicted probabilities (n, 2)
    entry_prices : price at which we'd enter (best_ask at signal time)
    exit_prices  : price at which we'd exit (future mid_price)
    threshold    : minimum confidence to take the trade
    fee          : round-trip fee per unit traded
    order_size   : units per trade

    Returns dict with total_pnl, n_trades, win_rate, avg_pnl_per_trade,
    max_drawdown, sharpe_like.
    """
    proba_pos = y_proba[:, 1]
    trade_mask = proba_pos >= threshold

    n_trades = int(trade_mask.sum())
    if n_trades == 0:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "threshold": threshold,
            "fee": fee,
        }

    entries = entry_prices[trade_mask]
    exits = exit_prices[trade_mask]
    pnls = (exits - entries - fee) * order_size

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    return {
        "n_trades": n_trades,
        "total_pnl": float(np.sum(pnls)),
        "win_rate": float(np.mean(pnls > 0)),
        "avg_pnl_per_trade": mean_pnl,
        "max_drawdown": max_dd,
        "sharpe_like": sharpe,
        "threshold": threshold,
        "fee": fee,
    }


def walk_forward_evaluate(
    results_per_fold: list[EvaluationResult],
) -> dict[str, Any]:
    """Aggregate walk-forward fold results into a summary."""
    if not results_per_fold:
        return {}

    accs = [r.accuracy for r in results_per_fold]
    losses = [r.log_loss_val for r in results_per_fold]
    aucs = [r.roc_auc for r in results_per_fold if r.roc_auc is not None]

    return {
        "n_folds": len(results_per_fold),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_log_loss": float(np.mean(losses)),
        "std_log_loss": float(np.std(losses)),
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "std_roc_auc": float(np.std(aucs)) if aucs else None,
    }
