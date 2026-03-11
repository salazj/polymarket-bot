"""
Report generation for model training and evaluation.

Produces a structured JSON report and a human-readable text summary
explaining feature importance, model weaknesses, overfitting risks,
and data improvement recommendations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.research.evaluation import EvaluationResult


def build_training_report(
    model_name: str,
    feature_names: list[str],
    train_size: int,
    val_size: int,
    test_size: int,
    eval_result: EvaluationResult,
    walk_forward_summary: dict[str, Any] | None = None,
    feature_importance: dict[str, float] | None = None,
    hyperparams: dict[str, Any] | None = None,
    profit_sim: dict[str, Any] | None = None,
    dataset_description: str = "",
) -> dict[str, Any]:
    """Build the full training report as a dict (serialisable to JSON)."""
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset": {
            "description": dataset_description,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "total": train_size + val_size + test_size,
            "n_features": len(feature_names),
            "features": feature_names,
        },
        "hyperparameters": hyperparams or {},
        "test_metrics": {
            "accuracy": eval_result.accuracy,
            "log_loss": eval_result.log_loss_val,
            "roc_auc": eval_result.roc_auc,
            "brier_score": eval_result.brier_score,
            "precision_at_60_pct": eval_result.precision_at_60,
            "recall_at_60_pct": eval_result.recall_at_60,
            "confusion_matrix": eval_result.confusion,
        },
        "classification_report": eval_result.classification_report_text,
        "calibration": {
            "true_probabilities": eval_result.calibration_true,
            "predicted_probabilities": eval_result.calibration_pred,
        },
    }

    if walk_forward_summary:
        report["walk_forward_cv"] = walk_forward_summary

    if feature_importance:
        sorted_feats = sorted(feature_importance.items(), key=lambda x: -abs(x[1]))
        report["feature_importance"] = dict(sorted_feats)

    if profit_sim:
        report["profit_simulation"] = profit_sim

    report["analysis"] = _generate_analysis(
        model_name=model_name,
        eval_result=eval_result,
        feature_importance=feature_importance,
        train_size=train_size,
        n_features=len(feature_names),
        profit_sim=profit_sim,
        wf_summary=walk_forward_summary,
    )

    return report


def _generate_analysis(
    model_name: str,
    eval_result: EvaluationResult,
    feature_importance: dict[str, float] | None,
    train_size: int,
    n_features: int,
    profit_sim: dict[str, Any] | None,
    wf_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Automated diagnostic commentary."""
    analysis: dict[str, Any] = {}

    # Overfitting risk
    risks = []
    if train_size < 200:
        risks.append(
            f"Very small training set ({train_size} rows). High overfitting risk. "
            "Collect more data before trusting this model."
        )
    if train_size < n_features * 20:
        risks.append(
            f"Low samples-per-feature ratio ({train_size}/{n_features} = "
            f"{train_size/n_features:.0f}). Consider reducing feature count."
        )
    if wf_summary and wf_summary.get("std_accuracy", 0) > 0.05:
        risks.append(
            f"Walk-forward accuracy has high variance "
            f"(std={wf_summary['std_accuracy']:.3f}). Model is unstable "
            "across time periods."
        )
    if eval_result.roc_auc is not None and eval_result.roc_auc < 0.55:
        risks.append(
            f"ROC-AUC is {eval_result.roc_auc:.3f}, barely above random (0.5). "
            "The model may have no real predictive power."
        )
    analysis["overfitting_risks"] = risks if risks else ["No major risks detected."]

    # Model weaknesses
    weaknesses = []
    if eval_result.precision_at_60 < 0.55:
        weaknesses.append(
            "Precision at 60% confidence threshold is below 55%. Many predicted "
            "'up' moves don't materialise — trading on this signal would produce "
            "too many false-positive trades."
        )
    if eval_result.brier_score > 0.3:
        weaknesses.append(
            f"Brier score ({eval_result.brier_score:.3f}) indicates poor probability "
            "calibration. The model's confidence values should not be trusted for "
            "position sizing."
        )
    if profit_sim and profit_sim.get("total_pnl", 0) <= 0:
        weaknesses.append(
            "Simulated profit is non-positive. Even under ideal (optimistic) "
            "assumptions, this model does not produce an edge after fees."
        )
    analysis["weaknesses"] = weaknesses if weaknesses else ["No major weaknesses detected."]

    # Feature importance commentary
    if feature_importance:
        top3 = list(feature_importance.keys())[:3]
        analysis["top_features"] = top3
        analysis["feature_commentary"] = (
            f"Top 3 features: {', '.join(top3)}. "
            "If these are all momentum/flow signals, the model may be "
            "momentum-chasing and vulnerable to mean reversion."
        )

    # Data improvement suggestions
    suggestions = [
        "Collect more historical data (ideally 10k+ rows per market).",
        "Add time-to-resolution feature if market end dates are available.",
        "Integrate sentiment/news data via the external provider interface.",
        "Add features from correlated markets (e.g., complementary Yes/No tokens).",
        "Engineer higher-frequency features (sub-minute) if the data cadence supports it.",
    ]
    analysis["data_improvement_suggestions"] = suggestions

    return analysis


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write the report to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def format_text_summary(report: dict[str, Any]) -> str:
    """Render a human-readable text summary from the report dict."""
    lines = [
        "=" * 72,
        f"  MODEL TRAINING REPORT — {report['model_name']}",
        f"  Generated: {report['generated_at']}",
        "=" * 72,
        "",
        f"Dataset: {report['dataset']['total']} rows "
        f"(train={report['dataset']['train_size']}, "
        f"val={report['dataset']['val_size']}, "
        f"test={report['dataset']['test_size']})",
        f"Features: {report['dataset']['n_features']}",
        "",
        "── Test Metrics ──",
        f"  Accuracy:        {report['test_metrics']['accuracy']:.4f}",
        f"  Log-loss:        {report['test_metrics']['log_loss']:.4f}",
        f"  ROC-AUC:         {report['test_metrics'].get('roc_auc', 'N/A')}",
        f"  Brier score:     {report['test_metrics']['brier_score']:.4f}",
        f"  Precision @60%:  {report['test_metrics']['precision_at_60_pct']:.4f}",
        f"  Recall @60%:     {report['test_metrics']['recall_at_60_pct']:.4f}",
        "",
    ]

    if "walk_forward_cv" in report:
        wf = report["walk_forward_cv"]
        lines.append("── Walk-Forward CV ──")
        lines.append(f"  Folds:           {wf.get('n_folds', 'N/A')}")
        lines.append(f"  Mean accuracy:   {wf.get('mean_accuracy', 0):.4f} ± {wf.get('std_accuracy', 0):.4f}")
        lines.append(f"  Mean log-loss:   {wf.get('mean_log_loss', 0):.4f} ± {wf.get('std_log_loss', 0):.4f}")
        lines.append("")

    if "feature_importance" in report:
        lines.append("── Feature Importance (top 10) ──")
        for i, (feat, imp) in enumerate(report["feature_importance"].items()):
            if i >= 10:
                break
            lines.append(f"  {feat:35s} {imp:+.4f}")
        lines.append("")

    if "profit_simulation" in report:
        ps = report["profit_simulation"]
        lines.append("── Profit Simulation (optimistic) ──")
        lines.append(f"  Trades taken:    {ps.get('n_trades', 0)}")
        lines.append(f"  Total PnL:       {ps.get('total_pnl', 0):.4f}")
        lines.append(f"  Win rate:        {ps.get('win_rate', 0):.2%}")
        lines.append(f"  Avg PnL/trade:   {ps.get('avg_pnl_per_trade', 0):.4f}")
        lines.append(f"  Max drawdown:    {ps.get('max_drawdown', 0):.4f}")
        lines.append(f"  Sharpe-like:     {ps.get('sharpe_like', 0):.3f}")
        lines.append("")

    analysis = report.get("analysis", {})

    lines.append("── Overfitting Risks ──")
    for risk in analysis.get("overfitting_risks", []):
        lines.append(f"  • {risk}")
    lines.append("")

    lines.append("── Model Weaknesses ──")
    for w in analysis.get("weaknesses", []):
        lines.append(f"  • {w}")
    lines.append("")

    lines.append("── Data Improvement Suggestions ──")
    for s in analysis.get("data_improvement_suggestions", []):
        lines.append(f"  • {s}")
    lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
