#!/usr/bin/env python3
"""
Evaluate a trained model artifact on new data.

Loads a saved model + preprocessing pipeline and evaluates on either:
  - The database (latest features not seen during training)
  - A synthetic dataset (for pipeline verification)

Outputs a full evaluation report with profit simulation.

Usage:
  python scripts/evaluate_model.py
  python scripts/evaluate_model.py --model model_artifacts/research_model.joblib
  python scripts/evaluate_model.py --synthetic --threshold 0.55
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np

from app.config.settings import get_settings
from app.monitoring import setup_logging
from app.research.dataset import generate_synthetic_dataset, load_features_df
from app.research.evaluation import evaluate_classifier, simulate_profit
from app.research.feature_eng import engineer_features
from app.research.models import load_model_artifact
from app.research.preprocessing import prepare_X
from app.research.report import (
    build_training_report,
    format_text_summary,
    save_report,
)
from app.research.targets import add_all_targets


@click.command()
@click.option("--model", default=None, help="Path to model .joblib artifact")
@click.option("--db", default=None, help="SQLite database path")
@click.option("--synthetic", is_flag=True, help="Evaluate on synthetic data")
@click.option("--n-samples", default=1000, type=int, help="Synthetic sample count")
@click.option("--horizon", default=6, type=int, help="Look-ahead rows")
@click.option("--min-edge", default=0.02, type=float, help="Min edge in cents")
@click.option("--fee", default=0.02, type=float, help="Round-trip fee")
@click.option("--threshold", default=0.6, type=float, help="Trading confidence threshold")
def main(
    model: str | None,
    db: str | None,
    synthetic: bool,
    n_samples: int,
    horizon: int,
    min_edge: float,
    fee: float,
    threshold: float,
) -> None:
    """Evaluate a trained model on held-out or new data."""
    setup_logging("INFO")
    settings = get_settings()
    settings.ensure_dirs()

    # ── Load model ────────────────────────────────────────────────────
    model_path = Path(model) if model else settings.model_artifacts_dir / "research_model.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train a model first: python scripts/train_model.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    classifier = artifact["classifier"]
    feature_names = artifact["feature_names"]
    print(f"  Model: {artifact['name']}, {len(feature_names)} features")

    # ── Load data ─────────────────────────────────────────────────────
    if synthetic:
        print(f"Generating {n_samples} synthetic evaluation samples...")
        df = generate_synthetic_dataset(n=n_samples, seed=99)  # different seed from training
        desc = "Synthetic evaluation data"
    else:
        db_path = db or settings.database_url
        print(f"Loading features from {db_path}...")
        df = load_features_df(db_path)
        if len(df) < 20:
            print(f"Only {len(df)} rows. Falling back to synthetic data.")
            df = generate_synthetic_dataset(n=n_samples, seed=99)
            desc = "Synthetic fallback"
        else:
            desc = f"Database: {db_path}"

    df = engineer_features(df)
    df = add_all_targets(df, horizon=horizon, min_edge=min_edge, fee=fee)
    df = df.dropna(subset=["target_direction"]).reset_index(drop=True)

    X, _ = prepare_X(df, feature_names)
    y = df["target_direction"].values.astype(int)

    print(f"Evaluation set: {len(df)} samples, {int(y.sum())} positive ({100*y.mean():.1f}%)")

    # ── Predict ───────────────────────────────────────────────────────

    X_t = pipeline.transform(X)
    proba = classifier.predict_proba(X_t)
    pred = classifier.predict(X_t)

    # ── Evaluate ──────────────────────────────────────────────────────

    result = evaluate_classifier(y, proba, pred)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({len(y)} samples)")
    print(f"{'='*60}")
    print(f"  Accuracy:       {result.accuracy:.4f}")
    print(f"  Log-loss:       {result.log_loss_val:.4f}")
    print(f"  ROC-AUC:        {result.roc_auc}")
    print(f"  Brier score:    {result.brier_score:.4f}")
    print(f"  Precision @60%: {result.precision_at_60:.4f}")
    print(f"  Recall @60%:    {result.recall_at_60:.4f}")
    print(result.classification_report_text)

    # ── Profit simulation ─────────────────────────────────────────────

    entry_prices = df["best_ask"].fillna(df["mid_price"]).values
    exit_prices = df["mid_price"].shift(-horizon).fillna(df["mid_price"]).values

    for thr in [0.55, 0.60, 0.65, 0.70]:
        ps = simulate_profit(y, proba, entry_prices, exit_prices, threshold=thr, fee=fee)
        marker = " ◀" if thr == threshold else ""
        print(
            f"  threshold={thr:.2f}  trades={ps['n_trades']:4d}  "
            f"PnL={ps['total_pnl']:+.4f}  win={ps['win_rate']:.1%}  "
            f"sharpe={ps['sharpe_like']:.3f}{marker}"
        )

    # ── Save report ───────────────────────────────────────────────────

    profit_main = simulate_profit(y, proba, entry_prices, exit_prices, threshold=threshold, fee=fee)

    report = build_training_report(
        model_name=artifact["name"],
        feature_names=feature_names,
        train_size=0,
        val_size=0,
        test_size=len(y),
        eval_result=result,
        profit_sim=profit_main,
        dataset_description=desc,
    )

    report_path = settings.reports_dir / "evaluation_report.json"
    save_report(report, report_path)
    print(f"\nReport saved: {report_path}")

    text = format_text_summary(report)
    text_path = settings.reports_dir / "evaluation_report.txt"
    text_path.write_text(text)
    print(f"Text report: {text_path}")


if __name__ == "__main__":
    main()
