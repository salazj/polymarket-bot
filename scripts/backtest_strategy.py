#!/usr/bin/env python3
"""
Run a backtest of a strategy against historical feature data.

Usage:
    python scripts/backtest_strategy.py --strategy passive_market_maker
    python scripts/backtest_strategy.py --strategy momentum_scalper --fee-rate 0.01
"""

import asyncio
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.data.models import MarketFeatures
from app.monitoring import setup_logging
from app.strategies.base import StrategyRegistry


@click.command()
@click.option("--strategy", "-s", default="passive_market_maker", help="Strategy name")
@click.option("--fee-rate", default=0.02, help="Trading fee rate")
@click.option("--slippage-bps", default=10.0, help="Slippage in basis points")
@click.option("--fill-prob", default=0.5, help="Fill probability for limit orders")
@click.option("--cash", default=100.0, help="Starting cash")
def main(strategy: str, fee_rate: float, slippage_bps: float, fill_prob: float, cash: float) -> None:
    """Run a backtest against stored or synthetic data."""
    settings = get_settings()
    setup_logging("INFO")

    print(f"Running backtest: strategy={strategy}")

    # Load features from database
    features = _load_features(settings.database_url)

    if not features:
        print("No stored features found. Generating synthetic data for demo...")
        features = _generate_synthetic_features()

    print(f"Feature snapshots: {len(features)}")

    strategy_cls = StrategyRegistry.get(strategy)
    strat = strategy_cls(settings)

    config = BacktestConfig(
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        fill_probability=fill_prob,
        starting_cash=cash,
    )

    engine = BacktestEngine(strat, settings, config)
    result = engine.run(features)

    print(f"\n{'='*50}")
    print(f"Strategy: {result.strategy_name}")
    print(f"Period: {result.start_time} to {result.end_time}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "Sharpe: N/A")
    print(f"{'='*50}")

    engine.save_report(result, settings.reports_dir)
    print(f"\nReport saved to {settings.reports_dir}")


def _load_features(db_url: str) -> list[MarketFeatures]:
    import sqlite3
    db_path = db_url.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT data_json FROM features ORDER BY id", conn)
        conn.close()
    except Exception:
        return []

    features = []
    for _, row in df.iterrows():
        data = json.loads(row["data_json"])
        features.append(MarketFeatures(**data))
    return features


def _generate_synthetic_features() -> list[MarketFeatures]:
    """Generate synthetic features for demo backtest."""
    from datetime import datetime, timedelta, timezone

    rng = np.random.default_rng(42)
    n = 1000
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    mid_prices = np.cumsum(rng.normal(0, 0.002, n)) + 0.50
    mid_prices = np.clip(mid_prices, 0.05, 0.95)

    features = []
    for i in range(n):
        mid = float(mid_prices[i])
        spread = float(rng.uniform(0.02, 0.08))
        features.append(MarketFeatures(
            market_id="synthetic-market",
            token_id="synthetic-token",
            timestamp=base_time + timedelta(minutes=i),
            best_bid=mid - spread / 2,
            best_ask=mid + spread / 2,
            spread=spread,
            mid_price=mid,
            microprice=mid + float(rng.normal(0, 0.005)),
            orderbook_imbalance=float(rng.uniform(-0.5, 0.5)),
            bid_depth_5c=float(rng.exponential(30)),
            ask_depth_5c=float(rng.exponential(30)),
            recent_trade_flow=float(rng.normal(0, 3)),
            volatility_1m=float(rng.exponential(0.008)),
            momentum_1m=float(mid_prices[i] - mid_prices[max(0, i - 1)]),
            momentum_5m=float(mid_prices[i] - mid_prices[max(0, i - 5)]),
            momentum_15m=float(mid_prices[i] - mid_prices[max(0, i - 15)]),
            trade_count_1m=int(rng.poisson(3)),
            seconds_since_last_update=float(rng.uniform(0, 5)),
        ))

    return features


if __name__ == "__main__":
    main()
