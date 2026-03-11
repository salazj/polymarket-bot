#!/usr/bin/env python3
"""
Replay a recorded market data session through a strategy.

Usage:
    python scripts/replay_session.py --input data/recorded_session.jsonl
    python scripts/replay_session.py --input data/recorded_session.jsonl --strategy momentum_scalper
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.monitoring import setup_logging
from app.replay.player import ReplayPlayer
from app.strategies.base import StrategyRegistry


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to recorded events JSONL")
@click.option("--strategy", "-s", default="passive_market_maker", help="Strategy to replay with")
@click.option("--output", "-o", default=None, help="Output results JSON path")
def main(input_path: str, strategy: str, output: str | None) -> None:
    """Replay recorded market data through a strategy."""
    settings = get_settings()
    setup_logging("INFO")

    strategy_cls = StrategyRegistry.get(strategy)
    strat = strategy_cls(settings)

    player = ReplayPlayer(strat, settings)

    print(f"Loading events from {input_path}...")
    events = player.load_events(Path(input_path))
    print(f"Loaded {len(events)} events")

    results = player.play(events)

    print(f"\nSignals generated: {results['signals_generated']}")
    print(f"Portfolio summary:")
    portfolio = results.get("portfolio", {})
    for key in ["cash", "total_exposure", "total_unrealized_pnl", "total_realized_pnl"]:
        if key in portfolio:
            print(f"  {key}: {portfolio[key]}")

    out = Path(output) if output else settings.reports_dir / f"replay_{strategy}.json"
    player.save_results(results, out)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
