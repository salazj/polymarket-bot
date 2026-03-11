#!/usr/bin/env python3
"""
Export PnL history and portfolio summary from the database.

Usage:
    python scripts/export_pnl_report.py
    python scripts/export_pnl_report.py --format csv --output reports/pnl.csv
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.monitoring import setup_logging
from app.storage.repository import Repository


@click.command()
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--limit", default=500, help="Max rows")
def main(fmt: str, output: str | None, limit: int) -> None:
    """Export PnL snapshots from database."""
    asyncio.run(_run(fmt, output, limit))


async def _run(fmt: str, output: str | None, limit: int) -> None:
    settings = get_settings()
    setup_logging("INFO")
    settings.ensure_dirs()

    repo = Repository(settings.database_url)
    await repo.initialize()

    snapshots = await repo.get_pnl_history(limit=limit)
    orders = await repo.get_orders(limit=limit)

    await repo.close()

    if not snapshots:
        print("No PnL snapshots found in database.")
        return

    out_path = Path(output) if output else settings.reports_dir / f"pnl_report.{fmt}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        report = {
            "pnl_snapshots": snapshots,
            "recent_orders": orders,
            "summary": {
                "total_snapshots": len(snapshots),
                "latest_cash": snapshots[0]["cash"] if snapshots else 0,
                "latest_daily_pnl": snapshots[0]["daily_pnl"] if snapshots else 0,
            },
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    elif fmt == "csv":
        if snapshots:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=snapshots[0].keys())
                writer.writeheader()
                writer.writerows(snapshots)

    print(f"Exported {len(snapshots)} PnL snapshots to {out_path}")
    if snapshots:
        latest = snapshots[0]
        print(f"Latest: cash=${latest['cash']:.2f}, daily_pnl=${latest['daily_pnl']:.2f}")


if __name__ == "__main__":
    main()
