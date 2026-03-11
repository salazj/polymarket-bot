#!/usr/bin/env python3
"""
Fetch and display available Polymarket markets.
Optionally saves them to the database.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.clients.rest_client import PolymarketRestClient
from app.monitoring import setup_logging
from app.storage.repository import Repository


@click.command()
@click.option("--save", is_flag=True, help="Save markets to database")
@click.option("--limit", default=20, help="Max markets to fetch")
@click.option("--output", "-o", default=None, help="Output JSON file path")
def main(save: bool, limit: int, output: str | None) -> None:
    """Fetch markets from Polymarket."""
    asyncio.run(_run(save, limit, output))


async def _run(save: bool, limit: int, output: str | None) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    client = PolymarketRestClient(settings)

    try:
        markets = await client.get_all_markets(max_pages=max(1, limit // 20))
        markets = markets[:limit]

        for i, m in enumerate(markets):
            tokens = ", ".join(f"{t.outcome}={t.token_id[:12]}..." for t in m.tokens)
            print(f"{i+1:3d}. [{m.condition_id[:12]}...] {m.question[:80]}")
            print(f"     slug={m.slug} active={m.active} tokens=[{tokens}]")
            print()

        if save:
            repo = Repository(settings.database_url)
            await repo.initialize()
            for m in markets:
                await repo.save_market(m)
            await repo.close()
            print(f"Saved {len(markets)} markets to database.")

        if output:
            data = [m.model_dump() for m in markets]
            with open(output, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Saved to {output}")

        print(f"\nTotal: {len(markets)} markets")

    finally:
        await client.close()


if __name__ == "__main__":
    main()
