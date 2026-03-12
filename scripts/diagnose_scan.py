"""Diagnose the events-based market scanning approach."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def main():
    from app.config.settings import get_settings
    from app.exchanges.kalshi.market_data import KalshiMarketDataClient

    settings = get_settings()
    client = KalshiMarketDataClient(settings)

    print("=== Events-Based Market Scan Test ===\n")

    t0 = time.time()
    markets = await client.get_all_markets(max_markets=2000)
    elapsed = time.time() - t0

    print(f"Result: {len(markets)} markets in {elapsed:.1f}s\n")

    if markets:
        cats: dict[str, int] = {}
        for m in markets:
            c = m.category or "uncategorized"
            cats[c] = cats.get(c, 0) + 1
        print("Categories:")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:15]:
            print(f"  {cat:25s} {count:5d}")

        print(f"\nFirst 10 markets:")
        for m in markets[:10]:
            print(f"  {m.market_id[:35]:35s} cat={m.category:15s} q={m.question[:50]}")

    # Test full pipeline
    print("\n--- Full Universe Pipeline ---")
    from app.universe.manager import UniverseManager
    mgr = UniverseManager(settings, client)
    t1 = time.time()
    result = await mgr.initial_selection()
    elapsed2 = time.time() - t1
    print(f"initial_selection returned {len(result)} markets in {elapsed2:.1f}s")
    print(f"Stats: {mgr.stats}")

    await client.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
