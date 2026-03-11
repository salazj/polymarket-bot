#!/usr/bin/env python3
"""
Record raw WebSocket market data to a JSONL file for later replay/backtesting.
Captures orderbook snapshots, deltas, and trades.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.clients.ws_client import PolymarketWSClient
from app.monitoring import setup_logging, get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--token-ids", "-t", multiple=True, required=True, help="Token IDs to subscribe to")
@click.option("--output", "-o", default="data/recorded_session.jsonl", help="Output file path")
@click.option("--duration", "-d", default=3600, help="Recording duration in seconds")
def main(token_ids: tuple[str, ...], output: str, duration: int) -> None:
    """Record market data from Polymarket WebSocket."""
    asyncio.run(_run(list(token_ids), output, duration))


async def _run(token_ids: list[str], output: str, duration: int) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ws = PolymarketWSClient(settings)
    ws.subscribe_book(token_ids)
    ws.subscribe_trades(token_ids)

    event_count = 0

    async def write_event(msg: dict[str, Any]) -> None:
        nonlocal event_count
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": msg.get("channel", msg.get("type", "unknown")),
            "data": msg,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        event_count += 1
        if event_count % 100 == 0:
            logger.info("events_recorded", count=event_count)

    ws.on("book", write_event)
    ws.on("trade", write_event)

    logger.info("recording_started", tokens=token_ids, output=output, duration=duration)

    ws_task = asyncio.create_task(ws.connect())

    try:
        await asyncio.sleep(duration)
    except KeyboardInterrupt:
        pass
    finally:
        await ws.disconnect()
        ws_task.cancel()
        logger.info("recording_stopped", total_events=event_count, output=output)


if __name__ == "__main__":
    main()
