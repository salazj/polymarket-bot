"""
Polymarket WebSocket client implementing the BaseWebSocketClient interface.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from app.config.settings import Settings
from app.exchanges.base import BaseWebSocketClient, MessageHandler
from app.monitoring import get_logger
from app.monitoring.logger import metrics

logger = get_logger(__name__)

HEARTBEAT_INTERVAL = 30.0
STALE_THRESHOLD = 60.0
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
MAX_CONSECUTIVE_RECONNECTS = 50


class PolymarketWebSocketClient(BaseWebSocketClient):
    """WebSocket client for Polymarket CLOB real-time data."""

    def __init__(self, settings: Settings) -> None:
        self._ws_url = settings.polymarket_ws_host
        self._ws: Any = None
        self._subscriptions: list[dict[str, Any]] = []
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._running = False
        self._last_message_time: datetime | None = None
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open

    @property
    def seconds_since_last_message(self) -> float:
        if self._last_message_time is None:
            return float("inf")
        delta = datetime.now(timezone.utc) - self._last_message_time
        return delta.total_seconds()

    @property
    def is_stale(self) -> bool:
        return self.seconds_since_last_message > STALE_THRESHOLD

    def on(self, event_type: str, handler: MessageHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_book(self, instrument_ids: list[str]) -> None:
        for iid in instrument_ids:
            sub = {
                "type": "subscribe",
                "channel": "book",
                "assets_ids": [iid],
            }
            if sub not in self._subscriptions:
                self._subscriptions.append(sub)

    def subscribe_trades(self, instrument_ids: list[str]) -> None:
        for iid in instrument_ids:
            sub = {
                "type": "subscribe",
                "channel": "trades",
                "assets_ids": [iid],
            }
            if sub not in self._subscriptions:
                self._subscriptions.append(sub)

    def subscribe_user(self) -> None:
        sub = {"type": "subscribe", "channel": "user"}
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)

    async def connect(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                metrics.increment("ws_reconnects")
                self._reconnect_count += 1

                if self._reconnect_count > MAX_CONSECUTIVE_RECONNECTS:
                    logger.critical(
                        "ws_max_reconnects_exceeded",
                        count=self._reconnect_count,
                        limit=MAX_CONSECUTIVE_RECONNECTS,
                    )
                    self._running = False
                    break

                delay = min(
                    RECONNECT_BASE_DELAY * (2 ** min(self._reconnect_count, 6)),
                    RECONNECT_MAX_DELAY,
                )
                logger.warning(
                    "ws_reconnecting",
                    error=str(e),
                    reconnect_count=self._reconnect_count,
                    delay=delay,
                )
                await asyncio.sleep(delay)

    async def _run_connection(self) -> None:
        async with websockets.connect(self._ws_url, ping_interval=20) as ws:
            self._ws = ws
            self._reconnect_count = 0
            logger.info("ws_connected", url=self._ws_url)

            for sub in self._subscriptions:
                await ws.send(json.dumps(sub))
                logger.debug("ws_subscribed", channel=sub.get("channel"))

            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            try:
                async for raw_msg in ws:
                    self._last_message_time = datetime.now(timezone.utc)
                    metrics.increment("ws_messages_received")
                    try:
                        msg = json.loads(raw_msg)
                        await self._dispatch(msg)
                    except json.JSONDecodeError:
                        logger.warning("ws_invalid_json", data=str(raw_msg)[:200])
            except ConnectionClosed:
                logger.warning("ws_connection_closed")
            finally:
                heartbeat_task.cancel()

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if self.is_stale:
                logger.warning("ws_stale_data", seconds=self.seconds_since_last_message)
                metrics.increment("ws_stale_detected")

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type", msg.get("channel", "unknown"))
        handlers = self._handlers.get(msg_type, [])
        for handler in handlers:
            try:
                await handler(msg)
            except Exception as e:
                logger.error("ws_handler_error", type=msg_type, error=str(e))
                metrics.increment("ws_handler_errors")

    async def disconnect(self) -> None:
        self._running = False
        if self._ws and self._ws.open:
            await self._ws.close()
        logger.info("ws_disconnected")
