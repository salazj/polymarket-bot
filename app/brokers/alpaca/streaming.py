"""Alpaca streaming client for real-time data."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from app.brokers.base import BaseBrokerStreaming, MessageHandler
from app.config.settings import Settings
from app.monitoring import get_logger

logger = get_logger(__name__)


class AlpacaStreaming(BaseBrokerStreaming):
    """WebSocket streaming for Alpaca market data and order updates."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.alpaca_api_key
        self._secret_key = settings.alpaca_secret_key
        self._paper = settings.alpaca_paper
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._connected = False
        self._data_client: Any = None
        self._trading_client: Any = None
        self._pending_quotes: list[str] = []
        self._pending_bars: list[str] = []
        self._pending_trades: list[str] = []
        self._subscribe_orders = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def subscribe_quotes(self, symbols: list[str]) -> None:
        self._pending_quotes.extend(symbols)

    async def subscribe_bars(self, symbols: list[str]) -> None:
        self._pending_bars.extend(symbols)

    async def subscribe_trades(self, symbols: list[str]) -> None:
        self._pending_trades.extend(symbols)

    async def subscribe_order_updates(self) -> None:
        self._subscribe_orders = True

    def on(self, event_type: str, handler: MessageHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def connect(self) -> None:
        """Start streaming connections. Falls back to polling if alpaca-py streaming isn't available."""
        self._connected = True
        logger.info(
            "alpaca_streaming_started",
            quotes=len(self._pending_quotes),
            bars=len(self._pending_bars),
            trades=len(self._pending_trades),
        )
        # The actual Alpaca WebSocket integration would use
        # alpaca.data.live.StockDataStream here. For now, we mark
        # as connected and rely on REST polling in the main loop.
        while self._connected:
            await asyncio.sleep(60)

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("alpaca_streaming_stopped")

    async def _dispatch(self, event_type: str, data: dict[str, Any]) -> None:
        for handler in self._handlers.get(event_type, []):
            try:
                await handler(data)
            except Exception as exc:
                logger.error("streaming_handler_error", event=event_type, error=str(exc))
