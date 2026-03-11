"""
Kalshi exchange adapter — wires together market data, execution, and WebSocket.

This is the top-level entry point for all Kalshi interactions.  The rest of
the system only sees the adapter's three sub-clients via the base interfaces.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.exchanges.base import (
    BaseExchangeAdapter,
    BaseExecutionClient,
    BaseMarketDataClient,
    BaseWebSocketClient,
    Exchange,
)
from app.exchanges.kalshi.execution import KalshiExecutionClient
from app.exchanges.kalshi.market_data import KalshiMarketDataClient
from app.exchanges.kalshi.websocket import KalshiWebSocketClient


class KalshiAdapter(BaseExchangeAdapter):
    """Full Kalshi adapter conforming to the exchange abstraction."""

    def __init__(self, settings: Settings) -> None:
        self._market_data = KalshiMarketDataClient(settings)
        self._execution = KalshiExecutionClient(settings)
        self._websocket = KalshiWebSocketClient(settings)

    @property
    def exchange(self) -> Exchange:
        return Exchange.KALSHI

    @property
    def market_data(self) -> BaseMarketDataClient:
        return self._market_data

    @property
    def execution(self) -> BaseExecutionClient:
        return self._execution

    @property
    def websocket(self) -> BaseWebSocketClient:
        return self._websocket

    async def close(self) -> None:
        await self._websocket.disconnect()
        await self._execution.close()
        await self._market_data.close()
