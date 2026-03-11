"""
Polymarket exchange adapter — wires together market data, execution, and WebSocket.
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
from app.exchanges.polymarket.execution import PolymarketExecutionClient
from app.exchanges.polymarket.market_data import PolymarketMarketDataClient
from app.exchanges.polymarket.websocket import PolymarketWebSocketClient


class PolymarketAdapter(BaseExchangeAdapter):
    """Full Polymarket adapter conforming to the exchange abstraction."""

    def __init__(self, settings: Settings) -> None:
        self._market_data = PolymarketMarketDataClient(settings)
        self._execution = PolymarketExecutionClient(settings)
        self._websocket = PolymarketWebSocketClient(settings)

    @property
    def exchange(self) -> Exchange:
        return Exchange.POLYMARKET

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
        await self._market_data.close()
