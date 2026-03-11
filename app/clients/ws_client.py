"""
Backward-compatibility shim — delegates to app.exchanges.polymarket.websocket.

DEPRECATED: Import from app.exchanges.polymarket instead.
"""

from app.exchanges.polymarket.websocket import PolymarketWebSocketClient as PolymarketWSClient  # noqa: F401

__all__ = ["PolymarketWSClient"]
