"""
Backward-compatibility shim — delegates to app.exchanges.polymarket.market_data.

DEPRECATED: Import from app.exchanges.polymarket instead.
"""

from app.exchanges.polymarket.market_data import PolymarketMarketDataClient as PolymarketRestClient  # noqa: F401

__all__ = ["PolymarketRestClient"]
