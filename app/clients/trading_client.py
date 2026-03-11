"""
Backward-compatibility shim — delegates to app.exchanges.polymarket.execution.

DEPRECATED: Import from app.exchanges.polymarket instead.
"""

from app.exchanges.polymarket.execution import PolymarketExecutionClient as TradingClient  # noqa: F401

__all__ = ["TradingClient"]
