"""Broker adapter layer for stock trading — pluggable broker implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.brokers.base import (
    BaseBrokerAdapter,
    BaseBrokerExecution,
    BaseBrokerMarketData,
    BaseBrokerStreaming,
)

if TYPE_CHECKING:
    from app.config.settings import Settings


def build_broker_adapter(settings: "Settings") -> BaseBrokerAdapter:
    """Factory: instantiate the correct broker adapter based on settings.broker."""
    from app.brokers.alpaca.adapter import AlpacaAdapter

    broker = settings.broker.lower()
    if broker == "alpaca":
        return AlpacaAdapter(settings)
    raise ValueError(f"Unknown broker: {broker!r}. Expected 'alpaca'.")


__all__ = [
    "BaseBrokerAdapter",
    "BaseBrokerExecution",
    "BaseBrokerMarketData",
    "BaseBrokerStreaming",
    "build_broker_adapter",
]
