"""Normalized market data models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.models.enums import AssetClass


class Quote(BaseModel):
    """Point-in-time quote for any instrument."""

    instrument_id: str
    asset_class: AssetClass
    exchange: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    timestamp: datetime

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0


class TradeTick(BaseModel):
    """Single trade execution tick."""

    instrument_id: str
    price: float
    size: float
    side: str = ""
    timestamp: datetime
