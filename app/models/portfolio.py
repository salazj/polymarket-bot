"""Normalized portfolio models shared across asset classes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.models.enums import AssetClass


class NormalizedPosition(BaseModel):
    """Cross-asset position representation."""

    instrument_id: str
    asset_class: AssetClass
    exchange: str
    symbol: str = ""
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class Balance(BaseModel):
    """Account balance snapshot."""

    cash: float = 0.0
    buying_power: float = 0.0
    portfolio_value: float = 0.0
    currency: str = "USD"


class PnLSnapshot(BaseModel):
    """Point-in-time PnL snapshot."""

    timestamp: datetime
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_equity: float = 0.0
