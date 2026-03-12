"""Alpaca-specific data models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AccountInfo(BaseModel):
    account_id: str = ""
    status: str = ""
    cash: float = 0.0
    buying_power: float = 0.0
    portfolio_value: float = 0.0
    currency: str = "USD"
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False


class AlpacaPosition(BaseModel):
    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    side: str = "long"


class AlpacaOrder(BaseModel):
    id: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    filled_qty: float = 0.0
    type: str = "market"
    time_in_force: str = "day"
    status: str = ""
    limit_price: float | None = None
    stop_price: float | None = None
    created_at: str = ""
