"""Normalized order and fill models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.models.enums import AssetClass, OrderType, TimeInForce


class OrderRequest(BaseModel):
    """Exchange-agnostic order request."""

    instrument_id: str
    asset_class: AssetClass
    exchange: str
    side: str
    quantity: float
    price: float | None = None
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.DAY
    stop_price: float | None = None
    metadata: dict = {}


class Fill(BaseModel):
    """Normalized fill report."""

    instrument_id: str
    order_id: str
    side: str
    price: float
    quantity: float
    commission: float = 0.0
    timestamp: datetime
