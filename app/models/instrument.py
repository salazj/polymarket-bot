"""Normalized instrument model usable across asset classes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.models.enums import AssetClass


class Instrument(BaseModel):
    """Unified instrument identifier across prediction markets and equities."""

    symbol: str
    asset_class: AssetClass
    exchange: str
    instrument_id: str
    name: str = ""
    metadata: dict[str, Any] = {}
