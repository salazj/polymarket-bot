"""Shared enumerations for multi-asset trading."""

from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    PREDICTION_MARKETS = "prediction_markets"
    EQUITIES = "equities"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


class StockAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"
