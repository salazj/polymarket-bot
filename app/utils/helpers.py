"""
Shared utility functions: time helpers, price rounding, identifiers.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def epoch_ms() -> int:
    return int(time.time() * 1000)


def generate_order_id() -> str:
    return uuid.uuid4().hex[:16]


# Polymarket prices are in USDC with 2-decimal (cent) precision.
TICK_SIZE = Decimal("0.01")


def round_price(price: float, tick: Decimal = TICK_SIZE) -> float:
    """Round price down to the nearest tick to avoid crossing the spread."""
    d = Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN)
    return float(d)


def round_size(size: float, min_size: float = 0.01) -> float:
    """Round order size down; return 0 if below minimum."""
    d = Decimal(str(size)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    result = float(d)
    return result if result >= min_size else 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator
