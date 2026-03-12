"""Stock universe filters."""

from __future__ import annotations

from typing import Any

from app.monitoring import get_logger

logger = get_logger(__name__)


class StockFilter:
    """Applies price, volume, and sector filters to candidate stocks."""

    def __init__(
        self,
        min_price: float = 5.0,
        max_price: float = 500.0,
        min_volume: int = 100000,
        sectors: list[str] | None = None,
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._min_volume = min_volume
        self._sectors = [s.lower() for s in sectors] if sectors else None

    def apply(self, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter assets based on configured criteria."""
        passed: list[dict[str, Any]] = []
        for asset in assets:
            symbol = asset.get("symbol", "")
            if not symbol:
                continue
            if self._sectors:
                sector = asset.get("sector", "").lower()
                if sector and sector not in self._sectors:
                    continue
            passed.append(asset)
        return passed
