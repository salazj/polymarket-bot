"""Stock universe scanner — queries broker for tradable assets."""

from __future__ import annotations

from typing import Any

from app.brokers.base import BaseBrokerMarketData
from app.monitoring import get_logger

logger = get_logger(__name__)


class StockUniverseScanner:
    """Discovers tradable stocks from broker API."""

    def __init__(self, market_data: BaseBrokerMarketData) -> None:
        self._market_data = market_data

    async def scan(
        self,
        min_price: float = 5.0,
        max_price: float = 500.0,
        min_volume: int = 100000,
        sectors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all tradable assets and apply basic filters."""
        try:
            assets = await self._market_data.get_tradable_assets(
                status="active",
                asset_class="us_equity",
            )
        except Exception as exc:
            logger.error("stock_scanner_error", error=str(exc))
            return []

        filtered: list[dict[str, Any]] = []
        for asset in assets:
            if not asset.get("tradable", True):
                continue
            if sectors:
                asset_sector = asset.get("sector", "").lower()
                if asset_sector and asset_sector not in [s.lower() for s in sectors]:
                    continue
            filtered.append(asset)

        logger.info(
            "stock_scan_complete",
            total=len(assets),
            filtered=len(filtered),
        )
        return filtered
