"""Stock universe manager — selects and maintains active stock list."""

from __future__ import annotations

from typing import Any

from app.brokers.base import BaseBrokerMarketData
from app.config.settings import Settings
from app.monitoring import get_logger
from app.stocks.universe.filters import StockFilter
from app.stocks.universe.scanner import StockUniverseScanner

logger = get_logger(__name__)


class StockUniverseManager:
    """Manages the active set of stock symbols for trading."""

    def __init__(self, settings: Settings, market_data: BaseBrokerMarketData) -> None:
        self._settings = settings
        self._scanner = StockUniverseScanner(market_data)
        self._filter = StockFilter(
            min_price=settings.stock_min_price,
            max_price=settings.stock_max_price,
            min_volume=settings.stock_min_volume,
            sectors=(
                [s.strip() for s in settings.stock_sector_include.split(",") if s.strip()]
                if settings.stock_sector_include
                else None
            ),
        )
        self._active_symbols: list[str] = []
        self._max_symbols = settings.max_stock_symbols

    @property
    def active_symbols(self) -> list[str]:
        return list(self._active_symbols)

    async def initial_selection(self) -> list[str]:
        """Select initial universe of stocks to trade."""
        mode = self._settings.stock_universe_mode.lower()

        if mode == "manual":
            tickers_str = self._settings.stock_tickers
            if tickers_str:
                self._active_symbols = [
                    t.strip().upper()
                    for t in tickers_str.split(",")
                    if t.strip()
                ][: self._max_symbols]
            logger.info(
                "stock_universe_manual",
                symbols=self._active_symbols,
                count=len(self._active_symbols),
            )
        else:
            assets = await self._scanner.scan(
                min_price=self._settings.stock_min_price,
                max_price=self._settings.stock_max_price,
                min_volume=self._settings.stock_min_volume,
            )
            filtered = self._filter.apply(assets)
            self._active_symbols = [
                a["symbol"] for a in filtered if "symbol" in a
            ][: self._max_symbols]
            logger.info(
                "stock_universe_filtered",
                scanned=len(assets),
                filtered=len(filtered),
                active=len(self._active_symbols),
            )

        return self._active_symbols

    async def refresh(self) -> list[str]:
        """Refresh the universe (re-run selection)."""
        return await self.initial_selection()
