"""Tests for stock universe selection."""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.stocks.universe.filters import StockFilter


class TestStockFilter:
    def test_filter_allows_all_without_sectors(self):
        f = StockFilter(min_price=5.0, max_price=500.0, min_volume=100)
        assets = [
            {"symbol": "AAPL", "sector": "tech"},
            {"symbol": "JPM", "sector": "finance"},
        ]
        result = f.apply(assets)
        assert len(result) == 2

    def test_filter_by_sector(self):
        f = StockFilter(sectors=["tech"])
        assets = [
            {"symbol": "AAPL", "sector": "tech"},
            {"symbol": "JPM", "sector": "finance"},
        ]
        result = f.apply(assets)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    def test_filter_empty_symbol_skipped(self):
        f = StockFilter()
        assets = [
            {"symbol": ""},
            {"symbol": "AAPL"},
        ]
        result = f.apply(assets)
        assert len(result) == 1


class TestStockUniverseSettings:
    def test_manual_mode_default(self):
        s = Settings()
        assert s.stock_universe_mode == "manual"

    def test_tickers_config(self):
        s = Settings(stock_tickers="AAPL,MSFT,NVDA")
        tickers = [t.strip() for t in s.stock_tickers.split(",")]
        assert len(tickers) == 3
        assert "AAPL" in tickers
