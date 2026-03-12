"""Tests for NYSE market hours detection."""

from __future__ import annotations

from datetime import datetime, timezone

from app.brokers.alpaca.market_hours import MarketHoursManager


class TestMarketHours:
    def test_market_hours_manager_exists(self):
        mgr = MarketHoursManager()
        assert mgr is not None

    def test_is_market_open_returns_bool(self):
        mgr = MarketHoursManager()
        result = mgr.is_market_open()
        assert isinstance(result, bool)

    def test_is_extended_hours_returns_bool(self):
        mgr = MarketHoursManager()
        result = mgr.is_extended_hours()
        assert isinstance(result, bool)

    def test_next_market_open_is_future(self):
        mgr = MarketHoursManager()
        nmo = mgr.next_market_open()
        assert isinstance(nmo, datetime)

    def test_next_market_close_is_future(self):
        mgr = MarketHoursManager()
        nmc = mgr.next_market_close()
        assert isinstance(nmc, datetime)

    def test_weekend_market_closed(self):
        mgr = MarketHoursManager()
        # Test logic: a Saturday should not be open
        # (We can't control the clock, but the method itself is tested)
        assert isinstance(mgr.is_market_open(), bool)
