"""Tests for stock feature engine."""

from __future__ import annotations

from datetime import datetime, timezone

from app.stocks.features import StockFeatureEngine
from app.stocks.models import StockBar


class TestStockFeatureEngine:
    def test_compute_empty(self):
        engine = StockFeatureEngine("AAPL")
        engine.start_new_day()
        engine.update_quote(bid=150.0, ask=150.5, last=150.25)
        features = engine.compute()
        assert features.symbol == "AAPL"
        assert features.last_price == 150.25

    def test_compute_with_bars(self):
        engine = StockFeatureEngine("AAPL")
        engine.start_new_day()
        for i in range(20):
            bar = StockBar(
                symbol="AAPL",
                open=148.0 + i * 0.1,
                high=149.0 + i * 0.1,
                low=147.0 + i * 0.1,
                close=148.5 + i * 0.1,
                volume=100000,
                timestamp=datetime.now(timezone.utc),
            )
            engine.add_bar(bar)
        engine.update_quote(bid=150.0, ask=150.5, last=150.25)
        features = engine.compute()
        assert features.sma_20 > 0
        assert features.ema_9 > 0
        assert features.volume_today == 20 * 100000

    def test_rsi_computation(self):
        assert StockFeatureEngine._compute_rsi([], 14) == 50.0
        assert StockFeatureEngine._compute_rsi(list(range(20)), 14) == 100.0

    def test_sma_computation(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert StockFeatureEngine._sma(values, 3) == 4.0

    def test_ema_computation(self):
        values = [1.0, 2.0, 3.0]
        result = StockFeatureEngine._ema(values, 3)
        assert result > 0
