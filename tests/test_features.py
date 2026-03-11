"""Tests for feature computation."""

from datetime import datetime, timedelta, timezone

import pytest

from app.data.features import (
    FeatureEngine,
    compute_depth_within,
    compute_microprice,
    compute_momentum,
    compute_orderbook_imbalance,
    compute_trade_flow,
    compute_volatility,
)
from app.data.models import OrderbookSnapshot, PriceLevel, Side, Trade


class TestMicroprice:
    def test_symmetric_book(self) -> None:
        book = OrderbookSnapshot(
            market_id="m", token_id="t",
            bids=[PriceLevel(price=0.50, size=100)],
            asks=[PriceLevel(price=0.60, size=100)],
        )
        mp = compute_microprice(book)
        assert mp is not None
        assert mp == pytest.approx(0.55, abs=0.001)

    def test_asymmetric_book(self) -> None:
        book = OrderbookSnapshot(
            market_id="m", token_id="t",
            bids=[PriceLevel(price=0.50, size=200)],
            asks=[PriceLevel(price=0.60, size=100)],
        )
        mp = compute_microprice(book)
        assert mp is not None
        # bid-heavy: microprice should be closer to the ask
        assert mp > 0.55

    def test_empty_book(self) -> None:
        book = OrderbookSnapshot(market_id="m", token_id="t", bids=[], asks=[])
        assert compute_microprice(book) is None


class TestOrderbookImbalance:
    def test_balanced(self) -> None:
        book = OrderbookSnapshot(
            market_id="m", token_id="t",
            bids=[PriceLevel(price=0.50, size=100)],
            asks=[PriceLevel(price=0.55, size=100)],
        )
        imb = compute_orderbook_imbalance(book)
        assert imb is not None
        assert imb == pytest.approx(0.0)

    def test_bid_heavy(self) -> None:
        book = OrderbookSnapshot(
            market_id="m", token_id="t",
            bids=[PriceLevel(price=0.50, size=300)],
            asks=[PriceLevel(price=0.55, size=100)],
        )
        imb = compute_orderbook_imbalance(book)
        assert imb is not None
        assert imb > 0

    def test_ask_heavy(self) -> None:
        book = OrderbookSnapshot(
            market_id="m", token_id="t",
            bids=[PriceLevel(price=0.50, size=50)],
            asks=[PriceLevel(price=0.55, size=200)],
        )
        imb = compute_orderbook_imbalance(book)
        assert imb is not None
        assert imb < 0


class TestDepthWithin:
    def test_basic(self) -> None:
        levels = [
            PriceLevel(price=0.50, size=100),
            PriceLevel(price=0.49, size=50),
            PriceLevel(price=0.45, size=200),
        ]
        # All three levels are within 5 cents of 0.50 (|0.45-0.50|=0.05 <= 0.05)
        depth = compute_depth_within(levels, reference=0.50, cents=0.05)
        assert depth == 350.0

    def test_excludes_distant_levels(self) -> None:
        levels = [
            PriceLevel(price=0.50, size=100),
            PriceLevel(price=0.49, size=50),
            PriceLevel(price=0.40, size=200),
        ]
        depth = compute_depth_within(levels, reference=0.50, cents=0.05)
        assert depth == 150.0


class TestVolatility:
    def test_constant_prices(self) -> None:
        assert compute_volatility([0.5, 0.5, 0.5]) == 0.0

    def test_varying_prices(self) -> None:
        vol = compute_volatility([0.5, 0.52, 0.48, 0.51])
        assert vol > 0

    def test_insufficient_data(self) -> None:
        assert compute_volatility([0.5]) == 0.0


class TestMomentum:
    def test_upward(self) -> None:
        assert compute_momentum([0.50, 0.52, 0.55]) == pytest.approx(0.05)

    def test_downward(self) -> None:
        assert compute_momentum([0.55, 0.52, 0.50]) == pytest.approx(-0.05)

    def test_flat(self) -> None:
        assert compute_momentum([0.50, 0.50, 0.50]) == 0.0


class TestFeatureEngine:
    def test_compute_features(self, sample_book: OrderbookSnapshot) -> None:
        engine = FeatureEngine("test-market", "test-token")
        features = engine.compute(sample_book)
        assert features.market_id == "test-market"
        assert features.best_bid == 0.50
        assert features.best_ask == 0.55
        assert features.spread == pytest.approx(0.05)
        assert features.microprice is not None
