"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config.settings import Settings
from app.data.models import (
    MarketFeatures,
    Order,
    OrderbookSnapshot,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    PriceLevel,
    Side,
    TokenSide,
)


@pytest.fixture
def settings() -> Settings:
    """Test settings with conservative defaults and dry-run enabled."""
    return Settings(
        dry_run=True,
        environment="test",
        max_position_per_market=10.0,
        max_total_exposure=50.0,
        max_daily_loss=10.0,
        max_orders_per_minute=10,
        max_slippage=0.03,
        min_liquidity_depth=10.0,
        min_spread_threshold=0.01,
        max_spread_threshold=0.15,
        default_order_size=1.0,
        strategy="passive_market_maker",
        log_level="WARNING",
    )


@pytest.fixture
def sample_book() -> OrderbookSnapshot:
    return OrderbookSnapshot(
        market_id="test-market",
        token_id="test-token",
        bids=[
            PriceLevel(price=0.50, size=100.0),
            PriceLevel(price=0.49, size=50.0),
            PriceLevel(price=0.48, size=30.0),
        ],
        asks=[
            PriceLevel(price=0.55, size=80.0),
            PriceLevel(price=0.56, size=40.0),
            PriceLevel(price=0.57, size=20.0),
        ],
    )


@pytest.fixture
def sample_features() -> MarketFeatures:
    return MarketFeatures(
        market_id="test-market",
        token_id="test-token",
        best_bid=0.50,
        best_ask=0.55,
        spread=0.05,
        mid_price=0.525,
        microprice=0.527,
        orderbook_imbalance=0.1,
        bid_depth_5c=180.0,
        ask_depth_5c=140.0,
        recent_trade_flow=2.0,
        volatility_1m=0.008,
        momentum_1m=0.003,
        momentum_5m=0.01,
        momentum_15m=0.02,
        trade_count_1m=5,
        seconds_since_last_update=1.0,
    )


@pytest.fixture
def empty_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(cash=100.0)


@pytest.fixture
def portfolio_with_position() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        cash=90.0,
        positions=[
            Position(
                market_id="test-market",
                token_id="test-token",
                token_side=TokenSide.YES,
                size=5.0,
                avg_entry_price=0.50,
            )
        ],
        total_exposure=2.5,
    )
