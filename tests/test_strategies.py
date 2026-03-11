"""Tests for strategy signal generation."""

import pytest

from app.config.settings import Settings
from app.data.models import MarketFeatures, PortfolioSnapshot, SignalAction
from app.strategies.passive_market_maker import PassiveMarketMaker
from app.strategies.momentum_scalper import MomentumScalper


class TestPassiveMarketMaker:
    def test_generates_signal_with_good_spread(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot
    ) -> None:
        mm = PassiveMarketMaker(settings)
        signal = mm.generate_signal(sample_features, empty_portfolio)
        assert signal is not None
        assert signal.action in {SignalAction.BUY_YES, SignalAction.SELL_YES}
        assert 0 < signal.confidence <= 1.0
        assert signal.suggested_price is not None
        assert signal.suggested_size is not None

    def test_no_signal_when_spread_too_tight(
        self, settings: Settings, empty_portfolio: PortfolioSnapshot
    ) -> None:
        features = MarketFeatures(
            market_id="m", token_id="t",
            best_bid=0.50, best_ask=0.505,
            spread=0.005,  # too tight
            mid_price=0.5025,
            bid_depth_5c=100, ask_depth_5c=100,
        )
        mm = PassiveMarketMaker(settings)
        assert mm.generate_signal(features, empty_portfolio) is None

    def test_no_signal_when_book_thin(
        self, settings: Settings, empty_portfolio: PortfolioSnapshot
    ) -> None:
        features = MarketFeatures(
            market_id="m", token_id="t",
            best_bid=0.50, best_ask=0.55,
            spread=0.05,
            mid_price=0.525,
            bid_depth_5c=2.0,  # too thin
            ask_depth_5c=2.0,
        )
        mm = PassiveMarketMaker(settings)
        assert mm.generate_signal(features, empty_portfolio) is None

    def test_no_signal_when_stale(
        self, settings: Settings, empty_portfolio: PortfolioSnapshot
    ) -> None:
        features = MarketFeatures(
            market_id="m", token_id="t",
            best_bid=0.50, best_ask=0.55,
            spread=0.05,
            bid_depth_5c=100, ask_depth_5c=100,
            seconds_since_last_update=60,  # stale
        )
        mm = PassiveMarketMaker(settings)
        assert mm.generate_signal(features, empty_portfolio) is None


class TestMomentumScalper:
    def test_no_signal_during_cooldown(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot
    ) -> None:
        ms = MomentumScalper(settings)
        # Generate one signal to start cooldown
        features = sample_features.model_copy()
        features.momentum_1m = 0.02
        features.recent_trade_flow = 5.0
        first = ms.generate_signal(features, empty_portfolio)
        # Should be in cooldown now
        second = ms.generate_signal(features, empty_portfolio)
        assert second is None

    def test_no_signal_without_momentum(
        self, settings: Settings, empty_portfolio: PortfolioSnapshot
    ) -> None:
        features = MarketFeatures(
            market_id="m", token_id="t",
            best_bid=0.50, best_ask=0.55,
            spread=0.05,
            bid_depth_5c=50, ask_depth_5c=50,
            momentum_1m=0.001,  # below threshold
            recent_trade_flow=0.5,
        )
        ms = MomentumScalper(settings)
        assert ms.generate_signal(features, empty_portfolio) is None
