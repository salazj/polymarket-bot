"""Tests for risk management checks."""

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.data.models import MarketFeatures, PortfolioSnapshot, Position, Side, TokenSide
from app.risk.manager import RiskManager


class TestRiskManager:
    def test_approves_valid_order(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot
    ) -> None:
        rm = RiskManager(settings)
        result = rm.check_order("test-token", Side.BUY, 0.52, 1.0, sample_features, empty_portfolio)
        assert result.approved is True

    def test_rejects_stale_data(self, settings: Settings, empty_portfolio: PortfolioSnapshot) -> None:
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05, bid_depth_5c=50, ask_depth_5c=50,
            seconds_since_last_update=120,
        )
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, features, empty_portfolio)
        assert result.approved is False
        assert "stale" in result.reason

    def test_rejects_wide_spread(self, settings: Settings, empty_portfolio: PortfolioSnapshot) -> None:
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.25,  # wider than max_spread_threshold
            bid_depth_5c=50, ask_depth_5c=50,
            seconds_since_last_update=1,
        )
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, features, empty_portfolio)
        assert result.approved is False
        assert "spread" in result.reason

    def test_rejects_low_liquidity(self, settings: Settings, empty_portfolio: PortfolioSnapshot) -> None:
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05,
            bid_depth_5c=2.0, ask_depth_5c=2.0,  # below threshold
            seconds_since_last_update=1,
        )
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, features, empty_portfolio)
        assert result.approved is False
        assert "liquidity" in result.reason

    def test_rejects_over_exposure(self, settings: Settings, sample_features: MarketFeatures) -> None:
        rm = RiskManager(settings)
        portfolio = PortfolioSnapshot(
            cash=10.0,
            total_exposure=49.0,  # near limit
        )
        result = rm.check_order("t", Side.BUY, 0.50, 5.0, sample_features, portfolio)
        assert result.approved is False
        assert "total_exposure" in result.reason

    def test_rejects_over_market_exposure(
        self, settings: Settings, sample_features: MarketFeatures
    ) -> None:
        rm = RiskManager(settings)
        # token_id matches the order token_id and position token_id
        # notional = size * avg_entry = 9.0 * 0.50 = 4.50, plus new size 7.0 = 11.50 > 10.0
        portfolio = PortfolioSnapshot(
            cash=50.0,
            positions=[
                Position(
                    market_id="m", token_id="test-token",
                    token_side=TokenSide.YES, size=9.0, avg_entry_price=0.50,
                )
            ],
        )
        result = rm.check_order("test-token", Side.BUY, 0.50, 7.0, sample_features, portfolio)
        assert result.approved is False
        assert "market_exposure" in result.reason

    def test_circuit_breaker(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot
    ) -> None:
        rm = RiskManager(settings)
        rm.trip_circuit_breaker("test")
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, sample_features, empty_portfolio)
        assert result.approved is False
        assert "circuit_breaker" in result.reason

    def test_circuit_breaker_reset(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot
    ) -> None:
        rm = RiskManager(settings)
        rm.trip_circuit_breaker("test")
        rm.reset_circuit_breaker()
        result = rm.check_order("t", Side.BUY, 0.52, 1.0, sample_features, empty_portfolio)
        assert result.approved is True

    def test_consecutive_losses_trigger(self, settings: Settings) -> None:
        settings.max_consecutive_losses = 3
        rm = RiskManager(settings)
        rm.record_fill(-1.0)
        rm.record_fill(-1.0)
        rm.record_fill(-1.0)
        assert rm.is_halted is True

    def test_order_frequency_limit(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot
    ) -> None:
        settings.max_orders_per_minute = 3
        rm = RiskManager(settings)
        for _ in range(3):
            rm.check_order("t", Side.BUY, 0.52, 1.0, sample_features, empty_portfolio)
        result = rm.check_order("t", Side.BUY, 0.52, 1.0, sample_features, empty_portfolio)
        assert result.approved is False
        assert "order_rate" in result.reason

    def test_emergency_stop_file(
        self, settings: Settings, sample_features: MarketFeatures, empty_portfolio: PortfolioSnapshot, tmp_path: Path
    ) -> None:
        stop_file = tmp_path / "EMERGENCY_STOP"
        stop_file.touch()
        settings.emergency_stop_file = stop_file
        rm = RiskManager(settings)
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, sample_features, empty_portfolio)
        assert result.approved is False
        assert "EMERGENCY_STOP" in result.reason

    def test_high_volatility_lockout(
        self, settings: Settings, empty_portfolio: PortfolioSnapshot
    ) -> None:
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05, bid_depth_5c=50, ask_depth_5c=50,
            volatility_1m=0.15,  # very high
            seconds_since_last_update=1,
        )
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, features, empty_portfolio)
        assert result.approved is False
        assert "volatility" in result.reason
