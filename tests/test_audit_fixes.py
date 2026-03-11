"""
Tests for issues found during the code audit.

Each test class maps to a specific audit finding.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.clients.trading_client import TradingClient
from app.config.settings import Settings
from app.data.models import (
    MarketFeatures,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    Side,
    Signal,
    SignalAction,
    TokenSide,
    OrderbookSnapshot,
    PriceLevel,
)
from app.data.orderbook import OrderbookManager
from app.execution.engine import ExecutionEngine
from app.portfolio.tracker import PortfolioTracker
from app.risk.manager import RiskManager


# ── A1: ENABLE_LIVE_TRADING guard ──────────────────────────────────────────


class TestLiveTradingGuard:
    def test_dry_run_is_default(self) -> None:
        s = Settings()
        assert s.dry_run is True
        assert s.enable_live_trading is False
        assert s.is_live is False

    def test_is_live_requires_both_flags(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=False)
        assert s.is_live is False

    def test_is_live_true_only_when_all_three_set(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=True,
                     private_key="x", poly_api_key="k", poly_api_secret="s")
        assert s.is_live is True

    def test_require_live_trading_raises_without_enable(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=False)
        with pytest.raises(RuntimeError, match="ENABLE_LIVE_TRADING"):
            s.require_live_trading()

    def test_require_live_trading_raises_without_acknowledged(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=False)
        with pytest.raises(RuntimeError, match="LIVE_TRADING_ACKNOWLEDGED"):
            s.require_live_trading()

    def test_require_live_trading_raises_without_creds(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=True)
        with pytest.raises(RuntimeError, match="PRIVATE_KEY"):
            s.require_live_trading()

    def test_trading_client_defaults_to_dry_run(self) -> None:
        """TradingClient resolves to dry-run unless both flags are set."""
        s = Settings(dry_run=False, enable_live_trading=False)
        tc = TradingClient(s)
        assert tc._dry_run is True

    def test_trading_client_dry_run_when_dry_run_true(self) -> None:
        s = Settings(dry_run=True, enable_live_trading=True)
        tc = TradingClient(s)
        assert tc._dry_run is True

    @pytest.mark.asyncio
    async def test_dry_run_order_gets_dry_prefix(self) -> None:
        s = Settings(dry_run=True)
        tc = TradingClient(s)
        order = Order(
            order_id="test", market_id="m", token_id="t",
            side=Side.BUY, price=0.50, size=1.0,
        )
        result = await tc.place_order(order)
        assert result.exchange_order_id is not None
        assert result.exchange_order_id.startswith("DRY-")


# ── A2: PnL accounting ────────────────────────────────────────────────────


class TestPnLAccounting:
    @pytest.fixture
    def tracker(self) -> PortfolioTracker:
        return PortfolioTracker(Settings(), starting_cash=100.0)

    def test_on_fill_returns_realized_pnl(self, tracker: PortfolioTracker) -> None:
        buy = Order(order_id="b1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=10.0)
        pnl_buy = tracker.on_fill(buy, 0.50, 10.0)
        assert pnl_buy == 0.0

        sell = Order(order_id="s1", market_id="m", token_id="t", side=Side.SELL, price=0.60, size=10.0)
        pnl_sell = tracker.on_fill(sell, 0.60, 10.0)
        assert pnl_sell == pytest.approx(1.0)

    def test_partial_sell_pnl_correct(self, tracker: PortfolioTracker) -> None:
        buy = Order(order_id="b1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=20.0)
        tracker.on_fill(buy, 0.50, 20.0)

        sell = Order(order_id="s1", market_id="m", token_id="t", side=Side.SELL, price=0.60, size=10.0)
        pnl = tracker.on_fill(sell, 0.60, 10.0)
        assert pnl == pytest.approx(1.0)

        pos = tracker.get_position("t")
        assert pos is not None
        assert pos.size == 10.0
        assert pos.avg_entry_price == 0.50

    def test_sell_without_position_returns_zero(self, tracker: PortfolioTracker) -> None:
        sell = Order(order_id="s1", market_id="m", token_id="t", side=Side.SELL, price=0.60, size=5.0)
        pnl = tracker.on_fill(sell, 0.60, 5.0)
        assert pnl == 0.0

    def test_cash_goes_negative_with_warning(self, tracker: PortfolioTracker) -> None:
        """Cash can go negative (logged as warning) — caller is responsible for pre-checks."""
        buy = Order(order_id="b1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=300.0)
        tracker.on_fill(buy, 0.50, 300.0)
        assert tracker.cash < 0


# ── A3: Orderbook snapshot immutability ────────────────────────────────────


class TestOrderbookSnapshotImmutability:
    def test_get_snapshot_returns_copy(self) -> None:
        mgr = OrderbookManager()
        mgr.apply_snapshot("m", "t", [{"price": "0.50", "size": "100"}], [{"price": "0.55", "size": "80"}])

        snap1 = mgr.get_snapshot("t")
        snap2 = mgr.get_snapshot("t")
        assert snap1 is not snap2
        assert snap1.bids is not snap2.bids

    def test_mutating_snapshot_does_not_affect_internal(self) -> None:
        mgr = OrderbookManager()
        mgr.apply_snapshot("m", "t", [{"price": "0.50", "size": "100"}], [{"price": "0.55", "size": "80"}])

        snap = mgr.get_snapshot("t")
        assert snap is not None
        snap.bids.clear()

        internal = mgr.get_snapshot("t")
        assert internal is not None
        assert len(internal.bids) == 1


# ── A7: Config cross-field validation ─────────────────────────────────────


class TestConfigCrossValidation:
    def test_min_spread_must_be_less_than_max(self) -> None:
        with pytest.raises(Exception):
            Settings(min_spread_threshold=0.20, max_spread_threshold=0.10)

    def test_max_exposure_must_exceed_per_market(self) -> None:
        with pytest.raises(Exception):
            Settings(max_position_per_market=100.0, max_total_exposure=50.0)

    def test_valid_spread_thresholds_accepted(self) -> None:
        s = Settings(min_spread_threshold=0.01, max_spread_threshold=0.15)
        assert s.min_spread_threshold < s.max_spread_threshold


# ── A4: Risk exposure check units ─────────────────────────────────────────


class TestRiskExposureUnits:
    def test_market_exposure_uses_token_count(self) -> None:
        """Exposure check should compare token counts, not notional values."""
        settings = Settings(max_position_per_market=10.0)
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05, bid_depth_5c=50, ask_depth_5c=50,
            seconds_since_last_update=1,
        )
        portfolio = PortfolioSnapshot(
            cash=50.0,
            positions=[
                Position(
                    market_id="m", token_id="t",
                    token_side=TokenSide.YES, size=8.0, avg_entry_price=0.50,
                )
            ],
        )
        # 8 existing + 3 new = 11 > 10 limit
        result = rm.check_order("t", Side.BUY, 0.50, 3.0, features, portfolio)
        assert result.approved is False
        assert "market_exposure" in result.reason

        # 8 existing + 1 new = 9 <= 10 limit
        result2 = rm.check_order("t", Side.BUY, 0.50, 1.0, features, portfolio)
        assert result2.approved is True


# ── A9: Secret redaction ──────────────────────────────────────────────────


class TestSecretRedaction:
    def test_repr_redacts_private_key(self) -> None:
        s = Settings(private_key="0xdeadbeef123456", poly_api_key="secret_key")
        r = repr(s)
        assert "0xdeadbeef123456" not in r
        assert "secret_key" not in r
        assert "***" in r


# ── A10: Daily loss not double-counted ────────────────────────────────────


class TestDailyLossAccounting:
    def test_daily_loss_uses_portfolio_pnl_only(self) -> None:
        """Verify the risk manager doesn't double-count daily losses."""
        settings = Settings(max_daily_loss=5.0)
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05, bid_depth_5c=50, ask_depth_5c=50,
            seconds_since_last_update=1,
        )
        # Simulate portfolio with -4.5 daily PnL (within limit)
        portfolio = PortfolioSnapshot(cash=95.5, daily_pnl=-4.5)
        result = rm.check_order("t", Side.BUY, 0.50, 1.0, features, portfolio)
        assert result.approved is True

        # Simulate portfolio with -5.1 daily PnL (exceeds limit)
        portfolio2 = PortfolioSnapshot(cash=94.9, daily_pnl=-5.1)
        result2 = rm.check_order("t", Side.BUY, 0.50, 1.0, features, portfolio2)
        assert result2.approved is False


# ── Additional edge cases ─────────────────────────────────────────────────


class TestOrderDeduplication:
    @pytest.fixture
    def engine(self) -> ExecutionEngine:
        s = Settings()
        tc = TradingClient(s)
        rm = RiskManager(s)
        return ExecutionEngine(s, tc, rm)

    @pytest.mark.asyncio
    async def test_different_prices_not_duplicate(self, engine: ExecutionEngine) -> None:
        features = MarketFeatures(
            market_id="m", token_id="t",
            best_bid=0.50, best_ask=0.55, spread=0.05,
            mid_price=0.525, bid_depth_5c=100, ask_depth_5c=100,
            seconds_since_last_update=1,
        )
        portfolio = PortfolioSnapshot(cash=100.0)

        sig1 = Signal(strategy_name="test", market_id="m", token_id="t",
                      action=SignalAction.BUY_YES, confidence=0.7,
                      suggested_price=0.50, suggested_size=1.0)
        sig2 = Signal(strategy_name="test", market_id="m", token_id="t",
                      action=SignalAction.BUY_YES, confidence=0.7,
                      suggested_price=0.51, suggested_size=1.0)

        o1 = await engine.process_signal(sig1, features, portfolio)
        o2 = await engine.process_signal(sig2, features, portfolio)
        assert o1 is not None
        assert o2 is not None  # different price, not a duplicate


class TestSlippageGuard:
    def test_slippage_rejects_far_price(self) -> None:
        settings = Settings(max_slippage=0.02)
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05, mid_price=0.50,
            bid_depth_5c=50, ask_depth_5c=50,
            seconds_since_last_update=1,
        )
        result = rm.check_order("t", Side.BUY, 0.53, 1.0, features, PortfolioSnapshot(cash=100))
        assert result.approved is False
        assert "slippage" in result.reason

    def test_slippage_allows_close_price(self) -> None:
        settings = Settings(max_slippage=0.03)
        rm = RiskManager(settings)
        features = MarketFeatures(
            market_id="m", token_id="t",
            spread=0.05, mid_price=0.50,
            bid_depth_5c=50, ask_depth_5c=50,
            seconds_since_last_update=1,
        )
        result = rm.check_order("t", Side.BUY, 0.52, 1.0, features, PortfolioSnapshot(cash=100))
        assert result.approved is True
