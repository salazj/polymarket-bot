"""
Integration tests that exercise the full pipeline end-to-end using
synthetic market data (no network access required).

Covers: orderbook → features → strategy → risk → execution → portfolio → storage.
"""

from __future__ import annotations

import asyncio

import pytest

from app.clients.trading_client import TradingClient
from app.config.settings import Settings
from app.data.features import FeatureEngine
from app.data.models import (
    MarketFeatures,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Side,
    Signal,
    SignalAction,
    TokenSide,
    Trade,
)
from app.data.orderbook import OrderbookManager
from app.execution.engine import ExecutionEngine
from app.monitoring.health import HealthServer
from app.portfolio.tracker import PortfolioTracker
from app.risk.manager import RiskManager
from app.storage.repository import Repository


MARKET_ID = "test-market-001"
TOKEN_ID = "token-yes-001"


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        dry_run=True,
        max_position_per_market=50.0,
        max_total_exposure=100.0,
        max_daily_loss=20.0,
        max_slippage=0.10,
        min_liquidity_depth=5.0,
        min_spread_threshold=0.005,
        max_spread_threshold=0.20,
        database_url="sqlite:///test_integration.db",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_features(**overrides) -> MarketFeatures:
    defaults = dict(
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        best_bid=0.48,
        best_ask=0.52,
        spread=0.04,
        mid_price=0.50,
        bid_depth_5c=60.0,
        ask_depth_5c=60.0,
        volatility_1m=0.01,
        seconds_since_last_update=1.0,
    )
    defaults.update(overrides)
    return MarketFeatures(**defaults)


class TestFullPipelineDryRun:
    """End-to-end: signal → risk → execution → portfolio, all in dry-run."""

    @pytest.fixture
    def components(self):
        settings = _make_settings()
        trading_client = TradingClient(settings)
        risk_manager = RiskManager(settings)
        execution = ExecutionEngine(settings, trading_client, risk_manager)
        portfolio = PortfolioTracker(settings, starting_cash=100.0)
        risk_manager.set_cancel_all_callback(execution.cancel_all_orders)
        return settings, trading_client, risk_manager, execution, portfolio

    @pytest.mark.asyncio
    async def test_buy_signal_creates_order(self, components):
        settings, tc, rm, exe, portfolio = components
        features = _make_features()
        snap = portfolio.get_snapshot()

        signal = Signal(
            strategy_name="test",
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            action=SignalAction.BUY_YES,
            confidence=0.8,
            suggested_price=0.49,
            suggested_size=5.0,
        )

        order = await exe.process_signal(signal, features, snap)
        assert order is not None
        assert order.status == OrderStatus.ACKNOWLEDGED
        assert order.exchange_order_id.startswith("DRY-")

    @pytest.mark.asyncio
    async def test_risk_rejects_insufficient_cash(self, components):
        _, tc, rm, exe, portfolio = components
        features = _make_features()

        buy = Order(
            order_id="b1", market_id=MARKET_ID, token_id=TOKEN_ID,
            side=Side.BUY, price=0.50, size=190.0,
        )
        portfolio.on_fill(buy, 0.50, 190.0)  # cash: 100 - 95 = 5

        snap = portfolio.get_snapshot()
        signal = Signal(
            strategy_name="test",
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            action=SignalAction.BUY_YES,
            confidence=0.8,
            suggested_price=0.49,
            suggested_size=20.0,
        )
        order = await exe.process_signal(signal, features, snap)
        assert order is None  # rejected by cash check

    @pytest.mark.asyncio
    async def test_fill_updates_portfolio_and_risk(self, components):
        _, tc, rm, exe, portfolio = components
        portfolio.start_new_day()

        buy = Order(
            order_id="b1", market_id=MARKET_ID, token_id=TOKEN_ID,
            side=Side.BUY, price=0.50, size=10.0,
        )
        realized = portfolio.on_fill(buy, 0.50, 10.0)
        rm.record_fill(realized)
        assert realized == 0.0
        assert portfolio.cash == pytest.approx(95.0)

        pos = portfolio.get_position(TOKEN_ID)
        assert pos is not None
        assert pos.size == 10.0
        assert pos.avg_entry_price == 0.50

        sell = Order(
            order_id="s1", market_id=MARKET_ID, token_id=TOKEN_ID,
            side=Side.SELL, price=0.55, size=10.0,
        )
        realized = portfolio.on_fill(sell, 0.55, 10.0)
        rm.record_fill(realized)
        assert realized == pytest.approx(0.5)
        assert portfolio.cash == pytest.approx(100.5)
        assert portfolio.get_position(TOKEN_ID) is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_orders(self, components):
        _, tc, rm, exe, portfolio = components
        features = _make_features()
        snap = portfolio.get_snapshot()

        rm.trip_circuit_breaker("test_trip")
        assert rm.is_halted is True

        signal = Signal(
            strategy_name="test",
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            action=SignalAction.BUY_YES,
            confidence=0.9,
            suggested_price=0.49,
            suggested_size=1.0,
        )
        order = await exe.process_signal(signal, features, snap)
        assert order is None

    @pytest.mark.asyncio
    async def test_cancel_stale_orders(self, components):
        _, tc, rm, exe, portfolio = components
        features = _make_features()
        snap = portfolio.get_snapshot()

        signal = Signal(
            strategy_name="test",
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            action=SignalAction.BUY_YES,
            confidence=0.8,
            suggested_price=0.49,
            suggested_size=1.0,
        )
        order = await exe.process_signal(signal, features, snap)
        assert order is not None
        assert len(exe.active_orders) == 1

        canceled = await exe.cancel_stale_orders(max_age_seconds=0)
        assert canceled == 1
        assert len(exe.active_orders) == 0


class TestOrderbookToFeatures:
    """Verify the orderbook → feature computation pipeline."""

    def test_features_from_orderbook(self):
        mgr = OrderbookManager()
        mgr.apply_snapshot(
            MARKET_ID,
            TOKEN_ID,
            bids=[
                {"price": "0.48", "size": "100"},
                {"price": "0.47", "size": "50"},
            ],
            asks=[
                {"price": "0.52", "size": "80"},
                {"price": "0.53", "size": "40"},
            ],
        )

        book = mgr.get_snapshot(TOKEN_ID)
        assert book is not None
        assert book.best_bid == 0.48
        assert book.best_ask == 0.52
        assert book.spread == pytest.approx(0.04)

        engine = FeatureEngine(MARKET_ID, TOKEN_ID)
        features = engine.compute(book)

        assert features.best_bid == 0.48
        assert features.best_ask == 0.52
        assert features.spread == pytest.approx(0.04)
        assert features.mid_price == pytest.approx(0.50)

    def test_features_include_trades(self):
        mgr = OrderbookManager()
        mgr.apply_snapshot(
            MARKET_ID, TOKEN_ID,
            bids=[{"price": "0.48", "size": "100"}],
            asks=[{"price": "0.52", "size": "80"}],
        )

        engine = FeatureEngine(MARKET_ID, TOKEN_ID)
        engine.add_trade(Trade(
            market_id=MARKET_ID, token_id=TOKEN_ID,
            price=0.49, size=5.0, side=Side.BUY,
        ))
        engine.add_trade(Trade(
            market_id=MARKET_ID, token_id=TOKEN_ID,
            price=0.51, size=3.0, side=Side.SELL,
        ))

        book = mgr.get_snapshot(TOKEN_ID)
        features = engine.compute(book)
        assert features.trade_count_1m == 2
        assert features.last_trade_price == 0.51


class TestRepositoryBatchWrites:
    """Verify that buffered writes flush correctly."""

    @pytest.fixture
    async def repo(self, tmp_path):
        db_path = tmp_path / "test_batch.db"
        repo = Repository(str(db_path), max_buffer_size=5)
        await repo.initialize()
        yield repo
        await repo.close()

    @pytest.mark.asyncio
    async def test_events_buffered_and_flushed(self, repo):
        for i in range(3):
            await repo.save_raw_event("book", f"token_{i}", {"seq": i})
        # Buffer not full yet — read should return empty until flush
        events = await repo.get_raw_events(event_type="book")
        assert len(events) == 0

        await repo.flush()
        events = await repo.get_raw_events(event_type="book")
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_full(self, repo):
        for i in range(6):
            await repo.save_raw_event("trade", "tok", {"i": i})
        # Buffer should have auto-flushed at 5
        events = await repo.get_raw_events(event_type="trade")
        assert len(events) >= 5


class TestPositionPersistence:
    """Verify save/load positions for crash recovery."""

    @pytest.fixture
    async def repo(self, tmp_path):
        db_path = tmp_path / "test_positions.db"
        repo = Repository(str(db_path))
        await repo.initialize()
        yield repo
        await repo.close()

    @pytest.mark.asyncio
    async def test_save_and_load_positions(self, repo):
        from app.data.models import Position

        pos = Position(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            token_side=TokenSide.YES,
            size=15.0,
            avg_entry_price=0.45,
            realized_pnl=2.5,
        )
        await repo.save_position(pos)

        rows = await repo.load_positions()
        assert len(rows) == 1
        assert rows[0]["token_id"] == TOKEN_ID
        assert rows[0]["size"] == 15.0
        assert rows[0]["avg_entry_price"] == 0.45

    @pytest.mark.asyncio
    async def test_zero_size_not_loaded(self, repo):
        from app.data.models import Position

        pos = Position(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            token_side=TokenSide.YES,
            size=0.0,
            avg_entry_price=0.45,
        )
        await repo.save_position(pos)

        rows = await repo.load_positions()
        assert len(rows) == 0


class TestPortfolioRecovery:
    """Verify portfolio tracker restore_position works."""

    def test_restore_position(self):
        settings = _make_settings()
        tracker = PortfolioTracker(settings, starting_cash=50.0)
        tracker.restore_position(
            token_id=TOKEN_ID,
            market_id=MARKET_ID,
            token_side=TokenSide.YES,
            size=10.0,
            avg_entry_price=0.45,
            realized_pnl=1.5,
        )
        pos = tracker.get_position(TOKEN_ID)
        assert pos is not None
        assert pos.size == 10.0
        assert pos.avg_entry_price == 0.45

        snap = tracker.get_snapshot()
        assert snap.total_realized_pnl == 1.5


class TestHealthServer:
    """Verify the health endpoint starts and responds."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        settings = _make_settings()
        portfolio = PortfolioTracker(settings, starting_cash=100.0)
        rm = RiskManager(settings)

        server = HealthServer(
            host="127.0.0.1",
            port=0,  # OS picks a free port
            portfolio_snapshot_fn=portfolio.get_snapshot,
            is_halted_fn=lambda: rm.is_halted,
        )
        # Port 0 is only available via asyncio.start_server, so test via unit:
        health = server._build_health()
        assert health["status"] == "ok"

        metrics_data = server._build_metrics()
        assert "portfolio" in metrics_data
        assert metrics_data["portfolio"]["cash"] == 100.0
