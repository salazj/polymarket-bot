"""Tests for the execution engine."""

import pytest

from app.clients.trading_client import TradingClient
from app.config.settings import Settings
from app.data.models import (
    MarketFeatures,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Signal,
    SignalAction,
    Side,
)
from app.execution.engine import ExecutionEngine
from app.risk.manager import RiskManager


class TestExecutionEngine:
    @pytest.fixture
    def engine(self, settings: Settings) -> ExecutionEngine:
        tc = TradingClient(settings)
        rm = RiskManager(settings)
        return ExecutionEngine(settings, tc, rm)

    @pytest.mark.asyncio
    async def test_process_valid_signal(
        self,
        engine: ExecutionEngine,
        sample_features: MarketFeatures,
        empty_portfolio: PortfolioSnapshot,
    ) -> None:
        signal = Signal(
            strategy_name="test",
            market_id="test-market",
            token_id="test-token",
            action=SignalAction.BUY_YES,
            confidence=0.7,
            suggested_price=0.52,
            suggested_size=1.0,
        )
        order = await engine.process_signal(signal, sample_features, empty_portfolio)
        assert order is not None
        assert order.status == OrderStatus.ACKNOWLEDGED
        assert order.side == Side.BUY
        assert order.price == 0.52
        assert "DRY-" in (order.exchange_order_id or "")

    @pytest.mark.asyncio
    async def test_rejects_hold_signal(
        self,
        engine: ExecutionEngine,
        sample_features: MarketFeatures,
        empty_portfolio: PortfolioSnapshot,
    ) -> None:
        signal = Signal(
            strategy_name="test",
            market_id="m",
            token_id="t",
            action=SignalAction.HOLD,
            confidence=0.5,
        )
        order = await engine.process_signal(signal, sample_features, empty_portfolio)
        assert order is None

    @pytest.mark.asyncio
    async def test_rejects_invalid_price(
        self,
        engine: ExecutionEngine,
        sample_features: MarketFeatures,
        empty_portfolio: PortfolioSnapshot,
    ) -> None:
        signal = Signal(
            strategy_name="test",
            market_id="m",
            token_id="t",
            action=SignalAction.BUY_YES,
            confidence=0.5,
            suggested_price=0.0,  # invalid
            suggested_size=1.0,
        )
        order = await engine.process_signal(signal, sample_features, empty_portfolio)
        assert order is None

    @pytest.mark.asyncio
    async def test_deduplication(
        self,
        engine: ExecutionEngine,
        sample_features: MarketFeatures,
        empty_portfolio: PortfolioSnapshot,
    ) -> None:
        signal = Signal(
            strategy_name="test",
            market_id="test-market",
            token_id="test-token",
            action=SignalAction.BUY_YES,
            confidence=0.7,
            suggested_price=0.52,
            suggested_size=1.0,
        )
        order1 = await engine.process_signal(signal, sample_features, empty_portfolio)
        assert order1 is not None

        order2 = await engine.process_signal(signal, sample_features, empty_portfolio)
        assert order2 is None  # duplicate

    @pytest.mark.asyncio
    async def test_cancel_order(
        self,
        engine: ExecutionEngine,
        sample_features: MarketFeatures,
        empty_portfolio: PortfolioSnapshot,
    ) -> None:
        signal = Signal(
            strategy_name="test",
            market_id="test-market",
            token_id="test-token",
            action=SignalAction.BUY_YES,
            confidence=0.7,
            suggested_price=0.52,
            suggested_size=1.0,
        )
        order = await engine.process_signal(signal, sample_features, empty_portfolio)
        assert order is not None

        canceled = await engine.cancel_order(order.order_id)
        assert canceled is not None
        assert canceled.status == OrderStatus.CANCELED
