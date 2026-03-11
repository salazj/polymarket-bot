"""Tests for the storage repository."""

import pytest

from app.data.models import Market, MarketFeatures, MarketToken, Order, OrderStatus, Side, Signal, SignalAction
from app.storage.repository import Repository


@pytest.fixture
async def repo(tmp_path) -> Repository:
    db_path = str(tmp_path / "test.db")
    r = Repository(db_path)
    await r.initialize()
    yield r
    await r.close()


class TestRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_market(self, repo: Repository) -> None:
        market = Market(
            condition_id="abc123",
            question="Will it rain?",
            slug="will-it-rain",
            tokens=[MarketToken(token_id="tok1", outcome="Yes")],
        )
        await repo.save_market(market)
        markets = await repo.get_markets()
        assert len(markets) == 1
        assert markets[0]["condition_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_save_raw_event(self, repo: Repository) -> None:
        await repo.save_raw_event("book", "tok1", {"bids": [], "asks": []})
        await repo.flush()  # buffered writes need explicit flush
        events = await repo.get_raw_events(token_id="tok1")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_save_and_get_features(self, repo: Repository) -> None:
        features = MarketFeatures(
            market_id="m1", token_id="t1",
            spread=0.05, mid_price=0.50,
        )
        await repo.save_features(features)
        await repo.flush()  # buffered writes need explicit flush
        rows = await repo.get_features("t1")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_save_signal(self, repo: Repository) -> None:
        signal = Signal(
            strategy_name="test",
            market_id="m1",
            token_id="t1",
            action=SignalAction.BUY_YES,
            confidence=0.7,
        )
        await repo.save_signal(signal)

    @pytest.mark.asyncio
    async def test_save_and_get_order(self, repo: Repository) -> None:
        order = Order(
            order_id="ord1",
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.50,
            size=1.0,
            status=OrderStatus.ACKNOWLEDGED,
        )
        await repo.save_order(order)
        orders = await repo.get_orders()
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ord1"

    @pytest.mark.asyncio
    async def test_save_pnl_snapshot(self, repo: Repository) -> None:
        await repo.save_pnl_snapshot(
            cash=100.0,
            total_exposure=10.0,
            total_unrealized=0.5,
            total_realized=1.0,
            daily_pnl=0.5,
        )
        history = await repo.get_pnl_history()
        assert len(history) == 1
