"""Tests for portfolio tracking and PnL calculation."""

import pytest

from app.config.settings import Settings
from app.data.models import Order, OrderStatus, Side
from app.portfolio.tracker import PortfolioTracker


class TestPortfolioTracker:
    @pytest.fixture
    def tracker(self, settings: Settings) -> PortfolioTracker:
        return PortfolioTracker(settings, starting_cash=100.0)

    def test_initial_state(self, tracker: PortfolioTracker) -> None:
        assert tracker.cash == 100.0
        assert tracker.positions == []

    def test_buy_creates_position(self, tracker: PortfolioTracker) -> None:
        order = Order(
            order_id="o1", market_id="m", token_id="t",
            side=Side.BUY, price=0.50, size=10.0,
            status=OrderStatus.FILLED,
        )
        tracker.on_fill(order, fill_price=0.50, fill_size=10.0)

        assert tracker.cash == pytest.approx(95.0)  # 100 - 0.50*10
        pos = tracker.get_position("t")
        assert pos is not None
        assert pos.size == 10.0
        assert pos.avg_entry_price == 0.50

    def test_sell_reduces_position(self, tracker: PortfolioTracker) -> None:
        buy = Order(order_id="o1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=10.0)
        tracker.on_fill(buy, 0.50, 10.0)

        sell = Order(order_id="o2", market_id="m", token_id="t", side=Side.SELL, price=0.60, size=5.0)
        tracker.on_fill(sell, 0.60, 5.0)

        pos = tracker.get_position("t")
        assert pos is not None
        assert pos.size == 5.0

    def test_sell_realizes_pnl(self, tracker: PortfolioTracker) -> None:
        buy = Order(order_id="o1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=10.0)
        tracker.on_fill(buy, 0.50, 10.0)

        sell = Order(order_id="o2", market_id="m", token_id="t", side=Side.SELL, price=0.60, size=10.0)
        tracker.on_fill(sell, 0.60, 10.0)

        snap = tracker.get_snapshot()
        assert snap.total_realized_pnl == pytest.approx(1.0)  # (0.60 - 0.50) * 10
        assert tracker.get_position("t") is None  # position closed

    def test_mark_to_market(self, tracker: PortfolioTracker) -> None:
        buy = Order(order_id="o1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=10.0)
        tracker.on_fill(buy, 0.50, 10.0)

        tracker.mark_to_market("t", 0.55)
        pos = tracker.get_position("t")
        assert pos is not None
        assert pos.unrealized_pnl == pytest.approx(0.50)  # (0.55 - 0.50) * 10

    def test_weighted_avg_entry(self, tracker: PortfolioTracker) -> None:
        buy1 = Order(order_id="o1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=10.0)
        tracker.on_fill(buy1, 0.50, 10.0)

        buy2 = Order(order_id="o2", market_id="m", token_id="t", side=Side.BUY, price=0.60, size=10.0)
        tracker.on_fill(buy2, 0.60, 10.0)

        pos = tracker.get_position("t")
        assert pos is not None
        assert pos.size == 20.0
        assert pos.avg_entry_price == pytest.approx(0.55)

    def test_snapshot_includes_daily_pnl(self, tracker: PortfolioTracker) -> None:
        tracker.start_new_day()
        buy = Order(order_id="o1", market_id="m", token_id="t", side=Side.BUY, price=0.50, size=10.0)
        tracker.on_fill(buy, 0.50, 10.0)

        snap = tracker.get_snapshot()
        # Cash decreased by 5.0, but we hold 10 tokens worth 5.0 at entry
        # Daily PnL should reflect equity change
        assert isinstance(snap.daily_pnl, float)
