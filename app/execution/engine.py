"""
Execution Engine

Converts approved signals into orders with comprehensive safety checks.
Manages order lifecycle (state machine), deduplication, and cancel/replace logic.

Invariants enforced:
- No duplicate orders for the same market+side+price
- No orders when data is stale
- No orders if any risk check fails
- All decisions are logged
- Only limit orders (no market orders)
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.clients.trading_client import TradingClient
from app.config.settings import Settings
from app.data.models import (
    MarketFeatures,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Side,
    Signal,
    SignalAction,
)
from app.monitoring import get_logger
from app.monitoring.logger import metrics
from app.utils.helpers import generate_order_id, round_price, round_size, utc_now

if TYPE_CHECKING:
    from app.risk.manager import RiskManager

logger = get_logger(__name__)

# Maps signal actions to order side
ACTION_SIDE_MAP: dict[SignalAction, Side] = {
    SignalAction.BUY_YES: Side.BUY,
    SignalAction.BUY_NO: Side.BUY,
    SignalAction.SELL_YES: Side.SELL,
    SignalAction.SELL_NO: Side.SELL,
}


class ExecutionEngine:
    """
    Processes signals into orders, enforces risk checks, manages order state.
    """

    def __init__(
        self,
        settings: Settings,
        trading_client: TradingClient,
        risk_manager: RiskManager,
    ) -> None:
        self._settings = settings
        self._trading_client = trading_client
        self._risk = risk_manager
        self._active_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._lock = threading.Lock()

    @property
    def active_orders(self) -> list[Order]:
        with self._lock:
            return [o for o in self._active_orders.values() if not o.is_terminal]

    @property
    def all_orders(self) -> list[Order]:
        with self._lock:
            return list(self._order_history)

    async def process_signal(
        self,
        signal: Signal,
        features: MarketFeatures,
        portfolio: PortfolioSnapshot,
    ) -> Order | None:
        """
        Main entry point: validate signal, run risk checks, place order if approved.
        Returns the Order object or None if rejected.
        """
        if signal.action == SignalAction.HOLD:
            return None

        if signal.action == SignalAction.CANCEL_ALL:
            await self.cancel_all_orders()
            return None

        side = ACTION_SIDE_MAP.get(signal.action)
        if side is None:
            logger.warning("unknown_signal_action", action=signal.action)
            return None

        # Validate price and size
        price = round_price(signal.suggested_price or 0.0)
        size = round_size(signal.suggested_size or self._settings.default_order_size)

        if price <= 0 or price >= 1.0:
            logger.warning("invalid_price", price=price, signal=signal.strategy_name)
            metrics.increment("orders_rejected_invalid_price")
            return None

        if size <= 0:
            logger.warning("invalid_size", size=size, signal=signal.strategy_name)
            metrics.increment("orders_rejected_invalid_size")
            return None

        # Deduplication: reject if we already have an active order at same market+side+price
        if self._is_duplicate(signal.token_id, side, price):
            logger.debug("duplicate_order_skipped", token_id=signal.token_id, side=side, price=price)
            metrics.increment("orders_rejected_duplicate")
            return None

        # Risk checks
        risk_result = self._risk.check_order(
            token_id=signal.token_id,
            side=side,
            price=price,
            size=size,
            features=features,
            portfolio=portfolio,
        )

        if not risk_result.approved:
            logger.warning(
                "order_rejected_by_risk",
                reason=risk_result.reason,
                token_id=signal.token_id,
                strategy=signal.strategy_name,
            )
            metrics.increment("orders_rejected_risk")
            return None

        # Build and place order
        order = Order(
            order_id=generate_order_id(),
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=side,
            price=price,
            size=size,
            signal_id=signal.strategy_name,
        )

        logger.info(
            "submitting_order",
            order_id=order.order_id,
            market_id=order.market_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            strategy=signal.strategy_name,
            confidence=signal.confidence,
            rationale=signal.rationale,
        )

        order = await self._trading_client.place_order(order)

        with self._lock:
            self._active_orders[order.order_id] = order
            self._order_history.append(order)

        if order.status == OrderStatus.ACKNOWLEDGED:
            metrics.increment("orders_placed")
        elif order.status == OrderStatus.REJECTED:
            metrics.increment("orders_rejected_exchange")

        return order

    async def cancel_order(self, order_id: str) -> Order | None:
        with self._lock:
            order = self._active_orders.get(order_id)
        if order is None or order.is_terminal:
            return None

        order = await self._trading_client.cancel_order(order)
        logger.info("order_canceled", order_id=order.order_id)
        return order

    async def cancel_all_orders(self) -> int:
        """Cancel all active orders. Returns count of orders canceled."""
        active = self.active_orders
        count = 0
        for order in active:
            result = await self._trading_client.cancel_order(order)
            if result.status == OrderStatus.CANCELED:
                count += 1
        logger.info("canceled_all_orders", count=count)
        return count

    async def cancel_stale_orders(self, max_age_seconds: float = 300) -> int:
        """Cancel orders older than max_age_seconds."""
        now = utc_now()
        count = 0
        for order in self.active_orders:
            age = (now - order.created_at).total_seconds()
            if age > max_age_seconds:
                await self.cancel_order(order.order_id)
                count += 1
        if count:
            logger.info("canceled_stale_orders", count=count, max_age=max_age_seconds)
        return count

    def update_order_status(self, order_id: str, new_status: OrderStatus, filled_size: float = 0) -> None:
        """Update order state from exchange callbacks."""
        with self._lock:
            order = self._active_orders.get(order_id)
            if order is None:
                return
            order.status = new_status
            if filled_size > 0:
                order.filled_size = min(order.filled_size + filled_size, order.size)
            order.updated_at = utc_now()

        logger.info(
            "order_status_updated",
            order_id=order_id,
            status=new_status.value,
            filled=order.filled_size,
        )

    def _is_duplicate(self, token_id: str, side: Side, price: float) -> bool:
        with self._lock:
            for o in self._active_orders.values():
                if (
                    not o.is_terminal
                    and o.token_id == token_id
                    and o.side == side
                    and abs(o.price - price) < 0.001
                ):
                    return True
        return False

    def get_fill_count(self) -> int:
        return sum(
            1 for o in self._order_history
            if o.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
        )
