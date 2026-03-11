"""
Polymarket execution client implementing the BaseExecutionClient interface.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.data.models import Order, OrderStatus, Side
from app.exchanges.base import BaseExecutionClient
from app.monitoring import get_logger
from app.monitoring.logger import metrics
from app.utils.helpers import generate_order_id

logger = get_logger(__name__)


class PolymarketExecutionClient(BaseExecutionClient):
    """Order execution for Polymarket via py-clob-client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._dry_run = settings.dry_run or not settings.enable_live_trading
        self._clob_client: Any = None

        if not self._dry_run:
            self._init_live_client()

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    def _init_live_client(self) -> None:
        self._settings.require_live_trading()
        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                self._settings.polymarket_host,
                key=self._settings.private_key,
                chain_id=self._settings.chain_id,
                creds={
                    "apiKey": self._settings.poly_api_key,
                    "secret": self._settings.poly_api_secret,
                    "passphrase": self._settings.poly_passphrase,
                },
            )
            logger.info("polymarket_live_client_initialized")
        except ImportError:
            raise RuntimeError(
                "py-clob-client is required for live Polymarket trading. "
                "Install with: pip install py-clob-client"
            )

    async def place_order(self, order: Order) -> Order:
        logger.info(
            "placing_order",
            order_id=order.order_id,
            instrument_id=order.instrument_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            dry_run=self._dry_run,
        )

        if self._dry_run:
            return self._simulate_place(order)
        return await self._live_place(order)

    def _simulate_place(self, order: Order) -> Order:
        order.status = OrderStatus.ACKNOWLEDGED
        order.exchange_order_id = f"DRY-{generate_order_id()}"
        metrics.increment("orders_placed_dry")
        logger.info("dry_run_order_placed", order_id=order.order_id, exchange_id=order.exchange_order_id)
        return order

    async def _live_place(self, order: Order) -> Order:
        if self._settings.dry_run or not self._settings.enable_live_trading:
            logger.critical(
                "LIVE_ORDER_BLOCKED_safety_recheck",
                order_id=order.order_id,
                dry_run=self._settings.dry_run,
                enable_live=self._settings.enable_live_trading,
            )
            order.status = OrderStatus.REJECTED
            return order

        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            side = BUY if order.side == Side.BUY else SELL
            signed_order = self._clob_client.create_and_post_order(
                {
                    "token_id": order.instrument_id,
                    "price": order.price,
                    "size": order.size,
                    "side": side,
                }
            )

            if signed_order and signed_order.get("orderID"):
                order.exchange_order_id = signed_order["orderID"]
                order.status = OrderStatus.ACKNOWLEDGED
                metrics.increment("orders_placed_live")
                logger.info("live_order_placed", order_id=order.order_id, exchange_id=order.exchange_order_id)
            else:
                order.status = OrderStatus.REJECTED
                metrics.increment("orders_rejected")
                logger.warning("live_order_rejected", order_id=order.order_id, response=str(signed_order))

        except Exception as e:
            order.status = OrderStatus.REJECTED
            metrics.increment("orders_rejected")
            logger.error("live_order_error", order_id=order.order_id, error=str(e))

        return order

    async def cancel_order(self, order: Order) -> Order:
        logger.info("canceling_order", order_id=order.order_id, exchange_id=order.exchange_order_id)

        if self._dry_run:
            order.status = OrderStatus.CANCELED
            metrics.increment("orders_canceled_dry")
            return order

        if not order.exchange_order_id:
            logger.warning("cannot_cancel_no_exchange_id", order_id=order.order_id)
            return order

        try:
            self._clob_client.cancel(order.exchange_order_id)
            order.status = OrderStatus.CANCELED
            metrics.increment("orders_canceled_live")
            logger.info("live_order_canceled", order_id=order.order_id)
        except Exception as e:
            logger.error("cancel_error", order_id=order.order_id, error=str(e))

        return order

    async def cancel_all(self) -> None:
        logger.info("canceling_all_orders", dry_run=self._dry_run)
        if not self._dry_run and self._clob_client:
            try:
                self._clob_client.cancel_all()
                metrics.increment("cancel_all_invoked")
            except Exception as e:
                logger.error("cancel_all_error", error=str(e))

    async def get_balance(self) -> float:
        if self._dry_run:
            return 0.0
        return 0.0

    async def get_open_positions(self) -> list[dict[str, Any]]:
        if self._dry_run or self._clob_client is None:
            return []
        try:
            return []
        except Exception as e:
            logger.warning("get_positions_failed", error=str(e))
            return []
