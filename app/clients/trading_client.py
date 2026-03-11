"""
Trading client wrapper for order placement, cancellation, and account queries.

In DRY_RUN mode, orders are simulated locally with no exchange interaction.
In LIVE mode, delegates to the py-clob-client for signed order submission.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.data.models import Order, OrderStatus, Side
from app.monitoring import get_logger
from app.monitoring.logger import metrics
from app.utils.helpers import epoch_ms, generate_order_id

logger = get_logger(__name__)


class TradingClient:
    """
    Abstraction over Polymarket CLOB order operations.

    Supports dry-run (simulated) and live modes behind the same interface.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Resolve to dry-run unless BOTH flags are set (defense in depth)
        self._dry_run = settings.dry_run or not settings.enable_live_trading
        self._clob_client: Any = None

        if not self._dry_run:
            self._init_live_client()

    def _init_live_client(self) -> None:
        """Initialize the py-clob-client for live trading."""
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
            logger.info("live_trading_client_initialized")
        except ImportError:
            raise RuntimeError(
                "py-clob-client is required for live trading. "
                "Install with: pip install py-clob-client"
            )

    async def place_order(self, order: Order) -> Order:
        """
        Place a limit order. In dry-run mode, immediately acknowledges.
        Returns the order with updated status and exchange ID.
        """
        logger.info(
            "placing_order",
            order_id=order.order_id,
            token_id=order.token_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            dry_run=self._dry_run,
        )

        if self._dry_run:
            return self._simulate_place(order)

        return await self._live_place(order)

    def _simulate_place(self, order: Order) -> Order:
        """Dry-run: mark order as acknowledged with a fake exchange ID."""
        order.status = OrderStatus.ACKNOWLEDGED
        order.exchange_order_id = f"DRY-{generate_order_id()}"
        metrics.increment("orders_placed_dry")
        logger.info("dry_run_order_placed", order_id=order.order_id, exchange_id=order.exchange_order_id)
        return order

    async def _live_place(self, order: Order) -> Order:
        """Live: submit signed order via py-clob-client."""
        # SAFETY: re-verify live trading is truly enabled at submission time.
        # Catches any state where dry_run was changed after init.
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
                    "token_id": order.token_id,
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
        """Cancel an active order."""
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
        """Cancel all open orders."""
        logger.info("canceling_all_orders", dry_run=self._dry_run)
        if not self._dry_run and self._clob_client:
            try:
                self._clob_client.cancel_all()
                metrics.increment("cancel_all_invoked")
            except Exception as e:
                logger.error("cancel_all_error", error=str(e))

    async def get_balance(self) -> float:
        """Get USDC balance. Returns 0 in dry-run mode (tracked by portfolio)."""
        if self._dry_run:
            return 0.0
        # TODO: Implement balance query via CLOB client or on-chain read
        return 0.0

    async def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch open positions from the exchange for reconciliation.

        Returns list of dicts with keys: token_id, size, avg_entry_price.
        Returns empty list in dry-run mode.
        """
        if self._dry_run or self._clob_client is None:
            return []
        try:
            # TODO: Replace with actual CLOB client positions endpoint when available.
            # The py-clob-client does not currently expose a positions endpoint natively;
            # this would call the Polymarket REST /positions endpoint.
            return []
        except Exception as e:
            logger.warning("get_positions_failed", error=str(e))
            return []
