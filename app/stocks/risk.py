"""Stock-specific risk manager with dollar-based limits and market-hours awareness."""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from app.brokers.base import BaseBrokerAdapter
from app.config.settings import Settings
from app.data.models import PortfolioSnapshot
from app.monitoring import get_logger

logger = get_logger(__name__)


@dataclass
class StockRiskCheckResult:
    approved: bool
    reason: str = ""


class StockRiskManager:
    """Pre-trade risk checks for stock trading."""

    def __init__(self, settings: Settings) -> None:
        self._max_position_dollars = settings.stock_max_position_dollars
        self._max_portfolio_dollars = settings.stock_max_portfolio_dollars
        self._max_daily_loss = settings.stock_max_daily_loss_dollars
        self._max_open_positions = settings.stock_max_open_positions
        self._max_orders_per_minute = settings.stock_max_orders_per_minute
        self._allow_extended_hours = settings.allow_extended_hours
        self._emergency_stop_file = settings.emergency_stop_file

        self._circuit_breaker_tripped = False
        self._halt_reason = ""
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._order_timestamps: deque[float] = deque(maxlen=120)
        self._lock = threading.Lock()
        self._cancel_all_callback = None

    @property
    def is_halted(self) -> bool:
        return self._circuit_breaker_tripped or self._emergency_stop_file.exists()

    def set_cancel_all_callback(self, cb) -> None:
        self._cancel_all_callback = cb

    def check_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        portfolio: PortfolioSnapshot,
        *,
        broker: BaseBrokerAdapter | None = None,
    ) -> StockRiskCheckResult:
        with self._lock:
            if self._emergency_stop_file.exists():
                return StockRiskCheckResult(False, "Emergency stop file exists")

            if self._circuit_breaker_tripped:
                return StockRiskCheckResult(False, f"Circuit breaker: {self._halt_reason}")

            if self._daily_pnl < -self._max_daily_loss:
                self.trip_circuit_breaker(f"Daily loss {self._daily_pnl:.2f} exceeds limit")
                return StockRiskCheckResult(False, "Daily loss limit exceeded")

            order_value = price * quantity
            if order_value > self._max_position_dollars:
                return StockRiskCheckResult(
                    False,
                    f"Order ${order_value:.2f} exceeds max position ${self._max_position_dollars:.2f}",
                )

            if portfolio.total_exposure + order_value > self._max_portfolio_dollars:
                return StockRiskCheckResult(False, "Would exceed max portfolio exposure")

            position_count = len(portfolio.positions) if hasattr(portfolio, "positions") else 0
            if position_count >= self._max_open_positions and side.upper() == "BUY":
                return StockRiskCheckResult(
                    False,
                    f"Max open positions ({self._max_open_positions}) reached",
                )

            now = time.time()
            self._order_timestamps.append(now)
            recent = [t for t in self._order_timestamps if now - t < 60]
            if len(recent) > self._max_orders_per_minute:
                return StockRiskCheckResult(False, "Order frequency limit exceeded")

            if broker and not self._allow_extended_hours and not broker.is_market_open():
                return StockRiskCheckResult(False, "Market is closed and extended hours disabled")

            if side.upper() == "BUY" and price * quantity > portfolio.cash:
                return StockRiskCheckResult(False, "Insufficient cash")

            return StockRiskCheckResult(True)

    def record_fill(self, pnl: float) -> None:
        with self._lock:
            self._daily_pnl += pnl
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def trip_circuit_breaker(self, reason: str) -> None:
        self._circuit_breaker_tripped = True
        self._halt_reason = reason
        logger.error("stock_circuit_breaker_tripped", reason=reason)
        if self._cancel_all_callback:
            try:
                import asyncio
                asyncio.create_task(self._cancel_all_callback())
            except RuntimeError:
                pass

    def reset_circuit_breaker(self) -> None:
        self._circuit_breaker_tripped = False
        self._halt_reason = ""
        logger.info("stock_circuit_breaker_reset")

    def reset_daily_counters(self) -> None:
        with self._lock:
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
