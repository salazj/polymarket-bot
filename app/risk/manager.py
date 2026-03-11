"""
Risk Manager

Enforces all risk limits before any order is placed. This is the last line
of defense between a signal and actual order submission.

Guards implemented:
1. Max exposure per market
2. Max total portfolio exposure
3. Max daily loss (realized + unrealized)
4. Max consecutive losses
5. Max order frequency (orders per minute)
6. Stale market data lockout
7. Volatility regime lockout
8. Slippage guard
9. Spread guard
10. Liquidity depth guard
11. Circuit breaker kill switch
12. Emergency stop file

ALL checks must pass for an order to proceed.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Coroutine, Any

from app.config.settings import Settings
from app.data.models import MarketFeatures, PortfolioSnapshot, Side
from app.monitoring import get_logger
from app.monitoring.logger import metrics

logger = get_logger(__name__)

# Callback type for auto-cancel on circuit breaker trip.
# Returns number of orders canceled. Set by ExecutionEngine after construction.
CancelAllCallback = Callable[[], Coroutine[Any, Any, int]]


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str = ""


class RiskManager:
    """
    Comprehensive pre-trade risk checks.
    Thread-safe. Every check logs its decision for auditability.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._circuit_breaker_tripped = False
        self._consecutive_losses = 0
        self._order_timestamps: deque[float] = deque(maxlen=1000)
        self._daily_realized_pnl = 0.0
        self._lock = threading.Lock()
        self._cancel_all_callback: CancelAllCallback | None = None

    def set_cancel_all_callback(self, cb: CancelAllCallback) -> None:
        """Register the execution engine's cancel-all function for auto-cancel on trip."""
        self._cancel_all_callback = cb

    # ── Public API ─────────────────────────────────────────────────────

    def check_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        features: MarketFeatures,
        portfolio: PortfolioSnapshot,
    ) -> RiskCheckResult:
        """Run all risk checks. Returns approved=True only if ALL pass."""
        checks = [
            self._check_emergency_stop(),
            self._check_circuit_breaker(),
            self._check_daily_loss(portfolio),
            self._check_consecutive_losses(),
            self._check_order_frequency(),
            self._check_stale_data(features),
            self._check_spread(features),
            self._check_liquidity(features),
            self._check_slippage(price, features),
            self._check_cash_sufficiency(side, price, size, portfolio),
            self._check_market_exposure(token_id, size, portfolio),
            self._check_total_exposure(size, portfolio),
            self._check_volatility(features),
        ]

        for result in checks:
            if not result.approved:
                metrics.increment("risk_rejections")
                logger.warning("risk_check_failed", reason=result.reason, token_id=token_id)
                return result

        # Record the order timestamp for frequency tracking
        with self._lock:
            self._order_timestamps.append(time.time())

        return RiskCheckResult(approved=True)

    def record_fill(self, pnl: float) -> None:
        """Called after a fill to update loss tracking."""
        with self._lock:
            self._daily_realized_pnl += pnl
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

        if self._consecutive_losses >= self._settings.max_consecutive_losses:
            self.trip_circuit_breaker("max_consecutive_losses_reached")

    def trip_circuit_breaker(self, reason: str) -> None:
        """
        Activate the kill switch and attempt to cancel all open orders.
        Requires manual reset to resume trading.
        """
        with self._lock:
            self._circuit_breaker_tripped = True
        logger.critical("CIRCUIT_BREAKER_TRIPPED", reason=reason)
        metrics.increment("circuit_breaker_trips")
        self._fire_cancel_all(reason)

    def _fire_cancel_all(self, reason: str) -> None:
        """Best-effort async cancel-all via registered callback."""
        if self._cancel_all_callback is None:
            return
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(self._safe_cancel_all(reason))
        else:
            logger.warning("cancel_all_skipped_no_event_loop", reason=reason)

    async def _safe_cancel_all(self, reason: str) -> None:
        try:
            assert self._cancel_all_callback is not None
            count = await self._cancel_all_callback()
            logger.warning("circuit_breaker_auto_canceled", count=count, reason=reason)
        except Exception as exc:
            logger.error("circuit_breaker_cancel_failed", error=str(exc))

    def reset_circuit_breaker(self) -> None:
        with self._lock:
            self._circuit_breaker_tripped = False
        logger.info("circuit_breaker_reset")

    def reset_daily_counters(self) -> None:
        """Call at the start of each trading day."""
        with self._lock:
            self._daily_realized_pnl = 0.0
            self._consecutive_losses = 0
        logger.info("daily_risk_counters_reset")

    @property
    def is_halted(self) -> bool:
        return self._circuit_breaker_tripped or self._emergency_stop_exists()

    # ── Individual Checks ──────────────────────────────────────────────

    def _check_emergency_stop(self) -> RiskCheckResult:
        """Check for EMERGENCY_STOP file on disk."""
        if self._emergency_stop_exists():
            return RiskCheckResult(False, "EMERGENCY_STOP file detected")
        return RiskCheckResult(True)

    def _emergency_stop_exists(self) -> bool:
        return self._settings.emergency_stop_file.exists()

    def _check_circuit_breaker(self) -> RiskCheckResult:
        if self._circuit_breaker_tripped:
            return RiskCheckResult(False, "circuit_breaker_active")
        return RiskCheckResult(True)

    def _check_daily_loss(self, portfolio: PortfolioSnapshot) -> RiskCheckResult:
        """
        Uses portfolio.daily_pnl which is equity-based (cash + positions - start_equity).
        This already incorporates both realized and unrealized PnL, so we do NOT
        add self._daily_realized_pnl again (that would double-count).
        """
        daily = portfolio.daily_pnl
        if daily < -self._settings.max_daily_loss:
            self.trip_circuit_breaker(f"daily_loss_exceeded: {daily:.2f}")
            return RiskCheckResult(False, f"daily_loss={daily:.2f} exceeds limit")
        return RiskCheckResult(True)

    def _check_consecutive_losses(self) -> RiskCheckResult:
        with self._lock:
            if self._consecutive_losses >= self._settings.max_consecutive_losses:
                return RiskCheckResult(
                    False,
                    f"consecutive_losses={self._consecutive_losses}",
                )
        return RiskCheckResult(True)

    def _check_order_frequency(self) -> RiskCheckResult:
        """Enforce max orders per minute."""
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            recent = sum(1 for t in self._order_timestamps if t > cutoff)
        if recent >= self._settings.max_orders_per_minute:
            return RiskCheckResult(False, f"order_rate={recent}/min exceeds limit")
        return RiskCheckResult(True)

    def _check_stale_data(self, features: MarketFeatures) -> RiskCheckResult:
        if features.seconds_since_last_update > 30:
            return RiskCheckResult(
                False,
                f"stale_data={features.seconds_since_last_update:.0f}s",
            )
        return RiskCheckResult(True)

    def _check_spread(self, features: MarketFeatures) -> RiskCheckResult:
        if features.spread is None:
            return RiskCheckResult(False, "no_spread_data")
        if features.spread > self._settings.max_spread_threshold:
            return RiskCheckResult(False, f"spread={features.spread:.4f} too wide")
        if features.spread < self._settings.min_spread_threshold:
            return RiskCheckResult(False, f"spread={features.spread:.4f} too tight")
        return RiskCheckResult(True)

    def _check_liquidity(self, features: MarketFeatures) -> RiskCheckResult:
        total_depth = features.bid_depth_5c + features.ask_depth_5c
        if total_depth < self._settings.min_liquidity_depth:
            return RiskCheckResult(False, f"liquidity_depth={total_depth:.1f} below minimum")
        return RiskCheckResult(True)

    def _check_slippage(self, price: float, features: MarketFeatures) -> RiskCheckResult:
        """Ensure our price isn't too far from the midpoint."""
        if features.mid_price is None:
            return RiskCheckResult(True)  # can't check, allow
        deviation = abs(price - features.mid_price)
        if deviation > self._settings.max_slippage:
            return RiskCheckResult(False, f"slippage={deviation:.4f} exceeds limit")
        return RiskCheckResult(True)

    def _check_cash_sufficiency(
        self, side: Side, price: float, size: float, portfolio: PortfolioSnapshot
    ) -> RiskCheckResult:
        """Reject BUY orders that would exceed available cash."""
        if side != Side.BUY:
            return RiskCheckResult(True)
        cost = price * size
        if cost > portfolio.cash:
            return RiskCheckResult(
                False,
                f"insufficient_cash: need={cost:.2f} have={portfolio.cash:.2f}",
            )
        return RiskCheckResult(True)

    def _check_market_exposure(
        self, token_id: str, size: float, portfolio: PortfolioSnapshot
    ) -> RiskCheckResult:
        """
        Compare in consistent units: existing notional (size*entry_price) for
        open positions, plus the token count of the proposed order.
        max_position_per_market is denominated in token count (shares).
        """
        existing_size = sum(
            p.size for p in portfolio.positions if p.token_id == token_id
        )
        proposed = existing_size + size
        if proposed > self._settings.max_position_per_market:
            return RiskCheckResult(
                False,
                f"market_exposure={proposed:.2f} shares exceeds {self._settings.max_position_per_market}",
            )
        return RiskCheckResult(True)

    def _check_total_exposure(self, size: float, portfolio: PortfolioSnapshot) -> RiskCheckResult:
        if portfolio.total_exposure + size > self._settings.max_total_exposure:
            return RiskCheckResult(
                False,
                f"total_exposure={portfolio.total_exposure + size:.2f} exceeds limit",
            )
        return RiskCheckResult(True)

    def _check_volatility(self, features: MarketFeatures) -> RiskCheckResult:
        """Lock out during extremely high short-term volatility."""
        if features.volatility_1m > 0.10:
            return RiskCheckResult(False, f"volatility_1m={features.volatility_1m:.4f} too high")
        return RiskCheckResult(True)
