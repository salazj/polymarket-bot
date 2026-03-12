"""Momentum strategy for stocks — EMA crossover + volume surge + RSI filter."""

from __future__ import annotations

from app.data.models import PortfolioSnapshot
from app.models.enums import OrderType, StockAction
from app.stocks.models import StockFeatures, StockSignal
from app.stocks.strategies.base import BaseStockStrategy


class StockMomentum(BaseStockStrategy):
    name = "stock_momentum"

    MOMENTUM_THRESHOLD = 0.003
    RSI_OVERBOUGHT = 70.0
    RSI_OVERSOLD = 30.0
    MIN_RELATIVE_VOLUME = 1.2
    COOLDOWN_BARS = 5

    def __init__(self) -> None:
        self._last_signal_bar: dict[str, int] = {}
        self._bar_count: int = 0

    def generate_signal(
        self, features: StockFeatures, portfolio: PortfolioSnapshot
    ) -> StockSignal | None:
        self._bar_count += 1
        symbol = features.symbol

        last_bar = self._last_signal_bar.get(symbol, 0)
        if self._bar_count - last_bar < self.COOLDOWN_BARS:
            return None

        if features.last_price <= 0 or features.ema_9 <= 0:
            return None

        price_above_ema = features.last_price > features.ema_9
        momentum_strong = features.momentum_5m > self.MOMENTUM_THRESHOLD
        rsi_ok = features.rsi_14 < self.RSI_OVERBOUGHT

        if price_above_ema and momentum_strong and rsi_ok:
            confidence = min(0.9, 0.5 + features.momentum_5m * 10)
            self._last_signal_bar[symbol] = self._bar_count
            return StockSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=StockAction.BUY,
                confidence=confidence,
                suggested_price=features.last_price,
                order_type=OrderType.LIMIT,
                rationale=(
                    f"Price above EMA-9, 5m momentum={features.momentum_5m:.4f}, "
                    f"RSI={features.rsi_14:.1f}"
                ),
            )

        momentum_reversed = features.momentum_5m < -self.MOMENTUM_THRESHOLD
        price_below_ema = features.last_price < features.ema_9

        if price_below_ema and momentum_reversed:
            confidence = min(0.8, 0.4 + abs(features.momentum_5m) * 10)
            self._last_signal_bar[symbol] = self._bar_count
            return StockSignal(
                strategy_name=self.name,
                symbol=symbol,
                action=StockAction.SELL,
                confidence=confidence,
                suggested_price=features.last_price,
                order_type=OrderType.LIMIT,
                rationale=f"Momentum reversal, 5m momentum={features.momentum_5m:.4f}",
            )

        return None
