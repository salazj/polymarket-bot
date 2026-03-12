"""Breakout strategy — buy on high-of-day breakout with volume confirmation."""

from __future__ import annotations

from app.data.models import PortfolioSnapshot
from app.models.enums import OrderType, StockAction
from app.stocks.models import StockFeatures, StockSignal
from app.stocks.strategies.base import BaseStockStrategy


class StockBreakout(BaseStockStrategy):
    name = "stock_breakout"

    MIN_RANGE_PCT = 0.005
    MIN_VOLUME_MULTIPLE = 1.5

    def generate_signal(
        self, features: StockFeatures, portfolio: PortfolioSnapshot
    ) -> StockSignal | None:
        if features.last_price <= 0 or features.high_of_day <= 0:
            return None

        day_range = features.high_of_day - features.low_of_day
        if features.low_of_day <= 0 or day_range / features.low_of_day < self.MIN_RANGE_PCT:
            return None

        breaking_high = features.last_price >= features.high_of_day * 0.999
        volume_surge = features.relative_volume >= self.MIN_VOLUME_MULTIPLE

        if breaking_high and volume_surge:
            stop_price = features.last_price - features.atr_14 * 1.5 if features.atr_14 > 0 else None
            confidence = min(0.8, 0.5 + features.relative_volume * 0.1)
            return StockSignal(
                strategy_name=self.name,
                symbol=features.symbol,
                action=StockAction.BUY,
                confidence=confidence,
                suggested_price=features.last_price,
                order_type=OrderType.MARKET,
                stop_price=stop_price,
                rationale=(
                    f"High-of-day breakout at {features.last_price:.2f}, "
                    f"volume={features.relative_volume:.1f}x, "
                    f"ATR stop={stop_price:.2f}" if stop_price else "no ATR stop"
                ),
            )

        return None
