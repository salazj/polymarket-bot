"""Mean reversion strategy — buy below VWAP with low RSI, sell at VWAP."""

from __future__ import annotations

from app.data.models import PortfolioSnapshot
from app.models.enums import OrderType, StockAction
from app.stocks.models import StockFeatures, StockSignal
from app.stocks.strategies.base import BaseStockStrategy


class StockMeanReversion(BaseStockStrategy):
    name = "stock_mean_reversion"

    RSI_ENTRY = 30.0
    RSI_EXIT = 50.0
    VWAP_DISCOUNT = 0.005

    def generate_signal(
        self, features: StockFeatures, portfolio: PortfolioSnapshot
    ) -> StockSignal | None:
        if features.last_price <= 0 or features.vwap <= 0:
            return None

        below_vwap = features.price_vs_vwap < -features.vwap * self.VWAP_DISCOUNT

        if below_vwap and features.rsi_14 < self.RSI_ENTRY:
            discount_pct = abs(features.price_vs_vwap / features.vwap)
            confidence = min(0.85, 0.4 + discount_pct * 5)
            return StockSignal(
                strategy_name=self.name,
                symbol=features.symbol,
                action=StockAction.BUY,
                confidence=confidence,
                suggested_price=features.last_price,
                order_type=OrderType.LIMIT,
                rationale=(
                    f"Price {discount_pct:.2%} below VWAP, RSI={features.rsi_14:.1f}"
                ),
            )

        above_vwap = features.price_vs_vwap > 0
        if above_vwap and features.rsi_14 > self.RSI_EXIT:
            return StockSignal(
                strategy_name=self.name,
                symbol=features.symbol,
                action=StockAction.SELL,
                confidence=0.6,
                suggested_price=features.last_price,
                order_type=OrderType.LIMIT,
                rationale=f"Price returned to VWAP, RSI={features.rsi_14:.1f}",
            )

        return None
