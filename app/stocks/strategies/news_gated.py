"""News-gated watchlist strategy — only allow entries with bullish NLP signals."""

from __future__ import annotations

from app.data.models import PortfolioSnapshot
from app.models.enums import StockAction
from app.stocks.models import StockFeatures, StockSignal
from app.stocks.strategies.base import BaseStockStrategy


class NewsGatedWatchlist(BaseStockStrategy):
    """
    Filter gate: only permits BUY signals for symbols that have recent
    bullish NLP sentiment. Acts as a pre-filter for other strategy signals.
    """

    name = "stock_news_gated"

    def __init__(self) -> None:
        self._bullish_symbols: set[str] = set()

    def update_sentiment(self, symbol: str, bullish: bool) -> None:
        """Called by the main loop when NLP signals arrive for a stock symbol."""
        if bullish:
            self._bullish_symbols.add(symbol.upper())
        else:
            self._bullish_symbols.discard(symbol.upper())

    def generate_signal(
        self, features: StockFeatures, portfolio: PortfolioSnapshot
    ) -> StockSignal | None:
        if features.symbol.upper() in self._bullish_symbols:
            return StockSignal(
                strategy_name=self.name,
                symbol=features.symbol,
                action=StockAction.BUY,
                confidence=0.3,
                rationale="News sentiment is bullish for this symbol",
            )
        return None
