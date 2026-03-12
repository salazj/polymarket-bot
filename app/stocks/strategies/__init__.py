"""Stock trading strategies."""

from app.stocks.strategies.base import BaseStockStrategy
from app.stocks.strategies.momentum import StockMomentum
from app.stocks.strategies.mean_reversion import StockMeanReversion
from app.stocks.strategies.breakout import StockBreakout
from app.stocks.strategies.news_gated import NewsGatedWatchlist

ALL_STOCK_STRATEGIES: list[type[BaseStockStrategy]] = [
    StockMomentum,
    StockMeanReversion,
    StockBreakout,
    NewsGatedWatchlist,
]

__all__ = [
    "BaseStockStrategy",
    "StockMomentum",
    "StockMeanReversion",
    "StockBreakout",
    "NewsGatedWatchlist",
    "ALL_STOCK_STRATEGIES",
]
