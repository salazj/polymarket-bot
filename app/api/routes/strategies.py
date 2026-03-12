"""Strategy listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import StrategyInfo

router = APIRouter(tags=["strategies"])

_STRATEGIES: list[StrategyInfo] = [
    StrategyInfo(
        name="passive_market_maker",
        description="Passive maker quotes when spread is wide enough",
        asset_class="prediction_markets",
    ),
    StrategyInfo(
        name="momentum_scalper",
        description="Short-horizon momentum and trade flow signals",
        asset_class="prediction_markets",
    ),
    StrategyInfo(
        name="event_probability_model",
        description="ML model for short-horizon probability changes",
        asset_class="prediction_markets",
    ),
    StrategyInfo(
        name="sentiment_adapter",
        description="Sentiment-based signal adapter",
        asset_class="prediction_markets",
    ),
    StrategyInfo(
        name="stock_momentum",
        description="EMA crossover + volume surge + RSI filter",
        asset_class="equities",
    ),
    StrategyInfo(
        name="stock_mean_reversion",
        description="Buy below VWAP with low RSI, sell at VWAP",
        asset_class="equities",
    ),
    StrategyInfo(
        name="stock_breakout",
        description="Buy on high-of-day breakout with volume confirmation",
        asset_class="equities",
    ),
    StrategyInfo(
        name="stock_news_gated",
        description="Gate entries on symbols with bullish NLP signals",
        asset_class="equities",
    ),
]


@router.get("/api/strategies", response_model=list[StrategyInfo])
async def list_strategies() -> list[StrategyInfo]:
    return _STRATEGIES
