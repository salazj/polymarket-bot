"""
Sentiment Adapter Strategy (Stub)

Architecture for future integration of sentiment signals from headlines,
social media, or other text sources. Provides an abstract provider interface
so multiple backends (free or paid) can be plugged in later.

Currently generates no signals — it is a scaffold for extension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.config.settings import Settings
from app.data.models import MarketFeatures, PortfolioSnapshot, Signal
from app.monitoring import get_logger
from app.strategies.base import BaseStrategy, StrategyRegistry
from app.utils.helpers import utc_now

logger = get_logger(__name__)


# ── Sentiment Provider Interface ───────────────────────────────────────────


class SentimentScore(BaseModel):
    """Standardized sentiment output from any provider."""

    market_id: str
    score: float = Field(ge=-1.0, le=1.0)  # -1 bearish to +1 bullish
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = ""
    headline: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class SentimentProvider(ABC):
    """Abstract interface for sentiment data sources."""

    @abstractmethod
    async def get_sentiment(self, market_id: str, query: str) -> list[SentimentScore]:
        """Fetch sentiment scores for a market/topic."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is reachable and configured."""
        ...


class NullSentimentProvider(SentimentProvider):
    """Default no-op provider that returns no data."""

    async def get_sentiment(self, market_id: str, query: str) -> list[SentimentScore]:
        return []

    async def is_available(self) -> bool:
        return False


# ── Strategy ───────────────────────────────────────────────────────────────


@StrategyRegistry.register
class SentimentAdapter(BaseStrategy):
    """
    Stub strategy that consults a SentimentProvider.
    Currently always returns None since no provider is configured.
    To activate, inject a real SentimentProvider implementation.
    """

    name = "sentiment_adapter"

    def __init__(self, settings: Settings, provider: SentimentProvider | None = None) -> None:
        super().__init__(settings)
        self._provider = provider or NullSentimentProvider()

    def generate_signal(
        self,
        features: MarketFeatures,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        # Sentiment is async — in a real implementation this would be
        # called from an async context with cached sentiment scores.
        # For now, this is a no-op stub.
        logger.debug("sentiment_adapter_no_op", market_id=features.market_id)
        return None
