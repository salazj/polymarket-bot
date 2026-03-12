"""
External data provider plugin system.

Architecture:
  BaseProvider       – abstract interface all providers implement
  ProviderRegistry   – discovers and manages provider instances
  SentimentProvider  – stub for sentiment analysis (no paid API required)
  NewsProvider       – stub for news/headline feeds (no paid API required)

The core bot never blocks on external providers.  If a provider is
unavailable, its columns are NaN and the ML pipeline imputes them.

To add a real provider later:
  1. Subclass BaseProvider
  2. Implement fetch_features()
  3. Register with @ProviderRegistry.register
  4. Set any API keys in .env (the bot will still run without them)
"""

from app.data.providers.base import BaseProvider, ProviderRegistry
from app.data.providers.sentiment import SentimentProvider
from app.data.providers.news import NewsProvider

__all__ = [
    "BaseProvider",
    "ProviderRegistry",
    "SentimentProvider",
    "NewsProvider",
]
