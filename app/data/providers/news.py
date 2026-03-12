"""
News/headline data provider — stub implementation.

Produces a ``news_intensity`` feature representing how many relevant
news articles/headlines appeared recently for a given market topic.

Current implementation: returns NaN (no-op).

To connect a real news API later:
  1. Set NEWS_API_KEY and NEWS_API_URL in .env
  2. Implement the fetch_features() method below
  3. The feature column auto-populates in the ML pipeline

Potential sources:
  - NewsAPI.org (free tier: 100 req/day)
  - GDELT Project (free, real-time news events)
  - Google News RSS (free but rate-limited)
  - Bing News Search (Azure, free tier available)
  - Custom RSS aggregator
"""

from __future__ import annotations

import math
import os

from app.data.providers.base import BaseProvider, ProviderRegistry


@ProviderRegistry.register
class NewsProvider(BaseProvider):
    name = "news"

    def __init__(self) -> None:
        self._api_key = os.environ.get("NEWS_API_KEY", "")
        self._api_url = os.environ.get("NEWS_API_URL", "")

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_url)

    def feature_names(self) -> list[str]:
        return ["news_intensity"]

    async def fetch_features(self, market_id: str, token_id: str) -> dict[str, float]:
        if not self.is_available():
            return {"news_intensity": math.nan}

        # TODO: Call the actual news API here.
        # Example skeleton:
        #   async with httpx.AsyncClient() as client:
        #       resp = await client.get(
        #           f"{self._api_url}/headlines",
        #           params={"query": market_id, "hours": 24},
        #           headers={"X-Api-Key": self._api_key},
        #           timeout=5.0,
        #       )
        #       data = resp.json()
        #       return {"news_intensity": float(len(data.get("articles", [])))}

        return {"news_intensity": math.nan}
