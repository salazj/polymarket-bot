"""
Sentiment data provider — stub implementation.

Produces a ``sentiment_score`` feature in [-1, 1] representing aggregated
sentiment about a market's question.

Current implementation: returns NaN (no-op).  The ML pipeline imputes
this with median/zero, so the core bot is unaffected.

To connect a real sentiment API later:
  1. Subclass or modify this provider
  2. Implement fetch_features() to call your API
     (e.g., Twitter/X API, Reddit API, a custom NLP service)
  3. Set credentials in .env: SENTIMENT_API_KEY, SENTIMENT_API_URL
  4. The provider auto-registers via @ProviderRegistry.register
  5. The research pipeline will include sentiment_score in the feature matrix

Potential free/low-cost sources:
  - Polymarket comment section scraping (check ToS)
  - Reddit API (free tier)
  - Farcaster protocol (public data)
  - HuggingFace zero-shot classification on headlines
"""

from __future__ import annotations

import math
import os

from app.data.providers.base import BaseProvider, ProviderRegistry


@ProviderRegistry.register
class SentimentProvider(BaseProvider):
    name = "sentiment"

    def __init__(self) -> None:
        self._api_key = os.environ.get("SENTIMENT_API_KEY", "")
        self._api_url = os.environ.get("SENTIMENT_API_URL", "")

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_url)

    def feature_names(self) -> list[str]:
        return ["sentiment_score"]

    async def fetch_features(self, market_id: str, token_id: str) -> dict[str, float]:
        if not self.is_available():
            return {"sentiment_score": math.nan}

        # TODO: Call the actual sentiment API here.
        # Example skeleton:
        #   async with httpx.AsyncClient() as client:
        #       resp = await client.get(
        #           f"{self._api_url}/sentiment",
        #           params={"market": market_id},
        #           headers={"Authorization": f"Bearer {self._api_key}"},
        #           timeout=5.0,
        #       )
        #       data = resp.json()
        #       return {"sentiment_score": data["score"]}

        return {"sentiment_score": math.nan}
