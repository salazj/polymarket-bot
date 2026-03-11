"""
Mock provider returning hardcoded test headlines.

Always available, useful for development, testing, and demos.
"""

from __future__ import annotations

import uuid

from app.news.models import NewsItem
from app.nlp.providers.base import BaseNlpProvider
from app.utils.helpers import utc_now


_SAMPLE_HEADLINES = [
    "Breaking: Federal judge rules in favor of defendant in landmark crypto case",
    "Election polls show candidate surging ahead in key battleground state",
    "SEC announces new regulatory framework for digital asset exchanges",
    "Bitcoin surges past key resistance level amid institutional buying",
    "Sports: Underdog team wins championship in stunning upset",
    "GDP growth beats expectations, economy showing strong recovery",
    "Celebrity scandal rocks entertainment industry",
    "NATO allies reach diplomatic agreement on disputed territory",
]


class MockProvider(BaseNlpProvider):
    """Returns rotating sample headlines for development and testing."""

    name: str = "mock"  # type: ignore[assignment]

    def __init__(self) -> None:
        self._index = 0

    async def fetch_items(self) -> list[NewsItem]:
        headline = _SAMPLE_HEADLINES[self._index % len(_SAMPLE_HEADLINES)]
        self._index += 1
        return [
            NewsItem(
                item_id=f"mock-{uuid.uuid4().hex[:8]}",
                source=self.name,
                text=headline,
                timestamp=utc_now(),
            )
        ]

    def is_available(self) -> bool:
        return True
