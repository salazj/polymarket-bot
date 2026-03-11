"""
Abstract base class for NLP text providers.

Every provider must implement fetch_items() and is_available().
The bot operates normally even when zero providers are configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.news.models import NewsItem


class BaseNlpProvider(ABC):
    """Interface for all text/news data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def fetch_items(self) -> list[NewsItem]:
        """Return new items since last poll. Empty list is perfectly valid."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is configured and ready."""
        ...
