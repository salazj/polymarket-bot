"""
RSS feed provider stub for future integration.

Requires ``feedparser`` (not included in base dependencies).
To enable: ``pip install feedparser`` and configure feed URLs.
"""

from __future__ import annotations

from app.news.models import NewsItem
from app.nlp.providers.base import BaseNlpProvider
from app.monitoring import get_logger

logger = get_logger(__name__)


class RssProvider(BaseNlpProvider):
    """RSS/Atom feed provider (stub — requires feedparser)."""

    name: str = "rss"  # type: ignore[assignment]

    def __init__(self, feed_urls: list[str] | None = None) -> None:
        self._urls = feed_urls or []

    async def fetch_items(self) -> list[NewsItem]:
        if not self._urls:
            return []
        try:
            import feedparser  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("rss_provider_missing_feedparser")
            return []

        items: list[NewsItem] = []
        for url in self._urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    items.append(NewsItem(
                        item_id=getattr(entry, "id", entry.get("link", "")),
                        source=f"rss:{url[:50]}",
                        text=getattr(entry, "title", ""),
                        url=getattr(entry, "link", ""),
                        raw_metadata={"summary": getattr(entry, "summary", "")},
                    ))
            except Exception:
                logger.exception("rss_provider_feed_error", url=url)
        return items

    def is_available(self) -> bool:
        if not self._urls:
            return False
        try:
            import feedparser  # noqa: F401  # type: ignore[import-untyped]
            return True
        except ImportError:
            return False
