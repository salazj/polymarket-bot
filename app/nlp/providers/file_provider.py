"""
File-based provider: reads JSON files from a directory for testing.

Each JSON file should contain a list of objects with at least
``{"text": "...", "source": "...", "timestamp": "..."}`` fields.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.news.models import NewsItem
from app.nlp.providers.base import BaseNlpProvider
from app.monitoring import get_logger
from app.utils.helpers import utc_now

logger = get_logger(__name__)


class FileProvider(BaseNlpProvider):
    """Reads news items from JSON files in a local directory."""

    name: str = "file"  # type: ignore[assignment]

    def __init__(self, directory: str | Path = "data/news") -> None:
        self._dir = Path(directory)
        self._seen_files: set[str] = set()

    async def fetch_items(self) -> list[NewsItem]:
        if not self._dir.exists():
            return []
        items: list[NewsItem] = []
        for path in sorted(self._dir.glob("*.json")):
            if path.name in self._seen_files:
                continue
            self._seen_files.add(path.name)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for entry in data:
                        items.append(self._parse_entry(entry))
                elif isinstance(data, dict):
                    items.append(self._parse_entry(data))
            except Exception:
                logger.exception("file_provider_read_error", path=str(path))
        return items

    def is_available(self) -> bool:
        return self._dir.exists()

    @staticmethod
    def _parse_entry(entry: dict) -> NewsItem:
        return NewsItem(
            item_id=entry.get("id", f"file-{uuid.uuid4().hex[:8]}"),
            source=entry.get("source", "file"),
            text=entry.get("text", ""),
            url=entry.get("url", ""),
            timestamp=utc_now(),
            raw_metadata=entry,
        )
