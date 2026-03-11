"""
Data models for the news / text ingestion subsystem.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.utils.helpers import utc_now


class NewsItem(BaseModel):
    """A single news/text item from any provider."""

    item_id: str
    source: str
    text: str
    url: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
