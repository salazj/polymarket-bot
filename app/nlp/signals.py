"""
NLP-specific signal and event types.

NlpSignal carries structured output from the text-processing pipeline
including relevance, sentiment, urgency, event classification, and
extracted entities — all mapped to candidate markets.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.utils.helpers import utc_now


class EventType(str, Enum):
    LEGAL_RULING = "legal_ruling"
    REGULATORY = "regulatory"
    ELECTION = "election"
    ECONOMIC = "economic"
    CELEBRITY = "celebrity"
    SPORTS = "sports"
    GEOPOLITICAL = "geopolitical"
    CRYPTO = "crypto"
    OTHER = "other"


class SentimentDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ClassificationResult(BaseModel):
    """Output of a single text classification step."""

    relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    sentiment: SentimentDirection = SentimentDirection.NEUTRAL
    sentiment_score: float = Field(ge=-1.0, le=1.0, default=0.0)
    event_type: EventType = EventType.OTHER
    urgency: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str = ""
    entities: list[str] = Field(default_factory=list)


class NlpSignal(BaseModel):
    """Structured signal produced by the NLP pipeline for the decision engine."""

    source_text_id: str
    source_provider: str
    source_timestamp: datetime = Field(default_factory=utc_now)
    text_snippet: str = ""

    market_ids: list[str] = Field(default_factory=list)
    relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    sentiment: SentimentDirection = SentimentDirection.NEUTRAL
    sentiment_score: float = Field(ge=-1.0, le=1.0, default=0.0)
    event_type: EventType = EventType.OTHER
    urgency: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str = ""
    entities: list[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
