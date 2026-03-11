"""
NLP event replay: load stored text events from the database or from
JSON files and re-process them through the current pipeline.

Use cases:
  - Evaluate a new classifier against historical events
  - Debug why a particular headline produced (or didn't produce) a signal
  - Backtest NLP signal quality without re-fetching from providers
  - Compare signal output across pipeline versions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.data.models import Market
from app.monitoring import get_logger
from app.news.models import NewsItem
from app.nlp.pipeline import NlpPipeline
from app.nlp.signals import NlpSignal
from app.utils.helpers import utc_now

logger = get_logger(__name__)


@dataclass
class ReplayResult:
    """Summary of one replay run."""
    total_items: int = 0
    items_with_signals: int = 0
    total_signals: int = 0
    signals_by_sentiment: dict[str, int] = field(default_factory=dict)
    signals_by_event_type: dict[str, int] = field(default_factory=dict)
    signals_by_market: dict[str, int] = field(default_factory=dict)
    per_item: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "items_with_signals": self.items_with_signals,
            "total_signals": self.total_signals,
            "signal_rate": (
                self.items_with_signals / self.total_items
                if self.total_items > 0 else 0.0
            ),
            "signals_by_sentiment": self.signals_by_sentiment,
            "signals_by_event_type": self.signals_by_event_type,
            "signals_by_market": self.signals_by_market,
        }


class NlpReplayEngine:
    """Replays stored or provided text events through the NLP pipeline."""

    def __init__(
        self,
        pipeline: NlpPipeline | None = None,
        active_markets: list[Market] | None = None,
    ) -> None:
        self._pipeline = pipeline or NlpPipeline()
        self._markets = active_markets or []

    def set_markets(self, markets: list[Market]) -> None:
        self._markets = markets

    def replay_items(self, items: list[NewsItem]) -> ReplayResult:
        """Process a list of NewsItems and return aggregated results."""
        result = ReplayResult(total_items=len(items))

        for item in items:
            try:
                signals = self._pipeline.process_item(item, self._markets)
                item_record: dict[str, Any] = {
                    "item_id": item.item_id,
                    "source": item.source,
                    "text": item.text[:200],
                    "signals_count": len(signals),
                    "signals": [],
                }

                if signals:
                    result.items_with_signals += 1
                    result.total_signals += len(signals)

                for sig in signals:
                    sent = sig.sentiment.value
                    etype = sig.event_type.value
                    result.signals_by_sentiment[sent] = result.signals_by_sentiment.get(sent, 0) + 1
                    result.signals_by_event_type[etype] = result.signals_by_event_type.get(etype, 0) + 1
                    for mid in sig.market_ids:
                        result.signals_by_market[mid] = result.signals_by_market.get(mid, 0) + 1

                    item_record["signals"].append({
                        "market_ids": sig.market_ids,
                        "sentiment": sent,
                        "sentiment_score": sig.sentiment_score,
                        "event_type": etype,
                        "urgency": sig.urgency,
                        "relevance": sig.relevance,
                        "confidence": sig.confidence,
                        "rationale": sig.rationale,
                    })

                result.per_item.append(item_record)

            except Exception:
                logger.exception("replay_item_error", item_id=item.item_id)
                result.per_item.append({
                    "item_id": item.item_id,
                    "error": True,
                })

        logger.info(
            "nlp_replay_complete",
            total_items=result.total_items,
            items_with_signals=result.items_with_signals,
            total_signals=result.total_signals,
        )
        return result

    def replay_from_json(self, path: str | Path) -> ReplayResult:
        """Load items from a JSON file and replay them."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Replay file not found: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            raw_items = data["items"]
        elif isinstance(data, list):
            raw_items = data
        else:
            raise ValueError(f"Unexpected JSON structure in {p}")

        items: list[NewsItem] = []
        for entry in raw_items:
            items.append(NewsItem(
                item_id=entry.get("id", entry.get("item_id", f"replay-{len(items)}")),
                source=entry.get("source", "replay"),
                text=entry.get("text", entry.get("headline", "")),
                url=entry.get("url", ""),
                timestamp=utc_now(),
                raw_metadata=entry,
            ))

        logger.info("replay_loaded_from_file", path=str(p), items=len(items))
        return self.replay_items(items)

    def replay_from_db_rows(self, rows: list[dict]) -> ReplayResult:
        """Replay from rows returned by Repository.get_nlp_events()."""
        items: list[NewsItem] = []
        for row in rows:
            items.append(NewsItem(
                item_id=row.get("item_id", ""),
                source=row.get("source", "db_replay"),
                text=row.get("text", ""),
                url=row.get("url", ""),
                timestamp=utc_now(),
            ))
        logger.info("replay_loaded_from_db", items=len(items))
        return self.replay_items(items)

    def save_result(self, result: ReplayResult, path: str | Path) -> None:
        """Save replay results to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(result.summary(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("replay_result_saved", path=str(p))
