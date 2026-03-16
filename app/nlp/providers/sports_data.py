"""
BetStack sports data provider — fetches live scores and consensus odds.

Free API from betstack.dev covering NFL, NBA, MLB, NHL, NCAA, soccer, etc.
Rate limit: 1 request per 60 seconds (data refreshes every 60s on edge).
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.monitoring import get_logger
from app.news.models import NewsItem
from app.nlp.providers.base import BaseNlpProvider

logger = get_logger(__name__)

_BASE_URL = "https://api.betstack.dev/api/v1"
_TIMEOUT = 15.0
_MIN_POLL_INTERVAL = 62.0


def moneyline_to_probability(ml: int | float) -> float:
    """Convert American moneyline odds to implied probability."""
    ml = float(ml)
    if ml == 0:
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


def _format_odds_line(odds: dict[str, Any]) -> str:
    """Build a human-readable odds summary from a BetStack line object."""
    parts: list[str] = []

    ml_home = odds.get("money_line_home")
    ml_away = odds.get("money_line_away")
    if ml_home is not None and ml_away is not None:
        home_prob = moneyline_to_probability(ml_home) * 100
        away_prob = moneyline_to_probability(ml_away) * 100
        parts.append(
            f"Moneyline: Home {ml_home:+d} ({home_prob:.0f}% implied) / "
            f"Away {ml_away:+d} ({away_prob:.0f}% implied)"
        )

    spread_home = odds.get("point_spread_home")
    if spread_home is not None:
        parts.append(f"Spread: Home {spread_home:+.1f}")

    total = odds.get("total_number")
    if total is not None:
        parts.append(f"Total: {total}")

    return " | ".join(parts) if parts else "No odds available"


class SportsDataProvider(BaseNlpProvider):
    """Fetches live sports events, scores, and consensus odds from BetStack."""

    name: str = "sports"  # type: ignore[assignment]

    def __init__(self, api_key: str, leagues: list[str] | None = None) -> None:
        self._api_key = api_key
        self._leagues = leagues or [
            "americanfootball_nfl",
            "basketball_nba",
            "baseball_mlb",
            "icehockey_nhl",
        ]
        self._seen: set[str] = set()
        self._last_poll: float = 0.0
        self._events_cache: list[dict[str, Any]] = []

    async def fetch_items(self) -> list[NewsItem]:
        now = time.monotonic()
        if now - self._last_poll < _MIN_POLL_INTERVAL and self._events_cache:
            return []

        all_events: list[dict[str, Any]] = []
        headers = {"X-API-Key": self._api_key}

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for league in self._leagues:
                try:
                    resp = await client.get(
                        f"{_BASE_URL}/events",
                        params={"league": league, "include_lines": "true"},
                        headers=headers,
                    )
                    if resp.status_code == 429:
                        logger.warning("betstack_rate_limited", league=league)
                        break
                    resp.raise_for_status()
                    events = resp.json()
                    if isinstance(events, list):
                        all_events.extend(events)
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "betstack_http_error",
                        league=league,
                        status=exc.response.status_code,
                    )
                except Exception as exc:
                    logger.warning("betstack_fetch_error", league=league, error=str(exc))

        self._last_poll = time.monotonic()
        self._events_cache = all_events

        items: list[NewsItem] = []
        for event in all_events:
            item = self._event_to_news_item(event)
            if item:
                items.append(item)

        logger.info("betstack_fetched", events=len(all_events), new_items=len(items))
        return items

    def _event_to_news_item(self, event: dict[str, Any]) -> NewsItem | None:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        if not home or not away:
            return None

        event_id = str(event.get("id", ""))
        league_info = event.get("league", {})
        league_name = league_info.get("name", "") if isinstance(league_info, dict) else str(league_info)
        league_key = league_info.get("key", "") if isinstance(league_info, dict) else ""

        content_key = f"{away}@{home}-{event_id}"
        content_hash = hashlib.sha256(content_key.encode()).hexdigest()[:16]
        if content_hash in self._seen:
            return None
        self._seen.add(content_hash)

        result = event.get("result") or {}
        completed = event.get("completed", False) or result.get("final", False)
        if completed:
            return None

        home_score = result.get("home_score")
        away_score = result.get("away_score")
        is_live = home_score is not None and away_score is not None and not completed

        title = f"{away} vs {home}"
        if league_name:
            title += f" — {league_name}"

        content_parts = [title]

        if is_live:
            content_parts.append(f"LIVE SCORE: {away} {away_score}, {home} {home_score}")

        lines = event.get("lines")
        odds_data: dict[str, Any] = {}
        if isinstance(lines, list) and lines:
            odds_data = lines[0]
        elif isinstance(lines, dict):
            odds_data = lines

        if odds_data:
            content_parts.append(f"Consensus odds: {_format_odds_line(odds_data)}")

            ml_home = odds_data.get("money_line_home")
            ml_away = odds_data.get("money_line_away")
            if ml_home is not None and ml_away is not None:
                home_prob = moneyline_to_probability(ml_home)
                away_prob = moneyline_to_probability(ml_away)
                content_parts.append(
                    f"Implied win probability: {home} {home_prob*100:.0f}%, "
                    f"{away} {away_prob*100:.0f}%"
                )

        commence_str = event.get("commence_time", "")
        try:
            ts = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        return NewsItem(
            item_id=f"betstack-{content_hash}",
            source=f"betstack:{league_key or league_name}",
            text=". ".join(content_parts),
            url="",
            timestamp=ts,
            raw_metadata={
                "provider": "betstack",
                "event_id": event_id,
                "home_team": home,
                "away_team": away,
                "league": league_key or league_name,
                "is_live": is_live,
                "home_score": home_score,
                "away_score": away_score,
                "odds": odds_data,
                "commence_time": commence_str,
                "completed": completed,
            },
        )

    @property
    def events_cache(self) -> list[dict[str, Any]]:
        return self._events_cache

    def is_available(self) -> bool:
        return bool(self._api_key)
