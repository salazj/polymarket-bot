"""
Polymarket CLOB REST client for market metadata and historical data.

Uses httpx with retry logic and rate-limit awareness.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.settings import Settings
from app.data.models import Market, MarketToken
from app.monitoring import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


class PolymarketRestClient:
    """Thin wrapper around the Polymarket CLOB REST API."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.polymarket_host
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=DEFAULT_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        self._settings = settings

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_markets(self, next_cursor: str = "") -> tuple[list[Market], str]:
        """
        Fetch a page of markets. Returns (markets, next_cursor).
        Cursor is empty string when no more pages.
        """
        params: dict[str, Any] = {}
        if next_cursor:
            params["next_cursor"] = next_cursor

        data = await self._get("/markets", params=params)
        cursor = data.get("next_cursor", "")
        raw_markets = data.get("data", data) if isinstance(data, dict) else data

        markets: list[Market] = []
        if isinstance(raw_markets, list):
            for m in raw_markets:
                markets.append(_parse_market(m))

        logger.info("fetched_markets", count=len(markets), next_cursor=cursor[:20] if cursor else "")
        return markets, cursor

    async def get_all_markets(self, max_pages: int = 50) -> list[Market]:
        """Page through all available markets."""
        all_markets: list[Market] = []
        cursor = ""
        for _ in range(max_pages):
            page, cursor = await self.get_markets(cursor)
            all_markets.extend(page)
            if not cursor or cursor == "LTE":
                break
        logger.info("fetched_all_markets", total=len(all_markets))
        return all_markets

    async def get_market(self, condition_id: str) -> Market | None:
        try:
            data = await self._get(f"/markets/{condition_id}")
            return _parse_market(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Fetch current orderbook snapshot for a token."""
        return await self._get(f"/book", params={"token_id": token_id})

    async def get_midpoint(self, token_id: str) -> float | None:
        """Fetch midpoint price for a token."""
        try:
            data = await self._get(f"/midpoint", params={"token_id": token_id})
            return float(data.get("mid", 0))
        except Exception:
            return None


def _parse_market(raw: dict[str, Any]) -> Market:
    tokens: list[MarketToken] = []
    for t in raw.get("tokens", []):
        tokens.append(
            MarketToken(
                token_id=t.get("token_id", ""),
                outcome=t.get("outcome", ""),
            )
        )

    return Market(
        condition_id=raw.get("condition_id", raw.get("id", "")),
        question=raw.get("question", ""),
        slug=raw.get("slug", ""),
        tokens=tokens,
        end_date=raw.get("end_date_iso"),
        active=raw.get("active", True),
        minimum_order_size=float(raw.get("minimum_order_size", 1.0)),
        minimum_tick_size=float(raw.get("minimum_tick_size", 0.01)),
    )
