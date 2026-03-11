"""
Polymarket market data client implementing the BaseMarketDataClient interface.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.settings import Settings
from app.data.models import Market, MarketToken
from app.exchanges.base import BaseMarketDataClient
from app.monitoring import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


class PolymarketMarketDataClient(BaseMarketDataClient):
    """REST client for Polymarket CLOB market data."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.polymarket_host
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=DEFAULT_TIMEOUT,
            headers={"Accept": "application/json"},
        )

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

    async def get_markets(self, cursor: str = "") -> tuple[list[Market], str]:
        params: dict[str, Any] = {}
        if cursor:
            params["next_cursor"] = cursor

        data = await self._get("/markets", params=params)
        next_cursor = data.get("next_cursor", "")
        raw_markets = data.get("data", data) if isinstance(data, dict) else data

        markets: list[Market] = []
        if isinstance(raw_markets, list):
            for m in raw_markets:
                markets.append(_parse_market(m))

        logger.info("fetched_markets", count=len(markets), next_cursor=next_cursor[:20] if next_cursor else "")
        return markets, next_cursor

    async def get_all_markets(self, max_pages: int = 50) -> list[Market]:
        all_markets: list[Market] = []
        cursor = ""
        for _ in range(max_pages):
            page, cursor = await self.get_markets(cursor)
            all_markets.extend(page)
            if not cursor or cursor == "LTE":
                break
        logger.info("fetched_all_markets", total=len(all_markets))
        return all_markets

    async def get_market(self, market_id: str) -> Market | None:
        try:
            data = await self._get(f"/markets/{market_id}")
            return _parse_market(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_orderbook(self, instrument_id: str) -> dict[str, Any]:
        return await self._get("/book", params={"token_id": instrument_id})

    async def get_midpoint(self, instrument_id: str) -> float | None:
        try:
            data = await self._get("/midpoint", params={"token_id": instrument_id})
            return float(data.get("mid", 0))
        except Exception:
            return None


def _parse_market(raw: dict[str, Any]) -> Market:
    tokens: list[MarketToken] = []
    for t in raw.get("tokens", []):
        tid = t.get("token_id", "")
        tokens.append(
            MarketToken(
                token_id=tid,
                instrument_id=tid,
                outcome=t.get("outcome", ""),
            )
        )

    cid = raw.get("condition_id", raw.get("id", ""))
    return Market(
        condition_id=cid,
        market_id=cid,
        question=raw.get("question", ""),
        slug=raw.get("slug", ""),
        tokens=tokens,
        end_date=raw.get("end_date_iso"),
        active=raw.get("active", True),
        minimum_order_size=float(raw.get("minimum_order_size", 1.0)),
        minimum_tick_size=float(raw.get("minimum_tick_size", 0.01)),
        exchange="polymarket",
    )
