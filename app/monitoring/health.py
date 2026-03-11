"""
Lightweight async HTTP health endpoint.

Exposes:
  GET /health  → 200 JSON with uptime, halted status, connections
  GET /metrics → 200 JSON with all MetricsCounter values + portfolio summary

Uses only the stdlib asyncio protocol server (no extra dependency).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from app.monitoring import get_logger
from app.monitoring.logger import metrics

logger = get_logger(__name__)


class HealthServer:
    """Minimal async HTTP/1.1 server serving health and metrics endpoints."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8880,
        portfolio_snapshot_fn: Callable[[], Any] | None = None,
        is_halted_fn: Callable[[], bool] | None = None,
        ws_connected_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._portfolio_fn = portfolio_snapshot_fn
        self._is_halted_fn = is_halted_fn
        self._ws_connected_fn = ws_connected_fn
        self._start_time = time.monotonic()
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        logger.info("health_server_started", host=self._host, port=self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("health_server_stopped")

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            path = self._parse_path(request_line.decode(errors="replace"))

            if path == "/health":
                body = self._build_health()
            elif path == "/metrics":
                body = self._build_metrics()
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            payload = json.dumps(body, default=str).encode()
            header = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                f"\r\n"
            )
            writer.write(header.encode() + payload)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    @staticmethod
    def _parse_path(request_line: str) -> str:
        parts = request_line.strip().split()
        if len(parts) >= 2:
            return parts[1].split("?")[0]
        return "/"

    def _build_health(self) -> dict[str, Any]:
        uptime = time.monotonic() - self._start_time
        halted = self._is_halted_fn() if self._is_halted_fn else False
        ws_ok = self._ws_connected_fn() if self._ws_connected_fn else None
        return {
            "status": "halted" if halted else "ok",
            "uptime_seconds": round(uptime, 1),
            "ws_connected": ws_ok,
        }

    def _build_metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {"counters": metrics.snapshot()}
        if self._portfolio_fn:
            try:
                snap = self._portfolio_fn()
                result["portfolio"] = {
                    "cash": snap.cash,
                    "total_exposure": snap.total_exposure,
                    "total_unrealized_pnl": snap.total_unrealized_pnl,
                    "total_realized_pnl": snap.total_realized_pnl,
                    "daily_pnl": snap.daily_pnl,
                    "position_count": len(snap.positions),
                }
            except Exception:
                result["portfolio"] = "unavailable"
        return result
