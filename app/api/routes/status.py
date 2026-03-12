"""Status, health, and log retrieval endpoints."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request

from app.api.log_broadcaster import log_broadcaster
from app.api.schemas import BotStatusResponse, HealthResponse, LogEntryResponse

router = APIRouter(tags=["status"])


@router.get("/api/status", response_model=BotStatusResponse)
async def get_status(request: Request) -> BotStatusResponse:
    mgr = request.app.state.bot_manager
    return mgr.get_status()


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    mgr = request.app.state.bot_manager
    status = mgr.get_status()
    return HealthResponse(
        status="ok" if not status.error else "degraded",
        version="3.0.0",
        bot_running=status.running,
        session_id=status.session_id,
        asset_class=status.asset_class,
        mode=status.mode,
        uptime_seconds=status.uptime_seconds,
        log_subscribers=len(log_broadcaster._subscribers),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/api/logs", response_model=list[LogEntryResponse])
async def get_logs(
    request: Request,
    limit: int = Query(default=200, ge=1, le=5000),
    level: str = Query(default="info"),
) -> list[LogEntryResponse]:
    """Fetch recent log entries from the ring buffer (REST complement to WS streaming)."""
    entries = log_broadcaster.get_recent(limit=limit, level_filter=level)
    return [LogEntryResponse(**e) for e in entries]
