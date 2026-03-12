"""WebSocket endpoint for live log streaming."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api.log_broadcaster import log_broadcaster

router = APIRouter()


@router.websocket("/ws/logs")
async def ws_logs(
    websocket: WebSocket,
    level: str = Query(default="info"),
) -> None:
    await websocket.accept()

    recent = log_broadcaster.get_recent(limit=200, level_filter=level)
    for entry in recent:
        await websocket.send_json(entry)

    queue = log_broadcaster.add_subscriber()
    try:
        while True:
            entry = await queue.get()
            from app.api.log_broadcaster import _level_set
            if entry.level in _level_set(level):
                await websocket.send_json(entry.to_dict())
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        log_broadcaster.remove_subscriber(queue)
