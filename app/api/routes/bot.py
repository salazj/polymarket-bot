"""Bot lifecycle endpoints — start, stop, restart."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import BotStatusResponse, RunConfig

router = APIRouter(prefix="/api/bot", tags=["bot"])


@router.post("/start", response_model=BotStatusResponse)
async def start_bot(request: Request, config: RunConfig) -> BotStatusResponse:
    mgr = request.app.state.bot_manager
    try:
        return await mgr.start(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/stop", response_model=BotStatusResponse)
async def stop_bot(request: Request) -> BotStatusResponse:
    mgr = request.app.state.bot_manager
    return await mgr.stop()


@router.post("/restart", response_model=BotStatusResponse)
async def restart_bot(
    request: Request, config: RunConfig | None = None
) -> BotStatusResponse:
    mgr = request.app.state.bot_manager
    try:
        return await mgr.restart(config)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
