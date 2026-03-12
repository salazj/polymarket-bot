"""Risk state and control endpoints.

All bot access goes through BotManager — no direct bot access.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import RiskStateResponse

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("", response_model=RiskStateResponse)
async def get_risk_state(request: Request) -> RiskStateResponse:
    mgr = request.app.state.bot_manager
    return mgr.get_risk_state()


@router.post("/reset-breaker")
async def reset_circuit_breaker(request: Request) -> dict:
    mgr = request.app.state.bot_manager
    try:
        mgr.reset_circuit_breaker()
        return {"status": "ok", "message": "Circuit breaker reset"}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/emergency-stop")
async def emergency_stop(request: Request) -> dict:
    mgr = request.app.state.bot_manager
    try:
        mgr.trip_emergency_stop()
        return {"status": "ok", "message": "Emergency stop activated"}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
