"""Configuration management endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ConfigPreset, RunConfig, ValidationResult
from app.config.settings import get_settings

router = APIRouter(prefix="/api/config", tags=["config"])

_PRESETS_DIR = Path("data/presets")


@router.get("")
async def get_config(request: Request) -> dict:
    mgr = request.app.state.bot_manager
    run_cfg = mgr.get_run_config()
    if run_cfg:
        return run_cfg.model_dump()
    settings = get_settings()
    return RunConfig(
        asset_class=settings.asset_class,
        exchange=settings.exchange,
        broker=settings.broker,
        dry_run=settings.dry_run,
        decision_mode=settings.decision_mode,
        nlp_provider=settings.nlp_provider,
        llm_provider=settings.llm_provider,
    ).model_dump()


@router.post("/validate", response_model=ValidationResult)
async def validate_config(request: Request, config: RunConfig) -> ValidationResult:
    mgr = request.app.state.bot_manager
    return mgr.validate_config(config)


@router.get("/presets", response_model=list[ConfigPreset])
async def list_presets() -> list[ConfigPreset]:
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    presets: list[ConfigPreset] = []
    for p in sorted(_PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            presets.append(ConfigPreset(
                name=p.stem,
                config=RunConfig(**data.get("config", data)),
                created_at=data.get("created_at", ""),
            ))
        except Exception:
            continue
    return presets


@router.post("/presets/{name}", response_model=ConfigPreset)
async def save_preset(name: str, config: RunConfig) -> ConfigPreset:
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid preset name")
    path = _PRESETS_DIR / f"{safe_name}.json"
    from datetime import datetime, timezone
    created = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps({"config": config.model_dump(), "created_at": created}, indent=2))
    return ConfigPreset(name=safe_name, config=config, created_at=created)


@router.delete("/presets/{name}")
async def delete_preset(name: str) -> dict:
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    path = _PRESETS_DIR / f"{safe_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preset not found")
    path.unlink()
    return {"status": "ok", "deleted": safe_name}
