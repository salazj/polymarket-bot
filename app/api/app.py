"""
FastAPI application factory for the $alazar-Trader platform.

Creates the app, mounts all REST and WebSocket routers, sets up CORS,
and configures the BotManager singleton on startup.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.bot_manager import BotManager
from app.api.log_broadcaster import log_broadcaster
from app.api.schemas import RunConfig
from app.monitoring import setup_logging
from app.monitoring.logger import get_logger

logger = get_logger(__name__)


def _build_auto_start_config() -> RunConfig:
    """Build a RunConfig from environment variables for auto-start."""
    env = os.environ.get

    dry_run = env("DRY_RUN", "true").lower() in ("true", "1", "yes")
    enable_live = env("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    live_ack = env("LIVE_TRADING_ACKNOWLEDGED", "false").lower() in ("true", "1", "yes")

    return RunConfig(
        asset_class=env("ASSET_CLASS", "prediction_markets"),
        exchange=env("EXCHANGE", "kalshi"),
        broker=env("BROKER", "alpaca"),
        dry_run=dry_run,
        enable_live_trading=enable_live,
        live_trading_acknowledged=live_ack,
        decision_mode=env("DECISION_MODE", "conservative"),
        ensemble_weight_l1=float(env("ENSEMBLE_WEIGHT_L1", "0.30")),
        ensemble_weight_l2=float(env("ENSEMBLE_WEIGHT_L2", "0.40")),
        ensemble_weight_l3=float(env("ENSEMBLE_WEIGHT_L3", "0.30")),
        min_ensemble_confidence=float(env("MIN_ENSEMBLE_CONFIDENCE", "0.50")),
        min_layers_agree=int(env("MIN_LAYERS_AGREE", "1")),
        min_evidence_signals=int(env("MIN_EVIDENCE_SIGNALS", "1")),
        nlp_provider=env("NLP_PROVIDER", "mock"),
        llm_provider=env("LLM_PROVIDER", "none"),
        max_tracked_markets=int(env("MAX_TRACKED_MARKETS", "50")),
        max_subscribed_markets=int(env("MAX_SUBSCRIBED_MARKETS", "20")),
        max_position_per_market=float(env("MAX_POSITION_PER_MARKET", "10.0")),
        max_total_exposure=float(env("MAX_TOTAL_EXPOSURE", "50.0")),
        max_daily_loss=float(env("MAX_DAILY_LOSS", "10.0")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    mgr = BotManager()
    app.state.bot_manager = mgr
    logger.info("api_server_started")

    auto_start = os.environ.get("AUTO_START_BOT", "true").lower() in ("true", "1", "yes")
    if auto_start:
        try:
            config = _build_auto_start_config()
            await mgr.start(config)
            logger.info("bot_auto_started", exchange=config.exchange, dry_run=config.dry_run)
        except Exception as exc:
            logger.error("bot_auto_start_failed", error=str(exc))

    yield

    if mgr.is_running:
        await mgr.stop()
    logger.info("api_server_stopped")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="$alazar-Trader API",
        version="3.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routes
    from app.api.routes.status import router as status_router
    from app.api.routes.bot import router as bot_router
    from app.api.routes.config import router as config_router
    from app.api.routes.portfolio import router as portfolio_router
    from app.api.routes.risk import router as risk_router
    from app.api.routes.exchanges import router as exchanges_router
    from app.api.routes.strategies import router as strategies_router

    app.include_router(status_router)
    app.include_router(bot_router)
    app.include_router(config_router)
    app.include_router(portfolio_router)
    app.include_router(risk_router)
    app.include_router(exchanges_router)
    app.include_router(strategies_router)

    # WebSocket routes
    from app.api.websocket.logs import router as ws_logs_router
    from app.api.websocket.status import router as ws_status_router
    from app.api.websocket.portfolio_ws import router as ws_portfolio_router

    app.include_router(ws_logs_router)
    app.include_router(ws_status_router)
    app.include_router(ws_portfolio_router)

    return app
