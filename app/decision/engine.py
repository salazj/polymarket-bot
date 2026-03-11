"""
Central decision engine: signal registry, normalization, and ensemble.

This module NEVER places orders.  It only produces TradeCandidate objects
(and full DecisionTrace records) for the risk manager and execution
engine to evaluate independently.
"""

from __future__ import annotations

from typing import Callable

from app.data.models import MarketFeatures, PortfolioSnapshot, Signal, SignalAction
from app.decision.ensemble import (
    EnsembleConfig,
    normalize_confidence,
    run_ensemble,
)
from app.decision.signals import (
    DecisionTrace,
    IntelligenceLayer,
    NormalizedSignal,
    LayeredSignal,
    TradeCandidate,
    Veto,
    VetoSource,
)
from app.monitoring import get_logger
from app.monitoring.logger import metrics
from app.utils.helpers import utc_now

logger = get_logger(__name__)


# ── Signal registry / dispatcher ───────────────────────────────────────


class SignalRegistry:
    """Collects and dispatches signals from all layers.

    Each layer registers its signal producers.  When ``collect()`` is
    called the registry runs each producer and normalizes the results
    into a common ``NormalizedSignal`` list per layer.
    """

    def __init__(self) -> None:
        self._producers: dict[IntelligenceLayer, list[Callable[..., list[NormalizedSignal]]]] = {
            IntelligenceLayer.RULES: [],
            IntelligenceLayer.ML: [],
            IntelligenceLayer.NLP: [],
        }

    def register(
        self,
        layer: IntelligenceLayer,
        producer: Callable[..., list[NormalizedSignal]],
    ) -> None:
        self._producers[layer].append(producer)

    def collect(
        self,
        features: MarketFeatures,
        portfolio: PortfolioSnapshot,
    ) -> dict[IntelligenceLayer, list[NormalizedSignal]]:
        result: dict[IntelligenceLayer, list[NormalizedSignal]] = {
            IntelligenceLayer.RULES: [],
            IntelligenceLayer.ML: [],
            IntelligenceLayer.NLP: [],
        }
        for layer, producers in self._producers.items():
            for fn in producers:
                try:
                    sigs = fn(features, portfolio)
                    result[layer].extend(sigs)
                except Exception:
                    logger.exception("signal_producer_error", layer=layer.value)
        return result


# ── Conversion helpers ─────────────────────────────────────────────────


def signal_to_normalized(
    signal: Signal, layer: IntelligenceLayer
) -> NormalizedSignal:
    """Convert a legacy ``Signal`` into a ``NormalizedSignal``."""
    direction = _action_direction(signal.action)
    normed = normalize_confidence(signal.confidence, layer)
    iid = signal.instrument_id or signal.token_id
    return NormalizedSignal(
        layer=layer,
        source_name=signal.strategy_name,
        market_id=signal.market_id,
        token_id=iid,
        instrument_id=iid,
        exchange=signal.exchange,
        action=signal.action,
        direction=direction,
        raw_confidence=signal.confidence,
        normalized_confidence=normed,
        expected_edge=0.0,
        suggested_price=signal.suggested_price,
        suggested_size=signal.suggested_size,
        rationale=signal.rationale,
        timestamp=signal.timestamp,
    )


# backward-compat alias used by main.py / backtesting
signal_to_layered = signal_to_normalized


# ── Decision engine ────────────────────────────────────────────────────


class DecisionEngine:
    """Evaluates all intelligence layers and produces trade candidates
    with full, inspectable traces."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self._config = config or EnsembleConfig()

    @property
    def config(self) -> EnsembleConfig:
        return self._config

    def evaluate(
        self,
        market_id: str,
        token_id: str = "",
        features: MarketFeatures | None = None,
        portfolio: PortfolioSnapshot | None = None,
        l1_signals: list[NormalizedSignal] | None = None,
        l2_signals: list[NormalizedSignal] | None = None,
        l3_signals: list[NormalizedSignal] | None = None,
        *,
        instrument_id: str = "",
        exchange: str = "",
    ) -> tuple[TradeCandidate, DecisionTrace]:
        """Run the full ensemble pipeline.

        Returns ``(candidate, trace)`` — the caller (risk + execution)
        decides what to do with the candidate.  The trace is a complete
        audit record.
        """
        iid = instrument_id or token_id
        candidate, trace = run_ensemble(
            market_id=market_id,
            token_id=iid,
            instrument_id=iid,
            exchange=exchange,
            l1_signals=l1_signals or [],
            l2_signals=l2_signals or [],
            l3_signals=l3_signals or [],
            config=self._config,
        )

        metrics.increment("decision_evaluations")
        if candidate.blocked:
            metrics.increment("decision_blocked")

        logger.info("decision_evaluated", **trace.to_log_dict())

        return candidate, trace

    def set_risk_outcome(
        self,
        trace: DecisionTrace,
        approved: bool,
        reason: str = "",
    ) -> None:
        """Called after the risk manager has rendered its verdict."""
        trace.risk_approved = approved
        trace.risk_reason = reason
        if not approved:
            trace.vetoes.append(
                Veto(source=VetoSource.RISK_MANAGER, reason=reason)
            )

    def set_execution_outcome(
        self,
        trace: DecisionTrace,
        decision: str,
    ) -> None:
        """Record what the execution engine did (submitted / skipped / error)."""
        trace.execution_decision = decision


# ── Private helpers ────────────────────────────────────────────────────


def _action_direction(action: SignalAction) -> int:
    if action in {SignalAction.BUY_YES, SignalAction.BUY_NO}:
        return 1
    if action in {SignalAction.SELL_YES, SignalAction.SELL_NO}:
        return -1
    return 0
