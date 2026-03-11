"""
Normalized signal schema shared by all three intelligence layers, plus
the rich DecisionTrace that records every step of the decision process.

Every piece of data that contributes to a trade/no-trade decision is
captured here so the system is fully transparent and inspectable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.data.models import SignalAction
from app.utils.helpers import utc_now


# ── Enums ──────────────────────────────────────────────────────────────


class IntelligenceLayer(str, Enum):
    RULES = "rules"
    ML = "ml"
    NLP = "nlp"


class VetoSource(str, Enum):
    """Who or what can veto a trade."""
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    AGREEMENT_GATE = "agreement_gate"
    L1_VETO = "l1_veto"
    L3_SUPPRESS = "l3_suppress"
    CONFLICT = "conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LARGE_TRADE_ALIGNMENT = "large_trade_alignment"
    RISK_MANAGER = "risk_manager"


# ── Normalized Signal (shared by L1, L2, L3) ──────────────────────────


class NormalizedSignal(BaseModel):
    """The common schema that every intelligence layer must produce.

    Confidence is always in [0, 1].  Direction is always in {-1, 0, +1}.
    This normalization is enforced at creation time so the ensemble never
    has to guess what a signal means.
    """
    layer: IntelligenceLayer
    source_name: str
    market_id: str
    token_id: str = ""
    instrument_id: str = ""
    exchange: str = ""

    action: SignalAction
    direction: int = Field(ge=-1, le=1)
    raw_confidence: float = Field(ge=0.0, le=1.0)
    normalized_confidence: float = Field(ge=0.0, le=1.0)
    expected_edge: float = 0.0

    suggested_price: float | None = None
    suggested_size: float | None = None
    rationale: str = ""
    features_used: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "instrument_id" not in data and "token_id" in data:
            data["instrument_id"] = data["token_id"]
        elif "token_id" not in data and "instrument_id" in data:
            data["token_id"] = data["instrument_id"]
        super().__init__(**data)


# kept as alias for backward compat used in backtesting/pipeline code
LayeredSignal = NormalizedSignal


# ── Per-Layer Summary ──────────────────────────────────────────────────


class LayerSummary(BaseModel):
    """Human-readable summary of one layer's contribution to a decision."""
    layer: IntelligenceLayer
    signal_count: int = 0
    direction: int = 0
    mean_confidence: float = 0.0
    weighted_score: float = 0.0
    edge: float = 0.0
    signals: list[NormalizedSignal] = Field(default_factory=list)
    synopsis: str = ""


# ── Veto Record ────────────────────────────────────────────────────────


class Veto(BaseModel):
    """Records one reason a trade was blocked."""
    source: VetoSource
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)


# ── Trade Candidate ────────────────────────────────────────────────────


class TradeCandidate(BaseModel):
    """The ensemble's final recommendation for one instrument."""
    market_id: str
    token_id: str = ""
    instrument_id: str = ""
    exchange: str = ""
    action: SignalAction
    final_confidence: float = Field(ge=0.0, le=1.0)
    expected_edge: float = 0.0
    suggested_price: float | None = None
    suggested_size: float | None = None

    direction: int = 0
    weighted_score: float = 0.0
    layer_contributions: dict[str, float] = Field(default_factory=dict)
    layers_agreeing: int = 0
    layers_total: int = 0

    vetoes: list[Veto] = Field(default_factory=list)
    blocked: bool = False

    rationale: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


# ── Decision Trace ─────────────────────────────────────────────────────


class DecisionTrace(BaseModel):
    """Complete, inspectable audit record for one evaluation cycle.

    Every trade (taken or not) produces one of these so you can always
    answer "exactly why was this trade taken / not taken?"
    """
    market_id: str
    token_id: str = ""
    instrument_id: str = ""
    exchange: str = ""
    timestamp: datetime = Field(default_factory=utc_now)

    # Per-layer summaries
    level_1_summary: LayerSummary = Field(default_factory=lambda: LayerSummary(layer=IntelligenceLayer.RULES))
    level_2_summary: LayerSummary = Field(default_factory=lambda: LayerSummary(layer=IntelligenceLayer.ML))
    level_3_summary: LayerSummary = Field(default_factory=lambda: LayerSummary(layer=IntelligenceLayer.NLP))

    # Ensemble computation
    weighted_scores: dict[str, float] = Field(default_factory=dict)
    ensemble_direction: int = 0
    ensemble_confidence: float = 0.0
    ensemble_edge: float = 0.0

    # Gate results
    confidence_result: str = ""
    agreement_result: str = ""
    conflict_result: str = ""
    evidence_result: str = ""
    large_trade_result: str = ""

    # Vetoes and risk
    vetoes: list[Veto] = Field(default_factory=list)
    risk_approved: bool | None = None
    risk_reason: str = ""

    # Final outcome
    final_action: SignalAction = SignalAction.HOLD
    final_confidence: float = 0.0
    execution_decision: str = ""
    candidate: TradeCandidate | None = None

    def add_veto(self, source: VetoSource, reason: str, **detail: Any) -> None:
        self.vetoes.append(Veto(source=source, reason=reason, detail=detail))

    @property
    def was_blocked(self) -> bool:
        return len(self.vetoes) > 0

    def to_log_dict(self) -> dict[str, Any]:
        """Flat dictionary suitable for structured logging."""
        return {
            "market_id": self.market_id,
            "instrument_id": self.instrument_id or self.token_id,
            "exchange": self.exchange,
            "l1_direction": self.level_1_summary.direction,
            "l1_confidence": round(self.level_1_summary.mean_confidence, 4),
            "l1_score": round(self.level_1_summary.weighted_score, 4),
            "l1_synopsis": self.level_1_summary.synopsis,
            "l2_direction": self.level_2_summary.direction,
            "l2_confidence": round(self.level_2_summary.mean_confidence, 4),
            "l2_score": round(self.level_2_summary.weighted_score, 4),
            "l2_synopsis": self.level_2_summary.synopsis,
            "l3_direction": self.level_3_summary.direction,
            "l3_confidence": round(self.level_3_summary.mean_confidence, 4),
            "l3_score": round(self.level_3_summary.weighted_score, 4),
            "l3_synopsis": self.level_3_summary.synopsis,
            "weighted_scores": {k: round(v, 4) for k, v in self.weighted_scores.items()},
            "ensemble_direction": self.ensemble_direction,
            "ensemble_confidence": round(self.ensemble_confidence, 4),
            "confidence_result": self.confidence_result,
            "agreement_result": self.agreement_result,
            "conflict_result": self.conflict_result,
            "evidence_result": self.evidence_result,
            "large_trade_result": self.large_trade_result,
            "vetoes": [f"{v.source.value}: {v.reason}" for v in self.vetoes],
            "risk_approved": self.risk_approved,
            "risk_reason": self.risk_reason,
            "final_action": self.final_action.value,
            "final_confidence": round(self.final_confidence, 4),
            "execution_decision": self.execution_decision,
        }
