"""
Comprehensive tests for the decision engine: confidence normalization,
ensemble aggregation, agreement gating, conflict detection, veto system,
minimum evidence threshold, large-trade alignment, L1/L3 vetoes,
decision modes, DecisionTrace audit completeness, and signal conversion.
"""

from __future__ import annotations

import pytest

from app.data.models import MarketFeatures, PortfolioSnapshot, Signal, SignalAction
from app.decision.engine import DecisionEngine, SignalRegistry, signal_to_normalized
from app.decision.ensemble import (
    DecisionMode,
    EnsembleConfig,
    detect_conflict,
    normalize_confidence,
    run_ensemble,
    summarize_layer,
)
from app.decision.signals import (
    DecisionTrace,
    IntelligenceLayer,
    LayerSummary,
    NormalizedSignal,
    TradeCandidate,
    Veto,
    VetoSource,
)

MARKET = "mkt-1"
TOKEN = "tok-1"


def _ns(
    layer: IntelligenceLayer,
    action: SignalAction,
    confidence: float,
    edge: float = 0.0,
    price: float | None = None,
    size: float | None = None,
    name: str = "test",
) -> NormalizedSignal:
    """Build a NormalizedSignal with direction computed from action."""
    if action in {SignalAction.BUY_YES, SignalAction.BUY_NO}:
        direction = 1
    elif action in {SignalAction.SELL_YES, SignalAction.SELL_NO}:
        direction = -1
    else:
        direction = 0
    normed = normalize_confidence(confidence, layer)
    return NormalizedSignal(
        layer=layer,
        source_name=name,
        market_id=MARKET,
        token_id=TOKEN,
        action=action,
        direction=direction,
        raw_confidence=confidence,
        normalized_confidence=normed,
        expected_edge=edge,
        suggested_price=price,
        suggested_size=size,
    )


def _make_features() -> MarketFeatures:
    return MarketFeatures(market_id=MARKET, token_id=TOKEN)


def _make_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(cash=100.0)


# ---------------------------------------------------------------------------
# Confidence normalization
# ---------------------------------------------------------------------------


class TestConfidenceNormalization:
    def test_zero_returns_zero(self) -> None:
        assert normalize_confidence(0.0, IntelligenceLayer.RULES) == 0.0

    def test_one_returns_one(self) -> None:
        assert normalize_confidence(1.0, IntelligenceLayer.ML) == 1.0

    def test_midpoint_maps_near_half(self) -> None:
        v = normalize_confidence(0.5, IntelligenceLayer.RULES)
        assert 0.45 <= v <= 0.55

    def test_high_confidence_maps_high(self) -> None:
        v = normalize_confidence(0.9, IntelligenceLayer.ML)
        assert v > 0.9

    def test_low_confidence_maps_low(self) -> None:
        v = normalize_confidence(0.1, IntelligenceLayer.NLP)
        assert v < 0.3

    def test_different_layers_give_different_results(self) -> None:
        r = normalize_confidence(0.6, IntelligenceLayer.RULES)
        m = normalize_confidence(0.6, IntelligenceLayer.ML)
        n = normalize_confidence(0.6, IntelligenceLayer.NLP)
        # At 0.6 the per-layer sigmoids should diverge
        assert r != m or m != n

    def test_monotonically_increasing(self) -> None:
        for layer in IntelligenceLayer:
            prev = 0.0
            for raw in [0.1, 0.3, 0.5, 0.7, 0.9]:
                v = normalize_confidence(raw, layer)
                assert v >= prev
                prev = v


# ---------------------------------------------------------------------------
# Layer summarization
# ---------------------------------------------------------------------------


class TestSummarizeLayer:
    def test_empty_signals(self) -> None:
        s = summarize_layer([], IntelligenceLayer.RULES, 0.3)
        assert s.signal_count == 0
        assert s.direction == 0
        assert "no signals" in s.synopsis

    def test_single_buy(self) -> None:
        sig = _ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8, edge=0.01)
        s = summarize_layer([sig], IntelligenceLayer.RULES, 0.3)
        assert s.signal_count == 1
        assert s.direction == 1
        assert s.mean_confidence > 0
        assert s.weighted_score > 0
        assert s.edge == pytest.approx(0.01)
        assert "BUY" in s.synopsis

    def test_single_sell(self) -> None:
        sig = _ns(IntelligenceLayer.ML, SignalAction.SELL_YES, 0.6)
        s = summarize_layer([sig], IntelligenceLayer.ML, 0.4)
        assert s.direction == -1
        assert s.weighted_score < 0

    def test_mixed_signals(self) -> None:
        s1 = _ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8)
        s2 = _ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.3)
        s = summarize_layer([s1, s2], IntelligenceLayer.RULES, 0.3)
        assert s.signal_count == 2
        assert s.direction == 1  # net positive


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_no_conflict_when_layers_agree(self) -> None:
        s1 = LayerSummary(layer=IntelligenceLayer.RULES, signal_count=1, direction=1, weighted_score=0.2)
        s2 = LayerSummary(layer=IntelligenceLayer.ML, signal_count=1, direction=1, weighted_score=0.3)
        is_c, _ = detect_conflict({"l1": s1, "l2": s2}, 0.15)
        assert is_c is False

    def test_conflict_when_disagreeing_and_weak_net(self) -> None:
        s1 = LayerSummary(layer=IntelligenceLayer.RULES, signal_count=1, direction=1, weighted_score=0.1)
        s2 = LayerSummary(layer=IntelligenceLayer.ML, signal_count=1, direction=-1, weighted_score=-0.08)
        is_c, desc = detect_conflict({"l1": s1, "l2": s2}, 0.15)
        assert is_c is True
        assert "disagree" in desc

    def test_no_conflict_when_net_score_strong(self) -> None:
        s1 = LayerSummary(layer=IntelligenceLayer.RULES, signal_count=1, direction=1, weighted_score=0.4)
        s2 = LayerSummary(layer=IntelligenceLayer.ML, signal_count=1, direction=-1, weighted_score=-0.05)
        is_c, _ = detect_conflict({"l1": s1, "l2": s2}, 0.15)
        assert is_c is False

    def test_no_conflict_single_layer(self) -> None:
        s1 = LayerSummary(layer=IntelligenceLayer.RULES, signal_count=1, direction=1, weighted_score=0.3)
        s2 = LayerSummary(layer=IntelligenceLayer.ML, signal_count=0, direction=0, weighted_score=0.0)
        is_c, _ = detect_conflict({"l1": s1, "l2": s2}, 0.15)
        assert is_c is False


# ---------------------------------------------------------------------------
# Full ensemble: basic directional behavior
# ---------------------------------------------------------------------------


class TestEnsembleDirection:
    def test_all_layers_buy(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.BUY_YES, 0.6)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        assert c.action == SignalAction.BUY_YES
        assert c.blocked is False
        assert c.layers_agreeing == 3

    def test_all_layers_sell(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.SELL_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.SELL_YES, 0.6)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        assert c.action == SignalAction.SELL_YES
        assert c.blocked is False

    def test_no_signals_produces_hold(self) -> None:
        cfg = EnsembleConfig(min_confidence=0.1, min_layers_agree=1, min_evidence_signals=0)
        c, _ = run_ensemble(MARKET, TOKEN, [], [], [], cfg)
        assert c.action == SignalAction.HOLD

    def test_weights_affect_outcome(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.9)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.5)]
        cfg = EnsembleConfig(
            weight_l1=0.8, weight_l2=0.2, weight_l3=0.0,
            min_confidence=0.05, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        assert c.action == SignalAction.SELL_YES


# ---------------------------------------------------------------------------
# Confidence thresholding (veto)
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_below_threshold_vetoes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.2)]
        cfg = EnsembleConfig(
            min_confidence=0.5, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        assert c.blocked is True
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.CONFIDENCE_THRESHOLD in veto_sources
        assert "FAIL" in t.confidence_result

    def test_above_threshold_passes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.9)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.9)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.BUY_YES, 0.9)]
        cfg = EnsembleConfig(
            min_confidence=0.3, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        assert c.blocked is False
        assert "PASS" in t.confidence_result


# ---------------------------------------------------------------------------
# Agreement gating
# ---------------------------------------------------------------------------


class TestAgreementGating:
    def test_two_of_three_required_passes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.HOLD, 0.3)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=2,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        assert c.blocked is False
        assert c.layers_agreeing >= 2
        assert "PASS" in t.agreement_result

    def test_two_of_three_required_blocks(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.SELL_YES, 0.6)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.SELL_YES, 0.5)]
        cfg = EnsembleConfig(
            min_confidence=0.01, min_layers_agree=3,
            require_l1_approval=False, min_evidence_signals=1,
            conflict_tolerance=1.0,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        # Only 2 layers can agree at most (sell side has L2+L3) so 3-agree blocks
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.AGREEMENT_GATE in veto_sources

    def test_single_layer_skip_if_below_min(self) -> None:
        """When only 1 layer is active but min_layers_agree=2, skip gate
        (don't penalize a market with sparse data)."""
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=2,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.AGREEMENT_GATE not in veto_sources


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    def test_opposing_layers_with_weak_net_vetoes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.5)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.SELL_YES, 0.45)]
        cfg = EnsembleConfig(
            min_confidence=0.01, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
            conflict_tolerance=0.5,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.CONFLICT in veto_sources
        assert "FAIL" in t.conflict_result

    def test_strong_agreement_no_conflict_veto(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.9)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.CONFLICT not in veto_sources


# ---------------------------------------------------------------------------
# Minimum evidence threshold
# ---------------------------------------------------------------------------


class TestMinimumEvidence:
    def test_insufficient_signals_vetoes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.9)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=3,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.INSUFFICIENT_EVIDENCE in veto_sources
        assert "FAIL" in t.evidence_result

    def test_sufficient_signals_passes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.9)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=2,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.INSUFFICIENT_EVIDENCE not in veto_sources
        assert "PASS" in t.evidence_result


# ---------------------------------------------------------------------------
# L1 veto
# ---------------------------------------------------------------------------


class TestL1Veto:
    def test_l1_opposes_blocks_when_required(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.9)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.01, min_layers_agree=1,
            require_l1_approval=True, min_evidence_signals=1,
            conflict_tolerance=1.0,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.L1_VETO in veto_sources

    def test_l1_agrees_no_veto(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.6)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=True, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.L1_VETO not in veto_sources

    def test_l1_absent_no_veto(self) -> None:
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=True, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, [], l2, [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.L1_VETO not in veto_sources


# ---------------------------------------------------------------------------
# L3 suppress
# ---------------------------------------------------------------------------


class TestL3Suppress:
    def test_l3_strong_opposition_vetoes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.SELL_YES, 0.95)]
        cfg = EnsembleConfig(
            min_confidence=0.01, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
            conflict_tolerance=1.0,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.L3_SUPPRESS in veto_sources

    def test_l3_mild_opposition_no_suppress(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.SELL_YES, 0.2)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
            conflict_tolerance=1.0,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.L3_SUPPRESS not in veto_sources


# ---------------------------------------------------------------------------
# Large trade alignment gate
# ---------------------------------------------------------------------------


class TestLargeTradeAlignment:
    def test_large_trade_insufficient_alignment_vetoes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8, size=5.0, price=0.5)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
            large_trade_threshold=3.0, large_trade_min_layers=2,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.LARGE_TRADE_ALIGNMENT in veto_sources
        assert "FAIL" in t.large_trade_result

    def test_large_trade_with_alignment_passes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8, size=5.0, price=0.5)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8, size=5.0, price=0.5)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
            large_trade_threshold=3.0, large_trade_min_layers=2,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.LARGE_TRADE_ALIGNMENT not in veto_sources
        assert "PASS" in t.large_trade_result

    def test_small_trade_skips_gate(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8, size=1.0, price=0.5)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
            large_trade_threshold=3.0, large_trade_min_layers=3,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        veto_sources = [v.source for v in c.vetoes]
        assert VetoSource.LARGE_TRADE_ALIGNMENT not in veto_sources
        assert "N/A" in t.large_trade_result


# ---------------------------------------------------------------------------
# Decision modes
# ---------------------------------------------------------------------------


class TestDecisionModes:
    def test_conservative_defaults(self) -> None:
        cfg = EnsembleConfig(mode=DecisionMode.CONSERVATIVE)
        cfg.apply_mode_defaults()
        assert cfg.min_confidence == 0.65
        assert cfg.require_l1_approval is True
        assert cfg.min_evidence_signals == 2
        assert cfg.large_trade_min_layers == 3

    def test_balanced_defaults(self) -> None:
        cfg = EnsembleConfig(mode=DecisionMode.BALANCED)
        cfg.apply_mode_defaults()
        assert cfg.min_confidence == 0.55
        assert cfg.min_layers_agree == 2
        assert cfg.require_l1_approval is False
        assert cfg.conflict_tolerance == 0.30

    def test_aggressive_defaults(self) -> None:
        cfg = EnsembleConfig(mode=DecisionMode.AGGRESSIVE)
        cfg.apply_mode_defaults()
        assert cfg.min_confidence == 0.40
        assert cfg.min_layers_agree == 1
        assert cfg.min_evidence_signals == 1
        assert cfg.large_trade_min_layers == 1


# ---------------------------------------------------------------------------
# Decision Trace audit completeness
# ---------------------------------------------------------------------------


class TestDecisionTrace:
    def test_trace_contains_all_layer_summaries(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.BUY_YES, 0.5)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, l3, cfg)

        assert t.level_1_summary.signal_count == 1
        assert t.level_2_summary.signal_count == 1
        assert t.level_3_summary.signal_count == 1
        assert t.level_1_summary.direction == 1
        assert "BUY" in t.level_1_summary.synopsis
        assert t.market_id == MARKET
        assert t.token_id == TOKEN

    def test_trace_weighted_scores_match_candidate(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.6)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        assert t.weighted_scores == c.layer_contributions

    def test_trace_records_vetoes(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.1)]
        cfg = EnsembleConfig(
            min_confidence=0.9, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, [], [], cfg)
        assert len(t.vetoes) > 0
        assert t.was_blocked is True
        assert t.final_action == SignalAction.HOLD

    def test_trace_to_log_dict(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        log = t.to_log_dict()
        assert "l1_direction" in log
        assert "l2_confidence" in log
        assert "ensemble_direction" in log
        assert "final_action" in log
        assert "vetoes" in log
        assert isinstance(log["vetoes"], list)

    def test_trace_candidate_roundtrip(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        assert t.candidate is not None
        assert t.candidate.action == c.action
        assert t.candidate.final_confidence == c.final_confidence


# ---------------------------------------------------------------------------
# DecisionEngine.evaluate integration
# ---------------------------------------------------------------------------


class TestDecisionEngineEvaluate:
    def test_evaluate_returns_trace_and_candidate(self) -> None:
        engine = DecisionEngine(config=EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        ))
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8)]
        l3 = [_ns(IntelligenceLayer.NLP, SignalAction.BUY_YES, 0.5)]

        c, t = engine.evaluate(
            MARKET, TOKEN, _make_features(), _make_portfolio(), l1, l2, l3,
        )
        assert isinstance(c, TradeCandidate)
        assert isinstance(t, DecisionTrace)
        assert t.level_1_summary.signal_count == 1

    def test_set_risk_outcome(self) -> None:
        engine = DecisionEngine(config=EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        ))
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8)]
        _, t = engine.evaluate(
            MARKET, TOKEN, _make_features(), _make_portfolio(), l1, [], [],
        )
        engine.set_risk_outcome(t, approved=False, reason="max_exposure_exceeded")
        assert t.risk_approved is False
        assert "max_exposure" in t.risk_reason
        veto_sources = [v.source for v in t.vetoes]
        assert VetoSource.RISK_MANAGER in veto_sources

    def test_set_execution_outcome(self) -> None:
        engine = DecisionEngine(config=EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        ))
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.8)]
        _, t = engine.evaluate(
            MARKET, TOKEN, _make_features(), _make_portfolio(), l1, [], [],
        )
        engine.set_execution_outcome(t, "order_submitted:BT-000001")
        assert t.execution_decision == "order_submitted:BT-000001"


# ---------------------------------------------------------------------------
# Signal conversion
# ---------------------------------------------------------------------------


class TestSignalToNormalized:
    def test_converts_correctly(self) -> None:
        sig = Signal(
            strategy_name="momentum_scalper",
            market_id=MARKET,
            token_id=TOKEN,
            action=SignalAction.BUY_YES,
            confidence=0.75,
            suggested_price=0.55,
            rationale="test",
        )
        ns = signal_to_normalized(sig, IntelligenceLayer.RULES)
        assert ns.layer == IntelligenceLayer.RULES
        assert ns.source_name == "momentum_scalper"
        assert ns.action == SignalAction.BUY_YES
        assert ns.direction == 1
        assert ns.raw_confidence == 0.75
        assert 0.0 < ns.normalized_confidence <= 1.0

    def test_sell_direction(self) -> None:
        sig = Signal(
            strategy_name="test",
            market_id=MARKET,
            token_id=TOKEN,
            action=SignalAction.SELL_YES,
            confidence=0.6,
        )
        ns = signal_to_normalized(sig, IntelligenceLayer.ML)
        assert ns.direction == -1

    def test_hold_direction(self) -> None:
        sig = Signal(
            strategy_name="test",
            market_id=MARKET,
            token_id=TOKEN,
            action=SignalAction.HOLD,
            confidence=0.3,
        )
        ns = signal_to_normalized(sig, IntelligenceLayer.RULES)
        assert ns.direction == 0


# ---------------------------------------------------------------------------
# SignalRegistry
# ---------------------------------------------------------------------------


class TestSignalRegistry:
    def test_register_and_collect(self) -> None:
        registry = SignalRegistry()

        def l1_producer(features: MarketFeatures, portfolio: PortfolioSnapshot) -> list[NormalizedSignal]:
            return [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]

        registry.register(IntelligenceLayer.RULES, l1_producer)
        result = registry.collect(_make_features(), _make_portfolio())

        assert len(result[IntelligenceLayer.RULES]) == 1
        assert result[IntelligenceLayer.RULES][0].action == SignalAction.BUY_YES
        assert len(result[IntelligenceLayer.ML]) == 0
        assert len(result[IntelligenceLayer.NLP]) == 0

    def test_multiple_producers(self) -> None:
        registry = SignalRegistry()

        def p1(f: MarketFeatures, p: PortfolioSnapshot) -> list[NormalizedSignal]:
            return [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7)]

        def p2(f: MarketFeatures, p: PortfolioSnapshot) -> list[NormalizedSignal]:
            return [_ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.3)]

        registry.register(IntelligenceLayer.RULES, p1)
        registry.register(IntelligenceLayer.RULES, p2)

        result = registry.collect(_make_features(), _make_portfolio())
        assert len(result[IntelligenceLayer.RULES]) == 2

    def test_producer_error_handled(self) -> None:
        registry = SignalRegistry()

        def bad_producer(f: MarketFeatures, p: PortfolioSnapshot) -> list[NormalizedSignal]:
            raise ValueError("boom")

        registry.register(IntelligenceLayer.ML, bad_producer)
        result = registry.collect(_make_features(), _make_portfolio())
        assert len(result[IntelligenceLayer.ML]) == 0


# ---------------------------------------------------------------------------
# Best price/size selection
# ---------------------------------------------------------------------------


class TestBestPriceSize:
    def test_buy_picks_lowest_price(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.BUY_YES, 0.7, price=0.55, size=2.0)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.8, price=0.50, size=3.0)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        assert c.suggested_price == pytest.approx(0.50)
        assert c.suggested_size == pytest.approx(2.0)

    def test_sell_picks_highest_price(self) -> None:
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.7, price=0.45, size=2.0)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.SELL_YES, 0.8, price=0.50, size=3.0)]
        cfg = EnsembleConfig(
            min_confidence=0.1, min_layers_agree=1,
            require_l1_approval=False, min_evidence_signals=1,
        )
        c, _ = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        assert c.suggested_price == pytest.approx(0.50)
        assert c.suggested_size == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Multiple vetoes accumulate
# ---------------------------------------------------------------------------


class TestMultipleVetoes:
    def test_multiple_vetoes_all_recorded(self) -> None:
        """A trade that fails multiple gates should have all vetoes logged."""
        l1 = [_ns(IntelligenceLayer.RULES, SignalAction.SELL_YES, 0.15)]
        l2 = [_ns(IntelligenceLayer.ML, SignalAction.BUY_YES, 0.15)]
        cfg = EnsembleConfig(
            min_confidence=0.5, min_layers_agree=2,
            require_l1_approval=True, min_evidence_signals=3,
            conflict_tolerance=0.5,
        )
        c, t = run_ensemble(MARKET, TOKEN, l1, l2, [], cfg)
        assert c.blocked is True
        assert len(c.vetoes) >= 2
        sources = {v.source for v in c.vetoes}
        # Should hit at least confidence + evidence thresholds
        assert VetoSource.INSUFFICIENT_EVIDENCE in sources
        assert VetoSource.CONFIDENCE_THRESHOLD in sources
