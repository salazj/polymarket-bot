# Decision Engine

## Overview

The decision engine is the central orchestration layer that combines signals from three intelligence levels into auditable, transparent trade candidates. It never places orders directly — it only produces `TradeCandidate` objects that pass through independent, deterministic risk controls.

Every evaluation produces a full `DecisionTrace` so you can always answer: *"exactly why was this trade taken or not taken?"*

## Signal Flow

```
L1 Rule Signals ─┐
L2 ML Signals ───┼→ Confidence Normalization → Layer Summaries → Weighted Aggregation
L3 NLP Signals ──┘                                                      │
                                                            ┌───────────┘
                                                            ▼
                                                   Gate Pipeline
                                                   ├─ Minimum evidence
                                                   ├─ Confidence threshold
                                                   ├─ Agreement gating
                                                   ├─ Conflict detection
                                                   ├─ L1 veto
                                                   ├─ L3 suppress
                                                   └─ Large trade alignment
                                                            │
                                                            ▼
                                               TradeCandidate + DecisionTrace
                                                            │
                                                            ▼
                                               Risk Manager (deterministic, never AI)
                                                            │
                                                            ▼
                                               Execution Engine (if approved)
```

## Intelligence Layers

### Level 1: Rule-Based (deterministic)
- Passive market maker, momentum scalper, spread/liquidity strategies
- Outputs: `Signal` objects → `NormalizedSignal` with `layer=RULES`
- Always available, no external dependencies
- Fully interpretable; can veto trades in conservative mode

### Level 2: ML Prediction (tabular models)
- Logistic regression, gradient boosting, random forest
- Trained offline via `scripts/train_model.py`
- Outputs predicted probability and expected edge
- Converted to `NormalizedSignal` with `layer=ML`

### Level 3: NLP/Event AI (text processing)
- Keyword classifier (baseline, always available)
- Optional LLM classifier adapter
- News ingestion → classify → map to markets → NLP signals
- Converted to `NormalizedSignal` with `layer=NLP`
- Can suppress trades when strongly opposing (breaking-news safety)

## Normalized Signal Schema

All three layers produce `NormalizedSignal` objects with a common schema:

| Field | Type | Description |
|-------|------|-------------|
| `layer` | enum | RULES / ML / NLP |
| `direction` | int | +1 (buy), -1 (sell), 0 (hold) |
| `raw_confidence` | float 0-1 | Layer's native confidence |
| `normalized_confidence` | float 0-1 | Sigmoid-squashed to common range |
| `expected_edge` | float | Predicted edge after costs |
| `rationale` | str | Why this signal was generated |
| `features_used` | list | What data the signal is based on |

### Confidence Normalization

Each layer has different native confidence semantics:
- **L1 rules**: threshold-based (often binary-ish)
- **L2 ML**: calibrated probability from sklearn
- **L3 NLP**: keyword-match × relevance (often low)

A per-layer sigmoid squash normalizes these so that a 0.7 from L2 and 0.7 from L1 mean roughly the same thing:

```
normalized = 1 / (1 + exp(-k × (raw - midpoint)))
```

| Layer | Midpoint | Steepness |
|-------|----------|-----------|
| RULES | 0.50 | 8.0 |
| ML | 0.55 | 10.0 |
| NLP | 0.40 | 6.0 |

## Ensemble Logic

### Step 1: Layer Summaries

Each layer's signals are summarized:
```
raw_score = mean(signal.direction × signal.normalized_confidence)
weighted_score = raw_score × (layer_weight / total_weight)
```

A `LayerSummary` captures signal count, direction, mean confidence, weighted score, edge, and a human-readable synopsis.

### Step 2: Weighted Aggregation

```
net_score = l1.weighted_score + l2.weighted_score + l3.weighted_score
direction = sign(net_score)
confidence = min(|net_score|, 1.0)
```

Default weights: L1=0.30, L2=0.40, L3=0.30 (configurable via env).

### Step 3: Gate Pipeline (Veto System)

Every gate that fails adds a `Veto` to the trace. All vetoes are accumulated — the trace shows *every* reason a trade was blocked, not just the first one.

1. **Minimum Evidence**: total signals across all layers must meet `MIN_EVIDENCE_SIGNALS`
2. **Confidence Threshold**: ensemble confidence must meet `MIN_ENSEMBLE_CONFIDENCE`
3. **Agreement Gating**: at least `MIN_LAYERS_AGREE` active layers must point in the same direction
4. **Conflict Detection**: if layers disagree and the net score is below `CONFLICT_TOLERANCE`, the trade is too ambiguous
5. **L1 Veto**: in conservative mode, if L1 rules oppose the ensemble direction, the trade is blocked
6. **L3 Suppress**: if L3 (NLP) strongly opposes the ensemble (confidence > 0.6 against), the trade is blocked
7. **Large Trade Alignment**: trades above `LARGE_TRADE_THRESHOLD` in size require `LARGE_TRADE_MIN_LAYERS` layers to agree

### Veto Sources

| Source | What it means |
|--------|---------------|
| `confidence_threshold` | Ensemble confidence too low |
| `agreement_gate` | Not enough layers agree |
| `l1_veto` | Rule layer opposes (conservative mode) |
| `l3_suppress` | NLP/news strongly opposes |
| `conflict` | Layers disagree, net score ambiguous |
| `insufficient_evidence` | Too few signals total |
| `large_trade_alignment` | Large trade needs more layer agreement |
| `risk_manager` | Post-ensemble deterministic risk check failed |

## Decision Modes

| Mode | Min Conf | Min Agree | L1 Veto | Min Evidence | Large Trade Layers | Conflict Tol |
|------|----------|-----------|---------|-------------|-------------------|-------------|
| Conservative | 0.65 | 2 | Yes | 2 | 3 | 0.15 |
| Balanced | 0.55 | 2 | No | 1 | 2 | 0.30 |
| Aggressive | 0.40 | 1 | No | 1 | 1 | 0.50 |

Set via `DECISION_MODE=conservative` (default) in `.env`.

## Decision Trace

Every evaluation produces a `DecisionTrace` with these fields:

```
DecisionTrace
├── market_id, token_id, timestamp
├── level_1_summary          ← LayerSummary (signal_count, direction, confidence, score, synopsis)
├── level_2_summary          ← LayerSummary
├── level_3_summary          ← LayerSummary
├── weighted_scores          ← {l1: ..., l2: ..., l3: ...}
├── ensemble_direction       ← +1 / -1 / 0
├── ensemble_confidence      ← 0.0 - 1.0
├── confidence_result        ← "PASS: 0.72 >= 0.65"
├── agreement_result         ← "PASS: 2/3 agree (need 2)"
├── conflict_result          ← "PASS: layers agree"
├── evidence_result          ← "PASS: 3 >= 2"
├── large_trade_result       ← "N/A: size=1.0 < threshold 3.0"
├── vetoes                   ← [{source, reason, detail}, ...]
├── risk_approved            ← true/false (filled after risk check)
├── risk_reason              ← "" or reason string
├── final_action             ← BUY_YES / SELL_YES / HOLD
├── final_confidence         ← 0.0 - 1.0
├── execution_decision       ← "order_submitted:..." or "skipped"
└── candidate                ← full TradeCandidate object
```

The `to_log_dict()` method produces a flat dictionary for structured logging.

## Signal Registry

The `SignalRegistry` class is a dispatcher that collects signal producers for each layer:

```python
registry = SignalRegistry()
registry.register(IntelligenceLayer.RULES, my_l1_producer)
registry.register(IntelligenceLayer.ML, my_l2_producer)
signals = registry.collect(features, portfolio)
```

Producer errors are caught and logged — a failing producer never crashes the decision cycle.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DECISION_MODE` | conservative | conservative / balanced / aggressive |
| `ENSEMBLE_WEIGHT_L1` | 0.30 | Weight for rule-based layer |
| `ENSEMBLE_WEIGHT_L2` | 0.40 | Weight for ML prediction layer |
| `ENSEMBLE_WEIGHT_L3` | 0.30 | Weight for NLP/event layer |
| `MIN_ENSEMBLE_CONFIDENCE` | 0.60 | Minimum ensemble confidence to proceed |
| `MIN_LAYERS_AGREE` | 2 | Minimum agreeing layers |
| `MIN_EVIDENCE_SIGNALS` | 2 | Minimum total signals required |
| `LARGE_TRADE_THRESHOLD` | 3.0 | Size above which extra alignment is needed |
| `LARGE_TRADE_MIN_LAYERS` | 3 | Layers required for large trades |
| `CONFLICT_TOLERANCE` | 0.15 | Net score below this with disagreement = conflict |

## Examples

**Strong buy (approved, all layers agree):**
```
level_1_summary: rules: 1 signal(s) | dir=BUY | conf=0.917 | score=+0.275
level_2_summary: ml: 1 signal(s) | dir=BUY | conf=0.953 | score=+0.381
level_3_summary: nlp: 1 signal(s) | dir=BUY | conf=0.858 | score=+0.257
weighted_scores: {l1: +0.275, l2: +0.381, l3: +0.257}
ensemble_direction: +1, confidence: 0.913
confidence_result: PASS
agreement_result: PASS (3/3 agree)
vetoes: []
final_action: BUY_YES
```

**Blocked — insufficient evidence + low confidence:**
```
level_1_summary: rules: 1 signal(s) | dir=BUY | conf=0.310 | score=+0.093
level_2_summary: ml: no signals
level_3_summary: nlp: no signals
evidence_result: FAIL: 1 < 2
confidence_result: FAIL: 0.093 < 0.65
vetoes:
  - insufficient_evidence: only 1 signal(s) present, need 2
  - confidence_threshold: confidence 0.093 < threshold 0.65
final_action: HOLD
```

**Blocked by L3 suppress (breaking news contradicts):**
```
level_1_summary: rules: 1 signal(s) | dir=BUY | score=+0.275
level_2_summary: ml: 1 signal(s) | dir=BUY | score=+0.381
level_3_summary: nlp: 1 signal(s) | dir=SELL | conf=0.950 | score=-0.285
vetoes:
  - l3_suppress: L3 NLP strongly opposes (dir=SELL, conf=0.950)
final_action: HOLD
```

**Blocked by conflict (ambiguous signals):**
```
level_1_summary: rules: 1 signal(s) | dir=BUY | score=+0.100
level_2_summary: ml: 1 signal(s) | dir=SELL | score=-0.080
vetoes:
  - conflict: layers disagree ({l1: 1, l2: -1}) and net score +0.020 is below tolerance 0.15
final_action: HOLD
```
