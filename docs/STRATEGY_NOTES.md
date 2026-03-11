# Strategy Notes

## Design Principles

1. **No strategy promises profits.** All strategies are research tools with uncertain outcomes.
2. **Strategies only produce signals.** They never place orders or mutate external state.
3. **Conservative by default.** Prefer fewer, higher-quality trades over constant action.
4. **Transparent reasoning.** Every signal includes a rationale string for debugging.

## Strategy 1: Passive Market Maker

**File**: `app/strategies/passive_market_maker.py`

### Idea
Place small maker quotes when the bid-ask spread is wide enough to capture edge after fees. This is a passive strategy: it joins existing levels rather than crossing the spread.

### Entry Conditions
- Spread between `MIN_EDGE_SPREAD` (2 cents) and `MAX_SPREAD_THRESHOLD` (15 cents)
- Both bid and ask depth above `MIN_DEPTH` ($5 within 5 cents)
- Data freshness under 30 seconds
- Orderbook imbalance not extreme

### Behavior
- In balanced books: joins the bid slightly inside the best level
- In imbalanced books: leans toward the lighter side (less adverse selection)
- Never crosses the spread

### Risk Profile
- Low aggression, low edge per trade
- Requires patient execution
- Main risk: being picked off when prices move sharply

---

## Strategy 2: Momentum Scalper

**File**: `app/strategies/momentum_scalper.py`

### Idea
Trade in the direction of short-term momentum when confirmed by trade flow. Only acts when momentum and flow agree in direction.

### Entry Conditions
- 1-minute momentum exceeds 0.5 cents
- Net trade flow exceeds $2 in the same direction
- Acceptable spread and liquidity
- Cooldown of 120 seconds between signals (prevents overtrading)

### Behavior
- BUY when momentum and flow are both positive (joins bid)
- SELL when both are negative (joins ask)
- Ignores ambiguous or conflicting signals

### Risk Profile
- Higher conviction per trade but fewer trades
- Cooldown prevents chasing
- Main risk: momentum reversal after entry

---

## Strategy 3: Event Probability Model

**File**: `app/strategies/event_probability_model.py`

### Idea
Use a trained tabular ML classifier to predict short-horizon probability direction from market features. Acts only when prediction confidence exceeds a threshold.

### Pipeline
1. Extract 11 engineered features from current market state
2. Feed through a pre-trained model (logistic regression or gradient boosting)
3. Model outputs probability of upward price movement
4. Signal generated only when confidence > 60%

### Training
- Run `python scripts/train_baseline_model.py`
- Uses chronological train/validation/test split (60/20/20)
- Compares logistic regression and gradient boosting
- Selects model with lowest validation log loss
- Saves to `model_artifacts/baseline_model.joblib`
- Generates feature importance and calibration reports

### Features Used
| Feature | Description |
|---------|-------------|
| spread | Current bid-ask spread |
| microprice | Size-weighted midpoint |
| orderbook_imbalance | Bid/ask volume ratio |
| bid_depth_5c | Bid liquidity within 5 cents |
| ask_depth_5c | Ask liquidity within 5 cents |
| recent_trade_flow | Net signed volume (1 min) |
| volatility_1m | Price change standard deviation |
| momentum_1m | Price change over 1 minute |
| momentum_5m | Price change over 5 minutes |
| momentum_15m | Price change over 15 minutes |
| trade_count_1m | Number of trades in 1 minute |

### Risk Profile
- Only trades when model is confident
- Requires sufficient training data for reliable predictions
- Main risk: model overfitting to historical patterns
- Mitigation: conservative threshold, small position sizes

---

## Strategy 4: Sentiment Adapter (Stub)

**File**: `app/strategies/sentiment_adapter.py`

### Status
Scaffold only — currently generates no signals.

### Intended Architecture
- Abstract `SentimentProvider` interface
- `NullSentimentProvider` as default (no-op)
- Future implementations can wrap free or paid APIs (news, social media, LLMs)
- Provider interface: `get_sentiment(market_id, query) -> list[SentimentScore]`
- Scores are normalized to [-1, +1] range

### Extension Points
- Implement `SentimentProvider` for your data source
- Inject it into `SentimentAdapter(settings, provider=your_provider)`
- Score thresholds and blending logic are customizable

---

## Intelligence Layer Integration

Strategies are organized into three intelligence layers within the decision engine:

### Level 1 — Rule-Based
All traditional strategies (passive market maker, momentum scalper) run as L1. Their signals are deterministic, interpretable, and always available. In conservative mode, L1 has veto power over the ensemble.

### Level 2 — Machine Learning
The event probability model runs as L2. It uses offline-trained tabular ML models (logistic regression, gradient boosting, random forest) with walk-forward validation and calibration. Train via `python scripts/train_model.py`.

### Level 3 — NLP/Event AI
The NLP subsystem classifies text (headlines, news items) and maps them to markets. It uses a keyword classifier baseline with an optional LLM adapter. Configure providers via `NLP_PROVIDER` env var. See `docs/NLP_PROVIDERS.md` for details.

All three layers feed signals into the `DecisionEngine`, which aggregates them via weighted voting, agreement gating, and confidence thresholding. See `docs/DECISION_ENGINE.md` for full details.

---

## Adding a New Strategy

1. Create a new file in `app/strategies/`
2. Subclass `BaseStrategy`
3. Set a unique `name` class attribute
4. Implement `generate_signal(features, portfolio) -> Signal | None`
5. Decorate with `@StrategyRegistry.register`
6. Select via `STRATEGY=your_name` in `.env` or `--strategy your_name` CLI flag
