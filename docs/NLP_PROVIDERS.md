# NLP / Level 3 Intelligence

## Overview

The NLP subsystem (Level 3) processes text from pluggable providers, normalizes it, classifies it across multiple dimensions, maps it to Polymarket markets, and generates structured trading signals. All providers and classifiers are optional — the bot runs normally with zero external APIs.

**Key design principle:** AI generates *structured signals*, never uncontrolled trade execution. The NLP layer feeds the decision engine, which evaluates signals alongside L1 and L2, subject to deterministic risk controls.

## Architecture

```
Providers (Mock / File / RSS / future APIs)
         │
         ▼
┌──────────────────────────────────────────────┐
│             NewsIngestionService              │
│  polls → deduplicates → caches               │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│                NLP Pipeline                   │
│                                              │
│  1. TextNormalizer                           │
│     ├─ strip HTML                            │
│     ├─ normalize Unicode                     │
│     ├─ replace URLs                          │
│     ├─ collapse whitespace                   │
│     ├─ truncate                              │
│     └─ near-duplicate detection (trigram)    │
│                                              │
│  2. Classifier (keyword or hybrid)           │
│     ├─ EventTypeClassifier                   │
│     ├─ SentimentClassifier (with negation)   │
│     ├─ UrgencyClassifier                     │
│     ├─ RelevanceClassifier (IDF-weighted)    │
│     ├─ EntityExtractor                       │
│     └─ Optional LLM adapter (see below)      │
│                                              │
│  3. MarketMapper                             │
│     ├─ IDF-weighted token overlap            │
│     ├─ entity matching                       │
│     ├─ manual overrides                      │
│     └─ ambiguity detection                   │
│                                              │
│  4. Signal Generator → NlpSignal             │
│                                              │
│  5. Processing Trace (logged per item)       │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│  nlp_signal_to_layered() → NormalizedSignal  │
│  (confidence normalized via per-layer        │
│   sigmoid, direction set, metadata attached) │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
            Decision Engine
```

## LLM Provider Architecture

The LLM is **optional**. When `LLM_PROVIDER=none` (default), the keyword classifier handles everything. When enabled, the `HybridClassifier` runs keywords first, then the LLM — using whichever result has higher confidence.

```
┌──────────────────────────────────────────────┐
│          build_llm_classifier()              │ ◀── Settings
│  (factory function in llm_provider.py)       │
└──────────┬───────────┬───────────┬───────────┘
           │           │           │
   ┌───────▼──┐ ┌──────▼──────┐ ┌─▼────────────┐
   │ OpenAI   │ │  LocalOS    │ │ MockLLM      │
   │ Compat.  │ │  Provider   │ │ Provider     │
   │ LLM      │ │ (Llama/    │ │ (testing)    │
   │ (hosted) │ │  Mistral/  │ │              │
   │          │ │  Qwen)     │ │              │
   └────┬─────┘ └─────┬──────┘ └──────┬───────┘
        │              │               │
        └──────┬───────┘               │
               ▼                       │
   ┌───────────────────┐               │
   │  ModelFamily      │               │
   │  (detects model   │               │
   │   → selects       │               │
   │   prompt template)│               │
   └───────────────────┘               │
               │                       │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  LlmOutputValidator   │
               │  ├─ strip fences      │
               │  ├─ extract JSON      │
               │  ├─ validate fields   │
               │  ├─ validate types    │
               │  ├─ validate ranges   │
               │  ├─ validate enums    │
               │  └─ return errors[]   │
               └───────────────────────┘
```

### Configuration

Set these in `.env`:

```bash
# Provider mode: none | mock | local_open_source | hosted_api
LLM_PROVIDER=none

# Model name (passed to the API)
LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

# API endpoint
LLM_BASE_URL=http://localhost:8000/v1

# API key (optional for local models)
LLM_API_KEY=

# Request timeout
LLM_TIMEOUT_SECONDS=30

# Confidence threshold: LLM result used only when above this
LLM_CONFIDENCE_THRESHOLD=0.5
```

### Provider Modes

| Mode | Class | Description |
|------|-------|-------------|
| `none` | — | No LLM, keyword classifiers only (default) |
| `mock` | `MockLLMProvider` | In-process mock for testing, no network |
| `local_open_source` | `LocalOpenSourceProvider` | Local model via OpenAI-compatible API |
| `hosted_api` | `OpenAICompatibleLLM` | Hosted API (OpenAI, Together, Groq, etc.) |

### Model Family Detection

When `local_open_source` or `hosted_api` is selected, the system auto-detects the model family from `LLM_MODEL_NAME` and selects an optimized prompt template:

| Family | Pattern Match | Prompt Style |
|--------|--------------|--------------|
| Llama | `llama` in name | Strict JSON-only instructions, no-fence rule |
| Mistral | `mistral`, `mixtral` | `[INST]` wrapper, explicit format rules |
| Qwen | `qwen` | JSON assistant framing |
| Generic | anything else | Standard classification prompt |

Each prompt requests the same JSON schema but uses wording patterns that each model family handles best.

### LLM Structured Output

The LLM is asked to produce **only** a JSON object with these fields:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `event_type` | string | enum | One of: legal_ruling, regulatory, election, economic, celebrity, sports, geopolitical, crypto, other |
| `sentiment` | string | enum | One of: bullish, bearish, neutral |
| `sentiment_score` | float | [-1.0, 1.0] | Directional impact strength |
| `urgency` | float | [0.0, 1.0] | Time-sensitivity |
| `relevance` | float | [0.0, 1.0] | How relevant to the market question |
| `confidence` | float | [0.0, 1.0] | Model's self-assessed confidence |
| `rationale` | string | — | One-sentence explanation |
| `entities` | list[str] | — | Key named entities mentioned |

The LLM **never**:
- Places trades
- Bypasses risk controls
- Makes execution decisions
- Accesses wallet keys or credentials

### Output Validation

`LlmOutputValidator` (`app/nlp/classifier.py`) validates every LLM response:

1. **Strips markdown fences** — many models wrap JSON in ` ```json ``` `
2. **Extracts JSON** — finds the first `{...}` object even if surrounded by text
3. **Validates JSON parsing** — catches malformed JSON gracefully
4. **Validates required fields** — all 8 fields must be present
5. **Validates types** — each field must match its expected type
6. **Validates enum values** — unknown event_type → `other`, unknown sentiment → `neutral`
7. **Validates ranges** — out-of-range floats are clamped and logged
8. **Validates entities** — non-string items in the entities list are filtered

Returns `(ClassificationResult, list[LlmValidationError])`. The result is always usable (safe defaults for invalid fields). The error list lets the `LlmClassifierAdapter` decide whether to trust or reject the result.

**Rejection threshold**: If the number of validation errors exceeds `max_validation_errors` (default 3), the entire LLM output is rejected and the hybrid classifier falls back to keyword-only classification.

### Example: Using a Local Llama Model

```bash
# 1. Start vLLM server (or Ollama, llama.cpp, etc.)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 2. Configure
LLM_PROVIDER=local_open_source
LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
LLM_BASE_URL=http://localhost:8000/v1
```

The system automatically uses the Llama-optimized prompt template. If the model's JSON output has minor issues (markdown wrapping, missing fields), the validator handles them gracefully.

### Example: Using Ollama with Mistral

```bash
# 1. Start Ollama
ollama run mistral:7b-instruct-v0.3

# 2. Configure
LLM_PROVIDER=local_open_source
LLM_MODEL_NAME=mistral:7b-instruct-v0.3
LLM_BASE_URL=http://localhost:11434/v1
```

### Example: Using a Hosted API

```bash
LLM_PROVIDER=hosted_api
LLM_MODEL_NAME=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
```

---

## Text Normalization Pipeline

`app/nlp/normalizer.py` — `TextNormalizer`

Every text item is cleaned before classification:

| Step | What it does |
|------|-------------|
| Strip HTML | Removes `<tags>`, keeps text content |
| Unicode NFKC | Normalizes smart quotes, ligatures, etc. |
| Replace URLs | Replaces `https://...` with `[URL]` token |
| Collapse whitespace | Multiple spaces/newlines → single space |
| Truncation | Max 1000 chars (configurable), word-boundary aware |
| Near-duplicate | Trigram Jaccard similarity against recent cache |

The normalizer returns a `NormalizationResult` with the cleaned text, content hash, duplicate flag, and list of steps applied.

## Classifiers

All classifiers live in `app/nlp/classifier.py`.

### Sub-Classifiers

| Classifier | Input | Output | Method |
|-----------|-------|--------|--------|
| `EventTypeClassifier` | text | `(EventType, confidence)` | Regex patterns with multi-match boosting |
| `SentimentClassifier` | text | `(SentimentDirection, score)` | Keyword counting with negation awareness |
| `UrgencyClassifier` | text | `float 0-1` | Phrase matching + all-caps + exclamation detection |
| `RelevanceClassifier` | text + question | `float 0-1` | IDF-weighted token overlap |
| `EntityExtractor` | text | `list[str]` | Title-case phrases + acronym detection |

### Composite Classifiers

**`KeywordClassifier`** (default): Chains all sub-classifiers, produces a complete `ClassificationResult`. Zero external dependencies, always works.

**`HybridClassifier`**: Runs keyword first, then optionally queries an LLM. Uses the LLM result when its confidence exceeds `LLM_CONFIDENCE_THRESHOLD`, otherwise falls back to keyword.

**`LlmClassifierAdapter`**: Pluggable adapter for any LLM API. Includes:
- Structured prompt template requesting JSON output
- Full output validation via `LlmOutputValidator`
- Configurable rejection threshold for malformed output
- Graceful fallback when no API key is set or API call fails
- Override `_call_llm_api()` in a subclass to connect to any provider

### Event Types

```
legal_ruling | regulatory | election | economic
celebrity    | sports     | geopolitical | crypto | other
```

### Sentiment

```
bullish (+1) | bearish (-1) | neutral (0)
```

Negation-aware: "not crash" → bullish signal, "not win" → bearish signal.

## Market Mapping

`app/nlp/market_mapper.py` — `MarketMapper`

Links text to candidate Polymarket markets using a multi-signal relevance score:

1. **IDF-weighted token overlap** (50%): Rare shared words between text and market question contribute more than common ones. IDF is computed across all active markets.

2. **Entity matching** (30%): Extracted entities (proper nouns, acronyms) that appear in market questions.

3. **Manual overrides** (20%): Configurable `{condition_id: [keywords]}` mappings for known text-to-market links.

Results are ranked by score, thresholded (`min_relevance=0.15`), and capped (`max_results=5`).

**Ambiguity detection**: When multiple markets match with very similar scores (gap < 0.10), all are flagged as `ambiguous=True` and logged.

## Signal Schema

### NlpSignal

```python
{
    "source_text_id": "ex-001",
    "source_provider": "file",
    "text_snippet": "Federal judge rules crypto staking...",
    "market_ids": ["cid-regulation"],
    "relevance": 0.72,
    "sentiment": "bullish",
    "sentiment_score": 0.65,
    "event_type": "legal_ruling",
    "urgency": 0.40,
    "confidence": 0.51,
    "rationale": "keyword: event=legal_ruling(0.85) | sentiment=bullish(+0.65) | urgency=0.40 | relevance=0.72",
    "entities": ["Federal"],
    "metadata": {
        "token_overlap_score": 0.35,
        "entity_score": 0.0,
        "matched_keywords": ["regulation", "crypto"],
        "ambiguous": false,
        "normalization_steps": ["strip_html"]
    }
}
```

### NormalizedSignal (for decision engine)

The `nlp_signal_to_layered()` function converts `NlpSignal` → `NormalizedSignal` with:
- `direction` from sentiment (+1/-1/0)
- `normalized_confidence` via per-layer sigmoid squash
- `metadata` preserving event_type, urgency, relevance, entities, and text snippet

## Storage

Two database tables (`app/storage/repository.py`):

### `nlp_events`

Stores every text item that was processed:

| Column | Type | Description |
|--------|------|-------------|
| item_id | TEXT | Provider's item ID |
| source | TEXT | Provider name |
| text | TEXT | Original text |
| content_hash | TEXT | SHA-256 hash for dedup |
| normalized_text | TEXT | After normalization |
| event_type | TEXT | Detected event type |
| sentiment | TEXT | bullish/bearish/neutral |
| sentiment_score | REAL | -1.0 to 1.0 |
| urgency | REAL | 0.0 to 1.0 |
| confidence | REAL | 0.0 to 1.0 |
| entities_json | TEXT | JSON array of entities |

### `nlp_signals`

Stores every signal generated from text:

| Column | Type | Description |
|--------|------|-------------|
| source_text_id | TEXT | Links to nlp_events.item_id |
| market_id | TEXT | Matched market |
| sentiment | TEXT | Direction |
| confidence | REAL | Combined confidence |
| metadata_json | TEXT | Full match details |

## Replay

`app/nlp/replay.py` — `NlpReplayEngine`

Re-process stored or provided text events through the current pipeline:

```python
from app.nlp.replay import NlpReplayEngine

engine = NlpReplayEngine(active_markets=my_markets)

# From JSON file
result = engine.replay_from_json("data/news/examples.json")

# From database rows
rows = await repo.get_nlp_events(event_type="crypto")
result = engine.replay_from_db_rows(rows)

# From NewsItems directly
result = engine.replay_items(my_items)

# Save results
engine.save_result(result, "reports/replay_result.json")
print(result.summary())
```

Output includes:
- Total items, signal rate, signals by sentiment/event_type/market
- Per-item breakdown showing classification and signal details

Use cases:
- Evaluate a new classifier against historical events
- Debug why a headline produced (or didn't produce) a signal
- Compare signal output across pipeline versions

## Providers

### MockProvider (default, `NLP_PROVIDER=mock`)
Rotating sample headlines. Always available.

### FileProvider (`NLP_PROVIDER=file`)
Reads JSON files from `NEWS_FILE_DIR`. Format:

```json
[
  {"id": "...", "text": "...", "source": "..."},
  {"text": "...", "source": "..."}
]
```

Or use the `examples.json` format with `items` key and `expected_output`.

### RssProvider (requires `feedparser`)
RSS/Atom feed reader stub. Install `feedparser` and configure feed URLs.

### Adding a New Provider

```python
from app.nlp.providers.base import BaseNlpProvider
from app.news.models import NewsItem

class MyProvider(BaseNlpProvider):
    name = "my_provider"

    async def fetch_items(self) -> list[NewsItem]:
        # Fetch from your source
        return [NewsItem(item_id="...", source=self.name, text="...")]

    def is_available(self) -> bool:
        return True  # Check credentials, connectivity
```

Register: `news_service.register_provider(MyProvider())`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NLP_PROVIDER` | mock | Text provider: mock, file, none |
| `NEWS_POLL_INTERVAL` | 300 | Seconds between polls |
| `NEWS_FILE_DIR` | data/news | FileProvider directory |
| `LLM_PROVIDER` | none | LLM mode: none, mock, local_open_source, hosted_api |
| `LLM_MODEL_NAME` | — | Model name passed to API |
| `LLM_BASE_URL` | — | API endpoint URL |
| `LLM_API_KEY` | — | API key (optional for local models) |
| `LLM_TIMEOUT_SECONDS` | 30 | Request timeout |
| `LLM_CONFIDENCE_THRESHOLD` | 0.5 | Below this, hybrid uses keyword result |

## Example Data

`data/news/examples.json` contains 15 sample headlines covering:
- Crypto (legal ruling, exchange breach, network upgrade)
- Elections (polls, candidate withdrawal)
- Economic (GDP, Fed rate decision)
- Sports (championship)
- Geopolitical (NATO)
- Regulatory (SEC enforcement)
- Irrelevant content (weather, celebrity)
- Edge cases (HTML stripping, negation, entity extraction)

Each item includes `expected_output` for validation.

## Adding a Custom LLM Adapter

If you need behavior beyond the OpenAI-compatible API (e.g., a native SDK), subclass `LlmClassifierAdapter`:

```python
from app.nlp.classifier import LlmClassifierAdapter

class MyCustomLLM(LlmClassifierAdapter):
    name = "my_llm"

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key=api_key, model=model)

    def _call_llm_api(self, user_prompt: str) -> str:
        # Call your LLM however you need to.
        # Return the raw text response — validation is handled
        # automatically by LlmOutputValidator in the base class.
        response = my_sdk.complete(prompt=user_prompt)
        return response.text
```

Then use with `HybridClassifier` for automatic keyword fallback:

```python
pipeline = NlpPipeline(
    classifier=HybridClassifier(
        llm=MyCustomLLM(api_key="...", model="..."),
    )
)
```

Neither paid integration is required. The system operates fully with the built-in keyword classifier and mock/file providers.
