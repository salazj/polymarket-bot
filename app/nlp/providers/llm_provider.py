"""
LLM-backed classification providers for Level 3 intelligence.

Architecture
~~~~~~~~~~~~
                  ┌───────────────────┐
                  │  build_llm_       │
                  │  classifier()     │ ◀── Settings
                  └────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
    ┌──────────────┐ ┌──────────┐ ┌────────────────┐
    │ OpenAICompat │ │ LocalOS  │ │ MockLLMProvider│
    │ ibleLLM      │ │ Provider │ │ (testing)      │
    └──────────────┘ └──────────┘ └────────────────┘
             │             │
             └──────┬──────┘
                    ▼
    ┌───────────────────────────────┐
    │ LlmOutputValidator           │
    │ (strip fences, parse JSON,   │
    │  validate fields/types/enums)│
    └───────────────────────────────┘

The provider is **optional**.  When ``LLM_PROVIDER=none`` (the default),
nothing is loaded and the bot runs on keyword classifiers alone.

The LLM only produces *structured classification signals*:
  - relevance score
  - directional impact (sentiment)
  - urgency
  - extracted entities
  - rationale

It never places trades or bypasses risk controls.

Supports any backend exposing ``POST /chat/completions``:
  - vLLM, Ollama, llama.cpp (local open-source)
  - OpenAI, Anthropic, Together, Groq (hosted)
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.monitoring import get_logger
from app.nlp.classifier import BaseClassifier, LlmClassifierAdapter
from app.nlp.signals import (
    ClassificationResult,
    EventType,
    SentimentDirection,
)

logger = get_logger(__name__)


# ── Model-family prompt templates ──────────────────────────────────────

_JSON_SCHEMA_BLOCK = """\
{
  "event_type": one of ["legal_ruling","regulatory","election","economic","celebrity","sports","geopolitical","crypto","other"],
  "sentiment": one of ["bullish","bearish","neutral"],
  "sentiment_score": float from -1.0 (very bearish) to 1.0 (very bullish),
  "urgency": float from 0.0 to 1.0,
  "relevance": float from 0.0 to 1.0,
  "confidence": float from 0.0 to 1.0,
  "rationale": one sentence explaining your reasoning,
  "entities": list of key named entities mentioned in the text
}"""


PROMPT_GENERIC = (
    "You are a financial news classifier for prediction markets.\n"
    "Given a headline or short text, output ONLY a valid JSON object "
    "with these fields:\n\n"
    f"{_JSON_SCHEMA_BLOCK}\n\n"
    "Output ONLY valid JSON. No markdown fences, no explanation outside the JSON."
)

PROMPT_LLAMA = (
    "You are a precise financial news classifier. "
    "Respond with a single JSON object — nothing else.\n\n"
    "Required JSON fields:\n"
    f"{_JSON_SCHEMA_BLOCK}\n\n"
    "Rules:\n"
    "- Output raw JSON only, no markdown code fences\n"
    "- All float fields must be plain numbers (not strings)\n"
    "- entities must be a JSON array of strings\n"
    "- Do not add any text before or after the JSON"
)

PROMPT_MISTRAL = (
    "[INST] You are a financial news classifier for prediction markets. "
    "Your task is to analyze text and produce structured classification.\n\n"
    "Output a single valid JSON object with exactly these fields:\n"
    f"{_JSON_SCHEMA_BLOCK}\n\n"
    "Important: respond with ONLY the JSON object. "
    "No markdown formatting, no explanation. [/INST]"
)

PROMPT_QWEN = (
    "You are a financial news classification assistant. "
    "You always respond in valid JSON format without any additional text.\n\n"
    "Analyze the provided headline and output a JSON object:\n"
    f"{_JSON_SCHEMA_BLOCK}\n\n"
    "Respond with only the JSON object."
)


class ModelFamily:
    """Identifies a model family from its name and selects the best prompt."""

    GENERIC = "generic"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"

    _FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"llama", re.I), LLAMA),
        (re.compile(r"mistral|mixtral", re.I), MISTRAL),
        (re.compile(r"qwen", re.I), QWEN),
    ]

    _PROMPTS: dict[str, str] = {
        GENERIC: PROMPT_GENERIC,
        LLAMA: PROMPT_LLAMA,
        MISTRAL: PROMPT_MISTRAL,
        QWEN: PROMPT_QWEN,
    }

    @classmethod
    def detect(cls, model_name: str) -> str:
        """Detect model family from model name string."""
        for pattern, family in cls._FAMILY_PATTERNS:
            if pattern.search(model_name):
                return family
        return cls.GENERIC

    @classmethod
    def get_system_prompt(cls, family: str) -> str:
        return cls._PROMPTS.get(family, cls._PROMPTS[cls.GENERIC])


# ── OpenAI-compatible LLM adapter ─────────────────────────────────────


class OpenAICompatibleLLM(LlmClassifierAdapter):
    """Concrete LLM classifier that calls an OpenAI-compatible API.

    Works with local servers (vLLM, Ollama, llama.cpp) and hosted
    services (OpenAI, Together, Groq) — anything that accepts
    ``POST /chat/completions`` with the standard request schema.
    """

    name: str = "llm"  # type: ignore[assignment]

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 30,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key or "no-key", model=model)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

        family = ModelFamily.detect(model)
        self._system_prompt = system_prompt or ModelFamily.get_system_prompt(family)
        self._family = family

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )
        logger.info(
            "llm_provider_initialized",
            base_url=self._base_url,
            model=self._model,
            family=self._family,
            has_key=bool(api_key),
        )

    def _call_llm_api(self, user_prompt: str) -> str:
        """Send a chat completion request and return the raw content string."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }

        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.debug(
                "llm_response_received",
                model=self._model,
                family=self._family,
                length=len(content),
            )
            return content
        except httpx.TimeoutException:
            logger.warning("llm_timeout", model=self._model, timeout=self._timeout)
            raise
        except httpx.HTTPStatusError as e:
            logger.warning("llm_http_error", status=e.response.status_code, model=self._model)
            raise
        except Exception:
            logger.exception("llm_call_failed", model=self._model)
            raise

    def close(self) -> None:
        self._client.close()


# ── Local open-source provider stub ───────────────────────────────────


class LocalOpenSourceProvider(OpenAICompatibleLLM):
    """Pre-configured adapter for local open-source models.

    Selects prompt template and parameters based on model family.
    Designed for use with:
      - Llama 3.1 8B Instruct  (via vLLM, Ollama, or llama.cpp)
      - Mistral 7B Instruct    (via vLLM, Ollama, or llama.cpp)
      - Qwen 2.5 Instruct      (via vLLM, Ollama, or llama.cpp)

    Usage::

        provider = LocalOpenSourceProvider(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
        # Automatically uses PROMPT_LLAMA template
    """

    name: str = "llm_local"  # type: ignore[assignment]

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_key: str = "",
        timeout: int = 60,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info(
            "local_open_source_provider_ready",
            model=self._model,
            family=self._family,
            base_url=self._base_url,
        )


# ── Mock LLM provider for testing ─────────────────────────────────────


_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_SKIP_ENTITIES = {
    "The", "This", "That", "There", "It", "He", "She", "They", "We",
    "You", "New", "Just", "Also", "But", "And", "However", "Meanwhile",
    "After", "Before", "Breaking", "Headline", "Market",
    "US", "UK", "EU", "AM", "PM", "CEO", "AI", "IT",
}


class MockLLMProvider(LlmClassifierAdapter):
    """In-process mock LLM for testing without any network calls.

    Returns deterministic classification results based on keyword
    detection, simulating what a real LLM would return.  Extracts
    entities from title-case words and acronyms.
    """

    name: str = "llm_mock"  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__(api_key="mock-key", model="mock-model")

    def _call_llm_api(self, user_prompt: str) -> str:
        text = user_prompt.lower()

        # ── Sentiment detection ────────────────────────────────────
        bullish_words = ("surge", "rally", "win", "approve", "pass", "gain", "rise", "soar")
        bearish_words = ("crash", "fail", "reject", "lose", "drop", "plunge", "decline", "collapse")

        bull = sum(1 for w in bullish_words if w in text)
        bear = sum(1 for w in bearish_words if w in text)

        if bull > bear:
            sentiment = "bullish"
            score = min(0.5 + bull * 0.1, 1.0)
        elif bear > bull:
            sentiment = "bearish"
            score = max(-0.5 - bear * 0.1, -1.0)
        else:
            sentiment = "neutral"
            score = 0.0

        # ── Event type detection ───────────────────────────────────
        event_type = "other"
        for kws, etype in [
            (("election", "poll", "vote", "ballot", "candidate"), "election"),
            (("bitcoin", "crypto", "btc", "eth", "blockchain"), "crypto"),
            (("sec", "regulation", "regulatory", "fcc", "fda"), "regulatory"),
            (("court", "ruling", "judge", "lawsuit", "verdict"), "legal_ruling"),
            (("gdp", "inflation", "unemployment", "fed", "interest rate"), "economic"),
            (("war", "conflict", "treaty", "nato", "military"), "geopolitical"),
            (("game", "match", "championship", "finals", "playoff"), "sports"),
        ]:
            if any(k in text for k in kws):
                event_type = etype
                break

        # ── Urgency ────────────────────────────────────────────────
        urgency = 0.3
        if "breaking" in text or "urgent" in text:
            urgency = 0.8
        elif "developing" in text or "just in" in text:
            urgency = 0.6

        # ── Relevance from market context ──────────────────────────
        relevance = 0.5
        if "market question:" in text:
            question_part = text.split("market question:")[-1].strip()
            text_part = text.split("\n")[0].replace("headline: ", "")
            shared = set(text_part.split()) & set(question_part.split())
            shared -= {"the", "a", "is", "will", "and", "to", "of", "in"}
            relevance = min(0.3 + len(shared) * 0.12, 1.0)

        # ── Entity extraction ──────────────────────────────────────
        entities: list[str] = []
        for match in _TITLE_CASE_RE.finditer(user_prompt):
            name = match.group()
            if name not in _SKIP_ENTITIES and name not in entities:
                entities.append(name)
        for match in _ACRONYM_RE.finditer(user_prompt):
            acr = match.group()
            if acr not in _SKIP_ENTITIES and acr not in entities:
                entities.append(acr)

        confidence = min(0.4 + abs(score) * 0.3 + (0.1 if event_type != "other" else 0.0), 1.0)

        return json.dumps({
            "event_type": event_type,
            "sentiment": sentiment,
            "sentiment_score": score,
            "urgency": urgency,
            "relevance": relevance,
            "confidence": confidence,
            "rationale": f"Mock LLM detected {event_type}/{sentiment}",
            "entities": entities[:10],
        })


# ── Factory ────────────────────────────────────────────────────────────


def build_llm_classifier(
    provider: str,
    model_name: str = "",
    base_url: str = "",
    api_key: str = "",
    timeout: int = 30,
) -> LlmClassifierAdapter | None:
    """Factory: build the appropriate LLM classifier from settings.

    Returns ``None`` when ``provider`` is ``"none"`` — the bot runs
    without any LLM and the keyword classifier handles everything.

    Supported providers:
      - ``none``             — disabled (default)
      - ``mock``             — in-process mock for testing
      - ``local_open_source``— local model via OpenAI-compatible API
      - ``hosted_api``       — hosted API (OpenAI, Together, Groq, etc.)
    """
    provider = provider.lower().strip()

    if provider == "none":
        logger.info("llm_provider_disabled")
        return None

    if provider == "mock":
        logger.info("llm_provider_mock")
        return MockLLMProvider()

    if provider in ("local_open_source", "hosted_api"):
        if not base_url:
            logger.error("llm_missing_base_url", provider=provider)
            raise ValueError(
                f"LLM_BASE_URL is required when LLM_PROVIDER={provider}. "
                f"Example: http://localhost:8000/v1 (local) or "
                f"https://api.openai.com/v1 (hosted)"
            )
        if not model_name:
            logger.error("llm_missing_model_name", provider=provider)
            raise ValueError(
                f"LLM_MODEL_NAME is required when LLM_PROVIDER={provider}. "
                f"Example: meta-llama/Llama-3.1-8B-Instruct"
            )
        if provider == "hosted_api" and not api_key:
            logger.warning(
                "llm_no_api_key_hosted",
                hint="Most hosted APIs require LLM_API_KEY",
            )

        if provider == "local_open_source":
            return LocalOpenSourceProvider(
                base_url=base_url,
                model=model_name,
                api_key=api_key,
                timeout=timeout,
            )

        return OpenAICompatibleLLM(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            timeout=timeout,
        )

    logger.error("llm_unknown_provider", provider=provider)
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        f"Use: none, mock, local_open_source, hosted_api"
    )
