"""
Text normalization pipeline for NLP ingestion.

Cleans, normalizes, and deduplicates incoming text before it reaches
the classifiers.  Every step is logged so you can trace exactly what
transformation happened to an input.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.monitoring import get_logger

logger = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


@dataclass
class NormalizationResult:
    """Records what the normalizer did to an input text."""
    original: str
    normalized: str
    content_hash: str
    is_duplicate: bool = False
    steps_applied: list[str] = field(default_factory=list)
    char_reduction: int = 0
    language_hint: str = "en"


class TextNormalizer:
    """Multi-step text cleaner with near-duplicate detection.

    Steps (in order):
    1. Strip HTML tags
    2. Unicode NFKC normalization
    3. Replace URLs with [URL] token
    4. Collapse whitespace
    5. Trim to max length
    6. Near-duplicate check against recent history
    """

    def __init__(
        self,
        max_length: int = 1000,
        dedup_cache_size: int = 500,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._max_length = max_length
        self._cache_size = dedup_cache_size
        self._sim_threshold = similarity_threshold
        self._seen: OrderedDict[str, str] = OrderedDict()

    def normalize(self, text: str) -> NormalizationResult:
        original = text
        steps: list[str] = []
        current = text

        # 1. HTML tag removal
        cleaned = _HTML_TAG_RE.sub(" ", current)
        if cleaned != current:
            steps.append("strip_html")
            current = cleaned

        # 2. Unicode NFKC normalization (smart quotes → ASCII quotes, etc.)
        nfkc = unicodedata.normalize("NFKC", current)
        if nfkc != current:
            steps.append("unicode_nfkc")
            current = nfkc

        # 3. URL replacement
        url_cleaned = _URL_RE.sub("[URL]", current)
        if url_cleaned != current:
            steps.append("replace_urls")
            current = url_cleaned

        # 4. Whitespace collapse
        ws = _MULTI_SPACE_RE.sub(" ", current).strip()
        if ws != current:
            steps.append("collapse_whitespace")
            current = ws

        # 5. Truncation
        if len(current) > self._max_length:
            current = current[:self._max_length].rsplit(" ", 1)[0] + "…"
            steps.append(f"truncated_to_{self._max_length}")

        content_hash = self._hash(current)
        is_dup = self._check_near_duplicate(current, content_hash)
        if is_dup:
            steps.append("near_duplicate_detected")

        lang = self._detect_language_hint(current)

        result = NormalizationResult(
            original=original,
            normalized=current,
            content_hash=content_hash,
            is_duplicate=is_dup,
            steps_applied=steps,
            char_reduction=len(original) - len(current),
            language_hint=lang,
        )

        logger.debug(
            "text_normalized",
            hash=content_hash[:8],
            steps=steps,
            reduction=result.char_reduction,
            duplicate=is_dup,
        )
        return result

    def reset_cache(self) -> None:
        self._seen.clear()

    def _check_near_duplicate(self, text: str, content_hash: str) -> bool:
        if content_hash in self._seen:
            return True

        # Trigram-based approximate duplicate check against recent texts
        text_trigrams = self._trigrams(text)
        if text_trigrams:
            for cached_hash, cached_text in reversed(list(self._seen.items())):
                cached_trigrams = self._trigrams(cached_text)
                if not cached_trigrams:
                    continue
                intersection = text_trigrams & cached_trigrams
                union = text_trigrams | cached_trigrams
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard >= self._sim_threshold:
                    logger.debug(
                        "near_duplicate_found",
                        hash_a=content_hash[:8],
                        hash_b=cached_hash[:8],
                        similarity=round(jaccard, 3),
                    )
                    return True

        # Register this text
        self._seen[content_hash] = text
        if len(self._seen) > self._cache_size:
            self._seen.popitem(last=False)
        return False

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        t = text.lower()
        if len(t) < 3:
            return set()
        return {t[i:i + 3] for i in range(len(t) - 2)}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _detect_language_hint(text: str) -> str:
        """Very rough heuristic — just checks for common non-English characters."""
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"
        if re.search(r"[\u0600-\u06ff]", text):
            return "ar"
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
            return "ja"
        return "en"
