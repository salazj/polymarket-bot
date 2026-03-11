"""
Market mapping: links incoming text/news items to candidate Polymarket markets.

Uses a multi-signal relevance score:
  1. Weighted token overlap (IDF-like: rare shared words matter more)
  2. Entity matching against market questions and descriptions
  3. Manual override rules from configuration
  4. Ambiguity detection and logging

Results are ranked and thresholded so only genuinely relevant markets
are considered by the NLP pipeline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.data.models import Market
from app.monitoring import get_logger

logger = get_logger(__name__)

_STOP_WORDS = frozenset({
    "the", "and", "for", "will", "this", "that", "with", "from",
    "are", "was", "has", "have", "been", "not", "but", "its",
    "who", "what", "when", "where", "how", "can", "does", "did",
    "would", "could", "should", "they", "their", "there", "than",
    "then", "into", "also", "just", "about", "more", "some",
    "any", "each", "every", "all", "most", "other", "after",
    "before", "between", "over", "under", "only", "much",
})


@dataclass
class MarketMatch:
    """One candidate market matched to a piece of text."""
    market: Market
    relevance_score: float
    token_overlap_score: float = 0.0
    entity_score: float = 0.0
    override_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    ambiguous: bool = False


class MarketMapper:
    """Scores incoming text against active markets and returns ranked matches.

    The mapper logs ambiguity when multiple markets match with similar
    scores, since this is a known hard problem for text-to-market linking.
    """

    def __init__(
        self,
        min_relevance: float = 0.15,
        ambiguity_gap: float = 0.10,
        manual_overrides: dict[str, list[str]] | None = None,
        max_results: int = 5,
    ) -> None:
        self._min_relevance = min_relevance
        self._ambiguity_gap = ambiguity_gap
        self._overrides = manual_overrides or {}
        self._max_results = max_results

    def find_matches(
        self,
        text: str,
        entities: list[str],
        markets: list[Market],
    ) -> list[MarketMatch]:
        text_lower = text.lower()
        text_tokens = _meaningful_tokens(text_lower)
        entity_lower = {e.lower() for e in entities}

        # Build IDF weights across all market questions (rarer words matter more)
        idf = self._build_idf(markets)

        results: list[MarketMatch] = []
        for market in markets:
            if not market.active:
                continue
            match = self._score_market(text_lower, text_tokens, entity_lower, market, idf)
            if match.relevance_score >= self._min_relevance:
                results.append(match)

        results.sort(key=lambda m: m.relevance_score, reverse=True)
        results = results[:self._max_results]

        # Flag ambiguity: multiple close-scoring matches
        self._check_ambiguity(results)

        for r in results:
            logger.debug(
                "market_match",
                market_id=r.market.condition_id,
                relevance=round(r.relevance_score, 3),
                token_score=round(r.token_overlap_score, 3),
                entity_score=round(r.entity_score, 3),
                keywords=r.matched_keywords[:5],
                entities=r.matched_entities[:5],
                ambiguous=r.ambiguous,
            )
        return results

    def _score_market(
        self,
        text_lower: str,
        text_tokens: set[str],
        entity_lower: set[str],
        market: Market,
        idf: dict[str, float],
    ) -> MarketMatch:
        question_lower = market.question.lower()
        question_tokens = _meaningful_tokens(question_lower)

        # 1. IDF-weighted token overlap
        overlap = question_tokens & text_tokens
        if not overlap or not question_tokens:
            token_score = 0.0
        else:
            overlap_weight = sum(idf.get(tok, 1.0) for tok in overlap)
            question_weight = sum(idf.get(tok, 1.0) for tok in question_tokens)
            token_score = overlap_weight / question_weight if question_weight else 0.0

        # 2. Entity matching
        entity_matches: list[str] = []
        for e in entity_lower:
            if e in question_lower:
                entity_matches.append(e)
            # Also check slug (often contains key proper nouns)
            if hasattr(market, "slug") and market.slug and e in market.slug.lower():
                if e not in entity_matches:
                    entity_matches.append(e)
        entity_score = min(len(entity_matches) * 0.30, 0.6)

        # 3. Manual override
        override_score = 0.0
        cid = market.condition_id
        if cid in self._overrides:
            for kw in self._overrides[cid]:
                if kw.lower() in text_lower:
                    override_score = 0.5
                    break

        # Weighted combination
        total = token_score * 0.50 + entity_score * 0.30 + override_score * 0.20

        return MarketMatch(
            market=market,
            relevance_score=min(total, 1.0),
            token_overlap_score=token_score,
            entity_score=entity_score,
            override_score=override_score,
            matched_keywords=list(overlap),
            matched_entities=entity_matches,
        )

    def _build_idf(self, markets: list[Market]) -> dict[str, float]:
        """IDF-like weight: tokens that appear in fewer market questions
        are more discriminative when they match."""
        n_docs = max(len(markets), 1)
        doc_freq: Counter[str] = Counter()
        for m in markets:
            tokens = _meaningful_tokens(m.question.lower())
            for t in tokens:
                doc_freq[t] += 1
        return {
            tok: math.log(n_docs / (1 + freq)) + 1.0
            for tok, freq in doc_freq.items()
        }

    def _check_ambiguity(self, results: list[MarketMatch]) -> None:
        """Flag and log when the top matches are too close to call."""
        if len(results) < 2:
            return
        top = results[0].relevance_score
        close = [r for r in results[1:] if top - r.relevance_score < self._ambiguity_gap]
        if close:
            for r in close:
                r.ambiguous = True
            results[0].ambiguous = True
            logger.info(
                "market_mapping_ambiguous",
                top_score=round(top, 3),
                close_count=len(close),
                markets=[r.market.condition_id for r in [results[0]] + close],
            )


def _meaningful_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w{3,}\b", text)) - _STOP_WORDS
