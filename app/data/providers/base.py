"""
Abstract base for external data providers.

Each provider produces a dict of named float features that get merged
into the ML feature matrix.  If a provider is disabled or unavailable,
its features are NaN — the preprocessing pipeline handles imputation.

Connection point for future paid APIs:
  - Social sentiment (Twitter/X, Reddit, Farcaster)
  - News headlines (NewsAPI, GDELT, etc.)
  - On-chain data (Dune, Flipside)
  - Macro indicators
  - Odds from other prediction markets
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.monitoring import get_logger

logger = get_logger(__name__)


class BaseProvider(ABC):
    """Interface that every external data provider must implement."""

    name: str = "base"

    @abstractmethod
    async def fetch_features(self, market_id: str, token_id: str) -> dict[str, float]:
        """Return a dict of feature_name → float value.

        MUST NOT raise — return empty dict or NaN values on failure.
        MUST NOT block the event loop for more than a few seconds.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if the provider has valid credentials and can fetch data."""
        ...

    def feature_names(self) -> list[str]:
        """List of column names this provider contributes."""
        return []


class ProviderRegistry:
    """Registry and lifecycle manager for external data providers."""

    _providers: dict[str, type[BaseProvider]] = {}
    _instances: dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, provider_class: type[BaseProvider]) -> type[BaseProvider]:
        cls._providers[provider_class.name] = provider_class
        return provider_class

    @classmethod
    def get_instance(cls, name: str) -> BaseProvider | None:
        if name in cls._instances:
            return cls._instances[name]
        if name in cls._providers:
            inst = cls._providers[name]()
            cls._instances[name] = inst
            return inst
        return None

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._providers.keys())

    @classmethod
    async def fetch_all(cls, market_id: str, token_id: str) -> dict[str, float]:
        """Aggregate features from all registered, available providers.

        Never raises.  Unavailable providers are skipped silently.
        """
        features: dict[str, float] = {}
        for name, provider_cls in cls._providers.items():
            inst = cls.get_instance(name)
            if inst is None or not inst.is_available():
                continue
            try:
                feats = await inst.fetch_features(market_id, token_id)
                features.update(feats)
            except Exception as e:
                logger.warning("provider_fetch_failed", provider=name, error=str(e))
        return features
