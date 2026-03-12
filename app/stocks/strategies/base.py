"""Base class for stock trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.data.models import PortfolioSnapshot
from app.stocks.models import StockFeatures, StockSignal


class BaseStockStrategy(ABC):
    """All stock strategies must implement this interface."""

    name: str = "base_stock"

    @abstractmethod
    def generate_signal(
        self,
        features: StockFeatures,
        portfolio: PortfolioSnapshot,
    ) -> StockSignal | None:
        """Evaluate stock features and return a signal or None."""
