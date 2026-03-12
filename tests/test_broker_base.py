"""Tests for broker adapter interface contracts."""

from __future__ import annotations

import pytest
from abc import ABC

from app.brokers.base import (
    BaseBrokerAdapter,
    BaseBrokerExecution,
    BaseBrokerMarketData,
    BaseBrokerStreaming,
)


class TestBrokerInterfaces:
    def test_market_data_is_abstract(self):
        assert issubclass(BaseBrokerMarketData, ABC)
        with pytest.raises(TypeError):
            BaseBrokerMarketData()

    def test_execution_is_abstract(self):
        assert issubclass(BaseBrokerExecution, ABC)
        with pytest.raises(TypeError):
            BaseBrokerExecution()

    def test_streaming_is_abstract(self):
        assert issubclass(BaseBrokerStreaming, ABC)
        with pytest.raises(TypeError):
            BaseBrokerStreaming()

    def test_adapter_is_abstract(self):
        assert issubclass(BaseBrokerAdapter, ABC)
        with pytest.raises(TypeError):
            BaseBrokerAdapter()

    def test_broker_factory_unknown_raises(self):
        from app.brokers import build_broker_adapter
        from app.config.settings import Settings

        settings = Settings(broker="unknown_broker", asset_class="equities")
        with pytest.raises(ValueError, match="Unknown broker"):
            build_broker_adapter(settings)
