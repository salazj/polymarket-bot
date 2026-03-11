"""
Tests for exchange adapter factory and config-based exchange selection.
"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.exchanges import build_exchange_adapter
from app.exchanges.base import Exchange


class TestExchangeFactory:
    def test_polymarket_selection(self, settings):
        settings.exchange = "polymarket"
        adapter = build_exchange_adapter(settings)
        assert adapter.exchange == Exchange.POLYMARKET

    def test_kalshi_selection(self, settings):
        settings.exchange = "kalshi"
        adapter = build_exchange_adapter(settings)
        assert adapter.exchange == Exchange.KALSHI

    def test_unknown_exchange_raises(self, settings):
        settings.exchange = "binance"
        with pytest.raises(ValueError, match="Unknown exchange"):
            build_exchange_adapter(settings)

    def test_case_insensitive(self, settings):
        settings.exchange = "Polymarket"
        adapter = build_exchange_adapter(settings)
        assert adapter.exchange == Exchange.POLYMARKET


class TestExchangeConfig:
    def test_default_exchange(self):
        s = Settings(
            dry_run=True,
            environment="test",
            log_level="WARNING",
            min_spread_threshold=0.01,
            max_spread_threshold=0.15,
        )
        assert s.exchange == "polymarket"

    def test_kalshi_exchange(self):
        s = Settings(
            dry_run=True,
            environment="test",
            exchange="kalshi",
            log_level="WARNING",
            min_spread_threshold=0.01,
            max_spread_threshold=0.15,
        )
        assert s.exchange == "kalshi"

    def test_invalid_exchange(self):
        with pytest.raises(Exception):
            Settings(
                dry_run=True,
                environment="test",
                exchange="invalid",
                log_level="WARNING",
                min_spread_threshold=0.01,
                max_spread_threshold=0.15,
            )

    def test_kalshi_credentials_check(self):
        s = Settings(
            dry_run=True,
            environment="test",
            exchange="kalshi",
            kalshi_api_key="test-key",
            kalshi_private_key_path="/tmp/test-key.pem",
            log_level="WARNING",
            min_spread_threshold=0.01,
            max_spread_threshold=0.15,
        )
        assert s.has_kalshi_credentials is True

    def test_kalshi_no_credentials(self):
        s = Settings(
            dry_run=True,
            environment="test",
            exchange="kalshi",
            log_level="WARNING",
            min_spread_threshold=0.01,
            max_spread_threshold=0.15,
        )
        assert s.has_kalshi_credentials is False

    def test_has_credentials_dispatch(self):
        s = Settings(
            dry_run=True,
            environment="test",
            exchange="kalshi",
            log_level="WARNING",
            min_spread_threshold=0.01,
            max_spread_threshold=0.15,
        )
        assert s.has_credentials is False

        s.exchange = "polymarket"
        assert s.has_credentials is False

    def test_kalshi_secret_redaction(self):
        s = Settings(
            dry_run=True,
            environment="test",
            exchange="kalshi",
            kalshi_api_key="super-secret",
            kalshi_private_key_path="/tmp/secret.pem",
            log_level="WARNING",
            min_spread_threshold=0.01,
            max_spread_threshold=0.15,
        )
        repr_str = repr(s)
        assert "super-secret" not in repr_str
        assert "/tmp/secret.pem" not in repr_str
