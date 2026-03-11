"""Tests for configuration loading and validation."""

import os
import pytest
from app.config.settings import Settings, Environment


class TestSettings:
    def test_default_is_dry_run(self) -> None:
        s = Settings()
        assert s.dry_run is True

    def test_default_environment(self) -> None:
        s = Settings()
        assert s.environment == Environment.DEVELOPMENT

    def test_is_live_requires_both_flags(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=False)
        assert s.is_live is False

    def test_is_live_requires_all_three_gates(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=False)
        assert s.is_live is False

    def test_is_live_true(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=True,
                     private_key="k", poly_api_key="a", poly_api_secret="s")
        assert s.is_live is True

    def test_has_credentials_false_by_default(self) -> None:
        s = Settings()
        assert s.has_credentials is False

    def test_has_credentials_true(self) -> None:
        s = Settings(private_key="0xabc", poly_api_key="key", poly_api_secret="secret")
        assert s.has_credentials is True

    def test_require_live_trading_raises_without_enable(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=False)
        with pytest.raises(RuntimeError, match="ENABLE_LIVE_TRADING"):
            s.require_live_trading()

    def test_require_live_trading_raises_without_acknowledged(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=False)
        with pytest.raises(RuntimeError, match="LIVE_TRADING_ACKNOWLEDGED"):
            s.require_live_trading()

    def test_require_live_trading_raises_without_creds(self) -> None:
        s = Settings(dry_run=False, enable_live_trading=True, live_trading_acknowledged=True)
        with pytest.raises(RuntimeError, match="PRIVATE_KEY"):
            s.require_live_trading()

    def test_require_live_trading_ok_for_dry_run(self) -> None:
        s = Settings(dry_run=True)
        # dry_run raises because dry_run=True conflicts with live trading
        with pytest.raises(RuntimeError, match="DRY_RUN"):
            s.require_live_trading()

    def test_log_level_validation(self) -> None:
        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_invalid_log_level(self) -> None:
        with pytest.raises(Exception):
            Settings(log_level="INVALID")

    def test_risk_limits_positive(self) -> None:
        s = Settings()
        assert s.max_position_per_market > 0
        assert s.max_total_exposure > 0
        assert s.max_daily_loss > 0

    # --- LLM settings ---

    def test_llm_provider_defaults_to_none(self) -> None:
        s = Settings()
        assert s.llm_provider == "none"

    def test_llm_provider_validation_accepts_valid(self) -> None:
        for p in ("none", "local_open_source", "hosted_api"):
            s = Settings(llm_provider=p)
            assert s.llm_provider == p

    def test_llm_provider_validation_rejects_invalid(self) -> None:
        with pytest.raises(Exception):
            Settings(llm_provider="gpt4_magic")

    def test_llm_api_key_redacted_in_repr(self) -> None:
        s = Settings(llm_api_key="super-secret-key")
        r = repr(s)
        assert "super-secret-key" not in r
        assert "***" in r

    def test_ml_model_name_default(self) -> None:
        s = Settings()
        assert s.ml_model_name == "gradient_boosting"

    # --- Docker/container safety ---

    def test_all_three_gates_default_to_safe(self) -> None:
        s = Settings()
        assert s.dry_run is True
        assert s.enable_live_trading is False
        assert s.live_trading_acknowledged is False
        assert s.is_live is False
