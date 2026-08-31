"""Tests for core settings: defaults, environment overrides, validation, immutability."""

import pytest
from pydantic import ValidationError

from zepto.core.settings import CoreSettings, get_core_settings


def test_defaults_are_development() -> None:
    settings = CoreSettings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.log_json is False
    assert settings.is_production is False


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_ENVIRONMENT", "production")
    monkeypatch.setenv("ZEPTO_LOG_JSON", "true")

    settings = CoreSettings()

    assert settings.environment == "production"
    assert settings.log_json is True
    assert settings.is_production is True


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the environment name must fail loudly at startup."""
    monkeypatch.setenv("ZEPTO_ENVIRONMENT", "prodction")

    with pytest.raises(ValidationError):
        CoreSettings()


def test_settings_are_immutable() -> None:
    settings = CoreSettings()

    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"


def test_get_core_settings_is_cached() -> None:
    assert get_core_settings() is get_core_settings()
