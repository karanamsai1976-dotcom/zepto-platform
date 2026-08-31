"""Tests for ingestion settings: defaults, overrides, validation, URL building."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from zepto.ingestion.settings import IngestionSettings


def test_defaults_are_sane() -> None:
    settings = IngestionSettings()

    assert settings.timeout_seconds == 10.0
    assert settings.max_retries == 3
    assert settings.gbp_to_inr == Decimal("105.50")
    assert len(settings.category_slugs) == 9


def test_conversion_rate_is_decimal_not_float() -> None:
    """Money must never be a float -- binary representation makes rounding
    non-deterministic, which produced a real discrepancy in v1."""
    settings = IngestionSettings()

    assert isinstance(settings.gbp_to_inr, Decimal)


def test_category_url_is_built_from_template() -> None:
    settings = IngestionSettings()

    url = settings.category_url("mystery_3")

    assert url.endswith("/mystery_3/index.html")
    assert "{slug}" not in url


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_INGESTION_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("ZEPTO_INGESTION_MAX_RETRIES", "5")

    settings = IngestionSettings()

    assert settings.timeout_seconds == 30.0
    assert settings.max_retries == 5


def test_non_positive_timeout_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero or negative timeout is meaningless and must fail at startup."""
    monkeypatch.setenv("ZEPTO_INGESTION_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValidationError):
        IngestionSettings()


def test_negative_retries_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_INGESTION_MAX_RETRIES", "-1")

    with pytest.raises(ValidationError):
        IngestionSettings()


def test_settings_are_immutable() -> None:
    settings = IngestionSettings()

    with pytest.raises(ValidationError):
        settings.timeout_seconds = 60.0


def test_get_ingestion_settings_is_cached() -> None:
    from zepto.ingestion.settings import get_ingestion_settings

    assert get_ingestion_settings() is get_ingestion_settings()
