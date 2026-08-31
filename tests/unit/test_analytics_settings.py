"""Tests for analytics settings: defaults, overrides, validation, immutability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zepto.analytics.settings import AnalyticsSettings, get_analytics_settings


def test_defaults_encode_the_documented_modelling_policy() -> None:
    settings = AnalyticsSettings()

    assert settings.target_column == "survived"
    assert "alive" in settings.leakage_columns
    assert "class" in settings.redundant_columns
    assert settings.test_size == 0.2
    assert settings.random_state == 42


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_ANALYTICS_TEST_SIZE", "0.3")
    monkeypatch.setenv("ZEPTO_ANALYTICS_RANDOM_STATE", "7")

    settings = AnalyticsSettings()

    assert settings.test_size == 0.3
    assert settings.random_state == 7


@pytest.mark.parametrize("value", ["0", "1", "1.5", "-0.2"])
def test_test_size_must_be_a_proper_fraction(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A split of 0 or 1 leaves one side empty and silently invalidates evaluation."""
    monkeypatch.setenv("ZEPTO_ANALYTICS_TEST_SIZE", value)

    with pytest.raises(ValidationError):
        AnalyticsSettings()


@pytest.mark.parametrize("value", ["0", "1.01", "-0.5"])
def test_leakage_threshold_must_be_a_valid_probability(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ZEPTO_ANALYTICS_MAX_SINGLE_FEATURE_ACCURACY", value)

    with pytest.raises(ValidationError):
        AnalyticsSettings()


def test_cv_folds_below_two_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_ANALYTICS_CV_FOLDS", "1")

    with pytest.raises(ValidationError):
        AnalyticsSettings()


def test_settings_are_immutable() -> None:
    settings = AnalyticsSettings()

    with pytest.raises(ValidationError):
        settings.random_state = 99


def test_get_analytics_settings_is_cached() -> None:
    assert get_analytics_settings() is get_analytics_settings()
