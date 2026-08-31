"""Tests for assistant settings: defaults, overrides, and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zepto.assistant.settings import AssistantSettings, get_assistant_settings


def test_mock_mode_is_the_default() -> None:
    """The default path must need no API key and make no external call."""
    settings = AssistantSettings()

    assert settings.mock_llm is True


def test_routing_keywords_cover_the_documented_set() -> None:
    settings = AssistantSettings()

    assert "delivery" in settings.policy_keywords
    assert "cancel" in settings.policy_keywords
    assert "gift card" in settings.policy_keywords


def test_input_limit_has_a_sane_default() -> None:
    settings = AssistantSettings()

    assert 0 < settings.max_query_length <= 2000


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_ASSISTANT_MOCK_LLM", "false")
    monkeypatch.setenv("ZEPTO_ASSISTANT_TOP_K", "5")

    settings = AssistantSettings()

    assert settings.mock_llm is False
    assert settings.top_k == 5


@pytest.mark.parametrize("value", ["0", "21", "-1"])
def test_top_k_outside_the_supported_range_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ZEPTO_ASSISTANT_TOP_K", value)

    with pytest.raises(ValidationError):
        AssistantSettings()


@pytest.mark.parametrize("value", ["-0.1", "1.5"])
def test_min_relevance_must_be_a_valid_score(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ZEPTO_ASSISTANT_MIN_RELEVANCE", value)

    with pytest.raises(ValidationError):
        AssistantSettings()


def test_zero_query_length_limit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPTO_ASSISTANT_MAX_QUERY_LENGTH", "0")

    with pytest.raises(ValidationError):
        AssistantSettings()


def test_non_positive_llm_timeout_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero timeout would mean an unbounded wait on an external service."""
    monkeypatch.setenv("ZEPTO_ASSISTANT_LLM_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValidationError):
        AssistantSettings()


def test_settings_are_immutable() -> None:
    settings = AssistantSettings()

    with pytest.raises(ValidationError):
        settings.top_k = 10


def test_get_assistant_settings_is_cached() -> None:
    assert get_assistant_settings() is get_assistant_settings()
