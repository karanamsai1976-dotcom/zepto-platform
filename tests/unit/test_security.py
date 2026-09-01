"""Tests for API key verification and rate limiting."""

from __future__ import annotations

import pytest

from zepto.assistant.security import (
    ApiKeyVerifier,
    SlidingWindowRateLimiter,
    client_identifier,
    fingerprint,
)
from zepto.assistant.settings import AssistantSettings
from zepto.core.errors import ConfigurationError


def _settings(**overrides: object) -> AssistantSettings:
    return AssistantSettings(**overrides)  # type: ignore[arg-type]


# --- fingerprints ---


def test_fingerprint_is_stable_and_short() -> None:
    assert fingerprint("secret-key") == fingerprint("secret-key")
    assert len(fingerprint("secret-key")) == 12


def test_fingerprint_does_not_contain_the_key() -> None:
    """Logging keys puts credentials into log aggregators, where they are
    searchable and retained."""
    assert "secret-key" not in fingerprint("secret-key")


def test_different_keys_produce_different_fingerprints() -> None:
    assert fingerprint("key-a") != fingerprint("key-b")


# --- key verification ---


def test_authentication_disabled_accepts_anything() -> None:
    """The default: the demo runs with no setup."""
    verifier = ApiKeyVerifier(_settings(require_api_key=False))

    assert verifier.enabled is False
    assert verifier.is_valid(None)
    assert verifier.is_valid("anything")


def test_a_configured_key_is_accepted() -> None:
    verifier = ApiKeyVerifier(_settings(require_api_key=True, api_keys=("alpha", "beta")))

    assert verifier.is_valid("alpha")
    assert verifier.is_valid("beta")


def test_an_unknown_key_is_rejected() -> None:
    verifier = ApiKeyVerifier(_settings(require_api_key=True, api_keys=("alpha",)))

    assert not verifier.is_valid("gamma")


def test_a_missing_key_is_rejected_when_required() -> None:
    verifier = ApiKeyVerifier(_settings(require_api_key=True, api_keys=("alpha",)))

    assert not verifier.is_valid(None)
    assert not verifier.is_valid("")


def test_requiring_auth_without_keys_fails_at_startup() -> None:
    """Failing closed. The dangerous alternative is an endpoint that believes it
    is protected and is not."""
    with pytest.raises(ConfigurationError):
        ApiKeyVerifier(_settings(require_api_key=True, api_keys=()))


def test_keys_can_be_supplied_comma_separated() -> None:
    """A comma-separated string is what a person types into an env var."""
    settings = _settings(require_api_key=True, api_keys="alpha, beta ,gamma")

    assert settings.api_keys == ("alpha", "beta", "gamma")
    assert ApiKeyVerifier(settings).is_valid("beta")


def test_keys_parse_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a tuple-typed setting is JSON-decoded by the environment source
    before any validator runs, so `a,b` raised a SettingsError until the field was
    marked NoDecode. Passing a string directly to the model does not exercise that
    path -- only reading it from the environment does."""
    monkeypatch.setenv("ZEPTO_ASSISTANT_API_KEYS", "alpha,beta")

    assert AssistantSettings().api_keys == ("alpha", "beta")


# --- rate limiting ---


def test_requests_within_the_limit_are_allowed() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)

    decisions = [limiter.check("client", now=1.0) for _ in range(3)]

    assert all(decision.allowed for decision in decisions)
    assert decisions[-1].remaining == 0


def test_exceeding_the_limit_is_refused() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    limiter.check("client", now=1.0)
    limiter.check("client", now=1.0)

    decision = limiter.check("client", now=1.0)

    assert not decision.allowed
    assert decision.remaining == 0
    assert decision.retry_after_seconds > 0


def test_the_window_slides_rather_than_resetting() -> None:
    """A fixed window permits twice the rate across a boundary: a client can
    spend its quota at the end of one window and again at the start of the next."""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
    limiter.check("client", now=0.0)
    limiter.check("client", now=9.0)

    assert not limiter.check("client", now=9.5).allowed
    # The first hit ages out at t=10, freeing exactly one slot.
    assert limiter.check("client", now=10.5).allowed
    assert not limiter.check("client", now=10.6).allowed


def test_clients_are_limited_independently() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("client-a", now=1.0)

    assert limiter.check("client-b", now=1.0).allowed


def test_retry_after_reports_when_a_slot_frees() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
    limiter.check("client", now=0.0)

    decision = limiter.check("client", now=4.0)

    assert decision.retry_after_seconds == pytest.approx(6.0)


def test_idle_clients_are_evicted() -> None:
    """Without eviction the map grows once per distinct client forever, which an
    attacker can accelerate by rotating source addresses."""
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=10)
    limiter.check("old-client", now=0.0)
    limiter.check("recent-client", now=100.0)

    evicted = limiter.evict_idle(now=105.0)

    assert evicted == 1
    assert limiter.tracked_clients == 1


def test_eviction_keeps_active_clients() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    limiter.check("active", now=100.0)

    assert limiter.evict_idle(now=101.0) == 0
    assert limiter.tracked_clients == 1


# --- client identity ---


def test_api_key_identifies_a_client_in_preference_to_address() -> None:
    """One client should get one quota regardless of which address it uses."""
    identifier = client_identifier("secret-key", "10.0.0.1")

    assert identifier.startswith("key:")
    assert "secret-key" not in identifier
    assert identifier == client_identifier("secret-key", "10.0.0.2")


def test_address_is_used_when_no_key_is_presented() -> None:
    assert client_identifier(None, "10.0.0.1") == "addr:10.0.0.1"


def test_an_unknown_address_still_yields_an_identifier() -> None:
    assert client_identifier(None, None) == "addr:unknown"
