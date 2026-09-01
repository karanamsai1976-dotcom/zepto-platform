"""API key verification and rate limiting.

Both are deliberately small and dependency-free, and both are honest about what
they are not.

Key comparison uses a constant-time function. String equality returns as soon as
it finds a differing byte, so how long a rejection takes leaks how much of the
key was correct -- enough, over many attempts, to recover it a character at a
time. That is a textbook attack and a one-line fix.

The rate limiter counts in process memory. That is correct for a single
instance and wrong the moment there are two: each replica would permit the full
quota independently, so N replicas allow N times the intended rate. A shared
store such as Redis is the fix, and this is stated here rather than discovered
in production.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import deque
from dataclasses import dataclass

from zepto.assistant.settings import AssistantSettings
from zepto.core.errors import ConfigurationError
from zepto.core.logging import get_logger

logger = get_logger(__name__)

#: Characters of the key digest used to identify a caller in logs. Enough to
#: correlate a caller's requests, far too little to reconstruct the key.
FINGERPRINT_LENGTH = 12


def fingerprint(api_key: str) -> str:
    """A short, non-reversible identifier for a key, safe to log.

    Logging keys themselves is how credentials end up in log aggregators, where
    they are searchable, retained, and visible to anyone with read access.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


class ApiKeyVerifier:
    """Checks presented keys against the configured set."""

    def __init__(self, settings: AssistantSettings) -> None:
        if settings.require_api_key and not settings.api_keys:
            raise ConfigurationError(
                "authentication is required but no API keys are configured",
                hint="set ZEPTO_ASSISTANT_API_KEYS, or leave ZEPTO_ASSISTANT_REQUIRE_API_KEY unset",
            )
        self._required = settings.require_api_key
        self._keys = tuple(settings.api_keys)

    @property
    def enabled(self) -> bool:
        return self._required

    def is_valid(self, presented: str | None) -> bool:
        """Return whether a presented key is accepted.

        Every configured key is compared even after a match is found, so the
        time taken does not depend on which key matched or how many were
        checked.
        """
        if not self._required:
            return True
        if not presented:
            return False

        matched = False
        for known in self._keys:
            if secrets.compare_digest(presented, known):
                matched = True
        return matched


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of one rate limit check."""

    allowed: bool
    remaining: int
    retry_after_seconds: float


class SlidingWindowRateLimiter:
    """Per-client request limiting over a sliding window, held in memory.

    A sliding window rather than a fixed one: fixed windows allow twice the
    intended rate across a boundary, since a client can spend its whole quota at
    the end of one window and again at the start of the next.

    Not shared between processes. See the module docstring.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, client_id: str, now: float | None = None) -> RateLimitDecision:
        """Record a request and report whether it is permitted."""
        current = time.monotonic() if now is None else now
        cutoff = current - self._window_seconds

        hits = self._hits.setdefault(client_id, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._max_requests:
            retry_after = hits[0] + self._window_seconds - current
            return RateLimitDecision(
                allowed=False,
                remaining=0,
                retry_after_seconds=max(0.0, round(retry_after, 3)),
            )

        hits.append(current)
        return RateLimitDecision(
            allowed=True,
            remaining=self._max_requests - len(hits),
            retry_after_seconds=0.0,
        )

    def evict_idle(self, now: float | None = None) -> int:
        """Drop clients with no recent requests, returning how many were removed.

        Without this the map grows once per distinct client forever, which is a
        slow memory leak on a public endpoint and something an attacker can
        accelerate by rotating source addresses.
        """
        current = time.monotonic() if now is None else now
        cutoff = current - self._window_seconds

        stale = [
            client_id for client_id, hits in self._hits.items() if not hits or hits[-1] <= cutoff
        ]
        for client_id in stale:
            del self._hits[client_id]
        return len(stale)

    @property
    def tracked_clients(self) -> int:
        return len(self._hits)


def client_identifier(api_key: str | None, remote_address: str | None) -> str:
    """Identify a caller for rate limiting purposes.

    Prefers the API key, so a single client is limited consistently regardless
    of which address it connects from. Falls back to the remote address, which
    is weaker: addresses are shared behind NAT and spoofable without a trusted
    proxy in front.
    """
    if api_key:
        return f"key:{fingerprint(api_key)}"
    return f"addr:{remote_address or 'unknown'}"


def client_address(
    forwarded_for: str | None,
    remote_address: str | None,
    trusted_proxy_count: int,
) -> str | None:
    """The caller's address, reading X-Forwarded-For only when it is trustworthy.

    Both directions of getting this wrong are real, and they fail in opposite
    ways.

    Trusting the header when nothing sets it lets any caller supply their own
    value, so every request looks like a new client and the rate limit stops
    limiting anything. Ignoring it behind a proxy is the mirror image: every
    request arrives from the proxy's address, all callers share one bucket, and
    one client can exhaust the quota for everyone.

    So the header is used only when the deployment declares how many proxies
    sit in front, and the address is taken by counting in from the right --
    each proxy appends the address it received from, so with N trusted proxies
    the (N)th entry from the end is the one the outermost proxy actually saw.
    Entries to the left of that were supplied by the caller and are ignored.

    A header shorter than the declared chain means the request did not arrive
    the way the configuration says it does, so it is discarded rather than
    guessed at.
    """
    if trusted_proxy_count <= 0 or not forwarded_for:
        return remote_address

    parts = [part.strip() for part in forwarded_for.split(",") if part.strip()]
    if len(parts) < trusted_proxy_count:
        return remote_address

    return parts[-trusted_proxy_count]
