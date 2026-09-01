"""Prometheus metrics for the assistant API.

Metric labels are drawn from the matched route template rather than the raw
request path. That distinction matters: labelling by raw path means every
unmatched URL creates a new time series, so anyone hitting random paths can
exhaust the metrics backend's memory. It is a real denial-of-service vector, and
the fix is to label with a bounded set of values.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

#: Buckets tuned to the observed shape of this service: retrieval runs around
#: 150ms, so the default Prometheus buckets would put almost everything in one
#: bin and tell you nothing about the distribution.
LATENCY_BUCKETS = (0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

UNMATCHED_ROUTE = "unmatched"


class Metrics:
    """The service's metric collectors.

    Held on an explicit registry rather than the global default, so tests can
    build an isolated instance instead of accumulating into shared state
    between test cases.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.requests = Counter(
            "zepto_requests_total",
            "HTTP requests handled.",
            labelnames=("method", "route", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "zepto_request_duration_seconds",
            "Time spent handling a request.",
            labelnames=("method", "route"),
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "zepto_rate_limited_total",
            "Requests refused by the rate limiter.",
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "zepto_auth_failures_total",
            "Requests refused for a missing or invalid API key.",
            registry=self.registry,
        )
        self.suspected_injection = Counter(
            "zepto_suspected_injection_total",
            "Questions matching a known prompt-injection phrasing. Telemetry, "
            "not a block: see sanitization.py for why this is not a filter.",
            registry=self.registry,
        )
        self.answers = Counter(
            "zepto_answers_total",
            "Questions answered, by whether the corpus covered them.",
            labelnames=("intent",),
            registry=self.registry,
        )
        self.documents_indexed = Gauge(
            "zepto_documents_indexed",
            "Chunks currently in the vector index.",
            registry=self.registry,
        )

    def observe_request(self, method: str, route: str, status: int, duration: float) -> None:
        self.requests.labels(method=method, route=route, status=str(status)).inc()
        self.request_duration.labels(method=method, route=route).observe(duration)


def route_label(request: Any) -> str:
    """The matched route template, or a constant for unmatched paths.

    Returning a constant for anything unmatched is what bounds label
    cardinality: without it, every 404 against a random URL would create a
    permanent new time series.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else UNMATCHED_ROUTE
