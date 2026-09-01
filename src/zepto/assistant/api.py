"""HTTP interface for the support assistant.

The vector store is built once during application startup and shared by every
request. This is what makes retrieval cost milliseconds rather than seconds: v1
rebuilt the embedding model inside each query, so its API paid a multi-second
penalty on every call.

Beyond answering, the surface here covers what a service needs to be operated.

Readiness. v1 would start with an empty index and fail every request
afterwards. The corpus is indexed at startup if the collection is empty, and
/ready reports whether this instance can actually answer.

Error containment. An unhandled exception in v1 returned a stack trace to the
caller, leaking file paths and internals. Errors map to status codes with a safe
message; detail goes to the logs.

Request correlation. Each request binds an id that appears on every log line it
produces and is echoed back in a header.

Access control and rate limiting, both off or permissive by default so the demo
runs with no setup, and both failing closed when switched on.

Metrics, on a separate registry, labelled by matched route so that unmatched
paths cannot inflate label cardinality.

Health, readiness, and metrics endpoints are exempt from authentication and rate
limiting: probes and scrapers hit them constantly, and a monitoring endpoint that
requires a credential tends to end up unmonitored.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from zepto.assistant.graph import build_graph
from zepto.assistant.metrics import Metrics, route_label
from zepto.assistant.retrieval import VectorStore, load_corpus
from zepto.assistant.sanitization import looks_like_injection
from zepto.assistant.schemas import AskRequest, AskResponse, HealthResponse, Source
from zepto.assistant.security import (
    ApiKeyVerifier,
    SlidingWindowRateLimiter,
    client_address,
    client_identifier,
    fingerprint,
)
from zepto.assistant.settings import get_assistant_settings
from zepto.assistant.web import LANDING_PAGE
from zepto.core.errors import ZeptoError
from zepto.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"
FORWARDED_FOR_HEADER = "X-Forwarded-For"

#: Paths that bypass authentication and rate limiting. Probes and scrapers hit
#: these constantly, and gating them tends to result in nothing being monitored.
UNGUARDED_PATHS = frozenset({"/", "/health", "/ready", "/metrics"})

#: Requests between sweeps of idle rate-limit entries. Sweeping on every request
#: is linear in tracked clients for no benefit; never sweeping leaks memory once
#: per distinct client.
EVICTION_INTERVAL = 100


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare the retrieval stack once, before the first request arrives."""
    configure_logging()
    settings = get_assistant_settings()

    # Constructed here so a misconfiguration -- authentication required with no
    # keys -- stops startup rather than surfacing on the first request.
    verifier = ApiKeyVerifier(settings)

    store = VectorStore(settings=settings)
    if store.count() == 0:
        logger.info("index_empty_bootstrapping", corpus=str(settings.corpus_dir))
        store.ingest(load_corpus(settings.corpus_dir))

    app.state.settings = settings
    app.state.store = store
    app.state.graph = build_graph(store, settings=settings)
    app.state.verifier = verifier
    app.state.limiter = SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    app.state.metrics = Metrics()
    app.state.request_count = 0
    app.state.metrics.documents_indexed.set(store.count())

    logger.info(
        "assistant_ready",
        documents=store.count(),
        mock_llm=settings.mock_llm,
        auth_required=verifier.enabled,
        rate_limit_enabled=settings.rate_limit_enabled,
    )
    yield


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Reject a request without a valid key, when keys are required."""
    verifier: ApiKeyVerifier = request.app.state.verifier
    if verifier.is_valid(x_api_key):
        return

    request.app.state.metrics.auth_failures.inc()
    logger.warning(
        "auth_rejected",
        path=request.url.path,
        key_presented=bool(x_api_key),
        key_fingerprint=fingerprint(x_api_key) if x_api_key else None,
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid API key is required.",
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton, so tests can construct an
    isolated instance instead of sharing global state.
    """
    app = FastAPI(title="Zepto Support Assistant", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe_and_guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Correlate, rate limit, time, and record every request."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        settings = request.app.state.settings
        metrics: Metrics = request.app.state.metrics
        started = time.monotonic()

        guarded = request.url.path not in UNGUARDED_PATHS
        if guarded and settings.rate_limit_enabled:
            limiter: SlidingWindowRateLimiter = request.app.state.limiter

            request.app.state.request_count += 1
            if request.app.state.request_count % EVICTION_INTERVAL == 0:
                limiter.evict_idle()

            client = client_identifier(
                request.headers.get(API_KEY_HEADER),
                client_address(
                    request.headers.get(FORWARDED_FOR_HEADER),
                    request.client.host if request.client else None,
                    settings.trusted_proxy_count,
                ),
            )
            decision = limiter.check(client)

            if not decision.allowed:
                metrics.rate_limited.inc()
                logger.warning("rate_limited", client=client, path=request.url.path)
                response: Response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded."},
                    headers={"Retry-After": str(int(decision.retry_after_seconds) + 1)},
                )
                response.headers["X-Request-ID"] = request_id
                metrics.observe_request(
                    request.method, route_label(request), 429, time.monotonic() - started
                )
                return response

        response = await call_next(request)
        duration = time.monotonic() - started

        metrics.observe_request(
            request.method, route_label(request), response.status_code, duration
        )
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ZeptoError)
    async def handle_known_error(request: Request, exc: ZeptoError) -> JSONResponse:
        """Report a known failure without exposing internals."""
        logger.warning("request_failed", error=type(exc).__name__, detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "The assistant is temporarily unable to answer."},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Contain an unexpected failure.

        v1 returned the stack trace to the caller, exposing file paths and
        library internals. The detail belongs in the logs, not the response.
        """
        logger.exception("unhandled_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    @app.get("/", include_in_schema=False)
    async def landing() -> HTMLResponse:
        """A browser page, so the root path is not a JSON 404.

        Unguarded like the probe endpoints: it is static markup that performs no
        retrieval, and gating the page while leaving /ask open would protect
        nothing while making the deployment look broken.
        """
        return HTMLResponse(content=LANDING_PAGE)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Liveness, plus enough detail to tell which mode is running."""
        store: VectorStore = request.app.state.store
        return HealthResponse(
            status="ok",
            mock_llm=request.app.state.settings.mock_llm,
            documents_indexed=store.count(),
        )

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness: can this instance actually answer a question?

        An empty index means the service is running but useless, a state v1
        could enter silently and serve from indefinitely.
        """
        store: VectorStore = request.app.state.store
        indexed = store.count()

        if indexed == 0:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "index is empty"},
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "documents_indexed": indexed},
        )

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        """Prometheus exposition, from this app's own registry."""
        settings = request.app.state.settings
        if not settings.metrics_enabled:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")

        payload = generate_latest(request.app.state.metrics.registry)
        return PlainTextResponse(content=payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: Request, payload: AskRequest) -> AskResponse:
        """Answer a question from the policy corpus."""
        require_api_key(request, request.headers.get(API_KEY_HEADER))

        # Recorded, not refused. Pattern matching cannot decide whether text is
        # an instruction, so blocking on it would turn a rephrasing away from a
        # real customer while stopping nothing determined. What contains the
        # risk is that the model has no tools -- see sanitization.py.
        if looks_like_injection(payload.query):
            request.app.state.metrics.suspected_injection.inc()
            logger.warning("suspected_injection", query_length=len(payload.query))

        graph = request.app.state.graph
        state = graph.invoke({"query": payload.query})

        request.app.state.metrics.answers.labels(intent=state["intent"]).inc()

        return AskResponse(
            answer=state["answer"],
            intent=state["intent"],
            sources=[
                Source(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    relevance=round(chunk.relevance, 4),
                )
                for chunk in state.get("chunks", [])
            ],
            confidence=round(state["confidence"], 4),
        )

    return app


app = create_app()


def main() -> None:
    """Console entry point: zepto-serve."""
    import uvicorn

    settings = get_assistant_settings()
    configure_logging()
    logger.info(
        "starting_server", mock_llm=settings.mock_llm, host=settings.host, port=settings.port
    )
    uvicorn.run("zepto.assistant.api:app", host=settings.host, port=settings.port)
