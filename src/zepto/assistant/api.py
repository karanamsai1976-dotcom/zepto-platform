"""HTTP interface for the support assistant.

The vector store is built once during application startup and shared by every
request. This is what makes the retrieval cost measured in milliseconds rather
than seconds: v1 rebuilt the embedding model inside each query, so its API paid
a multi-second penalty on every call.

Three things v1's API had no notion of are handled here.

Startup readiness: v1 would start happily with an empty index and fail every
request afterwards. Here the corpus is indexed during startup if the collection
is empty, and a readiness endpoint reports whether the service can actually
answer.

Error containment: an unhandled exception in v1 returned a stack trace to the
caller, leaking file paths and internals. Errors are mapped to status codes with
a safe message, and the detail goes to the logs instead.

Request correlation: every request binds an id that appears on each log line it
produces, so one request's activity can be followed through the system.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from zepto.assistant.graph import build_graph
from zepto.assistant.retrieval import VectorStore, load_corpus
from zepto.assistant.schemas import AskRequest, AskResponse, HealthResponse, Source
from zepto.assistant.settings import get_assistant_settings
from zepto.core.errors import ZeptoError
from zepto.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare the retrieval stack once, before the first request arrives."""
    configure_logging()
    settings = get_assistant_settings()

    store = VectorStore(settings=settings)
    if store.count() == 0:
        logger.info("index_empty_bootstrapping", corpus=str(settings.corpus_dir))
        store.ingest(load_corpus(settings.corpus_dir))

    app.state.settings = settings
    app.state.store = store
    app.state.graph = build_graph(store, settings=settings)

    logger.info("assistant_ready", documents=store.count(), mock_llm=settings.mock_llm)
    yield


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton, so tests can construct an
    isolated instance instead of sharing global state.
    """
    app = FastAPI(
        title="Zepto Support Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def correlate_and_time(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Bind a request id and record how long the request took."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - started) * 1000, 2)

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
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

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Liveness plus enough detail to tell which mode is running."""
        store: VectorStore = request.app.state.store
        settings = request.app.state.settings
        return HealthResponse(
            status="ok",
            mock_llm=settings.mock_llm,
            documents_indexed=store.count(),
        )

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness: can this instance actually answer a question?

        An empty index means the service is running but useless, which is a
        state v1 could enter silently and serve from indefinitely.
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

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: Request, payload: AskRequest) -> AskResponse:
        """Answer a question from the policy corpus."""
        graph = request.app.state.graph
        state = graph.invoke({"query": payload.query})

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
    logger.info("starting_server", mock_llm=settings.mock_llm)
    uvicorn.run("zepto.assistant.api:app", host="127.0.0.1", port=8000)
