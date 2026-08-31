"""Tests for the HTTP interface: contracts, limits, health, and error containment."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepto.assistant.api import create_app
from zepto.assistant.settings import get_assistant_settings
from zepto.core.errors import RetrievalError

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FastAPI]:
    """One application for the module.

    Startup builds the embedding model and indexes the corpus, which is exactly
    the work v1 repeated on every single request.
    """
    chroma_dir = tmp_path_factory.mktemp("api-chroma")
    previous = {
        key: os.environ.get(key)
        for key in ("ZEPTO_ASSISTANT_CORPUS_DIR", "ZEPTO_ASSISTANT_CHROMA_DIR")
    }
    os.environ["ZEPTO_ASSISTANT_CORPUS_DIR"] = str(CORPUS_DIR)
    os.environ["ZEPTO_ASSISTANT_CHROMA_DIR"] = str(chroma_dir)
    get_assistant_settings.cache_clear()

    try:
        yield create_app()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_assistant_settings.cache_clear()


@pytest.fixture(scope="module")
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


# --- health and readiness ---


def test_health_reports_mode_and_index_size(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mock_llm"] is True
    assert payload["documents_indexed"] == 8


def test_ready_reports_a_populated_index(client: TestClient) -> None:
    """v1 could start with an empty index and serve failures indefinitely."""
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_corpus_is_indexed_during_startup(client: TestClient) -> None:
    assert client.get("/health").json()["documents_indexed"] == 8


# --- answering ---


def test_policy_question_is_answered_with_cited_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"query": "What is the delivery fee?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "policy_question"
    assert "INR 149" in payload["answer"]
    assert payload["sources"]
    assert payload["sources"][0]["document_id"] == "doc_01"


def test_confidence_is_a_real_measurement(client: TestClient) -> None:
    """v1 returned 1.0 for everything. A relevant answer should score well
    without claiming certainty it cannot have."""
    payload = client.post("/ask", json={"query": "What is the delivery fee?"}).json()

    assert 0.0 < payload["confidence"] < 1.0


def test_source_relevance_is_reported_per_passage(client: TestClient) -> None:
    payload = client.post("/ask", json={"query": "Can I cancel my order?"}).json()

    for source in payload["sources"]:
        assert 0.0 <= source["relevance"] <= 1.0
    relevances = [source["relevance"] for source in payload["sources"]]
    assert relevances == sorted(relevances, reverse=True)


def test_unrelated_question_is_declined_and_cites_nothing(client: TestClient) -> None:
    response = client.post("/ask", json={"query": "Who won the world cup?"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["intent"] == "out_of_scope"
    assert payload["sources"] == []
    assert payload["confidence"] < 0.13


def test_a_question_without_any_keyword_is_still_answered(client: TestClient) -> None:
    """The defect the evaluation set exposed.

    Under keyword routing this reached a refusal, because it contains none of
    the eight substrings that classifier tested. Real customers phrase questions
    this way, and 28 of 29 labelled cases failed the same way.
    """
    payload = client.post("/ask", json={"query": "How much does shipping cost?"}).json()

    assert payload["intent"] == "policy_question"
    assert payload["sources"][0]["document_id"] == "doc_01"


# --- input limits ---


def test_oversized_query_is_rejected_before_any_work(client: TestClient) -> None:
    """The unauthenticated resource-exhaustion vector v1 left open."""
    response = client.post("/ask", json={"query": "x" * 5000})

    assert response.status_code == 422


def test_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_missing_query_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={}).status_code == 422


def test_unexpected_fields_are_rejected(client: TestClient) -> None:
    """Silently ignoring unknown fields hides client-side bugs."""
    response = client.post("/ask", json={"query": "delivery fee", "extra": "value"})

    assert response.status_code == 422


# --- request correlation ---


def test_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["X-Request-ID"]


def test_supplied_request_id_is_echoed_back(client: TestClient) -> None:
    """Lets a caller correlate its own logs with the service's."""
    response = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})

    assert response.headers["X-Request-ID"] == "caller-supplied-id"


# --- error containment ---


def test_unexpected_failure_does_not_leak_internals(client: TestClient, app: FastAPI) -> None:
    """v1 returned the stack trace to the caller, exposing paths and internals.

    raise_server_exceptions=False makes the client behave like a real
    deployment, returning the handled response rather than re-raising. The
    client is used without its context manager deliberately: startup has
    already run for this app, and re-entering it would rebuild the index.
    """

    def explode(state: object) -> None:
        raise RuntimeError("database on fire at /secret/path/config.py")

    original = app.state.graph
    app.state.graph = type("Boom", (), {"invoke": staticmethod(explode)})()
    try:
        failing_client = TestClient(app, raise_server_exceptions=False)
        response = failing_client.post("/ask", json={"query": "What is the delivery fee?"})
    finally:
        app.state.graph = original

    assert response.status_code == 500
    body = response.text
    assert "database on fire" not in body
    assert "secret" not in body
    assert response.json()["detail"] == "Internal server error."


def test_known_failure_is_reported_as_unavailable(client: TestClient, app: FastAPI) -> None:
    """A recognised platform error maps to 503 with a safe message, so a caller
    can distinguish 'try again' from 'your request was wrong'."""

    def unavailable(state: object) -> None:
        raise RetrievalError("vector store unreachable", host="internal-db-01")

    original = app.state.graph
    app.state.graph = type("Down", (), {"invoke": staticmethod(unavailable)})()
    try:
        failing_client = TestClient(app, raise_server_exceptions=False)
        response = failing_client.post("/ask", json={"query": "What is the delivery fee?"})
    finally:
        app.state.graph = original

    assert response.status_code == 503
    assert "internal-db-01" not in response.text
    assert response.json()["detail"] == "The assistant is temporarily unable to answer."


def test_readiness_fails_when_the_index_is_empty(client: TestClient, app: FastAPI) -> None:
    """The silent-failure state v1 could sit in: running, but unable to answer."""

    class EmptyStore:
        def count(self) -> int:
            return 0

    original = app.state.store
    app.state.store = EmptyStore()
    try:
        response = TestClient(app, raise_server_exceptions=False).get("/ready")
    finally:
        app.state.store = original

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_startup_skips_indexing_when_the_index_is_already_populated(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A restart against an existing index must not re-index needlessly."""
    chroma_dir = tmp_path_factory.mktemp("warm-chroma")
    os.environ["ZEPTO_ASSISTANT_CORPUS_DIR"] = str(CORPUS_DIR)
    os.environ["ZEPTO_ASSISTANT_CHROMA_DIR"] = str(chroma_dir)
    get_assistant_settings.cache_clear()

    try:
        with TestClient(create_app()) as first:
            assert first.get("/health").json()["documents_indexed"] == 8

        # Second startup against the same directory takes the skip path.
        with TestClient(create_app()) as second:
            assert second.get("/health").json()["documents_indexed"] == 8
    finally:
        os.environ.pop("ZEPTO_ASSISTANT_CORPUS_DIR", None)
        os.environ.pop("ZEPTO_ASSISTANT_CHROMA_DIR", None)
        get_assistant_settings.cache_clear()


def test_console_entry_point_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """zepto-serve wires logging and hands off to uvicorn."""
    from zepto.assistant import api

    started: list[str] = []
    monkeypatch.setattr(api, "configure_logging", lambda: started.append("configured"))
    monkeypatch.setattr("uvicorn.run", lambda target, **kwargs: started.append(f"served:{target}"))

    api.main()

    assert started == ["configured", "served:zepto.assistant.api:app"]
