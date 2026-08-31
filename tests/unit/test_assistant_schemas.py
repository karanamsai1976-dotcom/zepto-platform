"""Tests for the assistant's API contracts, especially its input limits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zepto.assistant.schemas import AskRequest, AskResponse, HealthResponse, Source

# --- request limits ---


def test_a_normal_question_is_accepted() -> None:
    request = AskRequest(query="What is the delivery fee?")

    assert request.query == "What is the delivery fee?"


def test_oversized_query_is_rejected() -> None:
    """v1 had no upper bound, so an arbitrarily large string reached the
    embedding model on an unauthenticated endpoint."""
    with pytest.raises(ValidationError):
        AskRequest(query="x" * 501)


def test_query_at_the_limit_is_accepted() -> None:
    request = AskRequest(query="x" * 500)

    assert len(request.query) == 500


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AskRequest(query="")


def test_unexpected_fields_are_rejected() -> None:
    """Silently ignoring unknown fields hides client bugs and typos."""
    with pytest.raises(ValidationError):
        AskRequest(query="hello", unexpected="value")  # type: ignore[call-arg]


# --- response contract ---


def test_response_accepts_a_valid_payload() -> None:
    response = AskResponse(
        answer="Standard delivery is free over INR 149.",
        intent="policy_question",
        sources=[Source(document_id="doc_01", chunk_id="doc_01#0", relevance=0.51)],
        confidence=0.51,
    )

    assert response.sources[0].document_id == "doc_01"
    assert response.confidence == 0.51


def test_sources_default_to_empty() -> None:
    """Questions answered without retrieval cite nothing."""
    response = AskResponse(
        answer="I can only answer policy questions.", intent="general_question", confidence=0.0
    )

    assert response.sources == []


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
def test_confidence_outside_the_unit_range_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        AskResponse(answer="a", intent="policy_question", confidence=confidence)


@pytest.mark.parametrize("relevance", [-0.5, 1.5])
def test_source_relevance_outside_the_unit_range_is_rejected(relevance: float) -> None:
    with pytest.raises(ValidationError):
        Source(document_id="doc_01", chunk_id="doc_01#0", relevance=relevance)


def test_response_is_immutable() -> None:
    response = AskResponse(answer="a", intent="policy_question", confidence=0.5)

    with pytest.raises(ValidationError):
        response.answer = "tampered"


def test_health_response_reports_mode_and_index_size() -> None:
    health = HealthResponse(status="ready", mock_llm=True, documents_indexed=8)

    assert health.status == "ready"
    assert health.mock_llm is True
    assert health.documents_indexed == 8
