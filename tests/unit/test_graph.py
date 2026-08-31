"""Tests for routing, answer composition, and the abstain path."""

from __future__ import annotations

import pytest

from zepto.assistant.graph import (
    ABSTAIN_ANSWER,
    GENERAL_ANSWER,
    GENERAL_INTENT,
    POLICY_INTENT,
    build_graph,
    classify,
    compose_policy_answer,
)
from zepto.assistant.retrieval import RetrievedChunk
from zepto.assistant.settings import AssistantSettings

KEYWORDS = AssistantSettings().policy_keywords


class StubRetriever:
    """Returns fixed chunks, so relevance can be controlled exactly."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks


def _chunk(
    relevance: float, document_id: str = "doc_01", text: str = "Delivery is free over INR 149."
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        chunk_id=f"{document_id}#0",
        text=text,
        relevance=relevance,
    )


# --- routing ---


@pytest.mark.parametrize(
    "query",
    [
        "What is the delivery fee?",
        "How do I get a refund?",
        "Can I cancel my order?",
        "What is your cancellation policy?",
        "Do you sell gift cards?",
        "What are your support hours?",
    ],
)
def test_policy_questions_route_to_retrieval(query: str) -> None:
    assert classify(query, KEYWORDS) == POLICY_INTENT


@pytest.mark.parametrize(
    "query",
    ["Who won the world cup?", "What is the capital of France?", "Tell me a joke"],
)
def test_unrelated_questions_route_to_the_direct_answer(query: str) -> None:
    assert classify(query, KEYWORDS) == GENERAL_INTENT


def test_matching_is_substring_based() -> None:
    """'cancellation' must match the 'cancel' keyword."""
    assert classify("What is your cancellation policy?", KEYWORDS) == POLICY_INTENT


def test_matching_ignores_case() -> None:
    assert classify("DELIVERY FEE?", KEYWORDS) == POLICY_INTENT


# --- answer composition ---


def test_policy_answer_quotes_the_matched_text_in_full() -> None:
    """v1 truncated at 200 characters, routinely cutting sentences mid-word."""
    long_text = "Zepto delivers within 10 to 30 minutes. " * 12
    answer = compose_policy_answer([_chunk(0.6, text=long_text)])

    assert long_text.strip() in answer
    assert not answer.endswith("...")


# --- graph behaviour ---


def test_policy_question_is_answered_from_retrieved_text() -> None:
    retriever = StubRetriever([_chunk(0.55)])
    graph = build_graph(retriever, settings=AssistantSettings())

    state = graph.invoke({"query": "What is the delivery fee?"})

    assert state["intent"] == POLICY_INTENT
    assert "INR 149" in state["answer"]
    assert state["confidence"] == pytest.approx(0.55)
    assert state["chunks"][0].document_id == "doc_01"


def test_unrelated_question_is_declined_without_retrieval() -> None:
    retriever = StubRetriever([_chunk(0.9)])
    graph = build_graph(retriever, settings=AssistantSettings())

    state = graph.invoke({"query": "Who won the world cup?"})

    assert state["intent"] == GENERAL_INTENT
    assert state["answer"] == GENERAL_ANSWER
    assert state["chunks"] == []
    assert state["confidence"] == 0.0
    assert retriever.queries == []


def test_weak_match_abstains_rather_than_answering() -> None:
    """The behaviour v1 lacked entirely.

    Retrieval always returns something -- asking a vector index about football
    still yields the nearest policy document. v1 answered from it and reported
    full confidence. Below the relevance floor, this declines instead.
    """
    retriever = StubRetriever([_chunk(0.05)])
    graph = build_graph(retriever, settings=AssistantSettings(min_relevance=0.25))

    state = graph.invoke({"query": "What is the delivery schedule on Mars?"})

    assert state["intent"] == POLICY_INTENT
    assert state["answer"] == ABSTAIN_ANSWER
    assert state["chunks"] == []
    assert state["confidence"] == pytest.approx(0.05)


def test_relevance_just_above_the_floor_is_answered() -> None:
    retriever = StubRetriever([_chunk(0.26)])
    graph = build_graph(retriever, settings=AssistantSettings(min_relevance=0.25))

    state = graph.invoke({"query": "What is the delivery fee?"})

    assert state["answer"] != ABSTAIN_ANSWER


def test_empty_retrieval_abstains() -> None:
    retriever = StubRetriever([])
    graph = build_graph(retriever, settings=AssistantSettings())

    state = graph.invoke({"query": "What is the delivery fee?"})

    assert state["answer"] == ABSTAIN_ANSWER
    assert state["confidence"] == 0.0


def test_confidence_reflects_the_best_match_not_a_constant() -> None:
    """v1 returned 1.0 for everything, which made the field actively misleading."""
    settings = AssistantSettings()

    strong = build_graph(StubRetriever([_chunk(0.82)]), settings=settings).invoke(
        {"query": "What is the delivery fee?"}
    )
    moderate = build_graph(StubRetriever([_chunk(0.41)]), settings=settings).invoke(
        {"query": "What is the delivery fee?"}
    )

    assert strong["confidence"] == pytest.approx(0.82)
    assert moderate["confidence"] == pytest.approx(0.41)
    assert strong["confidence"] > moderate["confidence"]


def test_real_llm_mode_delegates_generation_to_the_llm_module() -> None:
    """With mock mode off, the answer must come from the language model path,
    while retrieval and routing stay exactly the same."""
    import zepto.assistant.llm as llm_module

    original = llm_module.generate_grounded_answer
    llm_module.generate_grounded_answer = lambda question, chunks, settings=None, client=None: (
        "answer from the model"
    )
    try:
        graph = build_graph(
            StubRetriever([_chunk(0.6)]), settings=AssistantSettings(mock_llm=False)
        )
        state = graph.invoke({"query": "What is the delivery fee?"})
    finally:
        llm_module.generate_grounded_answer = original

    assert state["answer"] == "answer from the model"
    assert state["confidence"] == pytest.approx(0.6)
    assert state["chunks"][0].document_id == "doc_01"


def test_routing_does_not_depend_on_generation_mode() -> None:
    """Only generation differs between modes; routing must be identical."""
    chunks = [_chunk(0.6)]

    mock_state = build_graph(
        StubRetriever(chunks), settings=AssistantSettings(mock_llm=True)
    ).invoke({"query": "Who won the world cup?"})

    real_state = build_graph(
        StubRetriever(chunks), settings=AssistantSettings(mock_llm=False)
    ).invoke({"query": "Who won the world cup?"})

    assert mock_state["intent"] == real_state["intent"] == GENERAL_INTENT
