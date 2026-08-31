"""Tests for retrieval-based routing, answer composition, and the decline path."""

from __future__ import annotations

import pytest

from zepto.assistant.graph import (
    ANSWERED_INTENT,
    DECLINE_ANSWER,
    DECLINED_INTENT,
    build_graph,
    compose_policy_answer,
)
from zepto.assistant.retrieval import RetrievedChunk
from zepto.assistant.settings import AssistantSettings


class StubRetriever:
    """Returns fixed chunks, so relevance can be controlled exactly."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks


def _chunk(
    relevance: float,
    document_id: str = "doc_01",
    text: str = "Delivery is free over INR 149.",
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        chunk_id=f"{document_id}#0",
        text=text,
        relevance=relevance,
    )


# --- answer composition ---


def test_policy_answer_quotes_the_matched_text_in_full() -> None:
    """v1 truncated at 200 characters, routinely cutting sentences mid-word."""
    long_text = "Zepto delivers within 10 to 30 minutes. " * 12
    answer = compose_policy_answer([_chunk(0.6, text=long_text)])

    assert long_text.strip() in answer
    assert not answer.endswith("...")


# --- routing by relevance ---


def test_a_strong_match_is_answered_from_retrieved_text() -> None:
    retriever = StubRetriever([_chunk(0.55)])
    graph = build_graph(retriever, settings=AssistantSettings())

    state = graph.invoke({"query": "How much does shipping cost?"})

    assert state["intent"] == ANSWERED_INTENT
    assert "INR 149" in state["answer"]
    assert state["confidence"] == pytest.approx(0.55)
    assert state["chunks"][0].document_id == "doc_01"


def test_a_question_without_keywords_is_still_answered() -> None:
    """The defect this design replaced.

    Keyword routing sent this to a refusal because it contains none of the eight
    substrings it tested. Measured across 29 real questions, that classifier had
    3.4% recall.
    """
    retriever = StubRetriever([_chunk(0.42)])
    graph = build_graph(retriever, settings=AssistantSettings())

    state = graph.invoke({"query": "How much does shipping cost?"})

    assert state["intent"] == ANSWERED_INTENT
    assert state["answer"] != DECLINE_ANSWER


def test_a_weak_match_is_declined() -> None:
    """Retrieval always returns something -- asking a vector index about
    football still yields the nearest policy document."""
    retriever = StubRetriever([_chunk(0.05)])
    graph = build_graph(retriever, settings=AssistantSettings(min_relevance=0.13))

    state = graph.invoke({"query": "Who won the world cup?"})

    assert state["intent"] == DECLINED_INTENT
    assert state["answer"] == DECLINE_ANSWER
    assert state["chunks"] == []
    assert state["confidence"] == pytest.approx(0.05)


def test_relevance_exactly_at_the_floor_is_answered() -> None:
    retriever = StubRetriever([_chunk(0.13)])
    graph = build_graph(retriever, settings=AssistantSettings(min_relevance=0.13))

    assert graph.invoke({"query": "anything"})["intent"] == ANSWERED_INTENT


def test_relevance_just_below_the_floor_is_declined() -> None:
    retriever = StubRetriever([_chunk(0.129)])
    graph = build_graph(retriever, settings=AssistantSettings(min_relevance=0.13))

    assert graph.invoke({"query": "anything"})["intent"] == DECLINED_INTENT


def test_empty_retrieval_is_declined() -> None:
    retriever = StubRetriever([])
    graph = build_graph(retriever, settings=AssistantSettings())

    state = graph.invoke({"query": "anything"})

    assert state["intent"] == DECLINED_INTENT
    assert state["answer"] == DECLINE_ANSWER
    assert state["confidence"] == 0.0


def test_the_floor_is_configurable() -> None:
    """Raising the floor trades answered questions for stricter grounding."""
    chunks = [_chunk(0.3)]

    lenient = build_graph(StubRetriever(chunks), settings=AssistantSettings(min_relevance=0.13))
    strict = build_graph(StubRetriever(chunks), settings=AssistantSettings(min_relevance=0.5))

    assert lenient.invoke({"query": "q"})["intent"] == ANSWERED_INTENT
    assert strict.invoke({"query": "q"})["intent"] == DECLINED_INTENT


# --- confidence ---


def test_confidence_reflects_the_best_match_not_a_constant() -> None:
    """v1 returned 1.0 for everything, which made the field actively misleading."""
    settings = AssistantSettings()

    strong = build_graph(StubRetriever([_chunk(0.82)]), settings=settings).invoke({"query": "q"})
    moderate = build_graph(StubRetriever([_chunk(0.41)]), settings=settings).invoke({"query": "q"})

    assert strong["confidence"] == pytest.approx(0.82)
    assert moderate["confidence"] == pytest.approx(0.41)
    assert strong["confidence"] > moderate["confidence"]


def test_confidence_is_reported_even_when_declining() -> None:
    """Knowing how close a declined question came is useful for tuning."""
    graph = build_graph(
        StubRetriever([_chunk(0.09)]), settings=AssistantSettings(min_relevance=0.13)
    )

    state = graph.invoke({"query": "q"})

    assert state["intent"] == DECLINED_INTENT
    assert state["confidence"] == pytest.approx(0.09)


# --- generation modes ---


def test_real_llm_mode_delegates_generation_to_the_llm_module() -> None:
    """With mock mode off, the answer comes from the language model path, while
    retrieval and the scope decision stay exactly the same."""
    import zepto.assistant.llm as llm_module

    original = llm_module.generate_grounded_answer
    llm_module.generate_grounded_answer = lambda question, chunks, settings=None, client=None: (
        "answer from the model"
    )
    try:
        graph = build_graph(
            StubRetriever([_chunk(0.6)]), settings=AssistantSettings(mock_llm=False)
        )
        state = graph.invoke({"query": "How much does shipping cost?"})
    finally:
        llm_module.generate_grounded_answer = original

    assert state["answer"] == "answer from the model"
    assert state["confidence"] == pytest.approx(0.6)


def test_the_scope_decision_does_not_depend_on_generation_mode() -> None:
    """Only generation differs between modes; the decision to answer must not."""
    chunks = [_chunk(0.05)]

    mock_state = build_graph(
        StubRetriever(chunks), settings=AssistantSettings(mock_llm=True, min_relevance=0.13)
    ).invoke({"query": "q"})

    real_state = build_graph(
        StubRetriever(chunks), settings=AssistantSettings(mock_llm=False, min_relevance=0.13)
    ).invoke({"query": "q"})

    assert mock_state["intent"] == real_state["intent"] == DECLINED_INTENT
