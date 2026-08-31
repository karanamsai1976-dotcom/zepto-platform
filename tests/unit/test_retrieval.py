"""Tests for corpus loading, indexing, and semantic retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from zepto.assistant.retrieval import (
    Document,
    VectorStore,
    load_corpus,
    relevance_from_distance,
)
from zepto.assistant.settings import AssistantSettings
from zepto.core.errors import RetrievalError

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> VectorStore:
    """One store for the module: building it loads the embedding model, which
    is exactly the cost v1 paid on every single query."""
    chroma_dir = tmp_path_factory.mktemp("chroma")
    settings = AssistantSettings(chroma_dir=chroma_dir, corpus_dir=CORPUS_DIR)
    vector_store = VectorStore(settings=settings)
    vector_store.ingest(load_corpus(CORPUS_DIR))
    return vector_store


# --- relevance scoring ---


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0)],
)
def test_relevance_is_similarity_clamped_to_unit_range(distance: float, expected: float) -> None:
    assert relevance_from_distance(distance) == pytest.approx(expected)


def test_relevance_decreases_monotonically_with_distance() -> None:
    scores = [relevance_from_distance(d) for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]

    assert scores == sorted(scores, reverse=True)


# --- corpus loading ---


def test_corpus_loads_every_document() -> None:
    documents = load_corpus(CORPUS_DIR)

    assert len(documents) == 8
    assert {document.document_id for document in documents} == {
        f"doc_{index:02d}" for index in range(1, 9)
    }


def test_corpus_text_has_no_byte_order_mark() -> None:
    """A BOM survived into v1's indexed text and then into served answers."""
    documents = load_corpus(CORPUS_DIR)

    for document in documents:
        assert not document.text.startswith("﻿")
        assert document.text


def test_missing_corpus_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError):
        load_corpus(tmp_path / "absent")


def test_empty_corpus_directory_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RetrievalError):
        load_corpus(empty)


def test_byte_order_mark_is_stripped_on_read(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_01.txt").write_bytes(b"\xef\xbb\xbfPolicy text here.")

    documents = load_corpus(corpus)

    assert documents[0].text == "Policy text here."


# --- retrieval ---


def test_indexing_stores_every_document(store: VectorStore) -> None:
    assert store.count() == 8


@pytest.mark.parametrize(
    ("query", "expected_document"),
    [
        ("What is the delivery fee?", "doc_01"),
        ("Can I cancel my order?", "doc_05"),
        ("How do gift cards work?", "doc_07"),
        ("When is customer support available?", "doc_08"),
    ],
)
def test_queries_retrieve_the_right_document(
    store: VectorStore, query: str, expected_document: str
) -> None:
    """Retrieval must find the document that actually answers the question,
    not merely return something."""
    results = store.search(query, top_k=3)

    assert results[0].document_id == expected_document


def test_results_are_ordered_by_relevance(store: VectorStore) -> None:
    results = store.search("What is the delivery fee?", top_k=3)

    relevances = [result.relevance for result in results]
    assert relevances == sorted(relevances, reverse=True)


def test_relevant_and_irrelevant_queries_are_clearly_separated(
    store: VectorStore,
) -> None:
    """The score must discriminate, or it is no better than v1's hardcoded 1.0."""
    relevant = store.search("What is the delivery fee?", top_k=1)[0]
    irrelevant = store.search("Who won the football world cup?", top_k=1)[0]

    assert relevant.relevance > 0.4
    assert irrelevant.relevance < 0.2
    assert relevant.relevance > irrelevant.relevance * 3


def test_top_k_limits_the_result_count(store: VectorStore) -> None:
    assert len(store.search("delivery", top_k=2)) == 2
    assert len(store.search("delivery", top_k=5)) == 5


def test_results_carry_document_and_chunk_identity(store: VectorStore) -> None:
    result = store.search("What is the delivery fee?", top_k=1)[0]

    assert result.document_id == "doc_01"
    assert result.chunk_id == "doc_01#0"
    assert result.text


def test_empty_query_is_refused(store: VectorStore) -> None:
    with pytest.raises(RetrievalError):
        store.search("   ")


# --- ingestion behaviour ---


def test_reingesting_does_not_duplicate(tmp_path: Path) -> None:
    """v1 skipped ingestion when the collection looked full enough, which meant
    upserting by id was never exercised. Upsert makes repetition harmless."""
    settings = AssistantSettings(chroma_dir=tmp_path / "chroma", corpus_dir=CORPUS_DIR)
    store = VectorStore(settings=settings)
    documents = load_corpus(CORPUS_DIR)

    store.ingest(documents)
    store.ingest(documents)

    assert store.count() == 8


def test_reingesting_updates_changed_text(tmp_path: Path) -> None:
    """The bug v1's count-based skip caused: edited corpus text was never
    re-indexed, so the index silently diverged from the source of truth."""
    settings = AssistantSettings(chroma_dir=tmp_path / "chroma", corpus_dir=CORPUS_DIR)
    store = VectorStore(settings=settings)

    store.ingest([Document(document_id="doc_01", text="Original policy text.")])
    store.ingest([Document(document_id="doc_01", text="Revised policy text.")])

    assert store.count() == 1
    assert store.search("policy", top_k=1)[0].text == "Revised policy text."


def test_ingesting_nothing_is_refused(tmp_path: Path) -> None:
    settings = AssistantSettings(chroma_dir=tmp_path / "chroma", corpus_dir=CORPUS_DIR)
    store = VectorStore(settings=settings)

    with pytest.raises(RetrievalError):
        store.ingest([])
