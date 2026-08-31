"""Corpus ingestion and semantic retrieval.

Two v1 defects are fixed here.

The embedding model and database client are created once and reused. v1
constructed a fresh SentenceTransformer inside its retrieve() function, so every
single query paid the full model-loading cost -- roughly five seconds per
request, which is why its test suite took forty seconds to run nine tests.

Relevance is derived from the actual vector distance instead of being hardcoded.
v1 returned confidence=1.0 for every answer, including ones assembled from
barely related text, which made the field worse than useless: it looked
meaningful and never was.

Ingestion is idempotent by construction. v1 skipped ingestion whenever the
collection already held enough rows, which meant edited corpus text was silently
never re-indexed. Upserting by stable id re-indexes changed content and cannot
duplicate unchanged content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions

from zepto.assistant.settings import AssistantSettings, get_assistant_settings
from zepto.core.errors import RetrievalError
from zepto.core.logging import get_logger

logger = get_logger(__name__)

#: Cosine distance places identical direction at 0 and opposite at 2, so
#: subtracting from one yields cosine similarity directly.
COSINE_SPACE = "cosine"


@dataclass(frozen=True)
class Document:
    """One source document from the corpus."""

    document_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage returned by a search, with how well it matched."""

    document_id: str
    chunk_id: str
    text: str
    relevance: float


def relevance_from_distance(distance: float) -> float:
    """Convert a cosine distance into a relevance score in [0, 1].

    Cosine distance runs 0 (identical direction) to 2 (opposite), so cosine
    similarity is one minus the distance. Negative similarity means the vectors
    point away from each other, which is no less irrelevant than orthogonal, so
    the result is clamped at zero.

    This is a similarity measure, not a calibrated probability. It says how
    closely the question resembles the retrieved text; it does not say how
    likely the answer is to be correct. Calibrating that would require labelled
    question-answer pairs, which this project does not have. The distinction is
    documented rather than papered over, because a number presented as
    confidence will be read as one.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Read every document in the corpus directory.

    utf-8-sig strips a byte order mark if one is present. Files written by some
    Windows tooling carry one, and in v1 it survived into the indexed text and
    then into answers served to users.
    """
    if not corpus_dir.exists():
        raise RetrievalError("corpus directory not found", path=str(corpus_dir))

    documents = [
        Document(document_id=path.stem, text=path.read_text(encoding="utf-8-sig").strip())
        for path in sorted(corpus_dir.glob("*.txt"))
    ]

    if not documents:
        raise RetrievalError("corpus directory contains no documents", path=str(corpus_dir))

    return documents


class VectorStore:
    """A ChromaDB-backed semantic index over the policy corpus.

    The client, embedding function, and collection handle are built once when
    the store is constructed and reused for every subsequent call.
    """

    def __init__(
        self,
        settings: AssistantSettings | None = None,
        embedding_function: EmbeddingFunction[Embeddable] | None = None,
    ) -> None:
        self._settings = settings or get_assistant_settings()
        self._settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        # chromadb declares DefaultEmbeddingFunction more narrowly than the
        # signature of get_or_create_collection accepts, so the two do not line
        # up without help. The cast records that mismatch rather than widening
        # our own annotation to hide it.
        self._embedding_function: EmbeddingFunction[Embeddable] = (
            embedding_function
            if embedding_function is not None
            else cast(
                EmbeddingFunction[Embeddable],
                embedding_functions.DefaultEmbeddingFunction(),
            )
        )
        self._client = chromadb.PersistentClient(path=str(self._settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._settings.collection_name,
            metadata={"hnsw:space": COSINE_SPACE},
            embedding_function=self._embedding_function,
        )

        logger.info(
            "vector_store_ready",
            collection=self._settings.collection_name,
            path=str(self._settings.chroma_dir),
            documents=self.count(),
        )

    def count(self) -> int:
        """Number of indexed chunks."""
        return int(self._collection.count())

    def ingest(self, documents: list[Document]) -> int:
        """Index the given documents, replacing any previous version of each.

        Each document is treated as a single chunk: the policy documents are
        already a few sentences long, and splitting them further would separate
        a rule from its qualifying clause. The chunk id is kept distinct from
        the document id so real chunking can be introduced without changing the
        stored shape.
        """
        if not documents:
            raise RetrievalError("refusing to ingest an empty document set")

        self._collection.upsert(
            ids=[f"{document.document_id}#0" for document in documents],
            documents=[document.text for document in documents],
            metadatas=[{"document_id": document.document_id} for document in documents],
        )

        logger.info("corpus_ingested", documents=len(documents), total=self.count())
        return len(documents)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the closest passages to a query, best match first."""
        if not query.strip():
            raise RetrievalError("refusing to search for an empty query")

        limit = top_k if top_k is not None else self._settings.top_k
        results = self._collection.query(query_texts=[query], n_results=limit)

        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        chunks = [
            RetrievedChunk(
                document_id=str(metadata.get("document_id", chunk_id.split("#")[0])),
                chunk_id=str(chunk_id),
                text=str(text),
                relevance=relevance_from_distance(float(distance)),
            )
            for chunk_id, text, distance, metadata in zip(
                ids, documents, distances, metadatas, strict=False
            )
        ]

        logger.info(
            "retrieval_completed",
            query_length=len(query),
            results=len(chunks),
            best_relevance=round(chunks[0].relevance, 4) if chunks else 0.0,
        )
        return chunks
