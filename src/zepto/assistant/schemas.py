"""Request and response contracts for the assistant API.

v1's request model was `query: str` with no constraints, on an unauthenticated
endpoint. Any string of any length went straight into the embedding model. The
bounds here are the fix: they are enforced by the framework before a single line
of application code runs, so an oversized or empty payload is rejected at the
edge rather than consuming resources.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """An incoming question."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=500,
        description="The customer's question. Bounded so an oversized payload is "
        "rejected at the edge rather than reaching the embedding model.",
    )


class Source(BaseModel):
    """One retrieved passage that contributed to an answer."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_id: str
    relevance: float = Field(ge=0.0, le=1.0)


class AskResponse(BaseModel):
    """The assistant's answer, with the evidence it used."""

    model_config = ConfigDict(frozen=True)

    answer: str
    intent: str
    sources: list[Source] = Field(
        default_factory=list,
        description="Empty for questions answered without retrieval.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance of the best matching passage, on a 0-1 scale. This "
        "is a similarity score, not a calibrated probability that the answer is "
        "correct -- see the module documentation for what it does and does not mean.",
    )


class HealthResponse(BaseModel):
    """Liveness and readiness detail."""

    model_config = ConfigDict(frozen=True)

    status: str
    mock_llm: bool
    documents_indexed: int
