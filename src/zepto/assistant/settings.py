"""Configuration for the retrieval-augmented support assistant.

Environment variables are prefixed ZEPTO_ASSISTANT_, e.g.
ZEPTO_ASSISTANT_TOP_K=5.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AssistantSettings(BaseSettings):
    """Settings for corpus ingestion, retrieval, and answer generation."""

    model_config = SettingsConfigDict(
        env_prefix="ZEPTO_ASSISTANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Generation mode ---
    mock_llm: bool = Field(
        default=True,
        description="When true, answers are assembled deterministically from the "
        "retrieved text with no external call. Retrieval runs for real in both "
        "modes; only the final generation step differs.",
    )

    # --- Corpus and index ---
    corpus_dir: Path = Field(default=Path("data/corpus"))
    chroma_dir: Path = Field(default=Path("data/chroma"))
    collection_name: str = Field(default="zepto_policies")

    # --- Retrieval ---
    top_k: int = Field(default=3, ge=1, le=20)
    min_relevance: float = Field(
        default=0.13,
        ge=0.0,
        le=1.0,
        description="Below this relevance a question is treated as out of scope "
        "and declined. This is also the routing decision: retrieval relevance "
        "replaced the keyword classifier, which had 3.4% recall on real "
        "questions. Calibrated against the 34-case evaluation set, where "
        "in-scope questions score 0.186-0.567 and out-of-scope 0.000-0.089; the "
        "floor sits mid-gap. Note this is fitted on the same small set it is "
        "measured against -- as the set grows, recalibrate on a held-out split.",
    )

    # --- Input limits ---
    max_query_length: int = Field(
        default=500,
        ge=1,
        description="Upper bound on an incoming question. v1 had none, so an "
        "arbitrarily large string went straight into the embedding model on an "
        "unauthenticated endpoint -- a trivial resource-exhaustion vector.",
    )

    # --- Optional real-LLM path ---
    llm_model: str = Field(default="llama-3.1-8b-instant")
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_attempts: int = Field(
        default=3,
        ge=1,
        description="Total attempts, so the default allows two retries after the first failure.",
    )
    llm_max_output_tokens: int = Field(
        default=512,
        ge=1,
        description="Bounds cost and latency per call. An unbounded generation is "
        "an unbounded bill.",
    )


@lru_cache(maxsize=1)
def get_assistant_settings() -> AssistantSettings:
    """Return the process-wide assistant settings, parsed from the environment once."""
    return AssistantSettings()
