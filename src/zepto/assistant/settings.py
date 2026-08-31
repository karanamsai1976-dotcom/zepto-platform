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
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Below this relevance the assistant abstains rather than "
        "answering from weakly related text. Answering confidently from a poor "
        "match is worse than admitting the corpus does not cover the question.",
    )

    # --- Input limits ---
    max_query_length: int = Field(
        default=500,
        ge=1,
        description="Upper bound on an incoming question. v1 had none, so an "
        "arbitrarily large string went straight into the embedding model on an "
        "unauthenticated endpoint -- a trivial resource-exhaustion vector.",
    )

    # --- Routing ---
    policy_keywords: tuple[str, ...] = Field(
        default=(
            "delivery",
            "return",
            "refund",
            "membership",
            "tracking",
            "cancel",
            "gift card",
            "support hours",
        ),
        description="Substring matches that route a question to retrieval. Plain "
        "substring matching, so 'cancellation' matches 'cancel'.",
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
