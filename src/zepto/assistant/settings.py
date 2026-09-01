"""Configuration for the retrieval-augmented support assistant.

Environment variables are prefixed ZEPTO_ASSISTANT_, e.g.
ZEPTO_ASSISTANT_TOP_K=5.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    # --- Access control ---
    require_api_key: bool = Field(
        default=False,
        description="Off by default so the demo runs without setup. When enabled "
        "and no keys are configured, startup fails rather than silently serving "
        "an endpoint that believes it is protected.",
    )
    # NoDecode stops pydantic-settings from JSON-decoding the raw environment
    # value before validation. Without it, ZEPTO_ASSISTANT_API_KEYS="a,b" raises
    # a SettingsError inside the source and the validator below never runs.
    api_keys: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(),
        description="Accepted API keys. Supply as a comma-separated environment "
        "variable. Never logged; only a short fingerprint is.",
    )

    # --- Rate limiting ---
    rate_limit_requests: int = Field(
        default=60,
        ge=1,
        description="Requests permitted per client within the window.",
    )
    rate_limit_window_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Length of the fixed window, in seconds.",
    )
    rate_limit_enabled: bool = Field(default=True)
    trusted_proxy_count: int = Field(
        default=0,
        ge=0,
        description="How many reverse proxies sit in front of this service. Zero "
        "means X-Forwarded-For is ignored entirely, which is the only safe "
        "default: when nothing trustworthy sets that header, a caller can set it "
        "themselves and mint a fresh rate-limit bucket per request. Set it to 1 "
        "behind a single proxy (Hugging Face Spaces, a load balancer), where the "
        "opposite failure applies -- every caller arrives from the proxy's "
        "address and shares one bucket, so one client can exhaust everyone's quota.",
    )

    # --- Bind address ---
    host: str = Field(
        default="127.0.0.1",
        description="Loopback by default, so running the service locally does not "
        "silently expose it to the network. A container sets 0.0.0.0, where the "
        "network boundary is the container's rather than the process's.",
    )
    port: int = Field(default=8000, ge=1, le=65535)

    # --- Observability ---
    metrics_enabled: bool = Field(
        default=True,
        description="Exposes Prometheus metrics at /metrics.",
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        """Accept `a,b,c` as well as a JSON list, because a comma-separated
        string is what a person actually types into an environment variable."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

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
