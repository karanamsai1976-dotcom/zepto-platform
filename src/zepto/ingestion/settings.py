"""Configuration for the ingestion pipeline.

Lives alongside the module rather than in core, so that core never needs to
know which modules exist. Environment variables are prefixed
ZEPTO_INGESTION_, e.g. ZEPTO_INGESTION_TIMEOUT_SECONDS=30.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Settings for scraping, cleaning, and storing the book catalogue."""

    model_config = SettingsConfigDict(
        env_prefix="ZEPTO_INGESTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Source ---
    base_url: str = Field(
        default="http://books.toscrape.com/catalogue/category/books/{slug}/index.html",
        description="Category listing URL template. Category pages expose the "
        "category name directly, unlike the site's 'All products' listing.",
    )
    category_slugs: tuple[str, ...] = Field(
        default=(
            "fiction_10",
            "mystery_3",
            "historical-fiction_4",
            "sequential-art_5",
            "nonfiction_13",
            "fantasy_19",
            "young-adult_21",
            "romance_8",
            "childrens_11",
        ),
        description="Category slugs to scrape, including the site's numeric id suffix.",
    )

    # --- HTTP behaviour: all of these were absent in v1 ---
    timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Per-request timeout. Without one, a stalled connection hangs forever.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retry attempts for transient failures (connection errors, 5xx).",
    )
    backoff_factor: float = Field(
        default=0.5,
        ge=0,
        description="Exponential backoff multiplier between retries.",
    )
    request_delay_seconds: float = Field(
        default=0.5,
        ge=0,
        description="Politeness delay between successive requests.",
    )
    user_agent: str = Field(
        default="zepto-platform/0.1 (+https://github.com/karanamsai1976-dotcom/zepto-platform)",
        description="Identifies the scraper honestly. The default python-requests "
        "agent is blocked by many sites.",
    )

    # --- Domain ---
    gbp_to_inr: Decimal = Field(
        default=Decimal("105.50"),
        gt=0,
        description="Fixed project conversion rate. Decimal, not float, so that "
        "rounding is exact and deterministic.",
    )

    # --- Expectations, asserted after a run ---
    min_books: int = Field(default=60, ge=1)
    min_categories: int = Field(default=3, ge=1)

    # --- Storage ---
    database_path: Path = Field(default=Path("data/books.db"))

    def category_url(self, slug: str) -> str:
        """Build the listing URL for a category slug."""
        return self.base_url.format(slug=slug)


@lru_cache(maxsize=1)
def get_ingestion_settings() -> IngestionSettings:
    """Return the process-wide ingestion settings, parsed from the environment once."""
    return IngestionSettings()
