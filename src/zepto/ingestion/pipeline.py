"""End-to-end ingestion: scrape, clean, validate, store.

Validation runs before storage, so a scrape degraded by a site change cannot
overwrite a good catalogue with a bad one. Together with the repository's
atomic swap, that is two independent safeguards against publishing bad data.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from zepto.core.errors import IngestionError
from zepto.core.logging import configure_logging, get_logger
from zepto.ingestion.cleaning import clean_book
from zepto.ingestion.models import Book, ScrapedBook
from zepto.ingestion.repository import BookRepository
from zepto.ingestion.scraper import BookScraper
from zepto.ingestion.settings import IngestionSettings, get_ingestion_settings

logger = get_logger(__name__)


class Scraper(Protocol):
    """The subset of BookScraper the pipeline depends on."""

    def scrape_all(self) -> list[ScrapedBook]: ...


class Repository(Protocol):
    """The subset of BookRepository the pipeline depends on."""

    def replace_all(self, books: Sequence[Book]) -> None: ...


@dataclass(frozen=True)
class IngestionResult:
    """Structured summary of one pipeline run."""

    scraped: int
    stored: int
    rejected: int
    categories: int
    duration_seconds: float


class IngestionPipeline:
    """Orchestrates scrape -> clean -> validate -> store."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        scraper: Scraper | None = None,
        repository: Repository | None = None,
    ) -> None:
        self._settings = settings or get_ingestion_settings()
        self._scraper: Scraper = scraper or BookScraper(settings=self._settings)
        self._repository: Repository = repository or BookRepository(self._settings.database_path)

    def run(self) -> IngestionResult:
        """Run the full pipeline, returning a summary of what happened."""
        started = time.monotonic()

        scraped = self._scraper.scrape_all()
        books = self._clean(scraped)
        self._validate(books)
        self._repository.replace_all(books)

        result = IngestionResult(
            scraped=len(scraped),
            stored=len(books),
            rejected=len(scraped) - len(books),
            categories=len({book.category for book in books}),
            duration_seconds=round(time.monotonic() - started, 2),
        )
        logger.info("ingestion_complete", **asdict(result))
        return result

    def _clean(self, scraped: Sequence[ScrapedBook]) -> list[Book]:
        rate = self._settings.gbp_to_inr
        books = [book for book in (clean_book(item, rate) for item in scraped) if book is not None]

        rejected = len(scraped) - len(books)
        if rejected:
            logger.warning("books_rejected", rejected=rejected, scraped=len(scraped))

        return books

    def _validate(self, books: Sequence[Book]) -> None:
        """Refuse to publish a run that came back suspiciously thin.

        Runs before storage: a site change that halves the yield should fail
        the run, not quietly replace good data with bad.
        """
        categories = {book.category for book in books}

        if len(books) < self._settings.min_books:
            raise IngestionError(
                "too few books scraped",
                books=len(books),
                minimum=self._settings.min_books,
            )

        if len(categories) < self._settings.min_categories:
            raise IngestionError(
                "too few categories scraped",
                categories=len(categories),
                minimum=self._settings.min_categories,
            )


def main() -> None:
    """Console entry point: zepto-ingest."""
    configure_logging()
    IngestionPipeline().run()
