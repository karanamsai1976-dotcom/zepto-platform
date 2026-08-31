"""Tests for pipeline orchestration, with emphasis on the validation gate."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from zepto.core.errors import IngestionError
from zepto.ingestion.models import Book, ScrapedBook
from zepto.ingestion.pipeline import IngestionPipeline, IngestionResult, main
from zepto.ingestion.repository import BookRepository
from zepto.ingestion.settings import IngestionSettings


def _scraped(
    title: str = "Sharp Objects", category: str = "Mystery", **overrides: str
) -> ScrapedBook:
    defaults = {
        "title": title,
        "price_text": "£47.82",
        "rating_word": "Four",
        "availability_text": "In stock",
        "category": category,
    }
    defaults.update(overrides)
    return ScrapedBook(**defaults)


class FakeScraper:
    def __init__(self, books: list[ScrapedBook]) -> None:
        self._books = books

    def scrape_all(self) -> list[ScrapedBook]:
        return list(self._books)


class RecordingRepository:
    def __init__(self) -> None:
        self.stored: list[Book] = []
        self.calls = 0

    def replace_all(self, books: Sequence[Book]) -> None:
        self.calls += 1
        self.stored = list(books)


def _settings(**overrides: object) -> IngestionSettings:
    defaults: dict[str, object] = {"min_books": 1, "min_categories": 1}
    defaults.update(overrides)
    return IngestionSettings(**defaults)  # type: ignore[arg-type]


def test_pipeline_stores_cleaned_books() -> None:
    repository = RecordingRepository()
    pipeline = IngestionPipeline(
        settings=_settings(),
        scraper=FakeScraper([_scraped(title="A"), _scraped(title="B")]),
        repository=repository,
    )

    result = pipeline.run()

    assert isinstance(result, IngestionResult)
    assert result.scraped == 2
    assert result.stored == 2
    assert result.rejected == 0
    assert [book.title for book in repository.stored] == ["A", "B"]


def test_conversion_uses_the_configured_rate() -> None:
    repository = RecordingRepository()
    pipeline = IngestionPipeline(
        settings=_settings(gbp_to_inr=Decimal("100.00")),
        scraper=FakeScraper([_scraped(price_text="£10.00")]),
        repository=repository,
    )

    pipeline.run()

    assert repository.stored[0].price_inr == Decimal("1000.00")


def test_unparseable_books_are_rejected_and_counted() -> None:
    repository = RecordingRepository()
    pipeline = IngestionPipeline(
        settings=_settings(),
        scraper=FakeScraper([_scraped(title="Good"), _scraped(title="Bad", rating_word="Zero")]),
        repository=repository,
    )

    result = pipeline.run()

    assert result.scraped == 2
    assert result.stored == 1
    assert result.rejected == 1


def test_too_few_books_fails_the_run() -> None:
    pipeline = IngestionPipeline(
        settings=_settings(min_books=10),
        scraper=FakeScraper([_scraped()]),
        repository=RecordingRepository(),
    )

    with pytest.raises(IngestionError) as exc_info:
        pipeline.run()

    assert exc_info.value.context["minimum"] == 10


def test_too_few_categories_fails_the_run() -> None:
    pipeline = IngestionPipeline(
        settings=_settings(min_categories=3),
        scraper=FakeScraper([_scraped(category="Mystery"), _scraped(category="Mystery")]),
        repository=RecordingRepository(),
    )

    with pytest.raises(IngestionError):
        pipeline.run()


def test_validation_runs_before_storage() -> None:
    """The critical ordering: a thin run must never reach the repository."""
    repository = RecordingRepository()
    pipeline = IngestionPipeline(
        settings=_settings(min_books=100),
        scraper=FakeScraper([_scraped()]),
        repository=repository,
    )

    with pytest.raises(IngestionError):
        pipeline.run()

    assert repository.calls == 0
    assert repository.stored == []


def test_a_failed_run_leaves_a_real_catalogue_intact(tmp_path: Path) -> None:
    """End to end with real storage: the previous catalogue survives a bad run."""
    repository = BookRepository(tmp_path / "books.db")
    good = IngestionPipeline(
        settings=_settings(),
        scraper=FakeScraper([_scraped(title="Original")]),
        repository=repository,
    )
    good.run()

    thin = IngestionPipeline(
        settings=_settings(min_books=50),
        scraper=FakeScraper([_scraped(title="Replacement")]),
        repository=repository,
    )
    with pytest.raises(IngestionError):
        thin.run()

    assert repository.count_books() == 1
    assert repository.fetch_all()[0].title == "Original"


def test_result_reports_categories_and_duration() -> None:
    pipeline = IngestionPipeline(
        settings=_settings(),
        scraper=FakeScraper([_scraped(category="Mystery"), _scraped(category="Fiction")]),
        repository=RecordingRepository(),
    )

    result = pipeline.run()

    assert result.categories == 2
    assert result.duration_seconds >= 0


def test_console_entry_point_configures_logging_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zepto-ingest must configure logging before the pipeline emits anything."""
    events: list[str] = []

    def fake_configure_logging() -> None:
        events.append("configured")

    class StubPipeline:
        def run(self) -> None:
            events.append("ran")

    monkeypatch.setattr("zepto.ingestion.pipeline.configure_logging", fake_configure_logging)
    monkeypatch.setattr("zepto.ingestion.pipeline.IngestionPipeline", StubPipeline)

    main()

    assert events == ["configured", "ran"]
