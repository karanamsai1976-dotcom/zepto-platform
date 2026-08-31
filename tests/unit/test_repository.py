"""Tests for persistence: foreign key enforcement, atomicity, and exact money."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from zepto.core.errors import StorageError
from zepto.ingestion.models import Book
from zepto.ingestion.repository import (
    BookRepository,
    connect,
    from_minor_units,
    to_minor_units,
)


def _book(title: str = "Sharp Objects", category: str = "Mystery", **overrides: object) -> Book:
    defaults: dict[str, object] = {
        "title": title,
        "price_gbp": Decimal("47.82"),
        "price_inr": Decimal("5045.01"),
        "rating": 4,
        "in_stock": True,
        "category": category,
    }
    defaults.update(overrides)
    return Book(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def repository(tmp_path: Path) -> BookRepository:
    return BookRepository(tmp_path / "books.db")


@pytest.mark.parametrize(
    ("amount", "units"),
    [(Decimal("47.82"), 4782), (Decimal("0.01"), 1), (Decimal("1239.63"), 123963)],
)
def test_minor_units_round_trip_exactly(amount: Decimal, units: int) -> None:
    assert to_minor_units(amount) == units
    assert from_minor_units(units) == amount


def test_prices_survive_storage_without_precision_loss() -> None:
    """The whole point of integer storage: no float round-tripping."""
    original = Decimal("1239.63")

    assert from_minor_units(to_minor_units(original)) == original


def test_foreign_keys_are_enforced(repository: BookRepository) -> None:
    """v1 declared this constraint but never enabled the PRAGMA, so SQLite
    silently ignored it and orphan rows were insertable."""
    repository.replace_all([_book()])

    with connect(repository.database_path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO books
                (title, price_gbp_pence, price_inr_paise, rating, in_stock, category_id)
            VALUES ('Orphan', 100, 100, 3, 1, 9999)
            """
        )


def test_pragma_is_actually_on(repository: BookRepository) -> None:
    repository.replace_all([_book()])

    with connect(repository.database_path) as connection:
        (enabled,) = connection.execute("PRAGMA foreign_keys").fetchone()

    assert enabled == 1


@pytest.mark.parametrize(
    "bad_values",
    [
        "('X', 100, 100, 0, 1, 1)",
        "('X', 100, 100, 6, 1, 1)",
        "('X', 0, 100, 3, 1, 1)",
        "('X', 100, 100, 3, 2, 1)",
    ],
    ids=["rating_too_low", "rating_too_high", "zero_price", "invalid_boolean"],
)
def test_check_constraints_reject_invalid_rows(repository: BookRepository, bad_values: str) -> None:
    """The database enforces the same invariants as the model."""
    repository.replace_all([_book()])

    with connect(repository.database_path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO books "
            "(title, price_gbp_pence, price_inr_paise, rating, in_stock, category_id) "
            f"VALUES {bad_values}"
        )


def test_failed_load_leaves_previous_catalogue_intact(
    repository: BookRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1's DROP-then-insert left an empty database if the run died partway."""
    repository.replace_all([_book(title="Original")])
    assert repository.count_books() == 1

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(BookRepository, "_build_database", explode)

    with pytest.raises(RuntimeError):
        repository.replace_all([_book(title="Replacement"), _book(title="Another")])

    assert repository.count_books() == 1
    assert repository.fetch_all()[0].title == "Original"


def test_failed_load_removes_the_staging_file(
    repository: BookRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(BookRepository, "_build_database", explode)

    with pytest.raises(RuntimeError):
        repository.replace_all([_book()])

    staging = repository.database_path.with_suffix(repository.database_path.suffix + ".tmp")
    assert not staging.exists()


def test_empty_replacement_is_refused(repository: BookRepository) -> None:
    """Replacing a good catalogue with nothing is almost always a bug upstream."""
    with pytest.raises(StorageError):
        repository.replace_all([])


def test_books_round_trip_through_storage(repository: BookRepository) -> None:
    books = [
        _book(title="Sharp Objects", category="Mystery"),
        _book(title="Soumission", category="Fiction", price_gbp=Decimal("50.10")),
    ]

    repository.replace_all(books)
    loaded = repository.fetch_all()

    assert [book.title for book in loaded] == ["Sharp Objects", "Soumission"]
    assert loaded[1].price_gbp == Decimal("50.10")
    assert loaded[0].in_stock is True


def test_counts_reflect_stored_data(repository: BookRepository) -> None:
    repository.replace_all(
        [
            _book(title="A", category="Mystery"),
            _book(title="B", category="Mystery"),
            _book(title="C", category="Fiction"),
        ]
    )

    assert repository.count_books() == 3
    assert repository.count_categories() == 2


def test_replacing_twice_does_not_accumulate_rows(repository: BookRepository) -> None:
    """Idempotency: a second run replaces rather than appends."""
    repository.replace_all([_book(title="A")])
    repository.replace_all([_book(title="A")])

    assert repository.count_books() == 1
