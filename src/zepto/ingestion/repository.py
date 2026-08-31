"""SQLite persistence for the book catalogue.

Two v1 defects are addressed here.

Foreign keys were declared but never enforced: SQLite ignores foreign key
constraints unless PRAGMA foreign_keys is enabled on each connection, so v1's
constraint was decorative and orphan rows were insertable.

The loader dropped both tables and re-inserted row by row, so a crash partway
through left an empty database with no path back. Here the new database is
built beside the old one and swapped in with an atomic rename, so a failure at
any point leaves the previous data untouched.

Money is stored as integer minor units (pence, paise) rather than REAL. Binary
floating point cannot represent most decimal fractions exactly, and REAL
storage would reintroduce the rounding ambiguity that Decimal eliminates
upstream.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from zepto.core.errors import StorageError
from zepto.core.logging import get_logger
from zepto.ingestion.models import Book

logger = get_logger(__name__)

CREATE_CATEGORIES = """
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
)
"""

CREATE_BOOKS = """
CREATE TABLE books (
    book_id         INTEGER PRIMARY KEY,
    title           TEXT    NOT NULL,
    price_gbp_pence INTEGER NOT NULL CHECK (price_gbp_pence > 0),
    price_inr_paise INTEGER NOT NULL CHECK (price_inr_paise > 0),
    rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock        INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id     INTEGER NOT NULL REFERENCES categories(category_id)
)
"""

CREATE_BOOKS_CATEGORY_INDEX = "CREATE INDEX idx_books_category_id ON books(category_id)"


def to_minor_units(amount: Decimal) -> int:
    """Convert a decimal amount to exact integer minor units (pence, paise)."""
    return int((amount * 100).to_integral_value(rounding=ROUND_HALF_UP))


def from_minor_units(units: int) -> Decimal:
    """Convert integer minor units back to a two-place decimal amount."""
    return (Decimal(units) / 100).quantize(Decimal("0.01"))


@contextmanager
def connect(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with foreign key enforcement enabled.

    The PRAGMA is per-connection and off by default, which is why v1's foreign
    key constraint never actually did anything.
    """
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
    finally:
        connection.close()


class BookRepository:
    """Reads and writes the book catalogue."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    @property
    def database_path(self) -> Path:
        return self._database_path

    def replace_all(self, books: Sequence[Book]) -> None:
        """Replace the catalogue with the given books, atomically.

        The new database is built in a temporary file and swapped into place
        only once it is complete, so an interrupted run cannot destroy the
        previous catalogue.
        """
        if not books:
            raise StorageError("refusing to replace catalogue with an empty set")

        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = self._database_path.with_suffix(self._database_path.suffix + ".tmp")
        staging_path.unlink(missing_ok=True)

        try:
            self._build_database(staging_path, books)
        except Exception:
            staging_path.unlink(missing_ok=True)
            raise

        os.replace(staging_path, self._database_path)
        logger.info(
            "catalogue_replaced",
            books=len(books),
            categories=len({book.category for book in books}),
            path=str(self._database_path),
        )

    def _build_database(self, path: Path, books: Sequence[Book]) -> None:
        """Create a complete database at the given path."""
        with connect(path) as connection:
            connection.execute(CREATE_CATEGORIES)
            connection.execute(CREATE_BOOKS)
            connection.execute(CREATE_BOOKS_CATEGORY_INDEX)

            category_ids = self._insert_categories(connection, books)
            self._insert_books(connection, books, category_ids)
            connection.commit()

    def _insert_categories(
        self, connection: sqlite3.Connection, books: Sequence[Book]
    ) -> dict[str, int]:
        names = sorted({book.category for book in books})
        connection.executemany(
            "INSERT INTO categories (category_name) VALUES (?)",
            [(name,) for name in names],
        )
        rows = connection.execute("SELECT category_name, category_id FROM categories").fetchall()
        return {name: category_id for name, category_id in rows}

    def _insert_books(
        self,
        connection: sqlite3.Connection,
        books: Sequence[Book],
        category_ids: dict[str, int],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO books (
                title, price_gbp_pence, price_inr_paise, rating, in_stock, category_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    book.title,
                    to_minor_units(book.price_gbp),
                    to_minor_units(book.price_inr),
                    book.rating,
                    int(book.in_stock),
                    category_ids[book.category],
                )
                for book in books
            ],
        )

    def fetch_all(self) -> list[Book]:
        """Read every book back, reconstructing exact Decimal prices."""
        with connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT b.title, b.price_gbp_pence, b.price_inr_paise,
                       b.rating, b.in_stock, c.category_name
                FROM books b
                JOIN categories c ON b.category_id = c.category_id
                ORDER BY b.book_id
                """
            ).fetchall()

        return [
            Book(
                title=title,
                price_gbp=from_minor_units(price_gbp_pence),
                price_inr=from_minor_units(price_inr_paise),
                rating=rating,
                in_stock=bool(in_stock),
                category=category_name,
            )
            for title, price_gbp_pence, price_inr_paise, rating, in_stock, category_name in rows
        ]

    def count_books(self) -> int:
        with connect(self._database_path) as connection:
            (count,) = connection.execute("SELECT COUNT(*) FROM books").fetchone()
        return int(count)

    def count_categories(self) -> int:
        with connect(self._database_path) as connection:
            (count,) = connection.execute("SELECT COUNT(*) FROM categories").fetchone()
        return int(count)
