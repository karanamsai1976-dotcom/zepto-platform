"""Tests for ingestion data contracts: what they accept and what they refuse."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from zepto.ingestion.models import RATING_WORDS, Book, ScrapedBook


def _valid_book(**overrides: object) -> Book:
    defaults: dict[str, object] = {
        "title": "Sharp Objects",
        "price_gbp": Decimal("47.82"),
        "price_inr": Decimal("5045.01"),
        "rating": 4,
        "in_stock": True,
        "category": "Mystery",
    }
    defaults.update(overrides)
    return Book(**defaults)  # type: ignore[arg-type]


def test_valid_book_is_accepted() -> None:
    book = _valid_book()

    assert book.title == "Sharp Objects"
    assert book.rating == 4
    assert book.in_stock is True


def test_prices_stay_decimal() -> None:
    """Prices must not be silently coerced to float."""
    book = _valid_book()

    assert isinstance(book.price_gbp, Decimal)
    assert book.price_gbp == Decimal("47.82")


@pytest.mark.parametrize("rating", [0, 6, -1, 100])
def test_rating_outside_one_to_five_is_rejected(rating: int) -> None:
    """A rating outside 1-5 means parsing failed and produced a default."""
    with pytest.raises(ValidationError):
        _valid_book(rating=rating)


@pytest.mark.parametrize("price", [Decimal("0"), Decimal("-1.00")])
def test_non_positive_price_is_rejected(price: Decimal) -> None:
    with pytest.raises(ValidationError):
        _valid_book(price_gbp=price)


def test_empty_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _valid_book(title="")


def test_book_is_immutable() -> None:
    """A validated record must stay valid; mutation would bypass validation."""
    book = _valid_book()

    with pytest.raises(ValidationError):
        book.rating = 99


def test_scraped_book_requires_every_field() -> None:
    with pytest.raises(ValidationError):
        ScrapedBook(title="Sharp Objects", price_text="£47.82")  # type: ignore[call-arg]


def test_scraped_book_keeps_raw_text_uninterpreted() -> None:
    """The scraper extracts; it does not interpret."""
    scraped = ScrapedBook(
        title="Sharp Objects",
        price_text="£47.82",
        rating_word="Four",
        availability_text="In stock",
        category="Mystery",
    )

    assert scraped.price_text == "£47.82"
    assert scraped.rating_word == "Four"


def test_rating_words_cover_one_to_five() -> None:
    assert sorted(RATING_WORDS.values()) == [1, 2, 3, 4, 5]
