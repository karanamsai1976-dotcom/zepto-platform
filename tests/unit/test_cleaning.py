"""Tests for parsing rules and the scraped-to-validated conversion."""

from decimal import Decimal

import pytest

from zepto.ingestion.cleaning import (
    clean_book,
    convert_to_inr,
    parse_availability,
    parse_price,
    parse_rating,
)
from zepto.ingestion.models import ScrapedBook

RATE = Decimal("105.50")


def _scraped(**overrides: str) -> ScrapedBook:
    defaults = {
        "title": "Sharp Objects",
        "price_text": "£47.82",
        "rating_word": "Four",
        "availability_text": "In stock",
        "category": "Mystery",
    }
    defaults.update(overrides)
    return ScrapedBook(**defaults)


def test_parse_price_extracts_decimal() -> None:
    assert parse_price("£47.82") == Decimal("47.82")


def test_parse_price_returns_decimal_not_float() -> None:
    result = parse_price("£47.82")

    assert isinstance(result, Decimal)


def test_parse_price_returns_none_when_no_number_present() -> None:
    assert parse_price("price on request") is None


@pytest.mark.parametrize(
    ("word", "expected"),
    [("One", 1), ("Two", 2), ("Three", 3), ("Four", 4), ("Five", 5)],
)
def test_parse_rating_maps_known_words(word: str, expected: int) -> None:
    assert parse_rating(word) == expected


@pytest.mark.parametrize("word", ["Zero", "Six", "", "four"])
def test_parse_rating_rejects_unknown_words(word: str) -> None:
    assert parse_rating(word) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("In stock", True),
        ("In stock (19 available)", True),
        ("in stock", True),
        ("Out of stock", False),
        ("", False),
    ],
)
def test_parse_availability(text: str, expected: bool) -> None:
    assert parse_availability(text) is expected


def test_conversion_rounds_half_away_from_zero_on_exact_ties() -> None:
    """The v1 defect, fixed by construction.

    11.75 * 105.50 is exactly 1239.625. Python's round() gave 1239.62
    (banker's rounding), SQLite's ROUND() gave 1239.63, and reconciling them
    required a forensic investigation. Decimal with an explicit rounding mode
    makes the answer unambiguous.
    """
    assert convert_to_inr(Decimal("11.75"), RATE) == Decimal("1239.63")


@pytest.mark.parametrize(
    ("gbp", "expected"),
    [
        (Decimal("51.75"), Decimal("5459.63")),
        (Decimal("48.75"), Decimal("5143.13")),
        (Decimal("14.75"), Decimal("1556.13")),
    ],
)
def test_conversion_is_deterministic_for_other_tie_cases(gbp: Decimal, expected: Decimal) -> None:
    assert convert_to_inr(gbp, RATE) == expected


def test_conversion_result_has_two_decimal_places() -> None:
    result = convert_to_inr(Decimal("10.00"), RATE)

    assert result == Decimal("1055.00")
    assert result.as_tuple().exponent == -2


def test_clean_book_produces_validated_record() -> None:
    book = clean_book(_scraped(), RATE)

    assert book is not None
    assert book.title == "Sharp Objects"
    assert book.price_gbp == Decimal("47.82")
    assert book.price_inr == convert_to_inr(Decimal("47.82"), RATE)
    assert book.rating == 4
    assert book.in_stock is True


def test_clean_book_rejects_unparseable_price() -> None:
    assert clean_book(_scraped(price_text="price on request"), RATE) is None


def test_clean_book_rejects_unknown_rating() -> None:
    assert clean_book(_scraped(rating_word="Zero"), RATE) is None


def test_clean_book_rejects_zero_price_via_model_validation() -> None:
    """A zero price passes the regex but fails the Book contract's gt=0 rule."""
    assert clean_book(_scraped(price_text="£0.00"), RATE) is None


def test_clean_book_never_raises_on_bad_input() -> None:
    """One malformed listing must not abort a run of hundreds."""
    assert clean_book(_scraped(price_text="", rating_word="???"), RATE) is None
