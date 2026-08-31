"""Interpretation of scraped text into validated records.

Kept separate from scraping so parsing rules can be tested with plain strings,
and so a page-structure failure (the scraper's concern) stays distinguishable
from a value-format failure (this module's concern).
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from pydantic import ValidationError

from zepto.core.logging import get_logger
from zepto.ingestion.models import RATING_WORDS, Book, ScrapedBook

logger = get_logger(__name__)

_PRICE_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def parse_price(price_text: str) -> Decimal | None:
    """Extract a price as Decimal, or None when the text contains no number."""
    match = _PRICE_PATTERN.search(price_text)
    if match is None:
        return None

    try:
        return Decimal(match.group())
    except InvalidOperation:  # pragma: no cover - regex already guarantees digits
        return None


def parse_rating(rating_word: str) -> int | None:
    """Map the site's English rating word to an integer, or None if unrecognised."""
    return RATING_WORDS.get(rating_word)


def parse_availability(availability_text: str) -> bool:
    """Return True when the availability text indicates the item is in stock."""
    return "in stock" in availability_text.lower()


def convert_to_inr(price_gbp: Decimal, rate: Decimal) -> Decimal:
    """Convert GBP to INR at the fixed rate, rounding half away from zero.

    The rounding mode is explicit and applied in Decimal arithmetic. v1 used
    Python's round(), whose banker's rounding disagreed with SQLite's ROUND()
    on exact half-cent ties, producing a discrepancy that had to be
    investigated rather than prevented.
    """
    return (price_gbp * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def clean_book(scraped: ScrapedBook, rate: Decimal) -> Book | None:
    """Convert a ScrapedBook into a validated Book, or None if it cannot be.

    Returns None rather than raising so that a single malformed listing cannot
    abort a run spanning hundreds of pages. Every rejection is logged with its
    reason and the offending raw text, so drops are visible rather than silent.
    """
    price_gbp = parse_price(scraped.price_text)
    rating = parse_rating(scraped.rating_word)

    if price_gbp is None or rating is None:
        logger.warning(
            "book_rejected",
            title=scraped.title,
            reason="unparseable_price" if price_gbp is None else "unrecognised_rating",
            price_text=scraped.price_text,
            rating_word=scraped.rating_word,
        )
        return None

    try:
        return Book(
            title=scraped.title,
            price_gbp=price_gbp,
            price_inr=convert_to_inr(price_gbp, rate),
            rating=rating,
            in_stock=parse_availability(scraped.availability_text),
            category=scraped.category,
        )
    except ValidationError as exc:
        logger.warning(
            "book_rejected",
            title=scraped.title,
            reason="failed_validation",
            errors=exc.errors(),
        )
        return None
