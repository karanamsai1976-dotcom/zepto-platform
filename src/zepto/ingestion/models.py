"""Data contracts for the ingestion pipeline.

v1 passed plain dicts between scraping, cleaning, and storage, so a missing key
or wrong type surfaced layers away from its cause. These models validate at
each boundary: data that cannot satisfy the contract is rejected where it
enters, naming the offending field.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

#: The site encodes star ratings as an English word in a CSS class.
RATING_WORDS: dict[str, int] = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


class ScrapedBook(BaseModel):
    """A book exactly as it appeared on the page, before any interpretation.

    Strict about shape (every field present, every field a string), deliberately
    permissive about content -- interpreting the content is the cleaner's job.
    Retaining the raw text makes parse failures debuggable after the fact.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    price_text: str
    rating_word: str
    availability_text: str
    category: str = Field(min_length=1)


class Book(BaseModel):
    """A validated, typed book record, ready to be stored.

    Prices are Decimal rather than float: binary floating point makes rounding
    non-deterministic across languages, which produced a real cross-checking
    discrepancy in v1.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    price_gbp: Decimal = Field(gt=0, description="Listed price in GBP.")
    price_inr: Decimal = Field(gt=0, description="Converted price at the fixed project rate.")
    rating: int = Field(
        ge=1, le=5, description="Star rating; anything outside 1-5 is a parse failure."
    )
    in_stock: bool
    category: str = Field(min_length=1)
