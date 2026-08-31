"""Tests for the exception hierarchy: inheritance, context, and message rendering."""

import pytest

from zepto.core.errors import (
    AssistantError,
    FetchError,
    IngestionError,
    ParseError,
    RetrievalError,
    ZeptoError,
)


def test_all_errors_derive_from_zepto_error() -> None:
    """A caller can catch every platform error with one except clause."""
    for error_class in (IngestionError, FetchError, ParseError, AssistantError, RetrievalError):
        assert issubclass(error_class, ZeptoError)


def test_fetch_and_parse_are_distinguishable() -> None:
    """Network failure and structural failure must not be conflated."""
    assert not issubclass(FetchError, ParseError)
    assert not issubclass(ParseError, FetchError)


def test_context_is_preserved_as_data() -> None:
    error = FetchError("page fetch failed", url="http://example.com", status_code=503)

    assert error.message == "page fetch failed"
    assert error.context == {"url": "http://example.com", "status_code": 503}


def test_str_includes_context_when_present() -> None:
    error = FetchError("page fetch failed", url="http://example.com", status_code=503)

    rendered = str(error)
    assert "page fetch failed" in rendered
    assert "status_code=503" in rendered
    assert "url='http://example.com'" in rendered


def test_str_is_plain_message_when_context_is_empty() -> None:
    assert str(ZeptoError("something broke")) == "something broke"


def test_errors_can_be_caught_by_subclass() -> None:
    with pytest.raises(IngestionError):
        raise ParseError("unexpected page structure", selector="article.product_pod")
