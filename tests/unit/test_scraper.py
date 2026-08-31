"""Tests for fetching and parsing, with no network access.

A fake session returns real requests.Response objects, so raise_for_status and
encoding behave exactly as they would in production.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests
from requests.adapters import HTTPAdapter

from zepto.core.errors import FetchError, ParseError
from zepto.ingestion.scraper import BookScraper
from zepto.ingestion.settings import IngestionSettings


def _response(text: str = "", status_code: int = 200) -> requests.Response:
    """Build a real Response, so raise_for_status behaves genuinely."""
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode("utf-8")
    response.encoding = "utf-8"
    return response


class FakeSession:
    """Records requests and returns queued responses. Never touches the network."""

    def __init__(self, responses: list[requests.Response] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    def get(self, url: str, *, timeout: float) -> requests.Response:
        self.calls.append((url, {"timeout": timeout}))
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return _response()


def _article(
    title: str = "Sharp Objects",
    price: str = "£47.82",
    rating_class: str = "star-rating Four",
    availability: str = "In stock",
) -> str:
    return f"""
    <article class="product_pod">
      <h3><a href="catalogue/x.html" title="{title}">{title}</a></h3>
      <p class="{rating_class}"></p>
      <p class="price_color">{price}</p>
      <p class="instock availability">{availability}</p>
    </article>
    """


def _page(*articles: str, category: str = "Mystery", next_href: str | None = None) -> str:
    pager = f'<li class="next"><a href="{next_href}">next</a></li>' if next_href else ""
    return f"""
    <html><body>
      <ul class="breadcrumb">
        <li>Home</li><li>Books</li><li class="active">{category}</li>
      </ul>
      {"".join(articles)}
      <ul class="pager">{pager}</ul>
    </body></html>
    """


def _settings(**overrides: Any) -> IngestionSettings:
    defaults: dict[str, Any] = {
        "request_delay_seconds": 0.0,
        "category_slugs": ("mystery_3",),
        "timeout_seconds": 7.5,
    }
    defaults.update(overrides)
    return IngestionSettings(**defaults)


# --- fetch ---


def test_fetch_passes_an_explicit_timeout() -> None:
    """v1 called requests.get with no timeout, so a stalled connection hung forever."""
    session = FakeSession([_response("<html></html>")])
    scraper = BookScraper(settings=_settings(), session=session)

    scraper.fetch("http://example.com/page.html")

    _, kwargs = session.calls[0]
    assert kwargs["timeout"] == 7.5


def test_fetch_raises_on_error_status() -> None:
    """v1 had no raise_for_status, so a 503 error page was parsed as data."""
    session = FakeSession([_response("<html>Service Unavailable</html>", status_code=503)])
    scraper = BookScraper(settings=_settings(), session=session)

    with pytest.raises(FetchError):
        scraper.fetch("http://example.com/page.html")


def test_fetch_raises_on_connection_error() -> None:
    session = FakeSession()
    session.error = requests.ConnectionError("connection refused")
    scraper = BookScraper(settings=_settings(), session=session)

    with pytest.raises(FetchError):
        scraper.fetch("http://example.com/page.html")


def test_fetch_error_carries_the_url_as_context() -> None:
    session = FakeSession([_response("nope", status_code=500)])
    scraper = BookScraper(settings=_settings(), session=session)

    with pytest.raises(FetchError) as exc_info:
        scraper.fetch("http://example.com/page.html")

    assert exc_info.value.context["url"] == "http://example.com/page.html"


def test_fetch_decodes_pound_sign_correctly() -> None:
    session = FakeSession([_response("<p>£47.82</p>")])
    scraper = BookScraper(settings=_settings(), session=session)

    html = scraper.fetch("http://example.com/page.html")

    assert "£47.82" in html
    assert "Â£" not in html


# --- parsing ---


def test_category_name_is_read_from_the_breadcrumb() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())

    assert scraper.parse_category_name(_page(category="Historical Fiction")) == "Historical Fiction"


def test_missing_breadcrumb_raises_parse_error() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())

    with pytest.raises(ParseError):
        scraper.parse_category_name("<html><body>no breadcrumb</body></html>")


def test_listings_are_extracted_with_raw_text_preserved() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())

    books = scraper.parse_listings(_page(_article()), "Mystery")

    assert len(books) == 1
    assert books[0].title == "Sharp Objects"
    assert books[0].price_text == "£47.82"
    assert books[0].rating_word == "Four"
    assert books[0].availability_text == "In stock"
    assert books[0].category == "Mystery"


@pytest.mark.parametrize(
    "broken_article",
    [
        '<article class="product_pod"><h3><a title="X">X</a></h3></article>',
        _article(rating_class="star-rating"),
        _article(title=""),
    ],
    ids=["missing_elements", "no_rating_word", "empty_title"],
)
def test_malformed_listings_are_skipped_not_raised(broken_article: str) -> None:
    """v1 raised AttributeError here, discarding the entire run."""
    scraper = BookScraper(settings=_settings(), session=FakeSession())

    books = scraper.parse_listings(_page(broken_article), "Mystery")

    assert books == []


def test_one_bad_listing_does_not_discard_the_good_ones() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())
    html = _page(
        _article(title="Good One"),
        '<article class="product_pod"></article>',
        _article(title="Good Two"),
    )

    books = scraper.parse_listings(html, "Mystery")

    assert [book.title for book in books] == ["Good One", "Good Two"]


# --- pagination ---


def test_next_page_url_is_resolved_to_an_absolute_url() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())
    html = _page(_article(), next_href="page-2.html")

    next_url = scraper._next_page_url(html, "http://example.com/category/index.html")

    assert next_url == "http://example.com/category/page-2.html"


def test_next_page_url_is_none_on_the_last_page() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())

    assert scraper._next_page_url(_page(_article()), "http://example.com/x.html") is None


def test_scrape_category_follows_pagination_until_exhausted() -> None:
    session = FakeSession(
        [
            _response(_page(_article(title="Page One Book"), next_href="page-2.html")),
            _response(_page(_article(title="Page Two Book"))),
        ]
    )
    scraper = BookScraper(settings=_settings(), session=session)

    books = scraper.scrape_category("mystery_3")

    assert [book.title for book in books] == ["Page One Book", "Page Two Book"]
    assert len(session.calls) == 2


def test_scrape_all_covers_every_configured_category() -> None:
    session = FakeSession(
        [
            _response(_page(_article(title="Mystery Book"), category="Mystery")),
            _response(_page(_article(title="Fiction Book"), category="Fiction")),
        ]
    )
    scraper = BookScraper(
        settings=_settings(category_slugs=("mystery_3", "fiction_10")), session=session
    )

    books = scraper.scrape_all()

    assert len(books) == 2
    assert {book.category for book in books} == {"Mystery", "Fiction"}


# --- session construction ---


def test_built_session_sets_an_honest_user_agent() -> None:
    """The default python-requests agent is blocked by many sites."""
    scraper = BookScraper(settings=_settings())

    session = scraper._build_session()

    assert "zepto-platform" in session.headers["User-Agent"]


def test_built_session_mounts_a_retry_adapter() -> None:
    """v1 had no retries; one transient failure aborted the whole run."""
    scraper = BookScraper(settings=_settings(max_retries=5))

    session = scraper._build_session()
    adapter = session.get_adapter("http://example.com")

    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 5
    assert 503 in adapter.max_retries.status_forcelist


def test_next_page_url_is_none_when_href_is_missing() -> None:
    scraper = BookScraper(settings=_settings(), session=FakeSession())
    html = '<html><body><ul class="pager"><li class="next"><a>next</a></li></ul></body></html>'

    assert scraper._next_page_url(html, "http://example.com/x.html") is None


def test_politeness_delay_is_applied_between_pages_but_not_after_the_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limiting is a guarantee we make, so assert it actually happens --
    once between the two pages, and not redundantly after the final one."""
    sleeps: list[float] = []
    monkeypatch.setattr(
        "zepto.ingestion.scraper.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    session = FakeSession(
        [
            _response(_page(_article(), next_href="page-2.html")),
            _response(_page(_article())),
        ]
    )
    scraper = BookScraper(settings=_settings(request_delay_seconds=0.25), session=session)

    scraper.scrape_category("mystery_3")

    assert sleeps == [0.25]
