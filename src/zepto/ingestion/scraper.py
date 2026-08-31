"""HTTP fetching and HTML parsing for the book catalogue.

Addresses the network-layer weaknesses carried by v1: no timeout, no status
check, no retries, no connection reuse, a default User-Agent, and parse errors
that aborted an entire run.

The HTTP session is injected rather than constructed internally, so tests can
supply a fake and never touch the network.
"""

from __future__ import annotations

import time
from typing import Protocol
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter, Retry

from zepto.core.errors import FetchError, ParseError
from zepto.core.logging import get_logger
from zepto.ingestion.models import ScrapedBook
from zepto.ingestion.settings import IngestionSettings, get_ingestion_settings

logger = get_logger(__name__)

#: Transient statuses worth retrying. 4xx other than 429 are not retried:
#: a 404 will still be a 404 on the third attempt.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class HttpSession(Protocol):
    """The subset of requests.Session that BookScraper actually uses.

    Declaring the dependency structurally lets tests inject a fake without
    subclassing Session or monkeypatching module internals. The signature is
    deliberately narrow: it promises only what we actually call.
    """

    def get(self, url: str, *, timeout: float) -> requests.Response: ...


class BookScraper:
    """Scrapes book listings from category pages, following pagination."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        session: HttpSession | None = None,
    ) -> None:
        self._settings = settings or get_ingestion_settings()
        self._session: HttpSession = session if session is not None else self._build_session()

    def _build_session(self) -> requests.Session:
        """Build a session with connection reuse, retries, and an honest User-Agent."""
        session = requests.Session()
        session.headers.update({"User-Agent": self._settings.user_agent})

        retry = Retry(
            total=self._settings.max_retries,
            backoff_factor=self._settings.backoff_factor,
            status_forcelist=RETRYABLE_STATUS,
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch(self, url: str) -> str:
        """Fetch a page's HTML.

        Raises FetchError on any network failure or non-success status, rather
        than returning an error page that would later parse to zero books.
        """
        try:
            response = self._session.get(url, timeout=self._settings.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FetchError("page fetch failed", url=url, cause=str(exc)) from exc

        # Must be set before reading .text, or the pound sign decodes as mojibake.
        response.encoding = "utf-8"
        return response.text

    def parse_category_name(self, html: str) -> str:
        """Read the category's display name from the page's own breadcrumb.

        Preferred over deriving it from the URL slug, which would require
        guessing at capitalisation and word breaks.
        """
        soup = BeautifulSoup(html, "lxml")
        active = soup.select_one("ul.breadcrumb li.active")
        if active is None:
            raise ParseError("category name not found", selector="ul.breadcrumb li.active")
        return str(active.get_text(strip=True))

    def parse_listings(self, html: str, category_name: str) -> list[ScrapedBook]:
        """Extract every well-formed listing on a page.

        Listings whose markup does not match expectations are skipped and
        logged, not raised -- one bad entry must not discard the other
        nineteen on the page.
        """
        soup = BeautifulSoup(html, "lxml")
        books: list[ScrapedBook] = []

        for article in soup.select("article.product_pod"):
            book = self._parse_article(article, category_name)
            if book is not None:
                books.append(book)

        return books

    def _parse_article(self, article: Tag, category_name: str) -> ScrapedBook | None:
        """Extract one listing, or None if the markup is not as expected."""
        title_link = article.select_one("h3 a")
        price = article.select_one("p.price_color")
        rating = article.select_one("p.star-rating")
        availability = article.select_one("p.instock.availability")

        missing = [
            name
            for name, element in (
                ("title", title_link),
                ("price", price),
                ("rating", rating),
                ("availability", availability),
            )
            if element is None
        ]
        if missing or title_link is None or price is None or rating is None or availability is None:
            logger.warning("listing_skipped", reason="missing_elements", missing=missing)
            return None

        # The rating is the second CSS class, e.g. <p class="star-rating Three">.
        rating_classes = rating.get("class")
        if not isinstance(rating_classes, list) or len(rating_classes) < 2:
            logger.warning("listing_skipped", reason="rating_class_missing")
            return None

        title = title_link.get("title")
        if not title:
            logger.warning("listing_skipped", reason="empty_title")
            return None

        return ScrapedBook(
            title=str(title),
            price_text=price.get_text(strip=True),
            rating_word=str(rating_classes[1]),
            availability_text=availability.get_text(strip=True),
            category=category_name,
        )

    def _next_page_url(self, html: str, current_url: str) -> str | None:
        """Resolve the next page's absolute URL, or None on the last page."""
        soup = BeautifulSoup(html, "lxml")
        next_link = soup.select_one("li.next a")
        if next_link is None:
            return None

        href = next_link.get("href")
        if not href:
            return None

        return urljoin(current_url, str(href))

    def scrape_category(self, slug: str) -> list[ScrapedBook]:
        """Scrape every page of one category."""
        url: str | None = self._settings.category_url(slug)
        category_name: str | None = None
        books: list[ScrapedBook] = []
        pages = 0

        while url is not None:
            html = self.fetch(url)
            pages += 1

            if category_name is None:
                category_name = self.parse_category_name(html)

            books.extend(self.parse_listings(html, category_name))
            next_url = self._next_page_url(html, url)

            if next_url is not None and self._settings.request_delay_seconds > 0:
                time.sleep(self._settings.request_delay_seconds)

            url = next_url

        logger.info(
            "category_scraped", slug=slug, category=category_name, pages=pages, books=len(books)
        )
        return books

    def scrape_all(self) -> list[ScrapedBook]:
        """Scrape every configured category."""
        books: list[ScrapedBook] = []
        for slug in self._settings.category_slugs:
            books.extend(self.scrape_category(slug))

        logger.info("scrape_complete", total_books=len(books))
        return books
