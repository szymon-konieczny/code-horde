"""Web research toolkit for agents — search, fetch and extract web content.

Provides a lightweight ``httpx``-based fetcher with automatic fallback to
headless Playwright when a page requires JavaScript rendering.

Usage::

    wr = WebResearcher()
    results = await wr.search("FastAPI websocket tutorial", max_results=5)
    page    = await wr.fetch_page("https://docs.python.org/3/library/asyncio.html")
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ── Lightweight HTML → text extraction ────────────────────────────────

_TAG_RE = re.compile(r"<script[^>]*>.*?</script>", re.S | re.I)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Strip HTML tags, scripts and styles — return readable plain text."""
    text = _TAG_RE.sub("", html)
    text = _STYLE_RE.sub("", text)
    text = _HTML_RE.sub("\n", text)
    text = _WS_RE.sub("\n\n", text)
    return text.strip()


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Single search engine result."""
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class PageContent:
    """Fetched web page content."""
    url: str
    title: str = ""
    text: str = ""
    html: str = ""
    status_code: int = 0
    error: Optional[str] = None
    used_playwright: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text[:8000],
            "status_code": self.status_code,
            "error": self.error,
            "used_playwright": self.used_playwright,
        }


# ── WebResearcher ─────────────────────────────────────────────────────

# Common headers to avoid bot blocks
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
}


class WebResearcher:
    """Fetch and search the web for agent use.

    Args:
        timeout: HTTP timeout in seconds.
        max_text_length: Truncate extracted text to this length.
        playwright_fallback: Try headless Playwright when httpx fails or
            the page looks JS-heavy.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        max_text_length: int = 12_000,
        playwright_fallback: bool = True,
    ) -> None:
        self._timeout = timeout
        self._max_text = max_text_length
        self._pw_fallback = playwright_fallback

    # ── Search ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Search the web using DuckDuckGo HTML (no API key needed).

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of ``SearchResult`` objects.
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers=_HEADERS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("web_search_failed", query=query[:100], error=str(exc)[:200])
            return []

        return self._parse_ddg_results(resp.text, max_results)

    @staticmethod
    def _parse_ddg_results(html: str, limit: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML search results page."""
        results: list[SearchResult] = []

        # Each result lives in <a class="result__a" href="...">title</a>
        # with snippet in <a class="result__snippet">...</a>
        link_re = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            re.S,
        )
        snippet_re = re.compile(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.S,
        )

        links = link_re.findall(html)
        snippets = snippet_re.findall(html)

        for i, (href, title_html) in enumerate(links[:limit]):
            title = _HTML_RE.sub("", title_html).strip()
            snippet = _HTML_RE.sub("", snippets[i]).strip() if i < len(snippets) else ""
            # DDG wraps URLs in a redirect; extract the actual URL
            actual_url = href
            uddg_match = re.search(r"uddg=([^&]+)", href)
            if uddg_match:
                from urllib.parse import unquote
                actual_url = unquote(uddg_match.group(1))
            results.append(SearchResult(title=title, url=actual_url, snippet=snippet))

        return results

    # ── Fetch page ────────────────────────────────────────────────

    async def fetch_page(self, url: str) -> PageContent:
        """Fetch a web page and extract readable text.

        Tries httpx first (fast, lightweight). Falls back to Playwright
        headless browser when the page looks JS-heavy or httpx fails.

        Args:
            url: Full URL to fetch.

        Returns:
            ``PageContent`` with extracted text, title, status, etc.
        """
        page = await self._fetch_httpx(url)

        # Decide whether to retry with Playwright
        needs_js = (
            self._pw_fallback
            and page.error is None
            and self._looks_js_heavy(page)
        )
        httpx_failed = page.error is not None and self._pw_fallback

        if needs_js or httpx_failed:
            pw_page = await self._fetch_playwright(url)
            if pw_page.error is None and len(pw_page.text) > len(page.text):
                return pw_page
            # If Playwright also failed, return httpx result
            if page.error is not None and pw_page.error is not None:
                return page  # return original error

        return page

    # ── Multi-page research ───────────────────────────────────────

    async def research(
        self,
        query: str,
        max_pages: int = 3,
    ) -> dict[str, Any]:
        """Search and fetch top results — one-shot web research.

        Args:
            query: Search query.
            max_pages: How many search results to fetch in full.

        Returns:
            Dict with ``query``, ``results`` (search hits), and ``pages``
            (full content for top results).
        """
        results = await self.search(query, max_results=max_pages + 2)

        # Fetch top pages in parallel
        pages: list[PageContent] = []
        if results:
            fetch_tasks = [
                self.fetch_page(r.url) for r in results[:max_pages]
            ]
            pages = await asyncio.gather(*fetch_tasks, return_exceptions=False)

        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "pages": [p.to_dict() for p in pages if isinstance(p, PageContent)],
        }

    # ── Internal: httpx fetch ─────────────────────────────────────

    async def _fetch_httpx(self, url: str) -> PageContent:
        """Fetch with httpx (no JS rendering)."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers=_HEADERS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                html = resp.text
                text = _html_to_text(html)[:self._max_text]
                title = self._extract_title(html)
                return PageContent(
                    url=str(resp.url),
                    title=title,
                    text=text,
                    html=html[:50_000],
                    status_code=resp.status_code,
                )
        except Exception as exc:
            return PageContent(
                url=url,
                error=f"httpx: {exc}",
            )

    # ── Internal: Playwright fetch ────────────────────────────────

    async def _fetch_playwright(self, url: str) -> PageContent:
        """Fetch with headless Chromium (full JS rendering)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return PageContent(
                url=url,
                error="playwright not installed",
            )

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                # Wait a moment for JS to render
                await page.wait_for_timeout(1500)
                title = await page.title()
                html = await page.content()
                text = _html_to_text(html)[:self._max_text]
                await browser.close()
                return PageContent(
                    url=url,
                    title=title,
                    text=text,
                    html=html[:50_000],
                    status_code=200,
                    used_playwright=True,
                )
        except Exception as exc:
            logger.warning("playwright_fetch_failed", url=url[:200], error=str(exc)[:200])
            return PageContent(
                url=url,
                error=f"playwright: {exc}",
                used_playwright=True,
            )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        return _HTML_RE.sub("", m.group(1)).strip() if m else ""

    @staticmethod
    def _looks_js_heavy(page: PageContent) -> bool:
        """Heuristic: page is likely a JS-rendered SPA with little content."""
        if len(page.text) < 200:
            return True
        js_indicators = [
            "window.__NEXT_DATA__",
            "window.__NUXT__",
            "<noscript>",
            "id=\"__next\"",
            "id=\"app\"",
            "ng-app",
            "React.createElement",
        ]
        return any(ind in page.html for ind in js_indicators)
