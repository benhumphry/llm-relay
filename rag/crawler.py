"""
Website crawler for indexing web content into document stores.

Provides two crawler implementations:
- WebsiteCrawler: Built-in trafilatura-based crawler (free, runs locally)
- JinaCrawler: Jina Reader API crawler (handles JS, bot protection, different IP)

Both support:
- Crawl depth control (how many levels of links to follow)
- Max pages limit
- URL include/exclude patterns
- Same-domain restriction
"""

import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from trafilatura.settings import use_config

logger = logging.getLogger(__name__)

# Browser-like User-Agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Common browser headers
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


@dataclass
class CrawledPage:
    """A crawled web page."""

    url: str
    title: str
    content: str
    depth: int
    links: list[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class CrawlResult:
    """Result of a website crawl."""

    pages: list[CrawledPage]
    start_url: str
    pages_crawled: int
    pages_failed: int
    urls_discovered: int


class WebsiteCrawler:
    """
    Website crawler for indexing web content.

    Crawls a website starting from a URL, following links up to a specified depth,
    and extracts text content using trafilatura.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        min_delay: float = 1.0,
        max_delay: float = 3.0,
    ):
        """
        Initialize the crawler.

        Args:
            timeout: Request timeout in seconds
            min_delay: Minimum seconds to wait between requests
            max_delay: Maximum seconds to wait between requests
        """
        self.timeout = timeout
        self.min_delay = min_delay
        self.max_delay = max_delay

        # Configure trafilatura for better extraction
        self.traf_config = use_config()
        self.traf_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")

    def _get_headers(self) -> dict:
        """Get randomized browser-like headers."""
        headers = BROWSER_HEADERS.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    def _random_delay(self):
        """Wait a random amount of time between requests."""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def crawl(
        self,
        start_url: str,
        max_depth: int = 1,
        max_pages: int = 50,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        same_domain_only: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> CrawlResult:
        """
        Crawl a website starting from the given URL.

        Args:
            start_url: Starting URL for the crawl
            max_depth: Maximum depth of links to follow (0 = start page only)
            max_pages: Maximum number of pages to crawl
            include_pattern: Regex pattern - only crawl URLs matching this
            exclude_pattern: Regex pattern - skip URLs matching this
            same_domain_only: Only follow links to the same domain
            progress_callback: Called with (pages_crawled, pages_found) for progress updates

        Returns:
            CrawlResult with all crawled pages
        """
        # Normalize start URL
        start_url = self._normalize_url(start_url)
        start_domain = urlparse(start_url).netloc

        # Compile patterns
        include_re = re.compile(include_pattern) if include_pattern else None
        exclude_re = re.compile(exclude_pattern) if exclude_pattern else None

        # Track state
        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)
        pages: list[CrawledPage] = []
        urls_discovered = 0
        pages_failed = 0

        while to_visit and len(pages) < max_pages:
            url, depth = to_visit.pop(0)

            # Skip if already visited
            if url in visited:
                continue
            visited.add(url)

            # Check URL patterns
            if include_re and not include_re.search(url):
                logger.debug(f"Skipping {url} - doesn't match include pattern")
                continue
            if exclude_re and exclude_re.search(url):
                logger.debug(f"Skipping {url} - matches exclude pattern")
                continue

            # Crawl the page
            page = self._crawl_page(url, depth)

            if page.success:
                pages.append(page)
                logger.info(f"Crawled ({len(pages)}/{max_pages}): {url}")

                # Add discovered links to queue if not at max depth
                if depth < max_depth:
                    for link in page.links:
                        link_domain = urlparse(link).netloc

                        # Check same-domain restriction
                        if same_domain_only and link_domain != start_domain:
                            continue

                        if link not in visited:
                            to_visit.append((link, depth + 1))
                            urls_discovered += 1
            else:
                pages_failed += 1
                logger.warning(f"Failed to crawl: {url} - {page.error}")

            # Progress callback
            if progress_callback:
                progress_callback(len(pages), len(to_visit) + len(pages))

            # Rate limiting with random delay to avoid bot detection
            if to_visit:
                self._random_delay()

        return CrawlResult(
            pages=pages,
            start_url=start_url,
            pages_crawled=len(pages),
            pages_failed=pages_failed,
            urls_discovered=urls_discovered,
        )

    def _crawl_page(self, url: str, depth: int) -> CrawledPage:
        """
        Crawl a single page and extract content.

        Args:
            url: URL to crawl
            depth: Current crawl depth

        Returns:
            CrawledPage with extracted content
        """
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers=self._get_headers(),
            ) as client:
                response = client.get(url)
                response.raise_for_status()

                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip()
                )

                # Only handle HTML pages
                if "text/html" not in content_type and content_type:
                    return CrawledPage(
                        url=url,
                        title="",
                        content="",
                        depth=depth,
                        success=False,
                        error=f"Unsupported content type: {content_type}",
                    )

                html = response.text

                # Extract content with trafilatura
                content = trafilatura.extract(
                    html,
                    config=self.traf_config,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_precision=False,
                    favor_recall=True,
                    include_links=False,
                )

                # Extract title
                title = self._extract_title(html)

                # Extract links for further crawling
                links = self._extract_links(html, url)

                if not content:
                    return CrawledPage(
                        url=url,
                        title=title,
                        content="",
                        depth=depth,
                        links=links,
                        success=False,
                        error="No content extracted",
                    )

                return CrawledPage(
                    url=url,
                    title=title,
                    content=content,
                    depth=depth,
                    links=links,
                    success=True,
                )

        except httpx.TimeoutException:
            return CrawledPage(
                url=url,
                title="",
                content="",
                depth=depth,
                success=False,
                error="Request timed out",
            )
        except httpx.HTTPStatusError as e:
            return CrawledPage(
                url=url,
                title="",
                content="",
                depth=depth,
                success=False,
                error=f"HTTP {e.response.status_code}",
            )
        except Exception as e:
            return CrawledPage(
                url=url,
                title="",
                content="",
                depth=depth,
                success=False,
                error=str(e),
            )

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for consistency."""
        # Ensure scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Remove fragment
        parsed = urlparse(url)
        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            url += f"?{parsed.query}"

        # Remove trailing slash for consistency
        if url.endswith("/") and len(parsed.path) > 1:
            url = url[:-1]

        return url

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return unescape(match.group(1).strip())
        return ""

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """
        Extract all links from HTML.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        links = []
        seen = set()

        # Find all href attributes
        for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = match.group(1).strip()

            # Skip non-HTTP links
            if href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)

            # Normalize
            absolute_url = self._normalize_url(absolute_url)

            # Skip duplicates and non-HTTP
            if absolute_url in seen:
                continue
            if not absolute_url.startswith(("http://", "https://")):
                continue

            seen.add(absolute_url)
            links.append(absolute_url)

        return links

    def save_pages_to_temp_dir(self, pages: list[CrawledPage]) -> Path:
        """
        Save crawled pages to a temporary directory as text files.

        This is useful for feeding into the document indexer.

        Args:
            pages: List of crawled pages

        Returns:
            Path to the temporary directory containing the files
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="crawl_"))

        for i, page in enumerate(pages):
            if not page.success or not page.content:
                continue

            # Create filename from URL
            parsed = urlparse(page.url)
            path_part = parsed.path.strip("/").replace("/", "_") or "index"
            filename = f"{i:04d}_{path_part[:50]}.txt"

            # Write content with metadata header
            file_path = temp_dir / filename
            content = f"URL: {page.url}\n"
            if page.title:
                content += f"Title: {page.title}\n"
            content += f"\n{page.content}"

            file_path.write_text(content, encoding="utf-8")

        return temp_dir


class JinaCrawler:
    """
    Website crawler using Jina Reader API.

    Uses Jina's r.jina.ai service to fetch and parse web content.
    Handles JavaScript rendering and bot protection automatically.
    Requests come from Jina's IP addresses, not yours.

    Requires JINA_API_KEY environment variable for best results.
    """

    JINA_URL = "https://r.jina.ai/"

    def __init__(
        self,
        api_key: str | None = "USE_ENV",
        timeout: float = 60.0,
        delay_between_requests: float = 1.0,
    ):
        """
        Initialize the Jina crawler.

        Args:
            api_key: Jina API key. Use "USE_ENV" (default) to read from JINA_API_KEY env var,
                     None for free tier (no API key), or a specific key string.
            timeout: Request timeout in seconds
            delay_between_requests: Seconds to wait between requests
        """
        if api_key == "USE_ENV":
            self.api_key = os.environ.get("JINA_API_KEY")
        else:
            self.api_key = api_key
        self.timeout = timeout
        self.delay = delay_between_requests

        if not self.api_key:
            logger.info("Jina crawler using free tier (no API key)")

    def crawl(
        self,
        start_url: str,
        max_depth: int = 1,
        max_pages: int = 50,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        same_domain_only: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> CrawlResult:
        """
        Crawl a website using Jina Reader API.

        Args:
            start_url: Starting URL for the crawl
            max_depth: Maximum depth of links to follow (0 = start page only)
            max_pages: Maximum number of pages to crawl
            include_pattern: Regex pattern - only crawl URLs matching this
            exclude_pattern: Regex pattern - skip URLs matching this
            same_domain_only: Only follow links to the same domain
            progress_callback: Called with (pages_crawled, pages_found)

        Returns:
            CrawlResult with all crawled pages
        """
        # Normalize start URL
        start_url = self._normalize_url(start_url)
        start_domain = urlparse(start_url).netloc

        # Compile patterns
        include_re = re.compile(include_pattern) if include_pattern else None
        exclude_re = re.compile(exclude_pattern) if exclude_pattern else None

        # Track state
        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)
        pages: list[CrawledPage] = []
        urls_discovered = 0
        pages_failed = 0

        while to_visit and len(pages) < max_pages:
            url, depth = to_visit.pop(0)

            # Skip if already visited
            if url in visited:
                continue
            visited.add(url)

            # Check URL patterns
            if include_re and not include_re.search(url):
                logger.debug(f"Skipping {url} - doesn't match include pattern")
                continue
            if exclude_re and exclude_re.search(url):
                logger.debug(f"Skipping {url} - matches exclude pattern")
                continue

            # Crawl the page with Jina
            # Request links summary if we need to discover more pages
            get_links = depth < max_depth
            page, links = self._crawl_page(url, depth, get_links=get_links)

            if page.success:
                pages.append(page)
                logger.info(f"Crawled ({len(pages)}/{max_pages}): {url}")

                # Add discovered links to queue if not at max depth
                if depth < max_depth and links:
                    for link in links:
                        link_domain = urlparse(link).netloc

                        # Check same-domain restriction
                        if same_domain_only and link_domain != start_domain:
                            continue

                        if link not in visited:
                            to_visit.append((link, depth + 1))
                            urls_discovered += 1
            else:
                pages_failed += 1
                logger.warning(f"Failed to crawl: {url} - {page.error}")

            # Progress callback
            if progress_callback:
                progress_callback(len(pages), len(to_visit) + len(pages))

            # Rate limiting
            if to_visit and self.delay > 0:
                time.sleep(self.delay)

        return CrawlResult(
            pages=pages,
            start_url=start_url,
            pages_crawled=len(pages),
            pages_failed=pages_failed,
            urls_discovered=urls_discovered,
        )

    def _crawl_page(
        self, url: str, depth: int, get_links: bool = False
    ) -> tuple[CrawledPage, list[str]]:
        """
        Crawl a single page using Jina Reader API.

        Args:
            url: URL to crawl
            depth: Current crawl depth
            get_links: Whether to request links summary

        Returns:
            Tuple of (CrawledPage, list of discovered links)
        """
        links = []

        try:
            headers = {
                "Accept": "text/markdown",
                "X-Return-Format": "markdown",
            }

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Request links summary for link discovery
            if get_links:
                headers["X-With-Links-Summary"] = "all"

            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.JINA_URL}{url}",
                    headers=headers,
                )
                response.raise_for_status()

                content = response.text

                # Extract title from first H1
                title = ""
                if content.startswith("# "):
                    first_newline = content.find("\n")
                    if first_newline > 0:
                        title = content[2:first_newline].strip()

                # Extract links from the links summary section if present
                if get_links:
                    links = self._extract_links_from_summary(content, url)
                    # Remove the links summary section from content
                    content = self._remove_links_summary(content)

                return (
                    CrawledPage(
                        url=url,
                        title=title,
                        content=content,
                        depth=depth,
                        links=links,
                        success=True,
                    ),
                    links,
                )

        except httpx.TimeoutException:
            return (
                CrawledPage(
                    url=url,
                    title="",
                    content="",
                    depth=depth,
                    success=False,
                    error="Request timed out",
                ),
                [],
            )
        except httpx.HTTPStatusError as e:
            return (
                CrawledPage(
                    url=url,
                    title="",
                    content="",
                    depth=depth,
                    success=False,
                    error=f"HTTP {e.response.status_code}",
                ),
                [],
            )
        except Exception as e:
            return (
                CrawledPage(
                    url=url,
                    title="",
                    content="",
                    depth=depth,
                    success=False,
                    error=str(e),
                ),
                [],
            )

    def _extract_links_from_summary(self, content: str, base_url: str) -> list[str]:
        """
        Extract links from Jina's links summary section.

        Jina adds a section like:
        ## Links/Buttons
        - [Link Text](url)
        - [Another Link](url2)
        """
        links = []
        seen = set()

        # Find the links section (usually at the end)
        links_section_patterns = [
            r"## Links/Buttons\n(.*?)(?=\n## |\Z)",
            r"## Links\n(.*?)(?=\n## |\Z)",
            r"### Links\n(.*?)(?=\n### |\n## |\Z)",
        ]

        section_content = ""
        for pattern in links_section_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1)
                break

        if not section_content:
            # Try to find markdown links anywhere in content
            section_content = content

        # Extract markdown links [text](url)
        for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", section_content):
            url = match.group(2).strip()

            # Skip anchors and non-http
            if url.startswith("#") or url.startswith("javascript:"):
                continue
            if url.startswith("mailto:") or url.startswith("tel:"):
                continue

            # Resolve relative URLs
            if not url.startswith(("http://", "https://")):
                url = urljoin(base_url, url)

            # Normalize
            url = self._normalize_url(url)

            if url not in seen and url.startswith(("http://", "https://")):
                seen.add(url)
                links.append(url)

        return links

    def _remove_links_summary(self, content: str) -> str:
        """Remove the links summary section from content."""
        # Remove common link summary sections
        patterns = [
            r"\n## Links/Buttons\n.*$",
            r"\n## Links\n.*$",
            r"\n### Links\n.*$",
        ]

        for pattern in patterns:
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

        return content.strip()

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for consistency."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            url += f"?{parsed.query}"

        if url.endswith("/") and len(parsed.path) > 1:
            url = url[:-1]

        return url


def get_crawler(provider: str = "builtin") -> WebsiteCrawler | JinaCrawler:
    """
    Factory function to get a crawler instance.

    Args:
        provider: "builtin" for trafilatura crawler,
                  "jina" for Jina Reader API (free tier, no API key),
                  "jina-api" for Jina Reader API (with API key)

    Returns:
        Crawler instance
    """
    if provider == "jina":
        # Free tier - explicitly pass None to not use API key
        return JinaCrawler(api_key=None)
    if provider == "jina-api":
        # Use API key from environment
        return JinaCrawler()
    return WebsiteCrawler()
