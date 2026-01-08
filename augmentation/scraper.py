"""
Web scraper for fetching and extracting text content from URLs.

Uses httpx for fetching and a simple HTML-to-text conversion that
doesn't require external dependencies like BeautifulSoup.
"""

import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result of scraping a URL."""

    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None


class WebScraper:
    """
    Web scraper for fetching and extracting text content from URLs.

    Uses a simple HTML-to-text approach that extracts meaningful content
    while removing scripts, styles, and navigation elements.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 100000,
        user_agent: str = "Mozilla/5.0 (compatible; LLMRelay/1.0; +https://github.com/benhumphry/llm-relay)",
    ):
        """
        Initialize the web scraper.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to fetch (bytes)
            user_agent: User agent string for requests
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent

    def scrape(self, url: str) -> ScrapeResult:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape

        Returns:
            ScrapeResult with extracted content
        """
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = client.get(url)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return ScrapeResult(
                        url=url,
                        title="",
                        content="",
                        success=False,
                        error=f"Unsupported content type: {content_type}",
                    )

                # Check content length
                content = response.text
                if len(content) > self.max_content_length:
                    content = content[: self.max_content_length]

                # Extract title and content
                title = self._extract_title(content)
                text = self._html_to_text(content)

                logger.info(f"Scraped {len(text)} chars from {url}")
                return ScrapeResult(
                    url=url,
                    title=title,
                    content=text,
                    success=True,
                )

        except httpx.TimeoutException:
            logger.warning(f"Timeout scraping {url}")
            return ScrapeResult(
                url=url,
                title="",
                content="",
                success=False,
                error="Request timed out",
            )
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error scraping {url}: {e.response.status_code}")
            return ScrapeResult(
                url=url,
                title="",
                content="",
                success=False,
                error=f"HTTP {e.response.status_code}",
            )
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            return ScrapeResult(
                url=url,
                title="",
                content="",
                success=False,
                error=str(e),
            )

    def scrape_multiple(self, urls: list[str], max_urls: int = 3) -> list[ScrapeResult]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape
            max_urls: Maximum number of URLs to scrape

        Returns:
            List of ScrapeResult objects
        """
        results = []
        for url in urls[:max_urls]:
            result = self.scrape(url)
            results.append(result)
        return results

    def format_results(self, results: list[ScrapeResult], max_chars: int = 4000) -> str:
        """
        Format scrape results for injection into LLM context.

        Args:
            results: List of ScrapeResult objects
            max_chars: Maximum total characters for all results

        Returns:
            Formatted string suitable for context injection
        """
        if not results:
            return "No web content fetched."

        successful = [r for r in results if r.success]
        if not successful:
            return "Failed to fetch any web content."

        lines = ["## Web Content\n"]
        chars_used = 0
        chars_per_result = max_chars // len(successful)

        for result in successful:
            header = f"### {result.title or result.url}\n"
            header += f"Source: {result.url}\n\n"

            # Truncate content if needed
            available = chars_per_result - len(header)
            content = (
                result.content[:available]
                if len(result.content) > available
                else result.content
            )

            lines.append(header + content + "\n")
            chars_used += len(header) + len(content)

            if chars_used >= max_chars:
                break

        return "\n".join(lines)

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return unescape(match.group(1).strip())
        return ""

    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML to plain text.

        This is a simple implementation that:
        1. Removes script and style tags
        2. Removes HTML tags
        3. Cleans up whitespace
        4. Decodes HTML entities
        """
        # Remove script and style elements entirely
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.IGNORECASE | re.DOTALL
        )

        # Remove nav, header, footer, aside elements (often contain non-content)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(
            r"<header[^>]*>.*?</header>", "", html, flags=re.IGNORECASE | re.DOTALL
        )
        html = re.sub(
            r"<footer[^>]*>.*?</footer>", "", html, flags=re.IGNORECASE | re.DOTALL
        )
        html = re.sub(
            r"<aside[^>]*>.*?</aside>", "", html, flags=re.IGNORECASE | re.DOTALL
        )

        # Replace block-level elements with newlines
        html = re.sub(
            r"<(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE
        )

        # Remove all remaining HTML tags
        html = re.sub(r"<[^>]+>", "", html)

        # Decode HTML entities
        html = unescape(html)

        # Clean up whitespace
        lines = []
        for line in html.split("\n"):
            line = " ".join(line.split())  # Normalize whitespace
            if line:
                lines.append(line)

        # Join and limit consecutive newlines
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
