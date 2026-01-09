"""
Web scraper for fetching and extracting text content from URLs.

Uses:
- trafilatura for HTML web page extraction
- Docling for static documents (PDF, DOCX, PPTX, etc.)
"""

import logging
import re
import tempfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# Content types that Docling can handle
DOCLING_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/msword": ".doc",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.ms-excel": ".xls",
}

# File extensions that Docling can handle
DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"}


@dataclass
class ScrapeResult:
    """Result of scraping a URL."""

    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None
    content_type: Optional[str] = None


class WebScraper:
    """
    Web scraper for fetching and extracting text content from URLs.

    Uses trafilatura for HTML pages and Docling for static documents.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 10_000_000,  # 10MB for documents
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

        Automatically detects content type and uses the appropriate parser:
        - trafilatura for HTML pages
        - Docling for PDFs, DOCX, PPTX, etc.

        Args:
            url: URL to scrape

        Returns:
            ScrapeResult with extracted content
        """
        try:
            # Check if URL points to a document by extension
            parsed = urlparse(url)
            path_lower = parsed.path.lower()
            extension = Path(path_lower).suffix

            if extension in DOCLING_EXTENSIONS:
                return self._scrape_document(url, extension)

            # Fetch the URL to check content type
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = client.get(url)
                response.raise_for_status()

                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip()
                )

                # Check if it's a document type
                if content_type in DOCLING_CONTENT_TYPES:
                    return self._scrape_document_from_response(
                        url, response.content, content_type
                    )

                # Handle HTML with trafilatura
                if "text/html" in content_type or not content_type:
                    return self._scrape_html(url, response.text)

                # Handle plain text
                if "text/plain" in content_type:
                    return ScrapeResult(
                        url=url,
                        title=Path(parsed.path).name or url,
                        content=response.text[:100000],  # Limit plain text
                        success=True,
                        content_type=content_type,
                    )

                # Unsupported content type
                return ScrapeResult(
                    url=url,
                    title="",
                    content="",
                    success=False,
                    error=f"Unsupported content type: {content_type}",
                    content_type=content_type,
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

    def _scrape_html(self, url: str, html: str) -> ScrapeResult:
        """
        Extract content from HTML using trafilatura.

        Falls back to basic regex extraction if trafilatura fails.
        """
        title = ""
        content = ""

        try:
            import trafilatura

            # Extract with trafilatura
            downloaded = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=False,
                favor_recall=True,
            )

            if downloaded:
                content = downloaded

            # Extract title separately (trafilatura doesn't always get it)
            title = self._extract_title(html)

            if content:
                logger.info(
                    f"Extracted {len(content)} chars from {url} using trafilatura"
                )
                return ScrapeResult(
                    url=url,
                    title=title,
                    content=content,
                    success=True,
                    content_type="text/html",
                )

        except ImportError:
            logger.warning("trafilatura not installed, using fallback HTML extraction")
        except Exception as e:
            logger.warning(f"trafilatura extraction failed for {url}: {e}")

        # Fallback to basic extraction
        title = title or self._extract_title(html)
        content = self._html_to_text_fallback(html)

        logger.info(f"Extracted {len(content)} chars from {url} using fallback")
        return ScrapeResult(
            url=url,
            title=title,
            content=content,
            success=True,
            content_type="text/html",
        )

    def _scrape_document(self, url: str, extension: str) -> ScrapeResult:
        """
        Fetch and parse a document URL using Docling.
        """
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = client.get(url)
                response.raise_for_status()

                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip()
                )
                return self._scrape_document_from_response(
                    url,
                    response.content,
                    content_type or f"application/{extension[1:]}",
                )

        except Exception as e:
            logger.warning(f"Error fetching document {url}: {e}")
            return ScrapeResult(
                url=url,
                title="",
                content="",
                success=False,
                error=str(e),
            )

    def _scrape_document_from_response(
        self, url: str, content: bytes, content_type: str
    ) -> ScrapeResult:
        """
        Parse document content using Docling.

        Uses configurable vision model from settings for document parsing.
        """
        try:
            from rag.vision import (
                get_document_converter,
                get_vision_config_from_settings,
            )

            # Get file extension for content type
            extension = DOCLING_CONTENT_TYPES.get(content_type, ".pdf")

            # Write to temp file (Docling needs a file path)
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
                f.write(content)
                temp_path = Path(f.name)

            try:
                # Get vision config from settings and create converter
                vision_config = get_vision_config_from_settings()
                converter = get_document_converter(
                    vision_provider=vision_config.provider_type,
                    vision_model=vision_config.model_name,
                    vision_ollama_url=vision_config.base_url,
                )

                logger.debug(
                    f"Using vision provider '{vision_config.provider_type}' "
                    f"model '{vision_config.model_name}' for document parsing"
                )

                result = converter.convert(temp_path)

                # Extract text
                text = result.document.export_to_markdown()

                # Try to get title from metadata or filename
                title = Path(urlparse(url).path).stem or url

                logger.info(
                    f"Extracted {len(text)} chars from document {url} using Docling"
                )
                return ScrapeResult(
                    url=url,
                    title=title,
                    content=text,
                    success=True,
                    content_type=content_type,
                )

            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except ImportError as e:
            logger.warning(f"Docling not installed, cannot parse document: {e}")
            return ScrapeResult(
                url=url,
                title="",
                content="",
                success=False,
                error="Docling not installed for document parsing",
                content_type=content_type,
            )
        except Exception as e:
            logger.warning(f"Docling extraction failed for {url}: {e}")
            return ScrapeResult(
                url=url,
                title="",
                content="",
                success=False,
                error=f"Document parsing failed: {e}",
                content_type=content_type,
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

    def _html_to_text_fallback(self, html: str) -> str:
        """
        Fallback HTML to text conversion using regex.

        Used when trafilatura is not available or fails.
        """
        # Remove script and style elements entirely
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.IGNORECASE | re.DOTALL
        )

        # Remove nav, header, footer, aside elements
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
            line = " ".join(line.split())
            if line:
                lines.append(line)

        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
