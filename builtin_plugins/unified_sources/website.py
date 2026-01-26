"""
Website Crawler Unified Source Plugin.

A RAG-only unified source for crawling and indexing websites.
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

import httpx

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import PluginUnifiedSource, QueryAnalysis, QueryRouting

logger = logging.getLogger(__name__)


class WebsiteUnifiedSource(PluginUnifiedSource):
    """Website crawler source - RAG-only for indexed web content."""

    source_type = "website"
    display_name = "Website"
    description = "Crawl and index web pages for semantic search"
    category = "web"
    icon = "🌐"
    content_category = ContentCategory.WEBSITES

    # Document store types this unified source handles
    handles_doc_source_types = ["website"]

    supports_rag = True
    supports_live = False
    supports_actions = False
    supports_incremental = False
    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "url": store.website_url or "",
            "max_pages": store.website_max_pages or 50,
            "max_depth": store.website_crawl_depth or 1,
            "include_pattern": store.website_include_pattern or "",
            "exclude_pattern": store.website_exclude_pattern or "",
        }

    DEFAULT_EXCLUDE = [
        r"/login",
        r"/logout",
        r"/auth",
        r"/admin",
        r"/cart",
        r"/checkout",
        r"\?",
        r"#",
        r"\.pdf$",
        r"\.zip$",
    ]

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="url",
                label="Start URL",
                field_type=FieldType.TEXT,
                required=True,
                help_text="URL to start crawling from",
            ),
            FieldDefinition(
                name="max_pages",
                label="Max Pages",
                field_type=FieldType.INTEGER,
                default=100,
                help_text="Maximum number of pages to crawl",
            ),
            FieldDefinition(
                name="max_depth",
                label="Max Depth",
                field_type=FieldType.INTEGER,
                default=3,
                help_text="Maximum link depth to follow",
            ),
            FieldDefinition(
                name="stay_on_domain",
                label="Stay on Domain",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Only crawl pages on the same domain",
            ),
            FieldDefinition(
                name="exclude_patterns",
                label="Exclude URL Patterns",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated URL patterns to exclude (regex)",
            ),
            FieldDefinition(
                name="crawl_delay",
                label="Crawl Delay (ms)",
                field_type=FieldType.INTEGER,
                default=500,
                help_text="Delay between requests in milliseconds",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 0 * * *", "label": "Daily"},
                    {"value": "0 0 * * 0", "label": "Weekly"},
                ],
                help_text="How often to re-crawl",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        return []

    def __init__(self, config: dict):
        self.url = config.get("url", "").rstrip("/")
        self.max_pages = min(config.get("max_pages", 100), 1000)
        self.max_depth = min(config.get("max_depth", 3), 10)
        self.stay_on_domain = config.get("stay_on_domain", True)
        self.crawl_delay = config.get("crawl_delay", 500) / 1000.0
        self.index_schedule = config.get("index_schedule", "")
        exclude_str = config.get("exclude_patterns", "")
        self.exclude_patterns = (
            [re.compile(p.strip()) for p in exclude_str.split(",") if p.strip()]
            if exclude_str
            else [re.compile(p) for p in self.DEFAULT_EXCLUDE]
        )
        parsed = urlparse(self.url)
        self.base_domain = parsed.netloc
        self.base_scheme = parsed.scheme or "https"
        self._client = httpx.Client(timeout=30, follow_redirects=True)
        self._crawled: set[str] = set()

    def _should_crawl(self, url: str) -> bool:
        parsed = urlparse(url)
        if self.stay_on_domain and parsed.netloc != self.base_domain:
            return False
        if any(p.search(url) for p in self.exclude_patterns):
            return False
        return True

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        links = []
        for match in re.finditer(r'href=["\'](.*?)["\']', html, re.IGNORECASE):
            href = match.group(1)
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            if normalized not in self._crawled:
                links.append(normalized)
        return links

    def _extract_text(self, html: str) -> str:
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<(br|p|div|tr|li|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE
        )
        html = re.sub(r"<[^>]+>", "", html)
        import html as html_module

        html = html_module.unescape(html)
        html = re.sub(r"\n\s*\n", "\n\n", html)
        return html.strip()

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            import html as html_module

            return html_module.unescape(match.group(1).strip())
        return ""

    def list_documents(self) -> Iterator[DocumentInfo]:
        logger.info(f"Starting crawl from: {self.url}")
        self._crawled.clear()
        queue: list[tuple[str, int]] = [(self.url, 0)]
        pages_found = 0
        while queue and pages_found < self.max_pages:
            url, depth = queue.pop(0)
            normalized = url.rstrip("/")
            if normalized in self._crawled or not self._should_crawl(normalized):
                continue
            self._crawled.add(normalized)
            try:
                time.sleep(self.crawl_delay)
                response = self._client.get(url)
                content_type = response.headers.get("content-type", "")
                if (
                    "text/html" not in content_type.lower()
                    or response.status_code != 200
                ):
                    continue
                html = response.text
                title = self._extract_title(html) or url
                pages_found += 1
                yield DocumentInfo(
                    uri=f"website://{normalized}",
                    title=title[:100],
                    mime_type="text/html",
                    modified_at=datetime.now(timezone.utc).isoformat(),
                    metadata={"url": normalized, "depth": depth},
                )
                if depth < self.max_depth:
                    for link in self._extract_links(html, url):
                        if link not in self._crawled:
                            queue.append((link, depth + 1))
            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")
        logger.info(f"Crawl complete: found {pages_found} pages")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        if not uri.startswith("website://"):
            return None
        url = uri.replace("website://", "")
        try:
            response = self._client.get(url)
            if response.status_code != 200:
                return None
            html = response.text
            title = self._extract_title(html) or url
            text = self._extract_text(html)
            content = f"# {title}\n\nURL: {url}\n\n{text}"
            return DocumentContent(
                content=content,
                mime_type="text/plain",
                metadata={"url": url, "title": title, "source_type": "webpage"},
            )
        except Exception as e:
            logger.error(f"Failed to read page {url}: {e}")
            return None

    def fetch(self, params: dict) -> LiveDataResult:
        return LiveDataResult(
            success=False,
            error="Website source does not support live queries - use RAG search instead",
        )

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="Website source - RAG only",
            max_rag_results=20,
        )

    def is_available(self) -> bool:
        try:
            response = self._client.head(self.url, timeout=10)
            return response.status_code < 500
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        try:
            response = self._client.get(self.url, timeout=10)
            results = [f"URL: {self.url}", f"Status: {response.status_code}"]
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type.lower():
                    title = self._extract_title(response.text)
                    if title:
                        results.append(f"Title: {title}")
                    links = self._extract_links(response.text, self.url)
                    results.append(f"Links found: {len(links)}")
            return response.status_code == 200, "\n".join(results)
        except Exception as e:
            return False, f"Connection failed: {e}"

    def close(self) -> None:
        if self._client:
            self._client.close()
