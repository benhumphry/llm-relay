"""
Notion Unified Source Plugin.

Combines Notion database/page indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index pages and databases for semantic search
- Live side: Query recent pages, search, database queries
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "recently edited pages" -> Live only (real-time)
- "meeting notes from Q3" -> RAG only (historical)
- "pages about project X" -> Both, merge results
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class NotionUnifiedSource(PluginUnifiedSource):
    """
    Unified Notion source - RAG for pages/databases, Live for recent/search.

    Single configuration provides:
    - Document indexing: Pages and database entries indexed for RAG
    - Live queries: Recent pages, search, database queries
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "notion"
    display_name = "Notion"
    description = (
        "Notion workspace with page/database search (RAG) and real-time queries"
    )
    category = "productivity"
    icon = "📝"
    content_category = ContentCategory.FILES

    # Document store types this unified source handles
    handles_doc_source_types = ["notion"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # Could add page creation later
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results

    _abstract = False

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME Notion access. Use action='list' to enumerate ALL pages/notes "
            "(for 'what notes do I have?' queries), action='search' with query for specific searches, "
            "action='recent' for recently modified pages. For historical content search, RAG will be used automatically."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "api_token": os.environ.get("NOTION_TOKEN")
            or os.environ.get("NOTION_API_KEY", ""),
            "database_ids": store.notion_database_id or "",
            "root_page_id": store.notion_page_id or "",
        }

    # Notion API
    NOTION_API_BASE = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="api_token",
                label="Notion Integration Token",
                field_type=FieldType.PASSWORD,
                required=False,
                help_text="Notion integration token (leave empty to use env var)",
                env_var="NOTION_TOKEN",
            ),
            FieldDefinition(
                name="root_page_id",
                label="Root Page ID",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Root page to index (empty = all accessible pages)",
            ),
            FieldDefinition(
                name="database_ids",
                label="Database IDs",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated database IDs to index",
            ),
            FieldDefinition(
                name="include_databases",
                label="Index Databases",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index database entries",
            ),
            FieldDefinition(
                name="include_pages",
                label="Index Pages",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index regular pages",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                    {"value": "0 0 * * 0", "label": "Weekly"},
                ],
                help_text="How often to re-index",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=20,
                help_text="Maximum items to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: list (all pages), recent, search, database",
                param_type="string",
                required=False,
                default="recent",
                examples=["list", "recent", "search", "database"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for page content",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="database_id",
                description="Database ID for database queries",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="filter",
                description="Filter object for database queries (JSON)",
                param_type="object",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum items to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.api_token = (
            config.get("api_token")
            or os.environ.get("NOTION_TOKEN")
            or os.environ.get("NOTION_API_KEY", "")
        )
        self.root_page_id = config.get("root_page_id", "")
        self.include_databases = config.get("include_databases", True)
        self.include_pages = config.get("include_pages", True)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        # Parse database IDs
        db_str = config.get("database_ids", "")
        self.database_ids = [
            d.strip().replace("-", "") for d in db_str.split(",") if d.strip()
        ]

        self._client = httpx.Client(timeout=30)

    def _get_headers(self) -> dict:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Notion-Version": self.NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def _notion_request(
        self, method: str, endpoint: str, json_data: dict = None
    ) -> dict:
        """Make a Notion API request."""
        url = f"{self.NOTION_API_BASE}/{endpoint.lstrip('/')}"

        if method.upper() == "GET":
            response = self._client.get(url, headers=self._get_headers())
        else:
            response = self._client.post(
                url, headers=self._get_headers(), json=json_data or {}
            )

        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate pages and databases for indexing.

        Filtering logic:
        - If database_ids specified: ONLY index those databases (ignore include_pages)
        - If root_page_id specified: only index pages under that root
        - If neither: index all accessible pages and databases
        """
        logger.info("Listing Notion content")

        # If specific databases are configured, ONLY index those
        # This prevents bleeding between stores that target different databases
        if self.database_ids:
            logger.info(f"Indexing specific databases: {self.database_ids}")
            for db_id in self.database_ids:
                yield from self._list_database_entries(db_id)
            return  # Don't index pages when database filter is active

        # Otherwise, index pages (optionally filtered by root_page_id)
        if self.include_pages:
            yield from self._list_pages()

    def _list_pages(self) -> Iterator[DocumentInfo]:
        """List pages using search."""
        start_cursor = None

        while True:
            body = {
                "filter": {"property": "object", "value": "page"},
                "page_size": 100,
            }
            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                data = self._notion_request("POST", "/search", body)
                results = data.get("results", [])

                for page in results:
                    page_id = page.get("id", "").replace("-", "")
                    title = self._extract_title(page)
                    modified = page.get("last_edited_time", "")

                    yield DocumentInfo(
                        uri=f"notion://page/{page_id}",
                        title=title[:100] if title else page_id,
                        mime_type="text/markdown",
                        modified_at=modified,
                        metadata={
                            "type": "page",
                            "parent_type": page.get("parent", {}).get("type", ""),
                        },
                    )

                if not data.get("has_more"):
                    break
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.error(f"Failed to list pages: {e}")
                break

    def _list_database_entries(self, database_id: str) -> Iterator[DocumentInfo]:
        """List entries in a database."""
        start_cursor = None

        while True:
            body = {"page_size": 100}
            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                data = self._notion_request(
                    "POST", f"/databases/{database_id}/query", body
                )
                results = data.get("results", [])

                for entry in results:
                    entry_id = entry.get("id", "").replace("-", "")
                    title = self._extract_title(entry)
                    modified = entry.get("last_edited_time", "")

                    yield DocumentInfo(
                        uri=f"notion://database/{database_id}/entry/{entry_id}",
                        title=title[:100] if title else entry_id,
                        mime_type="text/markdown",
                        modified_at=modified,
                        metadata={
                            "type": "database_entry",
                            "database_id": database_id,
                        },
                    )

                if not data.get("has_more"):
                    break
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.error(f"Failed to list database entries for {database_id}: {e}")
                break

    def _extract_title(self, page: dict) -> str:
        """Extract title from a page or database entry."""
        properties = page.get("properties", {})

        # Check for title property
        for prop_name, prop_value in properties.items():
            prop_type = prop_value.get("type", "")

            if prop_type == "title":
                title_array = prop_value.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_array)

        # Fallback to Name property
        if "Name" in properties:
            name_prop = properties["Name"]
            if name_prop.get("type") == "title":
                title_array = name_prop.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_array)

        return "Untitled"

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read page or database entry content."""
        if not uri.startswith("notion://"):
            logger.error(f"Invalid Notion URI: {uri}")
            return None

        path = uri.replace("notion://", "")
        parts = path.split("/")

        try:
            if parts[0] == "page" and len(parts) >= 2:
                return self._read_page(parts[1])
            elif parts[0] == "database" and len(parts) >= 4:
                return self._read_database_entry(parts[1], parts[3])
        except Exception as e:
            logger.error(f"Failed to read {uri}: {e}")
            return None

        return None

    def _read_page(self, page_id: str) -> DocumentContent:
        """Read a page's content."""
        # Get page metadata
        page = self._notion_request("GET", f"/pages/{page_id}")
        title = self._extract_title(page)
        modified = page.get("last_edited_time", "")
        created = page.get("created_time", "")

        # Get page content (blocks)
        content = self._get_block_content(page_id)

        full_content = f"""# {title}

Created: {created}
Modified: {modified}

{content}
"""

        return DocumentContent(
            content=full_content,
            mime_type="text/markdown",
            metadata={
                "page_id": page_id,
                "title": title,
                "type": "page",
                "source_type": "page",
            },
        )

    def _read_database_entry(self, database_id: str, entry_id: str) -> DocumentContent:
        """Read a database entry."""
        entry = self._notion_request("GET", f"/pages/{entry_id}")
        title = self._extract_title(entry)
        modified = entry.get("last_edited_time", "")

        # Format properties
        properties = entry.get("properties", {})
        prop_lines = [f"# {title}", ""]

        for prop_name, prop_value in properties.items():
            if prop_name == "Name":
                continue
            formatted = self._format_property(prop_value)
            if formatted:
                prop_lines.append(f"**{prop_name}**: {formatted}")

        # Get content blocks
        content = self._get_block_content(entry_id)

        full_content = "\n".join(prop_lines) + "\n\n" + content

        return DocumentContent(
            content=full_content,
            mime_type="text/markdown",
            metadata={
                "entry_id": entry_id,
                "database_id": database_id,
                "title": title,
                "type": "database_entry",
                "source_type": "database_entry",
            },
        )

    def _get_block_content(self, block_id: str, depth: int = 0) -> str:
        """Recursively get block content."""
        if depth > 3:
            return ""

        content_parts = []
        start_cursor = None

        while True:
            params = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            try:
                url = f"/blocks/{block_id}/children"
                if start_cursor:
                    url += f"?start_cursor={start_cursor}"

                # Use GET for block children
                response = self._client.get(
                    f"{self.NOTION_API_BASE}{url}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

                blocks = data.get("results", [])

                for block in blocks:
                    block_type = block.get("type", "")
                    block_data = block.get(block_type, {})

                    text = self._extract_rich_text(block_data)

                    # Format based on block type
                    if block_type == "paragraph":
                        if text:
                            content_parts.append(text)
                    elif block_type.startswith("heading_"):
                        level = int(block_type[-1])
                        if text:
                            content_parts.append(f"{'#' * level} {text}")
                    elif block_type == "bulleted_list_item":
                        if text:
                            content_parts.append(f"• {text}")
                    elif block_type == "numbered_list_item":
                        if text:
                            content_parts.append(f"1. {text}")
                    elif block_type == "to_do":
                        checked = "x" if block_data.get("checked") else " "
                        if text:
                            content_parts.append(f"[{checked}] {text}")
                    elif block_type == "code":
                        lang = block_data.get("language", "")
                        if text:
                            content_parts.append(f"```{lang}\n{text}\n```")
                    elif block_type == "quote":
                        if text:
                            content_parts.append(f"> {text}")
                    elif block_type == "divider":
                        content_parts.append("---")
                    elif block_type == "callout":
                        emoji = block_data.get("icon", {}).get("emoji", "")
                        if text:
                            content_parts.append(f"{emoji} {text}")

                    # Handle children
                    if block.get("has_children"):
                        child_content = self._get_block_content(block["id"], depth + 1)
                        if child_content:
                            content_parts.append(child_content)

                if not data.get("has_more"):
                    break
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.warning(f"Failed to get blocks for {block_id}: {e}")
                break

        return "\n".join(content_parts)

    def _extract_rich_text(self, block_data: dict) -> str:
        """Extract plain text from rich text array."""
        rich_text = block_data.get("rich_text", [])
        if not rich_text:
            rich_text = block_data.get("text", [])

        return "".join(t.get("plain_text", "") for t in rich_text)

    def _format_property(self, prop_value: dict) -> str:
        """Format a property value for display."""
        prop_type = prop_value.get("type", "")

        if prop_type == "rich_text":
            return self._extract_rich_text(prop_value)
        elif prop_type == "number":
            return str(prop_value.get("number", ""))
        elif prop_type == "select":
            select = prop_value.get("select")
            return select.get("name", "") if select else ""
        elif prop_type == "multi_select":
            items = prop_value.get("multi_select", [])
            return ", ".join(i.get("name", "") for i in items)
        elif prop_type == "date":
            date = prop_value.get("date")
            if date:
                start = date.get("start", "")
                end = date.get("end", "")
                return f"{start} - {end}" if end else start
            return ""
        elif prop_type == "checkbox":
            return "Yes" if prop_value.get("checkbox") else "No"
        elif prop_type == "url":
            return prop_value.get("url", "")
        elif prop_type == "email":
            return prop_value.get("email", "")
        elif prop_type == "phone_number":
            return prop_value.get("phone_number", "")
        elif prop_type == "people":
            people = prop_value.get("people", [])
            return ", ".join(p.get("name", "") for p in people)
        elif prop_type == "status":
            status = prop_value.get("status")
            return status.get("name", "") if status else ""

        return ""

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live Notion data."""
        action = params.get("action", "recent")
        search_query = params.get("query", "")
        database_id = params.get("database_id", "")
        filter_obj = params.get("filter", {})
        max_results = params.get("max_results", self.live_max_results)

        try:
            if action == "list":
                # List all documents (pages/database entries)
                items = self._list_all_pages(max_results)
                formatted = self._format_list_results(items)
            elif action == "search" and search_query:
                items = self._search_pages(search_query, max_results)
                formatted = self._format_search_results(items)
            elif action == "database" and database_id:
                items = self._query_database(database_id, filter_obj, max_results)
                formatted = self._format_database_results(items, database_id)
            else:
                items = self._get_recent_pages(max_results)
                formatted = self._format_recent(items)

            return LiveDataResult(
                success=True,
                data=items,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Notion live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _list_all_pages(self, max_results: int = 100) -> list[dict]:
        """
        List pages scoped to this store's configuration (for 'list' action).

        Respects the store's root_page_id and database_ids settings:
        - If database_ids: list entries from those databases
        - If root_page_id: list child pages of that root
        - If neither: list all accessible pages (fallback)
        """
        all_pages = []

        # If database_ids configured, list entries from those databases
        if self.database_ids:
            for db_id in self.database_ids[:5]:  # Limit to 5 databases
                try:
                    entries = self._list_database_entries_simple(
                        db_id, max_results // len(self.database_ids)
                    )
                    all_pages.extend(entries)
                except Exception as e:
                    logger.warning(f"Failed to list database {db_id}: {e}")
            return all_pages[:max_results]

        # If root_page_id configured, list children of that page
        if self.root_page_id:
            return self._list_children_of_page(self.root_page_id, max_results)

        # Fallback: search all accessible pages
        start_cursor = None
        while len(all_pages) < max_results:
            body = {
                "filter": {"property": "object", "value": "page"},
                "page_size": min(100, max_results - len(all_pages)),
            }
            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                data = self._notion_request("POST", "/search", body)
                results = data.get("results", [])

                for page in results:
                    all_pages.append(
                        {
                            "id": page.get("id", "").replace("-", ""),
                            "title": self._extract_title(page),
                            "modified": page.get("last_edited_time", ""),
                            "url": page.get("url", ""),
                            "type": "page",
                        }
                    )

                if not data.get("has_more") or len(all_pages) >= max_results:
                    break
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.error(f"Failed to list pages: {e}")
                break

        return all_pages[:max_results]

    def _list_children_of_page(
        self, page_id: str, max_results: int = 100
    ) -> list[dict]:
        """List child pages of a specific page using the blocks API."""
        all_pages = []
        start_cursor = None

        # Normalize page_id (remove dashes)
        page_id = page_id.replace("-", "")

        while len(all_pages) < max_results:
            endpoint = f"/blocks/{page_id}/children"
            if start_cursor:
                endpoint += f"?start_cursor={start_cursor}"

            try:
                data = self._notion_request("GET", endpoint)
                results = data.get("results", [])

                for block in results:
                    # Check if this block is a child_page or child_database
                    block_type = block.get("type", "")
                    if block_type == "child_page":
                        child_page = block.get("child_page", {})
                        all_pages.append(
                            {
                                "id": block.get("id", "").replace("-", ""),
                                "title": child_page.get("title", "Untitled"),
                                "modified": block.get("last_edited_time", ""),
                                "url": f"https://notion.so/{block.get('id', '').replace('-', '')}",
                                "type": "page",
                            }
                        )
                    elif block_type == "child_database":
                        child_db = block.get("child_database", {})
                        all_pages.append(
                            {
                                "id": block.get("id", "").replace("-", ""),
                                "title": child_db.get("title", "Untitled Database"),
                                "modified": block.get("last_edited_time", ""),
                                "url": f"https://notion.so/{block.get('id', '').replace('-', '')}",
                                "type": "database",
                            }
                        )

                if not data.get("has_more") or len(all_pages) >= max_results:
                    break
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.error(f"Failed to list children of page {page_id}: {e}")
                break

        return all_pages[:max_results]

    def _list_database_entries_simple(
        self, database_id: str, max_results: int = 50
    ) -> list[dict]:
        """List entries from a database (simplified for list action)."""
        entries = []
        start_cursor = None

        while len(entries) < max_results:
            body = {"page_size": min(100, max_results - len(entries))}
            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                data = self._notion_request(
                    "POST", f"/databases/{database_id}/query", body
                )
                results = data.get("results", [])

                for page in results:
                    entries.append(
                        {
                            "id": page.get("id", "").replace("-", ""),
                            "title": self._extract_title(page),
                            "modified": page.get("last_edited_time", ""),
                            "url": page.get("url", ""),
                            "type": "database_entry",
                        }
                    )

                if not data.get("has_more") or len(entries) >= max_results:
                    break
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.error(f"Failed to query database {database_id}: {e}")
                break

        return entries[:max_results]

    def _get_recent_pages(self, max_results: int) -> list[dict]:
        """Get recently modified pages."""
        body = {
            "filter": {"property": "object", "value": "page"},
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
            "page_size": min(max_results, 100),
        }

        data = self._notion_request("POST", "/search", body)
        results = data.get("results", [])

        return [
            {
                "id": page.get("id", "").replace("-", ""),
                "title": self._extract_title(page),
                "modified": page.get("last_edited_time", ""),
                "url": page.get("url", ""),
                "type": "page",
            }
            for page in results[:max_results]
        ]

    def _search_pages(self, query: str, max_results: int) -> list[dict]:
        """Search for pages."""
        body = {
            "query": query,
            "page_size": min(max_results, 100),
        }

        data = self._notion_request("POST", "/search", body)
        results = data.get("results", [])

        return [
            {
                "id": item.get("id", "").replace("-", ""),
                "title": self._extract_title(item),
                "modified": item.get("last_edited_time", ""),
                "url": item.get("url", ""),
                "type": item.get("object", "page"),
            }
            for item in results[:max_results]
        ]

    def _query_database(
        self, database_id: str, filter_obj: dict, max_results: int
    ) -> list[dict]:
        """Query a database."""
        body = {"page_size": min(max_results, 100)}
        if filter_obj:
            body["filter"] = filter_obj

        data = self._notion_request("POST", f"/databases/{database_id}/query", body)
        results = data.get("results", [])

        return [
            {
                "id": entry.get("id", "").replace("-", ""),
                "title": self._extract_title(entry),
                "modified": entry.get("last_edited_time", ""),
                "url": entry.get("url", ""),
                "type": "database_entry",
                "properties": {
                    k: self._format_property(v)
                    for k, v in entry.get("properties", {}).items()
                    if k != "Name"
                },
            }
            for entry in results[:max_results]
        ]

    def _format_list_results(self, pages: list[dict]) -> str:
        """Format complete page list for LLM context (for 'list' action)."""
        if not pages:
            return "### Notion Pages\nNo pages found in this workspace."

        lines = ["### All Notion Pages", f"Total: {len(pages)} page(s)\n"]

        for page in pages:
            lines.append(f"• **{page['title']}**")
            lines.append(f"  Last modified: {page['modified']}")
            lines.append("")

        return "\n".join(lines)

    def _format_recent(self, pages: list[dict]) -> str:
        """Format recent pages for LLM context."""
        if not pages:
            return "### Notion Recent Pages\nNo recent pages found."

        lines = ["### Notion Recent Pages", f"Found {len(pages)} page(s):\n"]

        for page in pages:
            lines.append(f"**{page['title']}**")
            lines.append(f"  Modified: {page['modified']}")
            if page.get("url"):
                lines.append(f"  URL: {page['url']}")
            lines.append("")

        return "\n".join(lines)

    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results for LLM context."""
        if not results:
            return "### Notion Search\nNo results found."

        lines = ["### Notion Search Results", f"Found {len(results)} result(s):\n"]

        for item in results:
            item_type = "📄" if item["type"] == "page" else "🗃️"
            lines.append(f"{item_type} **{item['title']}**")
            lines.append(f"  Modified: {item['modified']}")
            lines.append("")

        return "\n".join(lines)

    def _format_database_results(self, entries: list[dict], database_id: str) -> str:
        """Format database query results for LLM context."""
        if not entries:
            return f"### Notion Database Query\nNo entries found."

        lines = [
            "### Notion Database Results",
            f"Found {len(entries)} entry(ies):\n",
        ]

        for entry in entries:
            lines.append(f"**{entry['title']}**")
            props = entry.get("properties", {})
            for prop_name, prop_value in props.items():
                if prop_value:
                    lines.append(f"  {prop_name}: {prop_value}")
            lines.append(f"  Modified: {entry['modified']}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine optimal routing."""
        query_lower = query.lower()
        action = params.get("action", "")

        # Recent/activity queries -> Live only
        recent_patterns = [
            "recent",
            "recently",
            "latest",
            "just edited",
            "today",
            "this week",
            "modified",
        ]
        if any(p in query_lower for p in recent_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "recent"},
                reason="Recent activity query - using live API only",
                max_live_results=self.live_max_results,
            )

        # Database queries -> Live only (need real-time filters)
        if action == "database" or params.get("database_id"):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "database"},
                reason="Database query requires live API",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"last month",
            r"archived",
        ]
        for pattern in historical_patterns:
            if re.search(pattern, query_lower):
                return QueryAnalysis(
                    routing=QueryRouting.RAG_ONLY,
                    rag_query=query,
                    reason="Historical reference - using RAG index",
                    max_rag_results=20,
                )

        # Search queries -> Both
        if action == "search" or params.get("query"):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.DEDUPE,
                reason="Search query - checking both sources",
                max_rag_results=15,
                max_live_results=self.live_max_results,
            )

        # Default -> Both with deduplication
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.DEDUPE,
            reason="General query - using both sources",
            max_rag_results=10,
            max_live_results=self.live_max_results,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Notion is accessible."""
        try:
            self._notion_request("GET", "/users/me")
            return True
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Notion API connection."""
        results = []
        overall_success = True

        try:
            # Test authentication
            user = self._notion_request("GET", "/users/me")
            user_type = user.get("type", "unknown")
            name = user.get("name", "Unknown")
            results.append(f"Connected as: {name} ({user_type})")

            # Test search
            try:
                search_results = self._notion_request(
                    "POST", "/search", {"page_size": 5}
                )
                count = len(search_results.get("results", []))
                results.append(f"Search: Found {count} accessible items")
            except Exception as e:
                results.append(f"Search: Error - {e}")

            # Test document listing
            if self.supports_rag:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found pages to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "recent", "max_results": 5})
                    if live_result.success:
                        count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {count} recent pages")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
