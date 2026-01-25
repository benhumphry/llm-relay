"""
OneNote Unified Source Plugin.

Combines OneNote indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index notebooks/sections/pages for semantic search
- Live side: Query recent pages, search notes, get page contents
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "what did I note today" -> Live only (real-time notes)
- "meeting notes from last month" -> RAG only (historical)
- "notes about project X" -> Both, merge results
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.oauth import MicrosoftOAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class OneNoteUnifiedSource(MicrosoftOAuthMixin, PluginUnifiedSource):
    """
    Unified OneNote source - RAG for notes, Live for recent/search.

    Single configuration provides:
    - Document indexing: Pages indexed for RAG semantic search
    - Live queries: Recent pages, search, specific page lookup
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "onenote"
    display_name = "OneNote"
    description = "OneNote notebooks with historical search (RAG) and real-time queries"
    category = "microsoft"
    icon = "📓"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:onenote"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # No actions for OneNote (yet)
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results

    _abstract = False

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME OneNote access. Actions: "
            "action='list' to enumerate ALL notes/pages "
            "(for 'what notes do I have?' queries), "
            "action='recent' for recently modified pages, "
            "action='search' with query='...' to search page content. "
            "Optional: notebook='NotebookName' to filter by notebook."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.microsoft_account_id,
            "notebook_ids": store.onenote_notebook_id or "",
            "index_schedule": store.index_schedule or "",
        }

    # Microsoft Graph API endpoint
    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Microsoft Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "microsoft", "scopes": ["notes"]},
                help_text="Select a connected Microsoft account with OneNote access",
            ),
            FieldDefinition(
                name="notebook_id",
                label="Notebook ID",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Specific notebook ID to index (empty = all notebooks)",
            ),
            FieldDefinition(
                name="section_id",
                label="Section ID",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Specific section ID to index (empty = all sections in notebook)",
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
                help_text="How often to re-index pages",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=20,
                help_text="Maximum pages to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: recent, search, list",
                param_type="string",
                required=False,
                default="recent",
                examples=["recent", "search", "list"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for page titles/content",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="notebook",
                description="Notebook name or ID to search within",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum pages to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.notebook_id = config.get("notebook_id", "")
        self.section_id = config.get("section_id", "")
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        self._init_oauth_client()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate pages for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot list pages - no valid access token")
            return

        logger.info("Listing OneNote pages")

        # If specific section is configured, list only its pages
        if self.section_id:
            yield from self._list_section_pages(self.section_id)
            return

        # If specific notebook is configured, list its sections
        if self.notebook_id:
            sections = self._get_notebook_sections(self.notebook_id)
            for section in sections:
                yield from self._list_section_pages(section["id"])
            return

        # Otherwise, list all notebooks and their sections
        notebooks = self._get_notebooks()
        for notebook in notebooks:
            sections = self._get_notebook_sections(notebook["id"])
            for section in sections:
                yield from self._list_section_pages(section["id"])

    def _get_notebooks(self) -> list[dict]:
        """Get all notebooks."""
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/onenote/notebooks",
                headers=self._get_auth_headers(),
                params={"$select": "id,displayName"},
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            logger.error(f"Failed to list notebooks: {e}")
            return []

    def _get_notebook_sections(self, notebook_id: str) -> list[dict]:
        """Get sections in a notebook."""
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/onenote/notebooks/{notebook_id}/sections",
                headers=self._get_auth_headers(),
                params={"$select": "id,displayName"},
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            logger.error(f"Failed to list sections for notebook {notebook_id}: {e}")
            return []

    def _list_section_pages(self, section_id: str) -> Iterator[DocumentInfo]:
        """List pages in a section."""
        endpoint = f"{self.GRAPH_API_BASE}/me/onenote/sections/{section_id}/pages"
        params = {
            "$select": "id,title,lastModifiedDateTime,parentSection",
            "$orderby": "lastModifiedDateTime desc",
            "$top": 100,
        }

        while endpoint:
            try:
                response = self._oauth_client.get(
                    endpoint,
                    headers=self._get_auth_headers(),
                    params=params if "?" not in endpoint else None,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to list pages in section {section_id}: {e}")
                break

            pages = data.get("value", [])

            for page in pages:
                title = page.get("title", "(Untitled)")
                modified = page.get("lastModifiedDateTime", "")
                parent_section = page.get("parentSection", {})

                yield DocumentInfo(
                    uri=f"onenote://{page['id']}",
                    title=title[:100] if title else page["id"],
                    mime_type="text/html",
                    modified_at=modified,
                    metadata={
                        "section_id": section_id,
                        "section_name": parent_section.get("displayName", ""),
                    },
                )

            # Handle pagination
            endpoint = data.get("@odata.nextLink")
            params = None

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read page content for indexing."""
        if not uri.startswith("onenote://"):
            logger.error(f"Invalid OneNote URI: {uri}")
            return None

        page_id = uri.replace("onenote://", "")

        if not self._refresh_token_if_needed():
            logger.error("Cannot read page - no valid access token")
            return None

        try:
            # Get page metadata
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/onenote/pages/{page_id}",
                headers=self._get_auth_headers(),
                params={
                    "$select": "id,title,createdDateTime,lastModifiedDateTime,parentSection,parentNotebook"
                },
            )
            response.raise_for_status()
            page_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch page metadata {page_id}: {e}")
            return None

        try:
            # Get page content (HTML)
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/onenote/pages/{page_id}/content",
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            logger.error(f"Failed to fetch page content {page_id}: {e}")
            return None

        # Extract fields
        title = page_data.get("title", "(Untitled)")
        created = page_data.get("createdDateTime", "")
        modified = page_data.get("lastModifiedDateTime", "")
        parent_section = page_data.get("parentSection", {})
        parent_notebook = page_data.get("parentNotebook", {})

        # Convert HTML to plain text
        text_content = self._strip_html(html_content)

        # Format for indexing
        content = f"""Title: {title}
Notebook: {parent_notebook.get("displayName", "Unknown")}
Section: {parent_section.get("displayName", "Unknown")}
Created: {created}
Modified: {modified}

{text_content}
"""

        # Parse date for metadata
        page_date = None
        if modified:
            try:
                parsed = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                page_date = parsed.strftime("%Y-%m-%d")
            except Exception:
                pass

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "page_id": page_id,
                "notebook_id": parent_notebook.get("id", ""),
                "notebook_name": parent_notebook.get("displayName", ""),
                "section_id": parent_section.get("id", ""),
                "section_name": parent_section.get("displayName", ""),
                "page_date": page_date,
                "title": title,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "source_type": "note",
            },
        )

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and decode entities."""
        import html as html_module

        # Remove script and style
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Replace block elements with newlines
        html = re.sub(
            r"<(br|p|div|tr|li|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE
        )
        # Remove remaining tags
        html = re.sub(r"<[^>]+>", "", html)
        # Decode entities
        html = html_module.unescape(html)
        # Clean whitespace
        html = re.sub(r"\n\s*\n", "\n\n", html)
        return html.strip()

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live note data."""
        if not self._refresh_token_if_needed():
            return LiveDataResult(
                success=False,
                error="No valid OneNote access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        notebook_filter = params.get("notebook", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            if action == "search" and search_query:
                pages = self._search_pages(search_query, max_results)
            else:
                pages = self._get_recent_pages(max_results)

            formatted = self._format_pages(pages, action)

            return LiveDataResult(
                success=True,
                data=pages,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"OneNote live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _get_recent_pages(self, max_results: int) -> list[dict]:
        """Get recently modified pages."""
        response = self._oauth_client.get(
            f"{self.GRAPH_API_BASE}/me/onenote/pages",
            headers=self._get_auth_headers(),
            params={
                "$select": "id,title,lastModifiedDateTime,parentSection,parentNotebook",
                "$orderby": "lastModifiedDateTime desc",
                "$top": min(max_results, 50),
            },
        )
        response.raise_for_status()

        pages_data = response.json().get("value", [])
        return self._parse_pages(pages_data, max_results)

    def _search_pages(self, query: str, max_results: int) -> list[dict]:
        """Search for pages by content."""
        # OneNote search requires different endpoint
        # Note: Graph API's OneNote search is limited
        # For now, we'll filter by title
        response = self._oauth_client.get(
            f"{self.GRAPH_API_BASE}/me/onenote/pages",
            headers=self._get_auth_headers(),
            params={
                "$select": "id,title,lastModifiedDateTime,parentSection,parentNotebook",
                "$filter": f"contains(title, '{query}')",
                "$top": min(max_results, 50),
            },
        )

        # If filter fails (not all fields support contains), fall back to fetching all and filtering
        if response.status_code == 400:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/onenote/pages",
                headers=self._get_auth_headers(),
                params={
                    "$select": "id,title,lastModifiedDateTime,parentSection,parentNotebook",
                    "$orderby": "lastModifiedDateTime desc",
                    "$top": 100,
                },
            )
            response.raise_for_status()
            all_pages = response.json().get("value", [])
            # Manual filter
            query_lower = query.lower()
            pages_data = [
                p
                for p in all_pages
                if query_lower in (p.get("title", "") or "").lower()
            ][:max_results]
        else:
            response.raise_for_status()
            pages_data = response.json().get("value", [])

        return self._parse_pages(pages_data, max_results)

    def _parse_pages(self, pages_data: list[dict], max_results: int) -> list[dict]:
        """Parse page data into standardized format."""
        pages = []
        for page in pages_data[:max_results]:
            parent_section = page.get("parentSection", {})
            parent_notebook = page.get("parentNotebook", {})

            pages.append(
                {
                    "id": page.get("id"),
                    "title": page.get("title", "(Untitled)"),
                    "modified": page.get("lastModifiedDateTime", ""),
                    "notebook": parent_notebook.get("displayName", ""),
                    "section": parent_section.get("displayName", ""),
                    "account_email": self.get_account_email(),
                }
            )

        return pages

    def _format_pages(self, pages: list[dict], action: str) -> str:
        """Format pages for LLM context."""
        account_email = self.get_account_email()

        if not pages:
            action_msgs = {
                "recent": "No recent pages.",
                "search": "No pages found matching your search.",
                "list": "No pages found.",
            }
            return (
                f"### OneNote ({action})\n{action_msgs.get(action, 'No pages found.')}"
            )

        action_titles = {
            "recent": "Recent Notes",
            "search": "Search Results",
            "list": "Notes",
        }

        lines = [f"### {action_titles.get(action, 'Notes')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(pages)} page(s):\n")

        for page in pages:
            title = page.get("title", "(Untitled)")
            notebook = page.get("notebook", "")
            section = page.get("section", "")
            modified = page.get("modified", "")

            lines.append(f"**{title}**")
            lines.append(f"  Notebook: {notebook}")
            lines.append(f"  Section: {section}")

            if modified:
                lines.append(f"  Modified: {modified}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine optimal routing."""
        query_lower = query.lower()
        action = params.get("action", "")

        # Recent notes queries -> Live only
        recent_patterns = [
            "today",
            "just wrote",
            "just added",
            "recent notes",
            "latest notes",
            "this week",
            "what did i note",
        ]
        if any(p in query_lower for p in recent_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "recent"},
                reason="Recent notes query - using live API only",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"last month",
            r"old notes",
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

        # Topic-based queries -> RAG primarily
        topic_patterns = ["about", "regarding", "notes on", "information about"]
        if any(p in query_lower for p in topic_patterns):
            return QueryAnalysis(
                routing=QueryRouting.RAG_THEN_LIVE,
                rag_query=query,
                live_params=params,
                merge_strategy=MergeStrategy.RAG_FIRST,
                reason="Topic search - RAG first, live supplement",
                max_rag_results=15,
                max_live_results=5,
            )

        # Default -> Both with deduplication
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.DEDUPE,
            reason="General query - using both sources",
            max_rag_results=10,
            max_live_results=10,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if OneNote is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test OneNote API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return False, "Failed to get access token - check OAuth configuration"

            # Test API access
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/onenote/notebooks",
                headers=self._get_auth_headers(),
                params={"$select": "id,displayName", "$top": 5},
            )
            response.raise_for_status()
            notebooks = response.json().get("value", [])
            results.append(f"Notebooks: Found {len(notebooks)} notebook(s)")

            # Test document listing (RAG side)
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
                        page_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {page_count} recent pages")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
