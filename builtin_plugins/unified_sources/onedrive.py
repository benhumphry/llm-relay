"""
OneDrive Unified Source Plugin.

Combines OneDrive document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index files/folders for semantic search
- Live side: Query recent files, search by name, get file contents
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "recent documents I edited" -> Live only (activity feed)
- "project proposal from Q3" -> RAG only (historical document)
- "find the latest budget spreadsheet" -> Both, prefer live
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


class OneDriveUnifiedSource(MicrosoftOAuthMixin, PluginUnifiedSource):
    """
    Unified OneDrive source - RAG for documents, Live for recent/search.

    Single configuration provides:
    - Document indexing: Files indexed for RAG semantic search
    - Live queries: Recent files, search, specific file lookup
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "onedrive"
    display_name = "OneDrive"
    description = "OneDrive/Microsoft 365 storage with document search (RAG) and real-time queries"
    category = "microsoft"
    icon = "☁️"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:onedrive"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # No actions for file storage (yet)
    supports_incremental = True

    default_cache_ttl = 600  # 10 minutes for live results

    # Supported file extensions for indexing
    INDEXABLE_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".doc",
        ".docx",
        ".odt",
        ".xls",
        ".xlsx",
        ".ods",
        ".csv",
        ".ppt",
        ".pptx",
        ".odp",
        ".pdf",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
    }

    _abstract = False

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME OneDrive access. Actions: "
            "action='recent' for recently modified files, "
            "action='search' with query='...' to search file names/content, "
            "action='list' with folder='path/to/folder' to list folder contents. "
            "Optional: max_results=N to limit results."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.microsoft_account_id,
            "folder_ids": store.onedrive_folder_id or "",
            "index_schedule": store.index_schedule or "",
            "live_max_results": 20,
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
                picker_options={"provider": "microsoft", "scopes": ["files"]},
                help_text="Select a connected Microsoft account with OneDrive access",
            ),
            FieldDefinition(
                name="folder_path",
                label="Folder Path",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Folder path to index (empty = root). Example: Documents/Projects",
            ),
            FieldDefinition(
                name="include_subfolders",
                label="Include Subfolders",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index files in subfolders recursively",
            ),
            FieldDefinition(
                name="file_extensions",
                label="File Extensions",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated extensions to index (empty = all supported). Example: .docx,.pdf,.txt",
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
                help_text="How often to re-index files",
            ),
            FieldDefinition(
                name="max_file_size_mb",
                label="Max File Size (MB)",
                field_type=FieldType.INTEGER,
                default=10,
                help_text="Maximum file size to index in megabytes",
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
                description="Search query for file names/content",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="folder",
                description="Folder path to list",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum files to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.folder_path = config.get("folder_path", "").strip("/")
        self.include_subfolders = config.get("include_subfolders", True)
        self.index_schedule = config.get("index_schedule", "")
        self.max_file_size_mb = config.get("max_file_size_mb", 10)

        # Parse file extensions filter
        ext_str = config.get("file_extensions", "")
        if ext_str:
            self.file_extensions = set(
                ext.strip().lower()
                if ext.strip().startswith(".")
                else f".{ext.strip().lower()}"
                for ext in ext_str.split(",")
                if ext.strip()
            )
        else:
            self.file_extensions = self.INDEXABLE_EXTENSIONS

        self._init_oauth_client()

    def _get_folder_endpoint(self) -> str:
        """Get the drive endpoint for the configured folder."""
        if self.folder_path:
            return f"{self.GRAPH_API_BASE}/me/drive/root:/{self.folder_path}:/children"
        return f"{self.GRAPH_API_BASE}/me/drive/root/children"

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate files for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot list files - no valid access token")
            return

        logger.info(f"Listing OneDrive files in: {self.folder_path or 'root'}")

        # Start recursive listing from root or specified folder
        yield from self._list_folder_recursive(self.folder_path)

    def _list_folder_recursive(self, folder_path: str) -> Iterator[DocumentInfo]:
        """Recursively list files in a folder."""
        # Build endpoint
        if folder_path:
            endpoint = f"{self.GRAPH_API_BASE}/me/drive/root:/{folder_path}:/children"
        else:
            endpoint = f"{self.GRAPH_API_BASE}/me/drive/root/children"

        params = {
            "$select": "id,name,file,folder,size,lastModifiedDateTime,parentReference",
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
                logger.error(f"OneDrive API error listing {folder_path}: {e}")
                break

            items = data.get("value", [])

            for item in items:
                if "folder" in item and self.include_subfolders:
                    # Recurse into subfolder
                    subfolder_path = (
                        f"{folder_path}/{item['name']}" if folder_path else item["name"]
                    )
                    yield from self._list_folder_recursive(subfolder_path)
                elif "file" in item:
                    # Check file extension
                    name = item.get("name", "")
                    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

                    if ext not in self.file_extensions:
                        continue

                    # Check file size
                    size = item.get("size", 0)
                    if size > self.max_file_size_mb * 1024 * 1024:
                        logger.debug(f"Skipping large file: {name} ({size} bytes)")
                        continue

                    parent = item.get("parentReference", {})
                    path = parent.get("path", "").replace("/drive/root:", "")

                    yield DocumentInfo(
                        uri=f"onedrive://{item['id']}",
                        title=name,
                        mime_type=item.get("file", {}).get(
                            "mimeType", "application/octet-stream"
                        ),
                        modified_at=item.get("lastModifiedDateTime", ""),
                        metadata={
                            "path": f"{path}/{name}" if path else name,
                            "size": size,
                        },
                    )

            # Handle pagination
            endpoint = data.get("@odata.nextLink")
            params = None

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read file content for indexing."""
        if not uri.startswith("onedrive://"):
            logger.error(f"Invalid OneDrive URI: {uri}")
            return None

        item_id = uri.replace("onedrive://", "")

        if not self._refresh_token_if_needed():
            logger.error("Cannot read file - no valid access token")
            return None

        try:
            # Get file metadata first
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/drive/items/{item_id}",
                headers=self._get_auth_headers(),
                params={
                    "$select": "id,name,file,size,lastModifiedDateTime,parentReference"
                },
            )
            response.raise_for_status()
            item_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch file metadata {item_id}: {e}")
            return None

        name = item_data.get("name", "")
        mime_type = item_data.get("file", {}).get(
            "mimeType", "application/octet-stream"
        )
        size = item_data.get("size", 0)
        parent = item_data.get("parentReference", {})
        path = parent.get("path", "").replace("/drive/root:", "")

        # Download content
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/drive/items/{item_id}/content",
                headers=self._get_auth_headers(),
                follow_redirects=True,
            )
            response.raise_for_status()
            content_bytes = response.content
        except Exception as e:
            logger.error(f"Failed to download file {item_id}: {e}")
            return None

        # Decode text content
        try:
            if mime_type.startswith("text/") or any(
                mime_type.endswith(t) for t in ["json", "xml", "yaml"]
            ):
                content = content_bytes.decode("utf-8", errors="replace")
            else:
                # For binary formats, return bytes for further processing
                # The indexer will handle PDF/Office docs with Docling
                content = content_bytes
        except Exception as e:
            logger.error(f"Failed to decode file {name}: {e}")
            return None

        return DocumentContent(
            content=content,
            mime_type=mime_type,
            metadata={
                "file_id": item_id,
                "filename": name,
                "path": f"{path}/{name}" if path else name,
                "size": size,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "source_type": "file",
            },
        )

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live file data."""
        if not self._refresh_token_if_needed():
            return LiveDataResult(
                success=False,
                error="No valid OneDrive access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        folder = params.get("folder", "")
        max_results = params.get("max_results", 20)

        try:
            if action == "search" and search_query:
                files = self._search_files(search_query, max_results)
            elif action == "list":
                files = self._list_folder(folder, max_results)
            else:
                files = self._get_recent_files(max_results)

            formatted = self._format_files(files, action)

            return LiveDataResult(
                success=True,
                data=files,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"OneDrive live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _get_recent_files(self, max_results: int) -> list[dict]:
        """Get recently modified files."""
        response = self._oauth_client.get(
            f"{self.GRAPH_API_BASE}/me/drive/recent",
            headers=self._get_auth_headers(),
            params={
                "$select": "id,name,file,folder,size,lastModifiedDateTime,webUrl,parentReference",
                "$top": min(max_results, 50),
            },
        )
        response.raise_for_status()

        items = response.json().get("value", [])
        return self._parse_items(items, max_results)

    def _search_files(self, query: str, max_results: int) -> list[dict]:
        """Search for files by name/content."""
        response = self._oauth_client.get(
            f"{self.GRAPH_API_BASE}/me/drive/root/search(q='{query}')",
            headers=self._get_auth_headers(),
            params={
                "$select": "id,name,file,folder,size,lastModifiedDateTime,webUrl,parentReference",
                "$top": min(max_results, 50),
            },
        )
        response.raise_for_status()

        items = response.json().get("value", [])
        return self._parse_items(items, max_results)

    def _list_folder(self, folder_path: str, max_results: int) -> list[dict]:
        """List files in a specific folder."""
        if folder_path:
            endpoint = f"{self.GRAPH_API_BASE}/me/drive/root:/{folder_path}:/children"
        else:
            endpoint = f"{self.GRAPH_API_BASE}/me/drive/root/children"

        response = self._oauth_client.get(
            endpoint,
            headers=self._get_auth_headers(),
            params={
                "$select": "id,name,file,folder,size,lastModifiedDateTime,webUrl,parentReference",
                "$top": min(max_results, 100),
            },
        )
        response.raise_for_status()

        items = response.json().get("value", [])
        return self._parse_items(items, max_results)

    def _parse_items(self, items: list[dict], max_results: int) -> list[dict]:
        """Parse OneDrive items into standardized format."""
        files = []
        for item in items[:max_results]:
            parent = item.get("parentReference", {})
            path = parent.get("path", "").replace("/drive/root:", "")
            name = item.get("name", "")

            is_folder = "folder" in item
            item_type = "folder" if is_folder else "file"

            files.append(
                {
                    "id": item.get("id"),
                    "name": name,
                    "type": item_type,
                    "path": f"{path}/{name}" if path else name,
                    "size": item.get("size", 0) if not is_folder else None,
                    "mime_type": item.get("file", {}).get("mimeType")
                    if not is_folder
                    else None,
                    "modified": item.get("lastModifiedDateTime", ""),
                    "web_url": item.get("webUrl", ""),
                    "account_email": self.get_account_email(),
                }
            )

        return files

    def _format_files(self, files: list[dict], action: str) -> str:
        """Format files for LLM context."""
        account_email = self.get_account_email()

        if not files:
            action_msgs = {
                "recent": "No recent files.",
                "search": "No files found matching your search.",
                "list": "No files in this folder.",
            }
            return (
                f"### OneDrive ({action})\n{action_msgs.get(action, 'No files found.')}"
            )

        action_titles = {
            "recent": "Recent Files",
            "search": "Search Results",
            "list": "Folder Contents",
        }

        lines = [f"### {action_titles.get(action, 'Files')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(files)} item(s):\n")

        for f in files:
            name = f.get("name", "")
            item_type = f.get("type", "file")
            path = f.get("path", "")
            size = f.get("size")
            modified = f.get("modified", "")

            type_icon = "📁" if item_type == "folder" else "📄"
            lines.append(f"{type_icon} **{name}**")
            lines.append(f"  Path: {path}")

            if size is not None:
                size_str = self._format_size(size)
                lines.append(f"  Size: {size_str}")

            if modified:
                lines.append(f"  Modified: {modified}")

            lines.append("")

        return "\n".join(lines)

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine optimal routing."""
        query_lower = query.lower()
        action = params.get("action", "")

        # Recent activity queries -> Live only
        recent_patterns = [
            "recent",
            "recently",
            "just edited",
            "just modified",
            "latest",
            "today",
            "this week",
            "working on",
        ]
        if any(p in query_lower for p in recent_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "recent"},
                reason="Recent activity query - using live API only",
                max_live_results=20,
            )

        # Explicit search -> Both
        if action == "search" or params.get("query"):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.DEDUPE,
                reason="Search query - checking both sources",
                max_rag_results=15,
                max_live_results=20,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"old version",
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

        # Content-based search -> RAG primarily
        content_patterns = ["about", "regarding", "contains", "mentions", "discusses"]
        if any(p in query_lower for p in content_patterns):
            return QueryAnalysis(
                routing=QueryRouting.RAG_THEN_LIVE,
                rag_query=query,
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.RAG_FIRST,
                reason="Content search - RAG first, live supplement",
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
            max_live_results=15,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if OneDrive is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test OneDrive API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return False, "Failed to get access token - check OAuth configuration"

            # Test API access
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/drive",
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            drive = response.json()
            quota = drive.get("quota", {})
            used = self._format_size(quota.get("used", 0))
            total = self._format_size(quota.get("total", 0))
            results.append(f"Drive: {used} / {total} used")

            # Test document listing (RAG side)
            if self.supports_rag:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found files to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "recent", "max_results": 5})
                    if live_result.success:
                        file_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {file_count} recent files")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
