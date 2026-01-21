"""
Google Drive Unified Source Plugin.

Combines Google Drive document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index files by folder for semantic search
- Live side: Query recent files, search by name/content, list folder contents
- Intelligent routing: Analyze queries to choose optimal data source

Query routing examples:
- "recent documents" -> Live only (real-time list)
- "find the quarterly report from last year" -> RAG only (historical search)
- "my project proposal doc" -> Both, prefer RAG for content match
- "files shared with me today" -> Live only (recent activity)
"""

import logging
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.oauth import OAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class GDriveUnifiedSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Google Drive source - RAG for content search, Live for file listing.

    Single configuration provides:
    - Document indexing: Files indexed by folder for RAG semantic search
    - Live queries: Recent files, search by name, folder contents
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "gdrive"
    display_name = "Google Drive"
    description = "Google Drive with content search (RAG) and real-time file queries"
    category = "google"
    icon = "📁"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:gdrive"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # No drive actions yet
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.google_account_id,
            "folder_ids": store.gdrive_folder_id or "",
            "index_schedule": store.index_schedule or "",
            "live_max_results": 20,
        }

    # Drive API endpoint
    DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"

    # Google Workspace MIME types that need export
    GOOGLE_EXPORT_TYPES = {
        "application/vnd.google-apps.document": ("text/plain", ".txt"),
        "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
        "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
        "application/vnd.google-apps.drawing": ("image/png", ".png"),
    }

    # Binary file types processed through Docling
    DOCLING_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "google", "scopes": ["drive"]},
                help_text="Select a connected Google account with Drive access",
            ),
            FieldDefinition(
                name="folder_ids",
                label="Folders to Index",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated folder IDs to index (empty = entire drive)",
            ),
            FieldDefinition(
                name="include_shared",
                label="Include Shared Files",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Include files shared with you",
            ),
            FieldDefinition(
                name="vision_provider",
                label="Vision Provider",
                field_type=FieldType.SELECT,
                default="local",
                options=[
                    {"value": "local", "label": "Local (SmolDocling)"},
                    {"value": "ollama", "label": "Ollama"},
                    {"value": "none", "label": "None (skip images/PDFs)"},
                ],
                help_text="Vision model for processing PDFs and images",
            ),
            FieldDefinition(
                name="vision_model",
                label="Vision Model",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Vision model name (e.g., granite3.2-vision:latest)",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 * * * *", "label": "Hourly"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                ],
                help_text="How often to re-index files",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=25,
                help_text="Maximum files to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: recent, search, folder, shared",
                param_type="string",
                required=False,
                default="recent",
                examples=["recent", "search", "folder", "shared"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for file name or content",
                param_type="string",
                required=False,
                examples=["quarterly report", "meeting notes", "project proposal"],
            ),
            ParamDefinition(
                name="folder_id",
                description="Specific folder ID to list",
                param_type="string",
                required=False,
                examples=["1a2b3c4d5e6f"],
            ),
            ParamDefinition(
                name="mime_type",
                description="Filter by MIME type",
                param_type="string",
                required=False,
                examples=[
                    "application/pdf",
                    "application/vnd.google-apps.document",
                    "image/*",
                ],
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum files to return",
                param_type="integer",
                required=False,
                default=25,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"

        self.folder_ids = []
        if config.get("folder_ids"):
            self.folder_ids = [
                f.strip() for f in config["folder_ids"].split(",") if f.strip()
            ]
        self.include_shared = config.get("include_shared", True)
        self.vision_provider = config.get("vision_provider", "local")
        self.vision_model = config.get("vision_model")
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 25)

        self._client = httpx.Client(timeout=60)
        self._init_oauth_client()
        self._document_converter = None

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate files for indexing.

        Lists files from configured folders or entire drive.
        """
        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot list files - no valid access token")
            return

        # Build queries based on folders
        if self.folder_ids:
            for folder_id in self.folder_ids:
                logger.info(f"Listing files in folder: {folder_id}")
                yield from self._list_files_in_folder(folder_id)
        else:
            logger.info("Listing all files in Drive")
            yield from self._list_all_files()

    def _list_files_in_folder(self, folder_id: str) -> Iterator[DocumentInfo]:
        """List files in a specific folder."""
        query = f"'{folder_id}' in parents and trashed = false"
        yield from self._list_files_with_query(query)

    def _list_all_files(self) -> Iterator[DocumentInfo]:
        """List all files in Drive."""
        query = "trashed = false"
        yield from self._list_files_with_query(query)

    def _list_files_with_query(self, query: str) -> Iterator[DocumentInfo]:
        """List files matching a query."""
        fields = "nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)"

        page_token = None
        total_files = 0

        while True:
            params = {
                "q": query,
                "fields": fields,
                "pageSize": 100,
                "orderBy": "modifiedTime desc",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                response = self._oauth_client.get(
                    f"{self.DRIVE_API_BASE}/files",
                    headers=self._get_auth_headers(),
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Drive API error: {e}")
                break

            files = data.get("files", [])

            for file_info in files:
                mime_type = file_info.get("mimeType", "")
                file_id = file_info["id"]
                name = file_info["name"]

                # Skip folders
                if mime_type == "application/vnd.google-apps.folder":
                    continue

                # Check if supported
                if not self._is_supported_file(mime_type, name):
                    continue

                total_files += 1
                yield DocumentInfo(
                    uri=f"gdrive://{file_id}",
                    title=name,
                    mime_type=mime_type,
                    size_bytes=int(file_info.get("size", 0))
                    if file_info.get("size")
                    else None,
                    modified_at=file_info.get("modifiedTime"),
                    metadata={"parents": ",".join(file_info.get("parents", []))},
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Found {total_files} files to index")

    def _is_supported_file(self, mime_type: str, name: str) -> bool:
        """Check if a file type is supported for indexing."""
        # Google Workspace types we can export
        if mime_type in self.GOOGLE_EXPORT_TYPES:
            return True

        # Binary types we can process through Docling
        if mime_type in self.DOCLING_MIME_TYPES:
            return self.vision_provider != "none"

        # Text types by extension
        name_lower = name.lower()
        text_extensions = {".txt", ".md", ".html", ".htm", ".asciidoc", ".adoc"}
        for ext in text_extensions:
            if name_lower.endswith(ext):
                return True

        return False

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read file content for indexing.

        Handles Google Docs export, binary file processing, and text download.
        """
        if not uri.startswith("gdrive://"):
            logger.error(f"Invalid Drive URI: {uri}")
            return None

        file_id = uri.replace("gdrive://", "")

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot read file - no valid access token")
            return None

        try:
            # Get file metadata
            response = self._oauth_client.get(
                f"{self.DRIVE_API_BASE}/files/{file_id}",
                headers=self._get_auth_headers(),
                params={"fields": "id, name, mimeType, size, modifiedTime, parents"},
            )
            response.raise_for_status()
            file_meta = response.json()
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            return None

        mime_type = file_meta.get("mimeType", "")
        name = file_meta.get("name", file_id)

        # Handle Google Workspace files (need export)
        if mime_type in self.GOOGLE_EXPORT_TYPES:
            return self._export_google_doc(file_id, name, mime_type)

        # Handle binary files through Docling
        if mime_type in self.DOCLING_MIME_TYPES:
            return self._download_and_process_binary(file_id, name, mime_type)

        # Handle text files
        return self._download_text(file_id, name, mime_type)

    def _export_google_doc(
        self, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Export a Google Workspace document."""
        export_mime, _ = self.GOOGLE_EXPORT_TYPES[mime_type]

        try:
            response = self._oauth_client.get(
                f"{self.DRIVE_API_BASE}/files/{file_id}/export",
                headers=self._get_auth_headers(),
                params={"mimeType": export_mime},
                timeout=60,
            )
            response.raise_for_status()
            content = response.text
        except Exception as e:
            logger.error(f"Failed to export {name}: {e}")
            return None

        return DocumentContent(
            content=content,
            mime_type=export_mime,
            metadata={
                "file_id": file_id,
                "name": name,
                "original_mime_type": mime_type,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "source_type": "drive_document",
            },
        )

    def _download_and_process_binary(
        self, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Download and process a binary file through Docling."""
        try:
            response = self._oauth_client.get(
                f"{self.DRIVE_API_BASE}/files/{file_id}",
                headers=self._get_auth_headers(),
                params={"alt": "media"},
                timeout=120,
            )
            response.raise_for_status()
            content_bytes = response.content
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            return None

        # Get file extension
        extension = self.DOCLING_MIME_TYPES.get(mime_type, ".bin")

        # Process with Docling
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            try:
                converter = self._get_document_converter()
                if converter:
                    result = converter.convert(tmp_path)
                    text = result.document.export_to_markdown()
                else:
                    # No vision - return placeholder for PDFs
                    text = f"[Binary file: {name}]"
            finally:
                import os

                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Docling processing failed for {name}: {e}")
            text = f"[Failed to process: {name}]"

        return DocumentContent(
            content=text,
            mime_type="text/plain",
            metadata={
                "file_id": file_id,
                "name": name,
                "original_mime_type": mime_type,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "source_type": "drive_document",
            },
        )

    def _download_text(
        self, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Download a text file."""
        try:
            response = self._oauth_client.get(
                f"{self.DRIVE_API_BASE}/files/{file_id}",
                headers=self._get_auth_headers(),
                params={"alt": "media"},
                timeout=60,
            )
            response.raise_for_status()
            content = response.text
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            return None

        return DocumentContent(
            content=content,
            mime_type=mime_type,
            metadata={
                "file_id": file_id,
                "name": name,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "source_type": "drive_document",
            },
        )

    def _get_document_converter(self):
        """Get or create a Docling DocumentConverter."""
        if self._document_converter is None and self.vision_provider != "none":
            try:
                from rag.vision import get_document_converter

                self._document_converter = get_document_converter(
                    vision_provider=self.vision_provider,
                    vision_model=self.vision_model,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize document converter: {e}")
                self._document_converter = None

        return self._document_converter

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live file data.

        Supports actions: recent, search, folder, shared
        """
        start_time = time.time()

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            return LiveDataResult(
                success=False,
                error="No valid Google Drive access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        folder_id = params.get("folder_id", "")
        mime_type_filter = params.get("mime_type", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            # Fetch files based on action
            files = self._fetch_files_live(
                action, search_query, folder_id, mime_type_filter, max_results
            )

            # Format for LLM context
            formatted = self._format_files(files, action, search_query)

            latency_ms = int((time.time() - start_time) * 1000)

            return LiveDataResult(
                success=True,
                data=files,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Drive live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_files_live(
        self,
        action: str,
        search_query: str,
        folder_id: str,
        mime_type_filter: str,
        max_results: int,
    ) -> list[dict]:
        """Fetch files from Google Drive API."""
        # Build query based on action
        query_parts = ["trashed = false"]

        if action == "folder" and folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        elif action == "shared":
            query_parts.append("sharedWithMe = true")
        elif action == "search" and search_query:
            # Full-text search
            query_parts.append(f"fullText contains '{search_query}'")
        elif action == "recent":
            # Recent is handled by orderBy, but limit to last 7 days
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            query_parts.append(f"modifiedTime > '{week_ago}'")

        if mime_type_filter:
            if mime_type_filter.endswith("/*"):
                # Wildcard like image/*
                prefix = mime_type_filter[:-2]
                query_parts.append(f"mimeType contains '{prefix}'")
            else:
                query_parts.append(f"mimeType = '{mime_type_filter}'")

        query = " and ".join(query_parts)

        fields = "files(id, name, mimeType, size, modifiedTime, webViewLink, owners)"

        try:
            response = self._oauth_client.get(
                f"{self.DRIVE_API_BASE}/files",
                headers=self._get_auth_headers(),
                params={
                    "q": query,
                    "fields": fields,
                    "pageSize": max_results,
                    "orderBy": "modifiedTime desc",
                },
            )
            response.raise_for_status()
            return response.json().get("files", [])
        except Exception as e:
            logger.error(f"Failed to fetch files: {e}")
            return []

    def _format_files(self, files: list[dict], action: str, search_query: str) -> str:
        """Format files for LLM context."""
        account_email = self.get_account_email()

        if not files:
            action_msgs = {
                "recent": "No recent files.",
                "search": "No files found matching your search.",
                "folder": "No files in this folder.",
                "shared": "No files shared with you.",
            }
            return f"### Google Drive\n{action_msgs.get(action, 'No files found.')}"

        action_titles = {
            "recent": "Recent Files",
            "search": "File Search Results",
            "folder": "Folder Contents",
            "shared": "Shared Files",
        }

        title = action_titles.get(action, "Files")
        if search_query:
            title = f"Search: {search_query}"

        lines = [f"### {title}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(files)} file(s):\n")

        for file in files:
            file_id = file.get("id", "")
            name = file.get("name", "(Untitled)")
            mime_type = file.get("mimeType", "")
            size = file.get("size")
            modified = file.get("modifiedTime", "")
            web_link = file.get("webViewLink", "")
            owners = file.get("owners", [])

            # Format file type
            type_name = self._get_type_display(mime_type)

            # Format size
            size_str = ""
            if size:
                size_bytes = int(size)
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes // 1024} KB"
                else:
                    size_str = f"{size_bytes // (1024 * 1024)} MB"

            # Format modified date
            modified_str = ""
            if modified:
                try:
                    dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                    modified_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    modified_str = modified[:16]

            lines.append(f"**{name}**")
            lines.append(f"   ID: {file_id}")
            lines.append(f"   Type: {type_name}")
            if size_str:
                lines.append(f"   Size: {size_str}")
            if modified_str:
                lines.append(f"   Modified: {modified_str}")
            if owners:
                owner_names = [
                    o.get("displayName") or o.get("emailAddress", "") for o in owners
                ]
                lines.append(f"   Owner: {', '.join(owner_names)}")
            lines.append("")

        return "\n".join(lines)

    def _get_type_display(self, mime_type: str) -> str:
        """Get human-readable file type."""
        type_map = {
            "application/vnd.google-apps.document": "Google Doc",
            "application/vnd.google-apps.spreadsheet": "Google Sheet",
            "application/vnd.google-apps.presentation": "Google Slides",
            "application/vnd.google-apps.folder": "Folder",
            "application/pdf": "PDF",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint",
            "image/png": "PNG Image",
            "image/jpeg": "JPEG Image",
            "text/plain": "Text",
            "text/html": "HTML",
        }
        return type_map.get(mime_type, mime_type.split("/")[-1].upper())

    # =========================================================================
    # Query Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze query to determine optimal routing.

        Routing logic:
        - File listing queries (recent, shared, folder) -> Live only
        - Content search queries -> RAG only (semantic search)
        - Name/title search -> Both, prefer RAG
        - Default -> RAG for content, Live for metadata
        """
        query_lower = query.lower()
        action = params.get("action", "")

        # Listing queries -> Live only
        listing_patterns = [
            "recent",
            "latest",
            "new files",
            "shared with me",
            "folder",
            "list",
            "what files",
        ]
        if action in ["recent", "folder", "shared"] or any(
            p in query_lower for p in listing_patterns
        ):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": action or "recent"},
                reason="File listing requires live API",
                max_live_results=self.live_max_results,
            )

        # Historical content search -> RAG only
        historical_patterns = [
            "last year",
            r"20\d{2}",
            "old",
            "archive",
            "content",
            "says",
            "contains",
            "about",
        ]
        if any(re.search(p, query_lower) for p in historical_patterns):
            return QueryAnalysis(
                routing=QueryRouting.RAG_ONLY,
                rag_query=query,
                reason="Content search uses RAG index",
                max_rag_results=20,
            )

        # Search queries with specific terms -> Both
        if params.get("query") or action == "search":
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=params.get("query", query),
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.RAG_FIRST,
                reason="Search query - RAG for content, Live for names",
                max_rag_results=15,
                max_live_results=self.live_max_results,
            )

        # Default -> RAG for semantic search
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="Document content search uses RAG",
            max_rag_results=20,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Google Drive is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Google Drive API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return (
                    False,
                    "Failed to get access token - check OAuth configuration",
                )

            # Test API access - get about info
            response = self._oauth_client.get(
                f"{self.DRIVE_API_BASE}/about",
                headers=self._get_auth_headers(),
                params={"fields": "user,storageQuota"},
            )
            response.raise_for_status()
            about = response.json()

            user = about.get("user", {})
            email = user.get("emailAddress", "unknown")
            results.append(f"Connected as: {email}")

            # Show storage info
            quota = about.get("storageQuota", {})
            if quota.get("limit"):
                used_gb = int(quota.get("usage", 0)) / (1024**3)
                limit_gb = int(quota.get("limit", 0)) / (1024**3)
                results.append(f"Storage: {used_gb:.1f} GB / {limit_gb:.1f} GB")

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
