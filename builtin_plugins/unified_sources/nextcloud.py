"""
Nextcloud Unified Source Plugin.

A unified source for indexing files from Nextcloud via WebDAV with live document lookup.

Features:
- Document side: Index files for RAG semantic search
- Live side: Fetch full file content by name, list files
- Fuzzy filename matching for document resolution
"""

import logging
import os
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree as ET

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class NextcloudUnifiedSource(PluginUnifiedSource):
    """Nextcloud source with live document lookup via WebDAV."""

    source_type = "nextcloud"
    display_name = "Nextcloud"
    description = "Index files from Nextcloud with full document lookup"
    category = "storage"
    icon = "☁️"

    # Document store types this unified source handles
    handles_doc_source_types = ["nextcloud"]

    supports_rag = True
    supports_live = True  # Enable live document lookup
    supports_actions = False
    supports_incremental = True

    default_cache_ttl = 3600  # 1 hour

    _abstract = False

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "Nextcloud file storage with document lookup. Use action='lookup' with filename "
            "param when user asks for FULL CONTENT of a specific file. "
            "Use action='list' to enumerate files. "
            "For semantic search across file contents, RAG will be used automatically."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        import os

        return {
            "url": os.environ.get("NEXTCLOUD_URL", ""),
            "username": os.environ.get("NEXTCLOUD_USERNAME", ""),
            "password": os.environ.get("NEXTCLOUD_PASSWORD", ""),
            "folder_path": store.nextcloud_folder or "",
        }

    DEFAULT_EXTENSIONS = {
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
    }
    DAV_NS = "{DAV:}"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="url",
                label="Nextcloud URL",
                field_type=FieldType.TEXT,
                required=True,
                help_text="URL of your Nextcloud instance",
            ),
            FieldDefinition(
                name="username",
                label="Username",
                field_type=FieldType.TEXT,
                required=True,
                help_text="Nextcloud username",
            ),
            FieldDefinition(
                name="password",
                label="Password/App Password",
                field_type=FieldType.PASSWORD,
                required=True,
                help_text="Password or app password for WebDAV access",
            ),
            FieldDefinition(
                name="folder_path",
                label="Folder Path",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Folder path to index (empty = root)",
            ),
            FieldDefinition(
                name="recursive",
                label="Include Subfolders",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Recursively index files in subfolders",
            ),
            FieldDefinition(
                name="file_extensions",
                label="File Extensions",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated extensions to index (empty = all supported)",
            ),
            FieldDefinition(
                name="max_file_size_mb",
                label="Max File Size (MB)",
                field_type=FieldType.INTEGER,
                default=10,
                help_text="Maximum file size to index",
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
                help_text="How often to re-index files",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters for live document lookup."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: lookup (full file), list (enumerate files)",
                param_type="string",
                required=False,
                default="lookup",
                examples=["lookup", "list"],
            ),
            ParamDefinition(
                name="filename",
                description="Filename or partial name to lookup (for action=lookup)",
                param_type="string",
                required=False,
                examples=["report.pdf", "meeting notes", "invoice 2024"],
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum files to list (for action=list)",
                param_type="integer",
                required=False,
                default=50,
            ),
        ]

    def __init__(self, config: dict):
        self.url = config.get("url", "").rstrip("/")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.folder_path = config.get("folder_path", "").strip("/")
        self.recursive = config.get("recursive", True)
        self.max_file_size_mb = config.get("max_file_size_mb", 10)
        self.index_schedule = config.get("index_schedule", "")
        ext_str = config.get("file_extensions", "")
        self.file_extensions = (
            set(
                ext.strip().lower()
                if ext.strip().startswith(".")
                else f".{ext.strip().lower()}"
                for ext in ext_str.split(",")
                if ext.strip()
            )
            if ext_str
            else self.DEFAULT_EXTENSIONS
        )
        self.webdav_base = f"{self.url}/remote.php/dav/files/{self.username}"
        self._client = httpx.Client(timeout=60, auth=(self.username, self.password))

    def _get_mime_type(self, ext: str) -> str:
        mime_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".xml": "application/xml",
            ".json": "application/json",
            ".yaml": "application/yaml",
            ".yml": "application/yaml",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _propfind(self, path: str, depth: int = 1) -> list[dict]:
        url = f"{self.webdav_base}/{path}".rstrip("/")
        body = '<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/><d:getcontentlength/><d:getlastmodified/><d:getcontenttype/></d:prop></d:propfind>'
        response = self._client.request(
            "PROPFIND",
            url,
            headers={"Depth": str(depth), "Content-Type": "application/xml"},
            content=body,
        )
        response.raise_for_status()
        items = []
        root = ET.fromstring(response.content)
        for response_elem in root.findall(f"{self.DAV_NS}response"):
            href = response_elem.find(f"{self.DAV_NS}href")
            if href is None:
                continue
            href_text = href.text or ""
            item_path = (
                href_text.split(f"/files/{self.username}/")[-1]
                if "/remote.php/dav/files/" in href_text
                else href_text.split("/")[-1]
            )
            propstat = response_elem.find(f"{self.DAV_NS}propstat")
            if propstat is None:
                continue
            prop = propstat.find(f"{self.DAV_NS}prop")
            if prop is None:
                continue
            resourcetype = prop.find(f"{self.DAV_NS}resourcetype")
            is_collection = (
                resourcetype is not None
                and resourcetype.find(f"{self.DAV_NS}collection") is not None
            )
            size_elem = prop.find(f"{self.DAV_NS}getcontentlength")
            size = (
                int(size_elem.text) if size_elem is not None and size_elem.text else 0
            )
            modified_elem = prop.find(f"{self.DAV_NS}getlastmodified")
            modified = modified_elem.text if modified_elem is not None else ""
            ctype_elem = prop.find(f"{self.DAV_NS}getcontenttype")
            content_type = ctype_elem.text if ctype_elem is not None else ""
            items.append(
                {
                    "path": item_path,
                    "is_collection": is_collection,
                    "size": size,
                    "modified": modified,
                    "content_type": content_type,
                }
            )
        return items

    def list_documents(self) -> Iterator[DocumentInfo]:
        logger.info(f"Listing Nextcloud files in: {self.folder_path or '/'}")
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        yield from (
            self._list_recursive(self.folder_path, max_size_bytes)
            if self.recursive
            else self._list_folder(self.folder_path, max_size_bytes)
        )

    def _list_folder(
        self, folder_path: str, max_size_bytes: int
    ) -> Iterator[DocumentInfo]:
        try:
            for item in self._propfind(folder_path):
                if item["is_collection"]:
                    continue
                ext = Path(item["path"]).suffix.lower()
                if ext not in self.file_extensions or item["size"] > max_size_bytes:
                    continue
                modified = item.get("modified", "")
                if modified:
                    try:
                        from email.utils import parsedate_to_datetime

                        modified = parsedate_to_datetime(modified).isoformat()
                    except Exception:
                        pass
                yield DocumentInfo(
                    uri=f"nextcloud://{item['path']}",
                    title=item["path"],
                    mime_type=item.get("content_type") or self._get_mime_type(ext),
                    modified_at=modified,
                    metadata={
                        "path": item["path"],
                        "size": item["size"],
                        "extension": ext,
                    },
                )
        except Exception as e:
            logger.error(f"Failed to list folder {folder_path}: {e}")

    def _list_recursive(
        self, folder_path: str, max_size_bytes: int
    ) -> Iterator[DocumentInfo]:
        try:
            for item in self._propfind(folder_path):
                item_path = item["path"]
                if item_path.rstrip("/") == folder_path.rstrip("/"):
                    continue
                if item["is_collection"]:
                    yield from self._list_recursive(item_path, max_size_bytes)
                else:
                    ext = Path(item_path).suffix.lower()
                    if ext not in self.file_extensions or item["size"] > max_size_bytes:
                        continue
                    modified = item.get("modified", "")
                    if modified:
                        try:
                            from email.utils import parsedate_to_datetime

                            modified = parsedate_to_datetime(modified).isoformat()
                        except Exception:
                            pass
                    yield DocumentInfo(
                        uri=f"nextcloud://{item_path}",
                        title=item_path,
                        mime_type=item.get("content_type") or self._get_mime_type(ext),
                        modified_at=modified,
                        metadata={
                            "path": item_path,
                            "size": item["size"],
                            "extension": ext,
                        },
                    )
        except Exception as e:
            logger.error(f"Failed to list folder {folder_path}: {e}")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        if not uri.startswith("nextcloud://"):
            return None
        file_path = uri.replace("nextcloud://", "")
        ext = Path(file_path).suffix.lower()
        mime_type = self._get_mime_type(ext)
        try:
            response = self._client.get(f"{self.webdav_base}/{file_path}")
            response.raise_for_status()
            content = (
                response.text
                if mime_type.startswith("text/")
                or mime_type
                in ("application/json", "application/xml", "application/yaml")
                else response.content
            )
            return DocumentContent(
                content=content,
                mime_type=mime_type,
                metadata={
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "extension": ext,
                    "source_type": "file",
                },
            )
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch full file content or list files."""
        action = params.get("action", "lookup")
        filename = params.get("filename", "")
        max_results = params.get("max_results", 50)
        two_pass_uri = params.get("_two_pass_uri")  # Direct URI from RAG metadata

        try:
            if action == "list":
                return self._fetch_list(max_results)
            elif action == "lookup" and (filename or two_pass_uri):
                return self._fetch_document(filename, two_pass_uri)
            else:
                return LiveDataResult(
                    success=False,
                    error="Specify action='lookup' with filename, or action='list'",
                )
        except Exception as e:
            logger.error(f"Nextcloud fetch error: {e}")
            return LiveDataResult(success=False, error=str(e))

    def _fetch_list(self, max_results: int) -> LiveDataResult:
        """List files in the configured folder."""
        files = []
        for doc in self.list_documents():
            files.append(
                {
                    "name": doc.title,
                    "path": doc.metadata.get("path", ""),
                    "size": doc.metadata.get("size", 0),
                    "modified": doc.modified_at,
                }
            )
            if len(files) >= max_results:
                break

        formatted = self._format_file_list(files)
        return LiveDataResult(
            success=True,
            data=files,
            formatted=formatted,
            cache_ttl=self.default_cache_ttl,
        )

    def _fetch_document(
        self, filename: str, two_pass_uri: str = None
    ) -> LiveDataResult:
        """
        Fetch full file content by filename.

        Args:
            filename: Filename or partial name to search for
            two_pass_uri: Direct file path from RAG metadata (two-pass retrieval)
        """
        resolved_path = None

        # If two_pass_uri provided, use it directly (from RAG metadata)
        if two_pass_uri:
            # two_pass_uri may be the source_uri which is "nextcloud://{path}"
            if two_pass_uri.startswith("nextcloud://"):
                resolved_path = two_pass_uri.replace("nextcloud://", "")
                logger.debug(f"Two-pass: Direct path resolution: {resolved_path}")

        # Otherwise, use fuzzy matching
        if not resolved_path and filename:
            # Build list of indexed files for fuzzy matching
            indexed_metadata = []
            for doc in self.list_documents():
                indexed_metadata.append(
                    {
                        "source_path": doc.metadata.get("path", ""),
                        "title": doc.title,
                        "name": os.path.basename(doc.metadata.get("path", "")),
                    }
                )

            # Use fuzzy matching to resolve filename
            resolved_path = self._resolve_document(filename, indexed_metadata)

        if not resolved_path:
            return LiveDataResult(
                success=False,
                error=f"Could not find file matching '{filename}'",
            )

        # Read the document
        content_result = self.read_document(f"nextcloud://{resolved_path}")
        if not content_result:
            return LiveDataResult(
                success=False,
                error=f"Failed to read file: {resolved_path}",
            )

        # Extract text content
        text_content = self._extract_text_content(content_result, resolved_path)

        formatted = f"### Full Content: {resolved_path}\n\n{text_content}"

        return LiveDataResult(
            success=True,
            data={"path": resolved_path, "content": text_content},
            formatted=formatted,
            cache_ttl=self.default_cache_ttl,
        )

    def _resolve_document(
        self, query: str, indexed_metadata: list[dict]
    ) -> Optional[str]:
        """Resolve filename query to actual path using fuzzy matching."""
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.warning("rapidfuzz not installed, using exact match")
            return self._exact_match_document(query, indexed_metadata)

        # Build candidates: {match_target: path}
        candidates: dict[str, str] = {}
        for meta in indexed_metadata:
            path = meta.get("source_path", "")
            if not path:
                continue

            # Use multiple match targets
            targets = [
                meta.get("title", ""),
                meta.get("name", ""),
                os.path.basename(path),
            ]
            for target in targets:
                if target:
                    target_lower = target.lower()
                    if target_lower not in candidates:
                        candidates[target_lower] = path

        if not candidates:
            return None

        # Fuzzy match - use partial_ratio for better substring matching
        query_lower = query.lower()
        match = process.extractOne(
            query_lower,
            candidates.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=70,
        )

        if match:
            matched_target, score, _ = match
            path = candidates[matched_target]
            logger.info(f"Resolved '{query}' -> '{path}' (score={score:.0f})")
            return path

        return None

    def _exact_match_document(
        self, query: str, indexed_metadata: list[dict]
    ) -> Optional[str]:
        """Fallback exact substring matching."""
        query_lower = query.lower()
        for meta in indexed_metadata:
            path = meta.get("source_path", "")
            if not path:
                continue
            targets = [
                meta.get("title", ""),
                meta.get("name", ""),
                os.path.basename(path),
            ]
            for target in targets:
                if target and (
                    query_lower in target.lower() or target.lower() in query_lower
                ):
                    return path
        return None

    def _extract_text_content(
        self, content_result: DocumentContent, file_path: str
    ) -> str:
        """Extract text from document content."""
        # If already text, return it
        if isinstance(content_result.content, str):
            return content_result.content

        # Binary content - need to process
        binary_content = content_result.content
        ext = Path(file_path).suffix.lower()
        mime_type = content_result.mime_type or ""

        try:
            # PDF
            if ext == ".pdf" or "pdf" in mime_type:
                return self._extract_pdf_text(binary_content)

            # DOCX
            if ext == ".docx" or "wordprocessingml" in mime_type:
                return self._extract_docx_text(binary_content)

            # Unknown binary
            return f"[Binary file: {ext}, {len(binary_content)} bytes]"

        except Exception as e:
            logger.warning(f"Failed to extract text from {file_path}: {e}")
            return f"[Failed to extract text: {e}]"

    def _extract_pdf_text(self, binary_content: bytes) -> str:
        """Extract text from PDF."""
        try:
            import io

            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(binary_content))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts) or "[No extractable text in PDF]"
        except ImportError:
            return "[pypdf not installed - cannot extract PDF text]"

    def _extract_docx_text(self, binary_content: bytes) -> str:
        """Extract text from DOCX."""
        try:
            import io

            from docx import Document

            doc = Document(io.BytesIO(binary_content))
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            return "[python-docx not installed - cannot extract DOCX text]"

    def _format_file_list(self, files: list[dict]) -> str:
        """Format file list for LLM context."""
        if not files:
            return "### Nextcloud Files\nNo files found."

        lines = ["### Nextcloud Files", f"Found {len(files)} file(s):\n"]
        for f in files:
            size_str = self._format_size(f.get("size", 0))
            lines.append(f"• **{f['name']}** ({size_str})")
            if f.get("modified"):
                lines.append(f"  Modified: {f['modified']}")
        return "\n".join(lines)

    def _format_size(self, size: int) -> str:
        """Format file size for display."""
        if size > 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} bytes"

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine RAG vs live lookup routing."""
        import re

        query_lower = query.lower()
        action = params.get("action", "")
        filename = params.get("filename", "")

        # Explicit lookup action -> Live only
        if action == "lookup" and filename:
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="Document lookup requested",
                max_live_results=1,
            )

        # Explicit list action -> Live only
        if action == "list":
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="File listing requested",
                max_live_results=params.get("max_results", 50),
            )

        # Detect document lookup intent from query
        # Two-pass: Use RAG to find relevant documents, then fetch full content
        lookup_patterns = [
            r"(full |entire |complete |whole )?content of",
            r"what('s| is| does) .* (say|contain)",
            r"show me .* (file|document)",
            r"read( me)? .* (file|document)",
            r"(can you )?(get|fetch|retrieve|pull) .* (file|document)",
        ]

        for pattern in lookup_patterns:
            if re.search(pattern, query_lower):
                # Use TWO_PASS: RAG finds documents, then live fetches full content
                return QueryAnalysis(
                    routing=QueryRouting.TWO_PASS,
                    rag_query=query,
                    live_params={"action": "lookup", "filename": query},
                    merge_strategy=MergeStrategy.LIVE_FIRST,
                    reason="Document lookup intent - using two-pass (RAG finds, live fetches)",
                    max_rag_results=5,  # Find top 5 relevant documents
                    max_live_results=3,  # Fetch full content of top 3
                    two_pass_fetch_full=True,  # Enable full document fetch in pass 2
                )

        # Default -> RAG only for semantic search
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="Semantic search - using RAG index",
            max_rag_results=20,
        )

    def is_available(self) -> bool:
        try:
            self._propfind("", depth=0)
            return True
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        try:
            items = self._propfind(self.folder_path or "", depth=1)
            folders = sum(1 for i in items if i["is_collection"])
            files = sum(1 for i in items if not i["is_collection"])
            results = [
                f"Nextcloud URL: {self.url}",
                f"WebDAV: {self.webdav_base}",
                f"Root contents: {folders} folder(s), {files} file(s)",
            ]
            return True, "\n".join(results)
        except Exception as e:
            return False, f"Connection failed: {e}"

    def close(self) -> None:
        if self._client:
            self._client.close()
