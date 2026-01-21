"""
Local Filesystem Unified Source Plugin.

A RAG-only unified source for indexing local files mounted via Docker volumes.

Features:
- Document side: Index files from local directories
- Supports common document formats (text, markdown, PDF, Office, etc.)
- Configurable file filtering by extension
- No live queries (local filesystem doesn't need real-time API)
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import (
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class LocalFilesystemUnifiedSource(PluginUnifiedSource):
    """
    Local filesystem source - RAG-only for mounted directories.

    Indexes files from Docker-mounted directories for semantic search.
    No live queries needed since files are local.
    """

    source_type = "local"
    display_name = "Local Filesystem"
    description = "Index files from local directories (Docker volumes)"
    category = "storage"
    icon = "📁"

    # Document store types this unified source handles
    handles_doc_source_types = ["local"]

    supports_rag = True
    supports_live = False  # No live API needed for local files
    supports_actions = False
    supports_incremental = True

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "path": store.source_path or "",
            "file_extensions": "",  # Use defaults
            "exclude_patterns": "",
        }

    # Supported file extensions for indexing
    DEFAULT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".text",
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
        ".jsx",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".rb",
        ".php",
        ".sh",
        ".bash",
        ".sql",
        ".graphql",
        ".log",
        ".ini",
        ".conf",
        ".cfg",
    }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="path",
                label="Directory Path",
                field_type=FieldType.TEXT,
                required=True,
                help_text="Path to the directory to index (must be mounted in Docker)",
            ),
            FieldDefinition(
                name="recursive",
                label="Include Subdirectories",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Recursively index files in subdirectories",
            ),
            FieldDefinition(
                name="file_extensions",
                label="File Extensions",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated extensions to index (empty = all supported). Example: .txt,.md,.pdf",
            ),
            FieldDefinition(
                name="exclude_patterns",
                label="Exclude Patterns",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated patterns to exclude. Example: __pycache__,*.pyc,.git",
            ),
            FieldDefinition(
                name="max_file_size_mb",
                label="Max File Size (MB)",
                field_type=FieldType.INTEGER,
                default=10,
                help_text="Maximum file size to index in megabytes",
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
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """No live parameters - this is a RAG-only source."""
        return []

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.path = config.get("path", "").rstrip("/")
        self.recursive = config.get("recursive", True)
        self.max_file_size_mb = config.get("max_file_size_mb", 10)
        self.index_schedule = config.get("index_schedule", "")

        # Parse file extensions
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
            self.file_extensions = self.DEFAULT_EXTENSIONS

        # Parse exclude patterns
        exclude_str = config.get("exclude_patterns", "")
        self.exclude_patterns = (
            [p.strip() for p in exclude_str.split(",") if p.strip()]
            if exclude_str
            else []
        )

    def _should_exclude(self, path: str) -> bool:
        """Check if path matches any exclude pattern."""
        for pattern in self.exclude_patterns:
            if "*" in pattern:
                regex = pattern.replace(".", r"\.").replace("*", ".*")
                if re.search(regex, path):
                    return True
            else:
                if pattern in path:
                    return True
        return False

    def _get_mime_type(self, ext: str) -> str:
        """Get MIME type for file extension."""
        mime_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".rst": "text/x-rst",
            ".html": "text/html",
            ".htm": "text/html",
            ".xml": "application/xml",
            ".json": "application/json",
            ".yaml": "application/yaml",
            ".yml": "application/yaml",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".ts": "text/typescript",
            ".java": "text/x-java-source",
            ".go": "text/x-go",
            ".rs": "text/x-rust",
            ".cpp": "text/x-c++src",
            ".c": "text/x-csrc",
            ".h": "text/x-chdr",
        }
        return mime_map.get(ext, "application/octet-stream")

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate files for indexing."""
        if not os.path.isdir(self.path):
            logger.error(f"Directory not found: {self.path}")
            return

        logger.info(f"Listing local files in: {self.path}")
        total_files = 0
        max_size_bytes = self.max_file_size_mb * 1024 * 1024

        if self.recursive:
            for root, dirs, files in os.walk(self.path):
                dirs[:] = [
                    d for d in dirs if not self._should_exclude(os.path.join(root, d))
                ]
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if self._should_exclude(file_path):
                        continue
                    doc = self._process_file(file_path, max_size_bytes)
                    if doc:
                        total_files += 1
                        yield doc
        else:
            for entry in os.scandir(self.path):
                if entry.is_file():
                    if self._should_exclude(entry.path):
                        continue
                    doc = self._process_file(entry.path, max_size_bytes)
                    if doc:
                        total_files += 1
                        yield doc

        logger.info(f"Found {total_files} files to index")

    def _process_file(
        self, file_path: str, max_size_bytes: int
    ) -> Optional[DocumentInfo]:
        """Process a single file for listing."""
        try:
            stat = os.stat(file_path)
            if stat.st_size > max_size_bytes:
                return None
            ext = Path(file_path).suffix.lower()
            if ext not in self.file_extensions:
                return None
            rel_path = os.path.relpath(file_path, self.path)
            modified_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            mime_type = self._get_mime_type(ext)
            return DocumentInfo(
                uri=f"file://{file_path}",
                title=rel_path,
                mime_type=mime_type,
                modified_at=modified_at,
                metadata={
                    "path": file_path,
                    "relative_path": rel_path,
                    "size": stat.st_size,
                    "extension": ext,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to process file {file_path}: {e}")
            return None

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read file content for indexing."""
        if not uri.startswith("file://"):
            return None
        file_path = uri.replace("file://", "")
        if not os.path.isfile(file_path):
            return None
        real_path = os.path.realpath(file_path)
        real_base = os.path.realpath(self.path)
        if not real_path.startswith(real_base):
            logger.error(f"File outside configured path: {file_path}")
            return None
        try:
            stat = os.stat(file_path)
            rel_path = os.path.relpath(file_path, self.path)
            ext = Path(file_path).suffix.lower()
            mime_type = self._get_mime_type(ext)
            if mime_type.startswith("text/") or mime_type in (
                "application/json",
                "application/xml",
                "application/yaml",
            ):
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                with open(file_path, "rb") as f:
                    content = f.read()
            return DocumentContent(
                content=content,
                mime_type=mime_type,
                metadata={
                    "path": file_path,
                    "relative_path": rel_path,
                    "filename": os.path.basename(file_path),
                    "extension": ext,
                    "size": stat.st_size,
                    "source_type": "file",
                },
            )
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def fetch(self, params: dict) -> LiveDataResult:
        """Local filesystem doesn't support live queries."""
        return LiveDataResult(
            success=False,
            error="Local filesystem does not support live queries - use RAG search instead",
        )

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """All queries go to RAG for local filesystem."""
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="Local filesystem source - RAG only",
            max_rag_results=20,
        )

    def is_available(self) -> bool:
        """Check if directory is accessible."""
        return os.path.isdir(self.path)

    def test_connection(self) -> tuple[bool, str]:
        """Test filesystem access."""
        results = []
        if not os.path.isdir(self.path):
            return False, f"Directory not found: {self.path}"
        results.append(f"Directory: {self.path}")
        file_count = 0
        total_size = 0
        for doc in self.list_documents():
            file_count += 1
            total_size += doc.metadata.get("size", 0)
            if file_count >= 100:
                break
        results.append(
            f"Files: Found {file_count}{'+ ' if file_count >= 100 else ' '}file(s) to index"
        )
        if total_size > 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        elif total_size > 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size} bytes"
        results.append(f"Total size: {size_str}")
        return True, "\n".join(results)
