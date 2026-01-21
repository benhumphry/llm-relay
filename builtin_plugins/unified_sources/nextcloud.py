"""
Nextcloud Unified Source Plugin.

A RAG-only unified source for indexing files from Nextcloud via WebDAV.
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
from plugin_base.unified_source import PluginUnifiedSource, QueryAnalysis, QueryRouting

logger = logging.getLogger(__name__)


class NextcloudUnifiedSource(PluginUnifiedSource):
    """Nextcloud source - RAG-only for file storage via WebDAV."""

    source_type = "nextcloud"
    display_name = "Nextcloud"
    description = "Index files from Nextcloud via WebDAV"
    category = "storage"
    icon = "☁️"

    # Document store types this unified source handles
    handles_doc_source_types = ["nextcloud"]

    supports_rag = True
    supports_live = False
    supports_actions = False
    supports_incremental = True
    _abstract = False

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
        return []

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
        return LiveDataResult(
            success=False,
            error="Nextcloud source does not support live queries - use RAG search instead",
        )

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="Nextcloud source - RAG only",
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
