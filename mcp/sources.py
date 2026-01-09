"""
Document source abstractions for Smart RAG.

Provides a unified interface for retrieving documents from:
- Local filesystem (existing functionality)
- MCP servers (new functionality)
"""

import fnmatch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)

# Supported file extensions (same as RAG indexer)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".asciidoc",
    ".adoc",
    # Images (OCR)
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}


@dataclass
class DocumentInfo:
    """Information about a document available for indexing."""

    uri: str  # Unique identifier (file path or MCP URI)
    name: str  # Display name
    mime_type: Optional[str] = None
    size: Optional[int] = None


@dataclass
class DocumentContent:
    """Content of a document."""

    uri: str
    name: str
    mime_type: Optional[str] = None
    text: Optional[str] = None  # Text content
    binary: Optional[bytes] = None  # Binary content (for PDFs, images, etc.)

    @property
    def is_binary(self) -> bool:
        """Check if this is binary content."""
        return self.binary is not None


class DocumentSource(ABC):
    """Abstract base class for document sources."""

    @abstractmethod
    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        List all available documents.

        Yields:
            DocumentInfo for each available document
        """
        pass

    @abstractmethod
    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read a specific document.

        Args:
            uri: Document URI

        Returns:
            DocumentContent or None if not found
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is available."""
        pass


class LocalDocumentSource(DocumentSource):
    """
    Document source for local filesystem.

    This wraps the existing local file functionality from the RAG indexer.
    """

    def __init__(self, source_path: str):
        """
        Initialize with a local directory path.

        Args:
            source_path: Path to directory containing documents
        """
        self.source_path = Path(source_path)

    def is_available(self) -> bool:
        """Check if the source path exists and is a directory."""
        return self.source_path.exists() and self.source_path.is_dir()

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all supported documents in the directory."""
        if not self.is_available():
            logger.warning(f"Source path not available: {self.source_path}")
            return

        for file_path in self.source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                relative_path = file_path.relative_to(self.source_path)
                yield DocumentInfo(
                    uri=str(file_path),
                    name=str(relative_path),
                    mime_type=self._get_mime_type(file_path.suffix),
                    size=file_path.stat().st_size,
                )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from the filesystem."""
        file_path = Path(uri)

        if not file_path.exists():
            logger.warning(f"File not found: {uri}")
            return None

        # Determine if binary or text
        suffix = file_path.suffix.lower()
        mime_type = self._get_mime_type(suffix)

        # Binary formats
        binary_formats = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

        if suffix in binary_formats:
            try:
                binary = file_path.read_bytes()
                return DocumentContent(
                    uri=uri,
                    name=file_path.name,
                    mime_type=mime_type,
                    binary=binary,
                )
            except Exception as e:
                logger.error(f"Failed to read binary file {uri}: {e}")
                return None
        else:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                return DocumentContent(
                    uri=uri,
                    name=file_path.name,
                    mime_type=mime_type,
                    text=text,
                )
            except Exception as e:
                logger.error(f"Failed to read text file {uri}: {e}")
                return None

    def _get_mime_type(self, suffix: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".html": "text/html",
            ".htm": "text/html",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".asciidoc": "text/asciidoc",
            ".adoc": "text/asciidoc",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".bmp": "image/bmp",
        }
        return mime_types.get(suffix.lower(), "application/octet-stream")


class MCPDocumentSource(DocumentSource):
    """
    Document source for MCP servers.

    Connects to an MCP server and retrieves resources for indexing.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize with MCP server configuration.

        Args:
            config: MCPServerConfig with connection details
        """
        self.config = config
        self._client: Optional[MCPClient] = None

    def is_available(self) -> bool:
        """Check if we can connect to the MCP server."""
        try:
            client = MCPClient(self.config)
            if client.connect():
                client.disconnect()
                return True
            return False
        except Exception as e:
            logger.warning(f"MCP server '{self.config.name}' not available: {e}")
            return False

    def _get_client(self) -> MCPClient:
        """Get or create a connected client."""
        if self._client is None:
            self._client = MCPClient(self.config)
            if not self._client.connect():
                raise RuntimeError(f"Failed to connect to MCP server '{self.config.name}'")
        return self._client

    def _close_client(self):
        """Close the client connection."""
        if self._client:
            self._client.disconnect()
            self._client = None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List resources from the MCP server."""
        try:
            client = self._get_client()
            resources = client.list_resources()

            for resource in resources:
                # Apply URI pattern filtering if configured
                if self.config.uri_patterns:
                    if not self._matches_patterns(resource.uri, self.config.uri_patterns):
                        continue

                # Check if it's a supported document type
                if not self._is_supported_resource(resource):
                    continue

                yield DocumentInfo(
                    uri=resource.uri,
                    name=resource.name or resource.uri,
                    mime_type=resource.mime_type,
                )

        except Exception as e:
            logger.error(f"Failed to list resources from MCP server: {e}")
        finally:
            self._close_client()

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a resource from the MCP server."""
        try:
            client = self._get_client()
            content = client.read_resource(uri)

            if not content:
                return None

            return DocumentContent(
                uri=uri,
                name=uri.split("/")[-1],
                mime_type=content.mime_type,
                text=content.text,
                binary=content.blob,
            )

        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            return None
        finally:
            self._close_client()

    def _matches_patterns(self, uri: str, patterns: list[str]) -> bool:
        """Check if a URI matches any of the configured patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(uri, pattern):
                return True
        return False

    def _is_supported_resource(self, resource) -> bool:
        """Check if a resource is a supported document type."""
        # Check MIME type first
        if resource.mime_type:
            supported_mimes = {
                "application/pdf",
                "text/plain",
                "text/markdown",
                "text/html",
                "image/png",
                "image/jpeg",
            }
            if any(resource.mime_type.startswith(m) for m in supported_mimes):
                return True

        # Fallback to checking URI extension
        uri_lower = resource.uri.lower()
        for ext in SUPPORTED_EXTENSIONS:
            if uri_lower.endswith(ext):
                return True

        # Google Docs exports (no extension, but MIME type indicates doc)
        if resource.mime_type and "google" in resource.mime_type.lower():
            return True

        return False


def get_document_source(
    source_type: str,
    source_path: Optional[str] = None,
    mcp_config: Optional[dict] = None,
) -> DocumentSource:
    """
    Factory function to create the appropriate document source.

    Args:
        source_type: "local" or "mcp"
        source_path: Path for local sources
        mcp_config: Configuration dict for MCP sources

    Returns:
        Appropriate DocumentSource instance
    """
    if source_type == "local":
        if not source_path:
            raise ValueError("source_path required for local document source")
        return LocalDocumentSource(source_path)

    elif source_type == "mcp":
        if not mcp_config:
            raise ValueError("mcp_config required for MCP document source")
        config = MCPServerConfig.from_dict(mcp_config)
        return MCPDocumentSource(config)

    else:
        raise ValueError(f"Unknown source type: {source_type}")
