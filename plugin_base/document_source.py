"""
Base class for document source plugins.

Document sources enumerate and fetch documents for RAG indexing.
Content is chunked, embedded, and stored in ChromaDB.

Full implementation in Phase 4.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

from plugin_base.common import FieldDefinition, ValidationResult, validate_config


@dataclass
class DocumentInfo:
    """
    Metadata for a document returned from list_documents().

    This is used by the indexer to track documents and detect changes.
    """

    uri: str  # Unique identifier within source
    title: str
    modified_at: Optional[datetime] = None
    mime_type: str = "text/plain"
    size_bytes: Optional[int] = None
    metadata: dict = field(default_factory=dict)  # Source-specific metadata

    # Compatibility properties for indexer (expects legacy field names)
    @property
    def name(self) -> str:
        """Alias for title (legacy compatibility)."""
        return self.title

    @property
    def modified_time(self) -> Optional[str]:
        """Alias for modified_at as ISO string (legacy compatibility)."""
        if self.modified_at is None:
            return None
        if isinstance(self.modified_at, str):
            return self.modified_at
        return self.modified_at.isoformat()

    @property
    def size(self) -> Optional[int]:
        """Alias for size_bytes (legacy compatibility)."""
        return self.size_bytes


@dataclass
class DocumentContent:
    """
    Content returned from read_document().

    For text documents, content is the raw text.
    For binary documents (PDF, DOCX), content may be base64 or bytes,
    and the indexer will process it through Docling.
    """

    content: str | bytes
    mime_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)

    # For binary content that needs vision processing
    needs_vision: bool = False

    # Compatibility property for indexer (expects legacy field name)
    @property
    def text(self) -> str:
        """Alias for content (legacy compatibility)."""
        if isinstance(self.content, bytes):
            return self.content.decode("utf-8", errors="replace")
        return self.content


class PluginDocumentSource(ABC):
    """
    Base class for document source plugins.

    Document sources enumerate and fetch documents for RAG indexing.
    Content is chunked, embedded, and stored in ChromaDB.

    Subclasses define:
    - source_type: Unique identifier (e.g., "notion", "gdrive")
    - display_name: Human-readable name for admin UI
    - description: Help text for users
    - category: Plugin category ("oauth", "api_key", "local", "crawler")
    - get_config_fields(): Configuration requirements
    - list_documents(): Enumerate available documents
    - read_document(): Fetch document content

    The indexer calls list_documents() to discover content, then
    read_document() for each document to get content for embedding.
    """

    # --- Required class attributes (override in subclass) ---
    source_type: str  # Unique identifier
    display_name: str  # Shown in admin UI
    description: str  # Help text
    category: str  # "oauth", "api_key", "local", "crawler"

    # --- Optional class attributes ---
    icon: str = "📄"
    supports_incremental: bool = True  # Can detect changed documents

    # Mark as abstract to prevent direct registration
    _abstract: bool = True

    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration fields for admin UI."""
        pass

    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """Validate configuration before saving."""
        return validate_config(cls.get_config_fields(), config)

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with validated config."""
        pass

    @abstractmethod
    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate all documents in the source.

        Yields DocumentInfo for each document. The indexer uses this to:
        1. Discover new documents
        2. Detect changed documents (via modified_at)
        3. Remove deleted documents

        For large sources, this should be a generator that fetches pages lazily.
        """
        pass

    @abstractmethod
    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Fetch document content by URI.

        Args:
            uri: The URI from DocumentInfo.uri

        Returns:
            DocumentContent with text/binary content, or None if not found
        """
        pass

    def is_available(self) -> bool:
        """Check if source is configured and available."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test the connection by listing documents."""
        try:
            # Just try to get first document
            docs = list(self.list_documents())
            return True, f"Connected. Found {len(docs)} documents."
        except Exception as e:
            return False, str(e)
