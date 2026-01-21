"""
Mock document source for testing the plugin loader.
"""

from typing import Iterator, Optional

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import (
    DocumentContent,
    DocumentInfo,
    PluginDocumentSource,
)


class MockDocumentSource(PluginDocumentSource):
    """Mock document source for testing."""

    source_type = "mock_docs"
    display_name = "Mock Documents"
    description = "Test document source for unit tests"
    category = "other"
    icon = "📝"

    # Override _abstract to allow registration
    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="doc_count",
                label="Document Count",
                field_type=FieldType.INTEGER,
                default=5,
                help_text="Number of mock documents to generate",
            ),
            FieldDefinition(
                name="prefix",
                label="Title Prefix",
                field_type=FieldType.TEXT,
                default="Doc",
            ),
        ]

    def __init__(self, config: dict):
        self.doc_count = config.get("doc_count", 5)
        self.prefix = config.get("prefix", "Doc")

    def list_documents(self) -> Iterator[DocumentInfo]:
        for i in range(self.doc_count):
            yield DocumentInfo(
                uri=f"doc_{i}",
                title=f"{self.prefix} {i}",
                mime_type="text/plain",
            )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        if uri.startswith("doc_"):
            return DocumentContent(
                content=f"Content of {uri}",
                mime_type="text/plain",
            )
        return None
