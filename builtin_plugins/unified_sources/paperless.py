"""
Paperless-ngx Unified Source Plugin.

A RAG-only unified source for indexing documents from Paperless-ngx.
"""

import logging
import os
from typing import Iterator, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import PluginUnifiedSource, QueryAnalysis, QueryRouting

logger = logging.getLogger(__name__)


class PaperlessUnifiedSource(PluginUnifiedSource):
    """Paperless-ngx source - RAG-only for document management."""

    source_type = "paperless"
    display_name = "Paperless-ngx"
    description = "Index documents from Paperless-ngx document management system"
    category = "documents"
    icon = "📄"

    # Document store types this unified source handles
    handles_doc_source_types = ["paperless"]

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
            "url": store.paperless_url or os.environ.get("PAPERLESS_URL", ""),
            "api_token": store.paperless_token or os.environ.get("PAPERLESS_TOKEN", ""),
            "tag_ids": str(store.paperless_tag_id) if store.paperless_tag_id else "",
        }

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="url",
                label="Paperless URL",
                field_type=FieldType.TEXT,
                required=True,
                help_text="URL of your Paperless-ngx instance",
            ),
            FieldDefinition(
                name="api_token",
                label="API Token",
                field_type=FieldType.PASSWORD,
                required=False,
                help_text="Paperless API token (or set PAPERLESS_TOKEN env var)",
            ),
            FieldDefinition(
                name="tag_ids",
                label="Tag IDs",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated tag IDs to filter documents",
            ),
            FieldDefinition(
                name="document_type_ids",
                label="Document Type IDs",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated document type IDs to filter",
            ),
            FieldDefinition(
                name="correspondent_ids",
                label="Correspondent IDs",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated correspondent IDs to filter",
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
                    {"value": "0 0 * * *", "label": "Daily"},
                ],
                help_text="How often to re-index documents",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        return []

    def __init__(self, config: dict):
        self.url = config.get("url", "").rstrip("/")
        self.api_token = config.get("api_token") or os.environ.get(
            "PAPERLESS_TOKEN", ""
        )
        self.index_schedule = config.get("index_schedule", "")
        self.tag_ids = self._parse_ids(config.get("tag_ids", ""))
        self.document_type_ids = self._parse_ids(config.get("document_type_ids", ""))
        self.correspondent_ids = self._parse_ids(config.get("correspondent_ids", ""))
        self._client = httpx.Client(timeout=60)
        self._tags_cache: dict[int, str] = {}
        self._types_cache: dict[int, str] = {}
        self._correspondents_cache: dict[int, str] = {}

    def _parse_ids(self, ids_str: str) -> list[int]:
        if not ids_str:
            return []
        return [int(i.strip()) for i in ids_str.split(",") if i.strip().isdigit()]

    def _get_headers(self) -> dict:
        return {"Authorization": f"Token {self.api_token}"} if self.api_token else {}

    def _api_request(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.url}/api{endpoint}"
        response = self._client.get(
            url, headers=self._get_headers(), params=params or {}
        )
        response.raise_for_status()
        return response.json()

    def _load_lookups(self) -> None:
        try:
            for tag in self._api_request("/tags/").get("results", []):
                self._tags_cache[tag["id"]] = tag["name"]
            for dtype in self._api_request("/document_types/").get("results", []):
                self._types_cache[dtype["id"]] = dtype["name"]
            for corr in self._api_request("/correspondents/").get("results", []):
                self._correspondents_cache[corr["id"]] = corr["name"]
        except Exception as e:
            logger.warning(f"Failed to load Paperless lookups: {e}")

    def list_documents(self) -> Iterator[DocumentInfo]:
        logger.info(f"Listing Paperless documents from: {self.url}")
        self._load_lookups()
        page = 1
        while True:
            params = {"page": page, "page_size": 100}
            if self.tag_ids:
                params["tags__id__in"] = ",".join(str(i) for i in self.tag_ids)
            if self.document_type_ids:
                params["document_type__id__in"] = ",".join(
                    str(i) for i in self.document_type_ids
                )
            if self.correspondent_ids:
                params["correspondent__id__in"] = ",".join(
                    str(i) for i in self.correspondent_ids
                )
            try:
                data = self._api_request("/documents/", params)
                documents = data.get("results", [])
                if not documents:
                    break
                for doc in documents:
                    doc_id = doc.get("id")
                    title = doc.get("title", f"Document {doc_id}")
                    tag_names = [
                        self._tags_cache.get(t, str(t)) for t in doc.get("tags", [])
                    ]
                    dtype_name = (
                        self._types_cache.get(doc.get("document_type"), "")
                        if doc.get("document_type")
                        else ""
                    )
                    corr_name = (
                        self._correspondents_cache.get(doc.get("correspondent"), "")
                        if doc.get("correspondent")
                        else ""
                    )
                    yield DocumentInfo(
                        uri=f"paperless://{doc_id}",
                        title=title[:100],
                        mime_type="text/plain",
                        modified_at=doc.get("modified", doc.get("created", "")),
                        metadata={
                            "document_id": doc_id,
                            "tags": tag_names,
                            "document_type": dtype_name,
                            "correspondent": corr_name,
                        },
                    )
                if not data.get("next"):
                    break
                page += 1
            except Exception as e:
                logger.error(f"Failed to list Paperless documents: {e}")
                break

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        if not uri.startswith("paperless://"):
            return None
        doc_id = uri.replace("paperless://", "")
        try:
            doc = self._api_request(f"/documents/{doc_id}/")
            title = doc.get("title", f"Document {doc_id}")
            content = doc.get("content", "")
            tag_names = [self._tags_cache.get(t, str(t)) for t in doc.get("tags", [])]
            dtype_name = (
                self._types_cache.get(doc.get("document_type"), "")
                if doc.get("document_type")
                else ""
            )
            corr_name = (
                self._correspondents_cache.get(doc.get("correspondent"), "")
                if doc.get("correspondent")
                else ""
            )
            full_content = f"# {title}\n\nDocument Type: {dtype_name or 'None'}\nCorrespondent: {corr_name or 'None'}\nTags: {', '.join(tag_names) if tag_names else 'None'}\nCreated: {doc.get('created', '')}\n\n---\n\n{content}"
            return DocumentContent(
                content=full_content,
                mime_type="text/plain",
                metadata={
                    "document_id": doc_id,
                    "title": title,
                    "tags": tag_names,
                    "document_type": dtype_name,
                    "correspondent": corr_name,
                    "source_type": "document",
                },
            )
        except Exception as e:
            logger.error(f"Failed to read Paperless document {doc_id}: {e}")
            return None

    def fetch(self, params: dict) -> LiveDataResult:
        return LiveDataResult(
            success=False,
            error="Paperless source does not support live queries - use RAG search instead",
        )

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        return QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            rag_query=query,
            reason="Paperless source - RAG only",
            max_rag_results=20,
        )

    def is_available(self) -> bool:
        try:
            self._api_request("/documents/", {"page_size": 1})
            return True
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        try:
            data = self._api_request("/documents/", {"page_size": 1})
            total = data.get("count", 0)
            self._load_lookups()
            results = [
                f"Paperless URL: {self.url}",
                f"Total documents: {total}",
                f"Tags: {len(self._tags_cache)}",
                f"Document types: {len(self._types_cache)}",
                f"Correspondents: {len(self._correspondents_cache)}",
            ]
            return True, "\n".join(results)
        except Exception as e:
            return False, f"Connection failed: {e}"

    def close(self) -> None:
        if self._client:
            self._client.close()
