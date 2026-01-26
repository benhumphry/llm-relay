"""
Paperless-ngx Unified Source Plugin.

A unified source for indexing documents from Paperless-ngx with live document lookup.

Features:
- Document side: Index documents for RAG semantic search
- Live side: Fetch full document content by title, list documents, search
- Fuzzy title matching for document resolution
"""

import logging
import os
from typing import Iterator, Optional

import httpx

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class PaperlessUnifiedSource(PluginUnifiedSource):
    """Paperless-ngx source with live document lookup."""

    source_type = "paperless"
    display_name = "Paperless-ngx"
    description = "Index documents from Paperless-ngx with full document lookup"
    category = "documents"
    icon = "📄"
    content_category = ContentCategory.FILES

    # Document store types this unified source handles
    handles_doc_source_types = ["paperless"]

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
            "Paperless document management with lookup. Use action='lookup' with title "
            "param when user asks for FULL CONTENT of a specific document. "
            "Use action='list' to enumerate documents. Use action='search' with query "
            "to search by content. For semantic search, RAG will be used automatically."
        )

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
                required=False,
                help_text="URL of your Paperless-ngx instance",
                env_var="PAPERLESS_URL",
            ),
            FieldDefinition(
                name="api_token",
                label="Paperless API Token",
                field_type=FieldType.PASSWORD,
                required=False,
                help_text="Paperless API token (leave empty to use env var)",
                env_var="PAPERLESS_TOKEN",
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
        """Parameters for live document lookup."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: lookup (full document), list (enumerate), search (by content)",
                param_type="string",
                required=False,
                default="lookup",
                examples=["lookup", "list", "search"],
            ),
            ParamDefinition(
                name="title",
                description="Document title or partial title to lookup (for action=lookup)",
                param_type="string",
                required=False,
                examples=["invoice", "contract", "receipt 2024"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for content search (for action=search)",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum documents to return",
                param_type="integer",
                required=False,
                default=50,
            ),
        ]

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
        """Fetch full document content or list/search documents."""
        action = params.get("action", "lookup")
        title = params.get("title", "")
        query = params.get("query", "")
        max_results = params.get("max_results", 50)
        two_pass_uri = params.get("_two_pass_uri")  # Direct URI from RAG metadata

        try:
            if action == "list":
                return self._fetch_list(max_results)
            elif action == "search" and query:
                return self._fetch_search(query, max_results)
            elif action == "lookup" and (title or two_pass_uri):
                return self._fetch_document(title, two_pass_uri)
            else:
                return LiveDataResult(
                    success=False,
                    error="Specify action='lookup' with title, action='search' with query, or action='list'",
                )
        except Exception as e:
            logger.error(f"Paperless fetch error: {e}")
            return LiveDataResult(success=False, error=str(e))

    def _fetch_list(self, max_results: int) -> LiveDataResult:
        """List documents."""
        self._load_lookups()
        documents = []
        page = 1

        while len(documents) < max_results:
            params = {"page": page, "page_size": min(100, max_results - len(documents))}
            if self.tag_ids:
                params["tags__id__in"] = ",".join(str(i) for i in self.tag_ids)

            try:
                data = self._api_request("/documents/", params)
                results = data.get("results", [])
                if not results:
                    break

                for doc in results:
                    tag_names = [
                        self._tags_cache.get(t, str(t)) for t in doc.get("tags", [])
                    ]
                    documents.append(
                        {
                            "id": doc.get("id"),
                            "title": doc.get("title", f"Document {doc.get('id')}"),
                            "correspondent": self._correspondents_cache.get(
                                doc.get("correspondent"), ""
                            )
                            if doc.get("correspondent")
                            else "",
                            "document_type": self._types_cache.get(
                                doc.get("document_type"), ""
                            )
                            if doc.get("document_type")
                            else "",
                            "tags": tag_names,
                            "created": doc.get("created", ""),
                        }
                    )

                if not data.get("next") or len(documents) >= max_results:
                    break
                page += 1
            except Exception as e:
                logger.error(f"Failed to list Paperless documents: {e}")
                break

        formatted = self._format_document_list(documents)
        return LiveDataResult(
            success=True,
            data=documents,
            formatted=formatted,
            cache_ttl=self.default_cache_ttl,
        )

    def _fetch_search(self, query: str, max_results: int) -> LiveDataResult:
        """Search documents by content."""
        self._load_lookups()
        params = {"query": query, "page_size": min(100, max_results)}
        if self.tag_ids:
            params["tags__id__in"] = ",".join(str(i) for i in self.tag_ids)

        try:
            data = self._api_request("/documents/", params)
            results = data.get("results", [])

            documents = []
            for doc in results[:max_results]:
                tag_names = [
                    self._tags_cache.get(t, str(t)) for t in doc.get("tags", [])
                ]
                documents.append(
                    {
                        "id": doc.get("id"),
                        "title": doc.get("title", f"Document {doc.get('id')}"),
                        "correspondent": self._correspondents_cache.get(
                            doc.get("correspondent"), ""
                        )
                        if doc.get("correspondent")
                        else "",
                        "tags": tag_names,
                        "created": doc.get("created", ""),
                    }
                )

            formatted = self._format_search_results(documents, query)
            return LiveDataResult(
                success=True,
                data=documents,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )
        except Exception as e:
            return LiveDataResult(success=False, error=str(e))

    def _fetch_document(self, title: str, two_pass_uri: str = None) -> LiveDataResult:
        """
        Fetch full document content by title.

        Args:
            title: Document title or partial title to search for
            two_pass_uri: Direct document URI from RAG metadata (two-pass retrieval)
        """
        self._load_lookups()
        doc_id = None

        # If two_pass_uri provided, extract doc_id directly (from RAG metadata)
        if two_pass_uri:
            # two_pass_uri is the source_uri which is "paperless://{id}"
            if two_pass_uri.startswith("paperless://"):
                doc_id = two_pass_uri.replace("paperless://", "")
                logger.debug(f"Two-pass: Direct doc_id resolution: {doc_id}")

        # Otherwise, use fuzzy matching
        if not doc_id and title:
            # Build list of documents for fuzzy matching
            indexed_metadata = []
            for doc_info in self.list_documents():
                indexed_metadata.append(
                    {
                        "doc_id": doc_info.uri.replace("paperless://", ""),
                        "title": doc_info.title,
                    }
                )

            # Resolve title to document ID
            doc_id = self._resolve_document(title, indexed_metadata)

        if not doc_id and title:
            # Try API search as fallback
            try:
                data = self._api_request(
                    "/documents/", {"query": title, "page_size": 1}
                )
                results = data.get("results", [])
                if results:
                    doc_id = str(results[0].get("id"))
            except Exception:
                pass

        if not doc_id:
            return LiveDataResult(
                success=False,
                error=f"Could not find document matching '{title}'",
            )

        # Read the document
        content_result = self.read_document(f"paperless://{doc_id}")
        if not content_result:
            return LiveDataResult(
                success=False,
                error=f"Failed to read document: {doc_id}",
            )

        formatted = f"### Full Document: {content_result.metadata.get('title', doc_id)}\n\n{content_result.content}"

        return LiveDataResult(
            success=True,
            data={"id": doc_id, "content": content_result.content},
            formatted=formatted,
            cache_ttl=self.default_cache_ttl,
        )

    def _resolve_document(
        self, query: str, indexed_metadata: list[dict]
    ) -> Optional[str]:
        """Resolve title query to document ID using fuzzy matching."""
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.warning("rapidfuzz not installed, using exact match")
            return self._exact_match_document(query, indexed_metadata)

        # Build candidates: {title: doc_id}
        candidates: dict[str, str] = {}
        for meta in indexed_metadata:
            doc_id = meta.get("doc_id", "")
            title = meta.get("title", "")
            if doc_id and title:
                candidates[title.lower()] = doc_id

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
            matched_title, score, _ = match
            doc_id = candidates[matched_title]
            logger.info(f"Resolved '{query}' -> doc {doc_id} (score={score:.0f})")
            return doc_id

        return None

    def _exact_match_document(
        self, query: str, indexed_metadata: list[dict]
    ) -> Optional[str]:
        """Fallback exact substring matching."""
        query_lower = query.lower()
        for meta in indexed_metadata:
            title = meta.get("title", "")
            if title and (query_lower in title.lower() or title.lower() in query_lower):
                return meta.get("doc_id")
        return None

    def _format_document_list(self, documents: list[dict]) -> str:
        """Format document list for LLM context."""
        if not documents:
            return "### Paperless Documents\nNo documents found."

        lines = ["### Paperless Documents", f"Found {len(documents)} document(s):\n"]
        for doc in documents:
            lines.append(f"• **{doc['title']}**")
            if doc.get("correspondent"):
                lines.append(f"  Correspondent: {doc['correspondent']}")
            if doc.get("document_type"):
                lines.append(f"  Type: {doc['document_type']}")
            if doc.get("tags"):
                lines.append(f"  Tags: {', '.join(doc['tags'])}")
            if doc.get("created"):
                lines.append(f"  Created: {doc['created']}")
            lines.append("")
        return "\n".join(lines)

    def _format_search_results(self, documents: list[dict], query: str) -> str:
        """Format search results for LLM context."""
        if not documents:
            return f"### Paperless Search: '{query}'\nNo documents found."

        lines = [
            f"### Paperless Search: '{query}'",
            f"Found {len(documents)} document(s):\n",
        ]
        for doc in documents:
            lines.append(f"• **{doc['title']}**")
            if doc.get("correspondent"):
                lines.append(f"  Correspondent: {doc['correspondent']}")
            if doc.get("tags"):
                lines.append(f"  Tags: {', '.join(doc['tags'])}")
            lines.append("")
        return "\n".join(lines)

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine RAG vs live lookup routing."""
        import re

        query_lower = query.lower()
        action = params.get("action", "")
        title = params.get("title", "")

        # Explicit lookup action -> Live only
        if action == "lookup" and title:
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
                reason="Document listing requested",
                max_live_results=params.get("max_results", 50),
            )

        # Explicit search action -> Live only
        if action == "search" and params.get("query"):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="Content search requested",
                max_live_results=params.get("max_results", 50),
            )

        # Detect document lookup intent from query
        # Two-pass: Use RAG to find relevant documents, then fetch full content
        lookup_patterns = [
            r"(full |entire |complete |whole )?content of",
            r"what('s| is| does) .* (say|contain)",
            r"show me .* (document|file|invoice|receipt|contract)",
            r"read( me)? .* (document|file)",
            r"(can you )?(get|fetch|retrieve|pull) .* (document|file)",
        ]

        for pattern in lookup_patterns:
            if re.search(pattern, query_lower):
                # Use TWO_PASS: RAG finds documents, then live fetches full content
                return QueryAnalysis(
                    routing=QueryRouting.TWO_PASS,
                    rag_query=query,
                    live_params={"action": "lookup", "title": query},
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
