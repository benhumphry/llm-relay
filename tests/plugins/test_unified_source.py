"""
Unit tests for the unified source plugin base class.

Tests cover:
- QueryAnalysis dataclass
- QueryRouting and MergeStrategy enums
- UnifiedResult dataclass
- PluginUnifiedSource base class methods
- Query routing logic
- Result merging strategies
"""

from typing import Iterator, Optional

import pytest

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
    UnifiedResult,
)

# =============================================================================
# Mock Implementation for Testing
# =============================================================================


class MockUnifiedSource(PluginUnifiedSource):
    """Mock unified source for testing."""

    source_type = "mock_unified"
    display_name = "Mock Unified"
    description = "Mock unified source for testing"
    category = "test"
    icon = "🧪"

    supports_rag = True
    supports_live = True
    supports_actions = False

    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="api_key",
                label="API Key",
                field_type=FieldType.PASSWORD,
                required=True,
            ),
            FieldDefinition(
                name="default_limit",
                label="Default Limit",
                field_type=FieldType.INTEGER,
                default=10,
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="query",
                description="Search query",
                param_type="string",
                required=True,
                examples=["search term", "keyword"],
            ),
            ParamDefinition(
                name="limit",
                description="Maximum results",
                param_type="integer",
                required=False,
                default=10,
            ),
        ]

    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.default_limit = config.get("default_limit", 10)
        self._documents = [
            DocumentInfo(uri="doc1", title="Document 1"),
            DocumentInfo(uri="doc2", title="Document 2"),
            DocumentInfo(uri="doc3", title="Document 3"),
        ]
        self._live_data = "Live result data"

    def list_documents(self) -> Iterator[DocumentInfo]:
        for doc in self._documents:
            yield doc

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        for doc in self._documents:
            if doc.uri == uri:
                return DocumentContent(content=f"Content of {uri}")
        return None

    def fetch(self, params: dict) -> LiveDataResult:
        query = params.get("query", "")
        return LiveDataResult(
            success=True,
            data={"query": query},
            formatted=f"Live results for: {query}" if query else self._live_data,
        )

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        query_lower = query.lower()

        # Very recent → live only
        if any(t in query_lower for t in ["last hour", "just now", "right now"]):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="Very recent time reference",
            )

        # Historical → RAG only
        if any(t in query_lower for t in ["last year", "2023", "2022", "historical"]):
            return QueryAnalysis(
                routing=QueryRouting.RAG_ONLY,
                rag_query=query,
                reason="Historical time reference",
            )

        # Latest/recent → both, prefer live
        if any(t in query_lower for t in ["latest", "recent", "new"]):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params=params,
                merge_strategy=MergeStrategy.LIVE_FIRST,
                freshness_priority=True,
                reason="Freshness keyword detected",
            )

        # Default → both, dedupe
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.DEDUPE,
            reason="Default routing",
        )


class RagOnlyUnifiedSource(MockUnifiedSource):
    """Unified source that only supports RAG."""

    source_type = "rag_only_unified"
    supports_rag = True
    supports_live = False
    _abstract = False


class LiveOnlyUnifiedSource(MockUnifiedSource):
    """Unified source that only supports live queries."""

    source_type = "live_only_unified"
    supports_rag = False
    supports_live = True
    _abstract = False


# =============================================================================
# QueryAnalysis Tests
# =============================================================================


class TestQueryAnalysis:
    """Tests for QueryAnalysis dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        analysis = QueryAnalysis()
        assert analysis.routing == QueryRouting.BOTH_MERGE
        assert analysis.merge_strategy == MergeStrategy.DEDUPE
        assert analysis.rag_query == ""
        assert analysis.live_params == {}
        assert analysis.reason == ""
        assert analysis.confidence == 1.0
        assert analysis.max_rag_results == 10
        assert analysis.max_live_results == 10
        assert analysis.freshness_priority is False

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        analysis = QueryAnalysis(
            routing=QueryRouting.LIVE_ONLY,
            merge_strategy=MergeStrategy.LIVE_FIRST,
            rag_query="test query",
            live_params={"key": "value"},
            reason="Testing",
            confidence=0.8,
            max_rag_results=5,
            freshness_priority=True,
        )
        assert analysis.routing == QueryRouting.LIVE_ONLY
        assert analysis.merge_strategy == MergeStrategy.LIVE_FIRST
        assert analysis.rag_query == "test query"
        assert analysis.live_params == {"key": "value"}
        assert analysis.reason == "Testing"
        assert analysis.confidence == 0.8
        assert analysis.max_rag_results == 5
        assert analysis.freshness_priority is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        analysis = QueryAnalysis(
            routing=QueryRouting.RAG_ONLY,
            merge_strategy=MergeStrategy.RAG_FIRST,
            rag_query="search",
            live_params={"limit": 5},
            reason="Test reason",
            confidence=0.9,
        )
        d = analysis.to_dict()
        assert d["routing"] == "rag_only"
        assert d["merge_strategy"] == "rag_first"
        assert d["rag_query"] == "search"
        assert d["live_params"] == {"limit": 5}
        assert d["reason"] == "Test reason"
        assert d["confidence"] == 0.9


# =============================================================================
# QueryRouting Tests
# =============================================================================


class TestQueryRouting:
    """Tests for QueryRouting enum."""

    def test_all_values_exist(self):
        """Test all expected routing values exist."""
        assert QueryRouting.RAG_ONLY.value == "rag_only"
        assert QueryRouting.LIVE_ONLY.value == "live_only"
        assert QueryRouting.RAG_THEN_LIVE.value == "rag_then_live"
        assert QueryRouting.LIVE_THEN_RAG.value == "live_then_rag"
        assert QueryRouting.BOTH_MERGE.value == "both_merge"

    def test_enum_count(self):
        """Test we have the expected number of routing options."""
        assert len(QueryRouting) == 5


# =============================================================================
# MergeStrategy Tests
# =============================================================================


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_all_values_exist(self):
        """Test all expected merge strategies exist."""
        assert MergeStrategy.DEDUPE.value == "dedupe"
        assert MergeStrategy.RAG_FIRST.value == "rag_first"
        assert MergeStrategy.LIVE_FIRST.value == "live_first"
        assert MergeStrategy.INTERLEAVE.value == "interleave"

    def test_enum_count(self):
        """Test we have the expected number of merge strategies."""
        assert len(MergeStrategy) == 4


# =============================================================================
# UnifiedResult Tests
# =============================================================================


class TestUnifiedResult:
    """Tests for UnifiedResult dataclass."""

    def test_success_result(self):
        """Test successful result creation."""
        result = UnifiedResult(
            success=True,
            formatted="Combined results",
            rag_results=["rag1", "rag2"],
            live_results=["live1"],
            routing_used=QueryRouting.BOTH_MERGE,
            merge_strategy_used=MergeStrategy.DEDUPE,
            rag_count=2,
            live_count=1,
        )
        assert result.success is True
        assert result.formatted == "Combined results"
        assert len(result.rag_results) == 2
        assert len(result.live_results) == 1
        assert result.error is None

    def test_failure_result(self):
        """Test failure result creation."""
        result = UnifiedResult(
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"
        assert result.formatted == ""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = UnifiedResult(success=True)
        assert result.rag_results == []
        assert result.live_results == []
        assert result.rag_count == 0
        assert result.live_count == 0
        assert result.dedupe_count == 0
        assert result.rag_time_ms == 0
        assert result.live_time_ms == 0
        assert result.total_time_ms == 0
        assert result.cache_ttl == 300


# =============================================================================
# PluginUnifiedSource Base Class Tests
# =============================================================================


class TestPluginUnifiedSourceConfig:
    """Tests for unified source configuration."""

    def test_get_config_fields(self):
        """Test config fields are returned correctly."""
        fields = MockUnifiedSource.get_config_fields()
        assert len(fields) == 2
        assert fields[0].name == "api_key"
        assert fields[0].required is True
        assert fields[1].name == "default_limit"
        assert fields[1].default == 10

    def test_get_live_params(self):
        """Test live params are returned correctly."""
        params = MockUnifiedSource.get_live_params()
        assert len(params) == 2
        assert params[0].name == "query"
        assert params[0].required is True
        assert params[1].name == "limit"
        assert params[1].required is False

    def test_validate_config_success(self):
        """Test valid config passes validation."""
        result = MockUnifiedSource.validate_config({"api_key": "secret"})
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_config_missing_required(self):
        """Test missing required field fails validation."""
        result = MockUnifiedSource.validate_config({})
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "api_key"

    def test_get_designator_hint(self):
        """Test designator hint generation."""
        hint = MockUnifiedSource.get_designator_hint()
        assert "query" in hint
        assert "limit" in hint
        assert "historical search" in hint
        assert "real-time" in hint


class TestPluginUnifiedSourceDocument:
    """Tests for document-side functionality."""

    def test_list_documents(self):
        """Test listing documents."""
        source = MockUnifiedSource({"api_key": "test"})
        docs = list(source.list_documents())
        assert len(docs) == 3
        assert docs[0].uri == "doc1"
        assert docs[1].title == "Document 2"

    def test_read_document_exists(self):
        """Test reading existing document."""
        source = MockUnifiedSource({"api_key": "test"})
        content = source.read_document("doc1")
        assert content is not None
        assert "doc1" in content.content

    def test_read_document_not_found(self):
        """Test reading non-existent document."""
        source = MockUnifiedSource({"api_key": "test"})
        content = source.read_document("nonexistent")
        assert content is None


class TestPluginUnifiedSourceLive:
    """Tests for live-side functionality."""

    def test_fetch_with_query(self):
        """Test fetching with query parameter."""
        source = MockUnifiedSource({"api_key": "test"})
        result = source.fetch({"query": "test search"})
        assert result.success is True
        assert "test search" in result.formatted

    def test_fetch_without_query(self):
        """Test fetching without query parameter."""
        source = MockUnifiedSource({"api_key": "test"})
        result = source.fetch({})
        assert result.success is True
        assert result.formatted == "Live result data"


class TestPluginUnifiedSourceRouting:
    """Tests for query analysis and routing."""

    def test_analyze_query_live_only(self):
        """Test routing to live only for recent queries."""
        source = MockUnifiedSource({"api_key": "test"})
        analysis = source.analyze_query("what happened in the last hour", {})
        assert analysis.routing == QueryRouting.LIVE_ONLY
        assert "recent" in analysis.reason.lower()

    def test_analyze_query_rag_only(self):
        """Test routing to RAG only for historical queries."""
        source = MockUnifiedSource({"api_key": "test"})
        analysis = source.analyze_query("emails from 2023", {})
        assert analysis.routing == QueryRouting.RAG_ONLY
        assert "historical" in analysis.reason.lower()

    def test_analyze_query_both_with_freshness(self):
        """Test routing to both for 'latest' queries."""
        source = MockUnifiedSource({"api_key": "test"})
        analysis = source.analyze_query("show me the latest updates", {})
        assert analysis.routing == QueryRouting.BOTH_MERGE
        assert analysis.merge_strategy == MergeStrategy.LIVE_FIRST
        assert analysis.freshness_priority is True

    def test_analyze_query_default(self):
        """Test default routing for general queries."""
        source = MockUnifiedSource({"api_key": "test"})
        analysis = source.analyze_query("find documents about testing", {})
        assert analysis.routing == QueryRouting.BOTH_MERGE
        assert analysis.merge_strategy == MergeStrategy.DEDUPE


class TestPluginUnifiedSourceQuery:
    """Tests for the main query() method."""

    def test_query_live_only(self):
        """Test query with live-only routing."""
        source = MockUnifiedSource({"api_key": "test"})
        result = source.query(
            "what happened just now",
            {"query": "recent events"},
        )
        assert result.success is True
        assert result.routing_used == QueryRouting.LIVE_ONLY
        assert len(result.live_results) > 0
        assert len(result.rag_results) == 0

    def test_query_rag_only_with_search_fn(self):
        """Test query with RAG-only routing and search function."""
        source = MockUnifiedSource({"api_key": "test"})

        def mock_rag_search(query: str, limit: int) -> list[str]:
            return [f"RAG result 1 for {query}", f"RAG result 2 for {query}"]

        result = source.query(
            "historical data from 2023",
            {},
            rag_search_fn=mock_rag_search,
        )
        assert result.success is True
        assert result.routing_used == QueryRouting.RAG_ONLY
        assert len(result.rag_results) == 2
        assert len(result.live_results) == 0

    def test_query_rag_only_without_search_fn(self):
        """Test query with RAG-only routing but no search function."""
        source = MockUnifiedSource({"api_key": "test"})
        result = source.query(
            "historical data from 2023",
            {},
            rag_search_fn=None,
        )
        # Should fail because RAG was needed but not available
        assert result.success is False

    def test_query_both_merge(self):
        """Test query with both sources merged."""
        source = MockUnifiedSource({"api_key": "test"})

        def mock_rag_search(query: str, limit: int) -> list[str]:
            return ["RAG result 1", "RAG result 2"]

        result = source.query(
            "find all documents",
            {"query": "documents"},
            rag_search_fn=mock_rag_search,
        )
        assert result.success is True
        assert result.routing_used == QueryRouting.BOTH_MERGE
        assert len(result.rag_results) == 2
        assert len(result.live_results) == 1
        assert result.rag_count == 2
        assert result.live_count == 1

    def test_query_timing(self):
        """Test that timing information is recorded."""
        source = MockUnifiedSource({"api_key": "test"})

        def mock_rag_search(query: str, limit: int) -> list[str]:
            return ["result"]

        result = source.query(
            "test query",
            {"query": "test"},
            rag_search_fn=mock_rag_search,
        )
        assert result.total_time_ms >= 0
        # Both should have been queried
        assert result.rag_time_ms >= 0
        assert result.live_time_ms >= 0


class TestPluginUnifiedSourceMerge:
    """Tests for result merging strategies."""

    def test_merge_rag_first(self):
        """Test RAG_FIRST merge strategy."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            ["rag1", "rag2"],
            ["live1", "live2"],
            MergeStrategy.RAG_FIRST,
        )
        assert "rag1" in merged
        assert merged.index("rag1") < merged.index("live1")
        assert dedupe_count == 0

    def test_merge_live_first(self):
        """Test LIVE_FIRST merge strategy."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            ["rag1", "rag2"],
            ["live1", "live2"],
            MergeStrategy.LIVE_FIRST,
        )
        assert "live1" in merged
        assert merged.index("live1") < merged.index("rag1")
        assert dedupe_count == 0

    def test_merge_interleave(self):
        """Test INTERLEAVE merge strategy."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            ["rag1", "rag2"],
            ["live1", "live2"],
            MergeStrategy.INTERLEAVE,
        )
        # Should interleave: live1, rag1, live2, rag2
        parts = merged.split("\n\n")
        assert parts[0] == "live1"
        assert parts[1] == "rag1"
        assert dedupe_count == 0

    def test_merge_dedupe(self):
        """Test DEDUPE merge strategy removes duplicates."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            ["unique rag result", "duplicate content"],
            ["unique live result", "Duplicate Content"],  # Same normalized
            MergeStrategy.DEDUPE,
        )
        assert dedupe_count == 1  # One duplicate removed

    def test_merge_empty_rag(self):
        """Test merging with empty RAG results."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            [],
            ["live1", "live2"],
            MergeStrategy.DEDUPE,
        )
        assert "live1" in merged
        assert "live2" in merged

    def test_merge_empty_live(self):
        """Test merging with empty live results."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            ["rag1", "rag2"],
            [],
            MergeStrategy.DEDUPE,
        )
        assert "rag1" in merged
        assert "rag2" in merged

    def test_merge_both_empty(self):
        """Test merging with both sources empty."""
        source = MockUnifiedSource({"api_key": "test"})
        merged, dedupe_count = source._merge_results(
            [],
            [],
            MergeStrategy.DEDUPE,
        )
        assert merged == ""
        assert dedupe_count == 0


class TestPluginUnifiedSourceAvailability:
    """Tests for availability and connection testing."""

    def test_is_available_default(self):
        """Test default availability returns True."""
        source = MockUnifiedSource({"api_key": "test"})
        assert source.is_available() is True

    def test_test_connection_success(self):
        """Test connection test with successful source."""
        source = MockUnifiedSource({"api_key": "test"})
        success, message = source.test_connection()
        assert success is True
        assert "Documents" in message
        assert "Live" in message

    def test_test_connection_rag_only(self):
        """Test connection test with RAG-only source."""
        source = RagOnlyUnifiedSource({"api_key": "test"})
        success, message = source.test_connection()
        assert success is True
        assert "Documents" in message
        # Live should not be tested

    def test_test_connection_live_only(self):
        """Test connection test with live-only source."""
        source = LiveOnlyUnifiedSource({"api_key": "test"})
        success, message = source.test_connection()
        assert success is True
        assert "Live" in message


class TestPluginUnifiedSourceCapabilities:
    """Tests for capability flags."""

    def test_full_capabilities(self):
        """Test source with all capabilities."""
        assert MockUnifiedSource.supports_rag is True
        assert MockUnifiedSource.supports_live is True
        assert MockUnifiedSource.supports_actions is False

    def test_rag_only_capabilities(self):
        """Test RAG-only source capabilities."""
        assert RagOnlyUnifiedSource.supports_rag is True
        assert RagOnlyUnifiedSource.supports_live is False

    def test_live_only_capabilities(self):
        """Test live-only source capabilities."""
        assert LiveOnlyUnifiedSource.supports_rag is False
        assert LiveOnlyUnifiedSource.supports_live is True


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestUnifiedSourceIntegration:
    """Integration-style tests combining multiple features."""

    def test_full_query_flow_with_rag_and_live(self):
        """Test complete query flow with both RAG and live."""
        source = MockUnifiedSource({"api_key": "test"})

        def mock_rag_search(query: str, limit: int) -> list[str]:
            return [
                f"Historical document about {query}",
                f"Archive entry for {query}",
            ]

        result = source.query(
            "latest updates on project",
            {"query": "project updates"},
            rag_search_fn=mock_rag_search,
        )

        assert result.success is True
        assert result.routing_used == QueryRouting.BOTH_MERGE
        # Should use LIVE_FIRST because query contains "latest"
        assert result.merge_strategy_used == MergeStrategy.LIVE_FIRST
        assert "Live results" in result.formatted
        assert "Historical" in result.formatted

    def test_fallback_when_rag_fails(self):
        """Test that live results are returned even if RAG fails."""
        source = MockUnifiedSource({"api_key": "test"})

        def failing_rag_search(query: str, limit: int) -> list[str]:
            raise Exception("RAG search failed")

        result = source.query(
            "general query",
            {"query": "test"},
            rag_search_fn=failing_rag_search,
        )

        # Should still succeed with live results
        assert result.success is True
        assert len(result.live_results) > 0

    def test_class_attributes(self):
        """Test that required class attributes are set."""
        assert MockUnifiedSource.source_type == "mock_unified"
        assert MockUnifiedSource.display_name == "Mock Unified"
        assert MockUnifiedSource.description != ""
        assert MockUnifiedSource.category == "test"
        assert MockUnifiedSource.icon == "🧪"
