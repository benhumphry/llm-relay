"""
Mock live source for testing the plugin loader.
"""

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)


class MockLiveSource(PluginLiveSource):
    """Mock live source for testing."""

    source_type = "mock_live"
    display_name = "Mock Live Data"
    description = "Test live source for unit tests"
    data_type = "testing"
    best_for = "Testing the plugin system"
    icon = "🔬"

    # Override _abstract to allow registration
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
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="query",
                description="Search query",
                param_type="string",
                required=True,
                examples=["test query"],
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

    def fetch(self, params: dict) -> LiveDataResult:
        query = params.get("query", "")
        limit = params.get("limit", 10)

        return LiveDataResult(
            success=True,
            data={"query": query, "limit": limit},
            formatted=f"Mock results for '{query}' (limit: {limit})",
            cache_ttl=60,
        )
