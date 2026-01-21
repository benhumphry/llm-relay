"""
Unit tests for plugin_base/live_source.py

Tests cover:
- ParamDefinition dataclass
- LiveDataResult dataclass
- PluginLiveSource base class
- Config validation
- Parameter validation
- Designator hint generation
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from plugin_base.common import FieldDefinition, FieldType, ValidationResult
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)

# =============================================================================
# Test Fixtures - Mock Live Source Plugin
# =============================================================================


class MockWeatherSource(PluginLiveSource):
    """Mock weather source for testing."""

    source_type = "mock_weather"
    display_name = "Mock Weather"
    description = "Mock weather data for testing"
    data_type = "weather"
    best_for = "Current weather conditions and forecasts"
    icon = "🌤️"
    default_cache_ttl = 600

    _abstract = False  # Allow registration

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="api_key",
                label="API Key",
                field_type=FieldType.PASSWORD,
                required=True,
                help_text="Your weather API key",
            ),
            FieldDefinition(
                name="units",
                label="Units",
                field_type=FieldType.SELECT,
                required=False,
                default="metric",
                options=[
                    {"value": "metric", "label": "Metric (°C)"},
                    {"value": "imperial", "label": "Imperial (°F)"},
                ],
            ),
        ]

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="location",
                description="City name or coordinates",
                param_type="string",
                required=True,
                examples=["London", "New York", "51.5,-0.1"],
            ),
            ParamDefinition(
                name="days",
                description="Number of forecast days",
                param_type="integer",
                required=False,
                default=1,
                examples=["1", "3", "7"],
            ),
        ]

    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        self.units = config.get("units", "metric")

    def fetch(self, params: dict) -> LiveDataResult:
        location = params.get("location")
        if not location:
            return LiveDataResult(
                success=False,
                error="Location is required",
            )

        # Mock response
        return LiveDataResult(
            success=True,
            data={"temp": 20, "condition": "sunny"},
            formatted=f"Weather in {location}: 20°C, sunny",
            cache_ttl=self.default_cache_ttl,
        )


class MockStocksSource(PluginLiveSource):
    """Mock stocks source with no required params."""

    source_type = "mock_stocks"
    display_name = "Mock Stocks"
    description = "Mock stock data for testing"
    data_type = "finance"
    best_for = "Stock prices and market data"

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
                name="symbol",
                description="Stock ticker symbol",
                param_type="string",
                required=False,
                examples=["AAPL", "GOOGL", "MSFT"],
            ),
        ]

    def __init__(self, config: dict):
        self.api_key = config.get("api_key")

    def fetch(self, params: dict) -> LiveDataResult:
        symbol = params.get("symbol", "AAPL")
        return LiveDataResult(
            success=True,
            data={"symbol": symbol, "price": 150.00},
            formatted=f"{symbol}: $150.00",
        )


class MockMinimalSource(PluginLiveSource):
    """Minimal source with no config or params."""

    source_type = "mock_minimal"
    display_name = "Mock Minimal"
    description = "Minimal source for testing"
    data_type = "other"
    best_for = "Testing edge cases"

    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return []

    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return []

    def __init__(self, config: dict):
        pass

    def fetch(self, params: dict) -> LiveDataResult:
        return LiveDataResult(
            success=True,
            formatted="Minimal data response",
        )


# =============================================================================
# ParamDefinition Tests
# =============================================================================


class TestParamDefinition:
    """Tests for ParamDefinition dataclass."""

    def test_basic_param(self):
        """Test basic parameter definition."""
        param = ParamDefinition(
            name="location",
            description="City name",
            param_type="string",
        )
        assert param.name == "location"
        assert param.description == "City name"
        assert param.param_type == "string"
        assert param.required is False
        assert param.default is None
        assert param.examples == []

    def test_required_param(self):
        """Test required parameter."""
        param = ParamDefinition(
            name="symbol",
            description="Stock symbol",
            param_type="string",
            required=True,
        )
        assert param.required is True

    def test_param_with_default(self):
        """Test parameter with default value."""
        param = ParamDefinition(
            name="days",
            description="Number of days",
            param_type="integer",
            default=7,
        )
        assert param.default == 7

    def test_param_with_examples(self):
        """Test parameter with examples."""
        param = ParamDefinition(
            name="location",
            description="City name",
            param_type="string",
            examples=["London", "Paris", "Tokyo"],
        )
        assert param.examples == ["London", "Paris", "Tokyo"]

    def test_to_dict_basic(self):
        """Test to_dict with basic parameter."""
        param = ParamDefinition(
            name="test",
            description="Test param",
            param_type="string",
        )
        d = param.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test param"
        assert d["param_type"] == "string"
        assert d["required"] is False
        assert d["default"] is None
        assert d["examples"] == []

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        param = ParamDefinition(
            name="location",
            description="City name or coordinates",
            param_type="string",
            required=True,
            default="London",
            examples=["London", "51.5,-0.1"],
        )
        d = param.to_dict()
        assert d["name"] == "location"
        assert d["required"] is True
        assert d["default"] == "London"
        assert d["examples"] == ["London", "51.5,-0.1"]

    def test_param_types(self):
        """Test various parameter types."""
        types = ["string", "integer", "number", "boolean", "datetime"]
        for t in types:
            param = ParamDefinition(name="test", description="Test", param_type=t)
            assert param.param_type == t
            assert param.to_dict()["param_type"] == t


# =============================================================================
# LiveDataResult Tests
# =============================================================================


class TestLiveDataResult:
    """Tests for LiveDataResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = LiveDataResult(
            success=True,
            data={"temp": 20},
            formatted="Temperature: 20°C",
        )
        assert result.success is True
        assert result.data == {"temp": 20}
        assert result.formatted == "Temperature: 20°C"
        assert result.error is None
        assert result.cache_ttl == 300  # Default

    def test_error_result(self):
        """Test error result."""
        result = LiveDataResult(
            success=False,
            error="API key invalid",
        )
        assert result.success is False
        assert result.error == "API key invalid"
        assert result.data is None
        assert result.formatted == ""

    def test_custom_cache_ttl(self):
        """Test custom cache TTL."""
        result = LiveDataResult(success=True, cache_ttl=3600)
        assert result.cache_ttl == 3600

    def test_no_cache(self):
        """Test no caching (TTL=0)."""
        result = LiveDataResult(success=True, cache_ttl=0)
        assert result.cache_ttl == 0

    def test_cache_forever(self):
        """Test cache forever (TTL=-1)."""
        result = LiveDataResult(success=True, cache_ttl=-1)
        assert result.cache_ttl == -1

    def test_timestamp_auto(self):
        """Test automatic timestamp."""
        before = datetime.now(timezone.utc)
        result = LiveDataResult(success=True)
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after

    def test_timestamp_custom(self):
        """Test custom timestamp."""
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = LiveDataResult(success=True, timestamp=ts)
        assert result.timestamp == ts

    def test_data_types(self):
        """Test various data types."""
        # Dict
        result1 = LiveDataResult(success=True, data={"key": "value"})
        assert result1.data == {"key": "value"}

        # List
        result2 = LiveDataResult(success=True, data=[1, 2, 3])
        assert result2.data == [1, 2, 3]

        # String
        result3 = LiveDataResult(success=True, data="raw data")
        assert result3.data == "raw data"


# =============================================================================
# PluginLiveSource Base Class Tests
# =============================================================================


class TestPluginLiveSourceClassAttributes:
    """Tests for class attributes."""

    def test_required_attributes(self):
        """Test required class attributes."""
        assert MockWeatherSource.source_type == "mock_weather"
        assert MockWeatherSource.display_name == "Mock Weather"
        assert MockWeatherSource.description == "Mock weather data for testing"
        assert MockWeatherSource.data_type == "weather"
        assert MockWeatherSource.best_for == "Current weather conditions and forecasts"

    def test_optional_attributes(self):
        """Test optional class attributes."""
        assert MockWeatherSource.icon == "🌤️"
        assert MockWeatherSource.default_cache_ttl == 600

    def test_default_optional_attributes(self):
        """Test default values for optional attributes."""
        assert MockMinimalSource.icon == "🔌"  # Default
        assert MockMinimalSource.default_cache_ttl == 300  # Default


class TestPluginLiveSourceConfigFields:
    """Tests for get_config_fields()."""

    def test_config_fields_returned(self):
        """Test config fields are returned correctly."""
        fields = MockWeatherSource.get_config_fields()
        assert len(fields) == 2
        assert fields[0].name == "api_key"
        assert fields[1].name == "units"

    def test_config_fields_types(self):
        """Test config field types."""
        fields = MockWeatherSource.get_config_fields()
        assert fields[0].field_type == FieldType.PASSWORD
        assert fields[1].field_type == FieldType.SELECT

    def test_empty_config_fields(self):
        """Test source with no config fields."""
        fields = MockMinimalSource.get_config_fields()
        assert fields == []


class TestPluginLiveSourceParamDefinitions:
    """Tests for get_param_definitions()."""

    def test_param_definitions_returned(self):
        """Test param definitions are returned correctly."""
        params = MockWeatherSource.get_param_definitions()
        assert len(params) == 2
        assert params[0].name == "location"
        assert params[1].name == "days"

    def test_required_params(self):
        """Test required parameter identification."""
        params = MockWeatherSource.get_param_definitions()
        assert params[0].required is True  # location
        assert params[1].required is False  # days

    def test_empty_param_definitions(self):
        """Test source with no param definitions."""
        params = MockMinimalSource.get_param_definitions()
        assert params == []


class TestPluginLiveSourceConfigValidation:
    """Tests for validate_config()."""

    def test_valid_config(self):
        """Test valid configuration."""
        result = MockWeatherSource.validate_config(
            {
                "api_key": "test-key-123",
                "units": "metric",
            }
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_required_field(self):
        """Test missing required field."""
        result = MockWeatherSource.validate_config(
            {
                "units": "metric",
            }
        )
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "api_key"

    def test_optional_field_omitted(self):
        """Test optional field can be omitted."""
        result = MockWeatherSource.validate_config(
            {
                "api_key": "test-key-123",
            }
        )
        assert result.valid is True

    def test_empty_config_for_minimal_source(self):
        """Test empty config for source with no required fields."""
        result = MockMinimalSource.validate_config({})
        assert result.valid is True


class TestPluginLiveSourceParamValidation:
    """Tests for validate_params()."""

    def test_valid_params(self):
        """Test valid parameters."""
        source = MockWeatherSource({"api_key": "test"})
        result = source.validate_params({"location": "London"})
        assert result.valid is True

    def test_missing_required_param(self):
        """Test missing required parameter."""
        source = MockWeatherSource({"api_key": "test"})
        result = source.validate_params({})
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "location"

    def test_optional_param_omitted(self):
        """Test optional parameter can be omitted."""
        source = MockWeatherSource({"api_key": "test"})
        result = source.validate_params({"location": "London"})
        # days is optional
        assert result.valid is True

    def test_all_params_provided(self):
        """Test all parameters provided."""
        source = MockWeatherSource({"api_key": "test"})
        result = source.validate_params({"location": "London", "days": 3})
        assert result.valid is True

    def test_no_params_for_minimal_source(self):
        """Test no params needed for minimal source."""
        source = MockMinimalSource({})
        result = source.validate_params({})
        assert result.valid is True


class TestPluginLiveSourceDesignatorHint:
    """Tests for get_designator_hint()."""

    def test_hint_with_params(self):
        """Test designator hint with parameters."""
        hint = MockWeatherSource.get_designator_hint()
        assert "location" in hint
        assert "City name or coordinates" in hint
        assert "London" in hint  # Example
        assert "days" in hint
        assert "optional" in hint
        assert "Current weather conditions and forecasts" in hint  # best_for

    def test_hint_with_required_param(self):
        """Test hint shows required params without 'optional' prefix."""
        hint = MockWeatherSource.get_designator_hint()
        # location is required, so should NOT have "optional" before it
        # The format is: name (optional, description) for optional
        # or: name (description) for required
        assert "location (" in hint
        # days is optional
        assert "days (optional" in hint

    def test_hint_no_params(self):
        """Test designator hint with no parameters."""
        hint = MockMinimalSource.get_designator_hint()
        assert "no parameters" in hint
        assert "Testing edge cases" in hint  # best_for

    def test_hint_with_examples(self):
        """Test examples appear in hint."""
        hint = MockWeatherSource.get_designator_hint()
        assert "London" in hint
        assert "New York" in hint


class TestPluginLiveSourceFetch:
    """Tests for fetch()."""

    def test_successful_fetch(self):
        """Test successful data fetch."""
        source = MockWeatherSource({"api_key": "test"})
        result = source.fetch({"location": "London"})
        assert result.success is True
        assert "London" in result.formatted
        assert result.data == {"temp": 20, "condition": "sunny"}

    def test_fetch_error(self):
        """Test fetch with missing required param."""
        source = MockWeatherSource({"api_key": "test"})
        result = source.fetch({})  # Missing location
        assert result.success is False
        assert "required" in result.error.lower()

    def test_fetch_with_optional_params(self):
        """Test fetch with optional parameters."""
        source = MockStocksSource({"api_key": "test"})
        result = source.fetch({"symbol": "GOOGL"})
        assert result.success is True
        assert "GOOGL" in result.formatted

    def test_fetch_minimal_source(self):
        """Test fetch with minimal source."""
        source = MockMinimalSource({})
        result = source.fetch({})
        assert result.success is True
        assert result.formatted == "Minimal data response"


class TestPluginLiveSourceTestConnection:
    """Tests for test_connection()."""

    def test_successful_connection(self):
        """Test successful connection test."""
        source = MockMinimalSource({})
        success, message = source.test_connection()
        assert success is True
        assert "OK" in message
        assert "Minimal data" in message

    def test_connection_with_preview(self):
        """Test connection shows preview."""
        source = MockWeatherSource({"api_key": "test"})
        # This will fail because fetch needs location
        success, message = source.test_connection()
        assert success is False
        assert "required" in message.lower()

    def test_connection_truncates_long_response(self):
        """Test long responses are truncated."""

        class LongResponseSource(MockMinimalSource):
            def fetch(self, params: dict) -> LiveDataResult:
                return LiveDataResult(
                    success=True,
                    formatted="x" * 200,  # Long response
                )

        source = LongResponseSource({})
        success, message = source.test_connection()
        assert success is True
        assert "..." in message
        assert len(message) < 150  # Truncated


class TestPluginLiveSourceIsAvailable:
    """Tests for is_available()."""

    def test_default_available(self):
        """Test default is_available returns True."""
        source = MockWeatherSource({"api_key": "test"})
        assert source.is_available() is True

    def test_custom_availability_check(self):
        """Test custom availability check."""

        class ConditionalSource(MockMinimalSource):
            def __init__(self, config: dict):
                self.enabled = config.get("enabled", True)

            def is_available(self) -> bool:
                return self.enabled

        source1 = ConditionalSource({"enabled": True})
        assert source1.is_available() is True

        source2 = ConditionalSource({"enabled": False})
        assert source2.is_available() is False


class TestPluginLiveSourceInitialization:
    """Tests for __init__()."""

    def test_init_stores_config(self):
        """Test initialization stores config values."""
        source = MockWeatherSource(
            {
                "api_key": "my-api-key",
                "units": "imperial",
            }
        )
        assert source.api_key == "my-api-key"
        assert source.units == "imperial"

    def test_init_with_defaults(self):
        """Test initialization uses defaults."""
        source = MockWeatherSource({"api_key": "test"})
        assert source.units == "metric"  # Default


# =============================================================================
# Integration Tests
# =============================================================================


class TestLiveSourceIntegration:
    """Integration tests for live source workflow."""

    def test_full_workflow(self):
        """Test complete workflow: validate -> init -> validate_params -> fetch."""
        # 1. Validate config
        config = {"api_key": "test-key"}
        validation = MockWeatherSource.validate_config(config)
        assert validation.valid is True

        # 2. Initialize
        source = MockWeatherSource(config)

        # 3. Validate params
        params = {"location": "Paris", "days": 3}
        param_validation = source.validate_params(params)
        assert param_validation.valid is True

        # 4. Fetch
        result = source.fetch(params)
        assert result.success is True
        assert "Paris" in result.formatted

    def test_designator_integration(self):
        """Test designator hint contains all needed info."""
        hint = MockWeatherSource.get_designator_hint()

        # Should contain param names
        for param in MockWeatherSource.get_param_definitions():
            assert param.name in hint

        # Should contain best_for
        assert MockWeatherSource.best_for in hint

    def test_metadata_for_admin_ui(self):
        """Test all metadata available for admin UI."""
        # Class attributes
        assert MockWeatherSource.source_type
        assert MockWeatherSource.display_name
        assert MockWeatherSource.description
        assert MockWeatherSource.data_type
        assert MockWeatherSource.icon

        # Config fields
        fields = MockWeatherSource.get_config_fields()
        assert all(hasattr(f, "to_dict") for f in fields)

        # Param definitions
        params = MockWeatherSource.get_param_definitions()
        assert all(hasattr(p, "to_dict") for p in params)
