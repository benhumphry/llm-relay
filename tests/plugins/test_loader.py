"""
Unit tests for plugin_base/loader.py

Tests the plugin discovery and registration system.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from plugin_base.action import ActionContext, ActionRisk, PluginActionHandler
from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import PluginDocumentSource
from plugin_base.live_source import PluginLiveSource
from plugin_base.loader import (
    PluginRegistry,
    _load_plugins_from_directory,
    action_registry,
    discover_plugins,
    document_source_registry,
    get_action_plugin,
    get_all_plugin_metadata,
    get_document_source_plugin,
    get_live_source_plugin,
    live_source_registry,
)
from tests.plugins.fixtures.mock_action import MockActionHandler

# Import fixtures for testing
from tests.plugins.fixtures.mock_document_source import MockDocumentSource
from tests.plugins.fixtures.mock_live_source import MockLiveSource


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        registry.register(MockDocumentSource)

        result = registry.get("mock_docs")
        assert result is MockDocumentSource

    def test_get_nonexistent(self):
        """Test retrieval of non-existent plugin."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        result = registry.get("nonexistent")
        assert result is None

    def test_get_all(self):
        """Test getting all registered plugins."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        registry.register(MockDocumentSource)

        all_plugins = registry.get_all()
        assert "mock_docs" in all_plugins
        assert all_plugins["mock_docs"] is MockDocumentSource

    def test_get_all_returns_copy(self):
        """Test that get_all returns a copy, not the internal dict."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        registry.register(MockDocumentSource)

        all_plugins = registry.get_all()
        all_plugins["fake"] = None  # Modify the copy

        # Original should be unchanged
        assert "fake" not in registry.get_all()

    def test_clear(self):
        """Test clearing the registry."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        registry.register(MockDocumentSource)
        assert len(registry.get_all()) == 1

        registry.clear()
        assert len(registry.get_all()) == 0

    def test_register_missing_type(self):
        """Test registration fails without source_type."""

        class BadPlugin(PluginDocumentSource):
            # Missing source_type
            display_name = "Bad"
            description = "Bad plugin"
            category = "other"
            _abstract = False

            @classmethod
            def get_config_fields(cls):
                return []

            def __init__(self, config):
                pass

            def list_documents(self):
                return iter([])

            def read_document(self, uri):
                return None

        registry = PluginRegistry(PluginDocumentSource, "test")
        with pytest.raises(ValueError, match="missing source_type"):
            registry.register(BadPlugin)

    def test_duplicate_registration_overwrites(self):
        """Test that duplicate registration overwrites existing."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        registry.register(MockDocumentSource)

        # Create another plugin with the same source_type
        class AnotherMock(MockDocumentSource):
            pass

        AnotherMock.source_type = "mock_docs"  # Same as MockDocumentSource
        registry.register(AnotherMock)

        # Should have overwritten
        result = registry.get("mock_docs")
        assert result is AnotherMock

    def test_get_all_metadata_document_source(self):
        """Test metadata retrieval for document source."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        registry.register(MockDocumentSource)

        metadata = registry.get_all_metadata()
        assert len(metadata) == 1

        m = metadata[0]
        assert m["source_type"] == "mock_docs"
        assert m["display_name"] == "Mock Documents"
        assert m["description"] == "Test document source for unit tests"
        assert m["category"] == "other"
        assert m["icon"] == "📝"
        assert len(m["fields"]) == 2  # doc_count and prefix

    def test_get_all_metadata_action(self):
        """Test metadata retrieval for action handler."""
        registry = PluginRegistry(PluginActionHandler, "test")
        registry.register(MockActionHandler)

        metadata = registry.get_all_metadata()
        assert len(metadata) == 1

        m = metadata[0]
        assert m["source_type"] == "mock_action"  # Uses action_type
        assert "actions" in m
        assert len(m["actions"]) == 3  # test, create, delete

    def test_get_all_metadata_live_source(self):
        """Test metadata retrieval for live source."""
        registry = PluginRegistry(PluginLiveSource, "test")
        registry.register(MockLiveSource)

        metadata = registry.get_all_metadata()
        assert len(metadata) == 1

        m = metadata[0]
        assert m["source_type"] == "mock_live"
        assert m["data_type"] == "testing"
        assert m["best_for"] == "Testing the plugin system"
        assert "params" in m
        assert len(m["params"]) == 2  # query and limit


class TestLoadPluginsFromDirectory:
    """Tests for _load_plugins_from_directory function."""

    def setup_method(self):
        """Create a temporary directory for test plugins."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_load_from_nonexistent_directory(self):
        """Test loading from non-existent directory returns 0."""
        registry = PluginRegistry(PluginDocumentSource, "test")
        count = _load_plugins_from_directory(
            Path("/nonexistent/path"), registry, PluginDocumentSource
        )
        assert count == 0

    def test_skip_private_files(self):
        """Test that files starting with _ are skipped."""
        registry = PluginRegistry(PluginDocumentSource, "test")

        # Create a private file
        private_file = self.temp_dir / "_private.py"
        private_file.write_text(
            """
from plugin_base.document_source import PluginDocumentSource
from plugin_base.common import FieldDefinition

class PrivatePlugin(PluginDocumentSource):
    source_type = "private"
    display_name = "Private"
    description = "Should not be loaded"
    category = "other"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    def __init__(self, config):
        pass

    def list_documents(self):
        return iter([])

    def read_document(self, uri):
        return None
"""
        )

        count = _load_plugins_from_directory(
            self.temp_dir, registry, PluginDocumentSource
        )
        assert count == 0
        assert registry.get("private") is None

    def test_load_valid_plugin(self):
        """Test loading a valid plugin from file."""
        registry = PluginRegistry(PluginDocumentSource, "test")

        # Create a valid plugin file
        plugin_file = self.temp_dir / "test_plugin.py"
        plugin_file.write_text(
            """
from plugin_base.document_source import PluginDocumentSource, DocumentInfo, DocumentContent
from plugin_base.common import FieldDefinition, FieldType

class TestPlugin(PluginDocumentSource):
    source_type = "test_file_plugin"
    display_name = "Test File Plugin"
    description = "Loaded from file"
    category = "other"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(
                name="setting",
                label="Setting",
                field_type=FieldType.TEXT,
            )
        ]

    def __init__(self, config):
        self.setting = config.get("setting", "")

    def list_documents(self):
        yield DocumentInfo(uri="test", title="Test Doc")

    def read_document(self, uri):
        return DocumentContent(content="Test content")
"""
        )

        count = _load_plugins_from_directory(
            self.temp_dir, registry, PluginDocumentSource
        )
        assert count == 1

        plugin = registry.get("test_file_plugin")
        assert plugin is not None
        assert plugin.display_name == "Test File Plugin"

    def test_skip_abstract_plugins(self):
        """Test that abstract plugins (with _abstract=True) are skipped."""
        registry = PluginRegistry(PluginDocumentSource, "test")

        # Create an abstract plugin
        plugin_file = self.temp_dir / "abstract_plugin.py"
        plugin_file.write_text(
            """
from plugin_base.document_source import PluginDocumentSource

class AbstractPlugin(PluginDocumentSource):
    source_type = "abstract"
    display_name = "Abstract"
    description = "Should not be registered"
    category = "other"
    _abstract = True  # Mark as abstract

    @classmethod
    def get_config_fields(cls):
        return []

    def __init__(self, config):
        pass

    def list_documents(self):
        return iter([])

    def read_document(self, uri):
        return None
"""
        )

        count = _load_plugins_from_directory(
            self.temp_dir, registry, PluginDocumentSource
        )
        assert count == 0
        assert registry.get("abstract") is None

    def test_handle_plugin_with_syntax_error(self):
        """Test that syntax errors are handled gracefully."""
        registry = PluginRegistry(PluginDocumentSource, "test")

        # Create a file with syntax error
        bad_file = self.temp_dir / "bad_syntax.py"
        bad_file.write_text("this is not valid python {{{{")

        # Should not raise, just log error
        count = _load_plugins_from_directory(
            self.temp_dir, registry, PluginDocumentSource
        )
        assert count == 0


class TestDiscoverPlugins:
    """Tests for discover_plugins function."""

    def setup_method(self):
        """Create temporary directories for test plugins."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.builtin_dir = self.temp_dir / "builtin_plugins"
        self.plugins_dir = self.temp_dir / "plugins"

        # Create directory structure
        (self.builtin_dir / "document_sources").mkdir(parents=True)
        (self.builtin_dir / "live_sources").mkdir(parents=True)
        (self.builtin_dir / "actions").mkdir(parents=True)
        (self.plugins_dir / "document_sources").mkdir(parents=True)
        (self.plugins_dir / "live_sources").mkdir(parents=True)
        (self.plugins_dir / "actions").mkdir(parents=True)

        # Clear global registries before each test
        document_source_registry.clear()
        live_source_registry.clear()
        action_registry.clear()

    def teardown_method(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
        # Clear registries after test
        document_source_registry.clear()
        live_source_registry.clear()
        action_registry.clear()

    def test_discover_empty_directories(self):
        """Test discovery with empty directories."""
        counts = discover_plugins(
            plugins_dir=self.plugins_dir, builtin_dir=self.builtin_dir
        )
        assert counts["document_sources"] == 0
        assert counts["live_sources"] == 0
        assert counts["actions"] == 0

    def test_discover_builtin_document_source(self):
        """Test discovery of builtin document source."""
        # Create a builtin plugin
        plugin_file = self.builtin_dir / "document_sources" / "builtin_docs.py"
        plugin_file.write_text(
            """
from plugin_base.document_source import PluginDocumentSource, DocumentInfo
from plugin_base.common import FieldDefinition

class BuiltinDocs(PluginDocumentSource):
    source_type = "builtin_docs"
    display_name = "Builtin Docs"
    description = "A builtin document source"
    category = "other"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    def __init__(self, config):
        pass

    def list_documents(self):
        return iter([])

    def read_document(self, uri):
        return None
"""
        )

        counts = discover_plugins(
            plugins_dir=self.plugins_dir, builtin_dir=self.builtin_dir
        )
        assert counts["document_sources"] == 1

        plugin = get_document_source_plugin("builtin_docs")
        assert plugin is not None
        assert plugin.display_name == "Builtin Docs"

    def test_user_plugin_overrides_builtin(self):
        """Test that user plugins override builtin plugins."""
        # Create a builtin plugin
        builtin_file = self.builtin_dir / "document_sources" / "override_test.py"
        builtin_file.write_text(
            """
from plugin_base.document_source import PluginDocumentSource
from plugin_base.common import FieldDefinition

class BuiltinVersion(PluginDocumentSource):
    source_type = "override_test"
    display_name = "Builtin Version"
    description = "Original builtin"
    category = "other"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    def __init__(self, config):
        pass

    def list_documents(self):
        return iter([])

    def read_document(self, uri):
        return None
"""
        )

        # Create a user plugin with the same source_type
        user_file = self.plugins_dir / "document_sources" / "override_test.py"
        user_file.write_text(
            """
from plugin_base.document_source import PluginDocumentSource
from plugin_base.common import FieldDefinition

class UserVersion(PluginDocumentSource):
    source_type = "override_test"
    display_name = "User Version"  # Different display name
    description = "User override"
    category = "other"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    def __init__(self, config):
        pass

    def list_documents(self):
        return iter([])

    def read_document(self, uri):
        return None
"""
        )

        counts = discover_plugins(
            plugins_dir=self.plugins_dir, builtin_dir=self.builtin_dir
        )
        # Both are loaded (count includes override)
        assert counts["document_sources"] == 2

        # User version should win
        plugin = get_document_source_plugin("override_test")
        assert plugin.display_name == "User Version"

    def test_discover_all_plugin_types(self):
        """Test discovery of all three plugin types."""
        # Create document source
        (self.builtin_dir / "document_sources" / "test_docs.py").write_text(
            """
from plugin_base.document_source import PluginDocumentSource
from plugin_base.common import FieldDefinition

class TestDocs(PluginDocumentSource):
    source_type = "test_docs"
    display_name = "Test Docs"
    description = "Test"
    category = "other"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    def __init__(self, config):
        pass

    def list_documents(self):
        return iter([])

    def read_document(self, uri):
        return None
"""
        )

        # Create live source
        (self.builtin_dir / "live_sources" / "test_live.py").write_text(
            """
from plugin_base.live_source import PluginLiveSource, LiveDataResult, ParamDefinition
from plugin_base.common import FieldDefinition

class TestLive(PluginLiveSource):
    source_type = "test_live"
    display_name = "Test Live"
    description = "Test"
    data_type = "test"
    best_for = "Testing"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    @classmethod
    def get_param_definitions(cls):
        return []

    def __init__(self, config):
        pass

    def fetch(self, params):
        return LiveDataResult(success=True, formatted="test")
"""
        )

        # Create action
        (self.builtin_dir / "actions" / "test_action.py").write_text(
            """
from plugin_base.action import PluginActionHandler, ActionDefinition, ActionRisk, ActionResult, ActionContext
from plugin_base.common import FieldDefinition

class TestAction(PluginActionHandler):
    action_type = "test_action"
    display_name = "Test Action"
    description = "Test"
    _abstract = False

    @classmethod
    def get_config_fields(cls):
        return []

    @classmethod
    def get_actions(cls):
        return []

    def __init__(self, config):
        pass

    def execute(self, action, params, context):
        return ActionResult(success=True, message="OK")
"""
        )

        counts = discover_plugins(
            plugins_dir=self.plugins_dir, builtin_dir=self.builtin_dir
        )
        assert counts["document_sources"] == 1
        assert counts["live_sources"] == 1
        assert counts["actions"] == 1

        assert get_document_source_plugin("test_docs") is not None
        assert get_live_source_plugin("test_live") is not None
        assert get_action_plugin("test_action") is not None


class TestGetAllPluginMetadata:
    """Tests for get_all_plugin_metadata function."""

    def setup_method(self):
        """Clear registries and register test plugins."""
        document_source_registry.clear()
        live_source_registry.clear()
        action_registry.clear()

        document_source_registry.register(MockDocumentSource)
        live_source_registry.register(MockLiveSource)
        action_registry.register(MockActionHandler)

    def teardown_method(self):
        """Clear registries."""
        document_source_registry.clear()
        live_source_registry.clear()
        action_registry.clear()

    def test_get_all_plugin_metadata(self):
        """Test getting metadata for all plugin types."""
        metadata = get_all_plugin_metadata()

        assert "document_sources" in metadata
        assert "live_sources" in metadata
        assert "actions" in metadata

        assert len(metadata["document_sources"]) == 1
        assert len(metadata["live_sources"]) == 1
        assert len(metadata["actions"]) == 1

        # Check document source metadata
        doc_meta = metadata["document_sources"][0]
        assert doc_meta["source_type"] == "mock_docs"

        # Check live source metadata
        live_meta = metadata["live_sources"][0]
        assert live_meta["source_type"] == "mock_live"
        assert "params" in live_meta

        # Check action metadata
        action_meta = metadata["actions"][0]
        assert action_meta["source_type"] == "mock_action"
        assert "actions" in action_meta


class TestMockPluginFunctionality:
    """Tests that mock plugins work correctly when instantiated."""

    def test_mock_document_source(self):
        """Test MockDocumentSource functionality."""
        config = {"doc_count": 3, "prefix": "Test"}
        source = MockDocumentSource(config)

        docs = list(source.list_documents())
        assert len(docs) == 3
        assert docs[0].title == "Test 0"
        assert docs[1].title == "Test 1"

        content = source.read_document("doc_0")
        assert content is not None
        assert "doc_0" in content.content

    def test_mock_live_source(self):
        """Test MockLiveSource functionality."""
        config = {"api_key": "test-key"}
        source = MockLiveSource(config)

        result = source.fetch({"query": "test", "limit": 5})
        assert result.success is True
        assert "test" in result.formatted
        assert result.data["limit"] == 5

    def test_mock_action_handler(self):
        """Test MockActionHandler functionality."""
        config = {"api_key": "test-key", "verbose": True}
        handler = MockActionHandler(config)

        # Test action
        result = handler.execute("test", {"message": "Hello"}, ActionContext())
        assert result.success is True
        assert "Hello" in result.message

        # Create action
        result = handler.execute("create", {"name": "Item"}, ActionContext())
        assert result.success is True
        assert result.data["name"] == "Item"

        # Unknown action
        result = handler.execute("unknown", {}, ActionContext())
        assert result.success is False
        assert "Unknown action" in result.error

    def test_mock_action_llm_instructions(self):
        """Test LLM instructions generation."""
        instructions = MockActionHandler.get_llm_instructions()

        assert "Mock Actions" in instructions
        assert "mock_action:test" in instructions
        assert "mock_action:create" in instructions
        assert "mock_action:delete" in instructions
        assert "message (required)" in instructions
