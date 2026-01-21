# Plugin framework public exports
# All base classes and utilities for building plugins

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
)
from plugin_base.common import (
    FieldDefinition,
    FieldType,
    SelectOption,
    ValidationError,
    ValidationResult,
    validate_config,
    validate_field_value,
)
from plugin_base.document_source import (
    DocumentContent,
    DocumentInfo,
    PluginDocumentSource,
)
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)
from plugin_base.loader import (
    PluginRegistry,
    action_registry,
    discover_plugins,
    document_source_registry,
    get_action_plugin,
    get_all_plugin_metadata,
    get_document_source_plugin,
    get_live_source_plugin,
    get_unified_source_plugin,
    live_source_registry,
    unified_source_registry,
)
from plugin_base.oauth import OAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
    UnifiedResult,
)

__all__ = [
    # Common types
    "FieldType",
    "SelectOption",
    "FieldDefinition",
    "ValidationError",
    "ValidationResult",
    "validate_field_value",
    "validate_config",
    # Document source
    "PluginDocumentSource",
    "DocumentInfo",
    "DocumentContent",
    # Live source
    "PluginLiveSource",
    "ParamDefinition",
    "LiveDataResult",
    # Action
    "PluginActionHandler",
    "ActionRisk",
    "ActionDefinition",
    "ActionResult",
    "ActionContext",
    # Unified source
    "PluginUnifiedSource",
    "QueryRouting",
    "MergeStrategy",
    "QueryAnalysis",
    "UnifiedResult",
    # Loader
    "PluginRegistry",
    "document_source_registry",
    "live_source_registry",
    "action_registry",
    "unified_source_registry",
    "discover_plugins",
    "get_document_source_plugin",
    "get_live_source_plugin",
    "get_action_plugin",
    "get_unified_source_plugin",
    "get_all_plugin_metadata",
    # OAuth
    "OAuthMixin",
]
