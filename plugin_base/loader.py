"""
Plugin discovery and registration.

This module handles:
- Scanning plugin directories for Python files
- Loading plugin classes from modules
- Registering plugins in type-specific registries
- Providing access to registered plugins

Plugins are discovered from two directories:
1. builtin_plugins/ - Shipped with the app
2. plugins/ - User plugins (can override builtins)
"""

import importlib.util
import logging
from pathlib import Path
from typing import Dict, Generic, Optional, Type, TypeVar

from plugin_base.action import PluginActionHandler
from plugin_base.document_source import PluginDocumentSource
from plugin_base.live_source import PluginLiveSource
from plugin_base.unified_source import PluginUnifiedSource

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """
    Generic registry for plugins of a specific type.

    Stores plugin classes by their type identifier (source_type or action_type).
    Provides methods for registration, lookup, and metadata retrieval.
    """

    def __init__(self, base_class: Type[T], plugin_type_name: str):
        """
        Initialize the registry.

        Args:
            base_class: The base class for this plugin type
            plugin_type_name: Human-readable name for logging
        """
        self.base_class = base_class
        self.plugin_type_name = plugin_type_name
        self._plugins: Dict[str, Type[T]] = {}
        self._builtin: Dict[str, bool] = {}  # Track if plugin is builtin

    def register(self, plugin_class: Type[T], is_builtin: bool = False) -> None:
        """
        Register a plugin class.

        Args:
            plugin_class: The plugin class to register
            is_builtin: Whether this is a builtin plugin

        Raises:
            ValueError: If the plugin doesn't have a type identifier
        """
        # Get the type identifier (source_type for sources, action_type for actions)
        source_type = getattr(plugin_class, "source_type", None) or getattr(
            plugin_class, "action_type", None
        )
        if not source_type:
            raise ValueError(
                f"Plugin {plugin_class.__name__} missing source_type/action_type"
            )

        if source_type in self._plugins:
            logger.warning(
                f"Duplicate {self.plugin_type_name} '{source_type}', "
                f"overwriting {self._plugins[source_type].__name__} with {plugin_class.__name__}"
            )

        self._plugins[source_type] = plugin_class
        self._builtin[source_type] = is_builtin
        logger.info(
            f"Registered {self.plugin_type_name}: {source_type} ({plugin_class.__name__})"
            + (" [builtin]" if is_builtin else "")
        )

    def get(self, source_type: str) -> Optional[Type[T]]:
        """
        Get a plugin class by source_type.

        Args:
            source_type: The type identifier

        Returns:
            The plugin class, or None if not found
        """
        return self._plugins.get(source_type)

    def get_all(self) -> Dict[str, Type[T]]:
        """
        Get all registered plugins.

        Returns:
            Dict mapping source_type to plugin class
        """
        return self._plugins.copy()

    def get_all_metadata(self) -> list[dict]:
        """
        Get metadata for all plugins (for admin UI).

        Returns:
            List of dicts with plugin metadata including fields
        """
        result = []
        for source_type, cls in self._plugins.items():
            metadata = {
                "source_type": source_type,
                "display_name": getattr(cls, "display_name", source_type),
                "description": getattr(cls, "description", ""),
                "category": getattr(cls, "category", "other"),
                "icon": getattr(cls, "icon", ""),
                "is_builtin": self._builtin.get(source_type, False),
                "fields": [f.to_dict() for f in cls.get_config_fields()],
            }

            # Add action-specific metadata
            if hasattr(cls, "get_actions"):
                metadata["actions"] = [a.to_dict() for a in cls.get_actions()]

            # Add live source-specific metadata
            if hasattr(cls, "data_type"):
                metadata["data_type"] = getattr(cls, "data_type", "")
                metadata["best_for"] = getattr(cls, "best_for", "")

            if hasattr(cls, "get_param_definitions"):
                metadata["params"] = [p.to_dict() for p in cls.get_param_definitions()]

            result.append(metadata)

        return result

    def clear(self) -> None:
        """Clear all registered plugins. Useful for testing."""
        self._plugins.clear()
        self._builtin.clear()


# Global registries for each plugin type
document_source_registry: PluginRegistry[PluginDocumentSource] = PluginRegistry(
    PluginDocumentSource, "document_source"
)
live_source_registry: PluginRegistry[PluginLiveSource] = PluginRegistry(
    PluginLiveSource, "live_source"
)
action_registry: PluginRegistry[PluginActionHandler] = PluginRegistry(
    PluginActionHandler, "action"
)
unified_source_registry: PluginRegistry[PluginUnifiedSource] = PluginRegistry(
    PluginUnifiedSource, "unified_source"
)


def _load_plugins_from_directory(
    directory: Path,
    registry: PluginRegistry,
    base_class: type,
    is_builtin: bool = False,
) -> int:
    """
    Load all plugins from a directory.

    Scans for .py files, imports them, and registers any subclasses
    of the base class found in the module.

    Args:
        directory: Directory to scan
        registry: Registry to add plugins to
        base_class: Base class to look for subclasses of
        is_builtin: Whether these are builtin plugins

    Returns:
        Number of plugins successfully registered
    """
    if not directory.exists():
        logger.debug(f"Plugin directory does not exist: {directory}")
        return 0

    count = 0
    for py_file in sorted(directory.glob("*.py")):
        # Skip __init__.py and other private files
        if py_file.name.startswith("_"):
            continue

        try:
            # Create a unique module name to avoid conflicts
            module_name = f"_plugin_{directory.name}_{py_file.stem}"

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all plugin subclasses in the module
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, base_class)
                        and obj is not base_class
                        and not getattr(obj, "_abstract", False)
                    ):
                        try:
                            registry.register(obj, is_builtin=is_builtin)
                            count += 1
                        except ValueError as e:
                            logger.error(
                                f"Failed to register {name} from {py_file}: {e}"
                            )

        except Exception as e:
            logger.error(f"Failed to load plugin {py_file}: {e}", exc_info=True)

    return count


def discover_plugins(
    plugins_dir: Optional[Path] = None,
    builtin_dir: Optional[Path] = None,
    app_root: Optional[Path] = None,
) -> dict:
    """
    Discover and register all plugins.

    Loads from both builtin_plugins/ (shipped with app) and plugins/ (user plugins).
    User plugins can override builtin plugins with the same source_type.

    Args:
        plugins_dir: Path to user plugins directory (default: plugins/)
        builtin_dir: Path to builtin plugins directory (default: builtin_plugins/)
        app_root: Application root directory (default: current working directory)

    Returns:
        Dict with counts: {"document_sources": N, "live_sources": N, "actions": N}
    """
    if app_root is None:
        app_root = Path.cwd()

    if plugins_dir is None:
        plugins_dir = app_root / "plugins"

    if builtin_dir is None:
        builtin_dir = app_root / "builtin_plugins"

    counts = {
        "document_sources": 0,
        "live_sources": 0,
        "actions": 0,
        "unified_sources": 0,
    }

    # Define the loading configuration
    load_config = [
        ("document_sources", document_source_registry, PluginDocumentSource),
        ("live_sources", live_source_registry, PluginLiveSource),
        ("actions", action_registry, PluginActionHandler),
        ("unified_sources", unified_source_registry, PluginUnifiedSource),
    ]

    # Load builtin plugins first
    for subdir, registry, base_class in load_config:
        loaded = _load_plugins_from_directory(
            builtin_dir / subdir, registry, base_class, is_builtin=True
        )
        counts[subdir] += loaded

    # Load user plugins (can override builtins)
    for subdir, registry, base_class in load_config:
        loaded = _load_plugins_from_directory(
            plugins_dir / subdir, registry, base_class, is_builtin=False
        )
        counts[subdir] += loaded

    logger.info(
        f"Plugin discovery complete: {counts['document_sources']} document sources, "
        f"{counts['live_sources']} live sources, {counts['actions']} actions, "
        f"{counts['unified_sources']} unified sources"
    )

    return counts


def get_document_source_plugin(
    source_type: str,
) -> Optional[Type[PluginDocumentSource]]:
    """
    Get a document source plugin by type.

    Args:
        source_type: The source type identifier

    Returns:
        The plugin class, or None if not found
    """
    return document_source_registry.get(source_type)


def get_live_source_plugin(source_type: str) -> Optional[Type[PluginLiveSource]]:
    """
    Get a live source plugin by type.

    Args:
        source_type: The source type identifier

    Returns:
        The plugin class, or None if not found
    """
    return live_source_registry.get(source_type)


def get_action_plugin(action_type: str) -> Optional[Type[PluginActionHandler]]:
    """
    Get an action plugin by type.

    Args:
        action_type: The action type identifier

    Returns:
        The plugin class, or None if not found
    """
    return action_registry.get(action_type)


def get_unified_source_plugin(source_type: str) -> Optional[Type[PluginUnifiedSource]]:
    """
    Get a unified source plugin by type.

    Args:
        source_type: The source type identifier

    Returns:
        The plugin class, or None if not found
    """
    return unified_source_registry.get(source_type)


def get_all_plugin_metadata() -> dict:
    """
    Get metadata for all registered plugins.

    Returns:
        Dict with metadata for each plugin type
    """
    return {
        "document_sources": document_source_registry.get_all_metadata(),
        "live_sources": live_source_registry.get_all_metadata(),
        "actions": action_registry.get_all_metadata(),
        "unified_sources": unified_source_registry.get_all_metadata(),
    }


def get_unified_source_for_doc_type(
    doc_source_type: str,
) -> Optional[Type[PluginUnifiedSource]]:
    """
    Find the unified source plugin that handles a given document store source_type.

    Args:
        doc_source_type: The document store's source_type (e.g., "mcp:gmail", "local")

    Returns:
        The unified source plugin class, or None if no match
    """
    for source_type, plugin_class in unified_source_registry.get_all().items():
        handled_types = plugin_class.get_handled_doc_source_types()
        if doc_source_type in handled_types:
            return plugin_class
    return None


def get_doc_source_to_unified_map() -> dict[str, str]:
    """
    Build a mapping from document store source_type to unified source_type.

    This is dynamically built from registered unified source plugins,
    so custom plugins are automatically included.

    Returns:
        Dict mapping doc store source_type to unified source_type
    """
    mapping = {}
    for source_type, plugin_class in unified_source_registry.get_all().items():
        handled_types = plugin_class.get_handled_doc_source_types()
        for doc_type in handled_types:
            mapping[doc_type] = source_type
    return mapping


def get_unified_source_for_live_type(
    live_source_type: str,
) -> Optional[Type[PluginUnifiedSource]]:
    """
    Find the unified source plugin that handles a given live data source type.

    Args:
        live_source_type: The live data source type (e.g., "google_gmail_live")

    Returns:
        The unified source plugin class, or None if no match
    """
    for source_type, plugin_class in unified_source_registry.get_all().items():
        handled_types = plugin_class.get_handled_live_source_types()
        if live_source_type in handled_types:
            return plugin_class
    return None


def get_live_source_to_unified_map() -> dict[str, str]:
    """
    Build a mapping from live data source type to unified source_type.

    This is dynamically built from registered unified source plugins,
    so custom plugins are automatically included.

    Returns:
        Dict mapping live source type to unified source_type
    """
    mapping = {}
    for source_type, plugin_class in unified_source_registry.get_all().items():
        handled_types = plugin_class.get_handled_live_source_types()
        for live_type in handled_types:
            mapping[live_type] = source_type
    return mapping


def get_doc_to_live_source_info(doc_source_type: str) -> Optional[dict]:
    """
    Get live source info for a document store type.

    Used when auto-creating live sources for document stores.
    Returns metadata needed to create the corresponding live source.

    Args:
        doc_source_type: The document store's source_type (e.g., "mcp:gmail")

    Returns:
        Dict with live_type, data_type, description, best_for - or None if no mapping
    """
    # Find the unified source that handles this doc type
    plugin_class = get_unified_source_for_doc_type(doc_source_type)
    if not plugin_class:
        return None

    # Check if it has any live source types
    live_types = plugin_class.get_handled_live_source_types()
    if not live_types:
        return None

    # Return info for the first live type (typically there's only one)
    live_type = live_types[0]

    # Build the info dict - use plugin metadata
    return {
        "live_type": live_type,
        "data_type": plugin_class.category,
        "description": f"{plugin_class.display_name} (real-time)",
        "best_for": plugin_class.description,
    }
