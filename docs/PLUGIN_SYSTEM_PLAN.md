# LLM Relay Plugin System - Implementation Plan

## Overview

This document outlines the implementation plan for a modular plugin system that allows users to add custom document sources, live data sources, and action handlers without modifying core code.

### Goals

1. **Extensibility** - Drop Python files into `plugins/` folder, auto-discovered on startup
2. **Consistency** - All three plugin types share common patterns and base classes
3. **Self-documenting** - Plugins define their own config fields, parameters, and LLM instructions
4. **Type-safe** - Dataclasses with validation, clear interfaces
5. **Backwards compatible** - Existing built-in sources continue to work during migration

### Non-Goals

- Hot reloading (restart required for new plugins)
- Plugin marketplace/remote installation
- Sandboxing/security isolation (plugins are trusted code)

---

## Architecture

### Directory Structure

```
llm-relay/
├── plugins/                          # User plugins (gitignored in production)
│   ├── document_sources/
│   │   └── example_wiki.py
│   ├── live_sources/
│   │   └── example_weather.py
│   ├── actions/
│   │   └── example_todoist.py
│   └── README.md                     # Plugin development guide
│
├── plugin_base/                      # Plugin framework (core code)
│   ├── __init__.py                   # Public exports
│   ├── common.py                     # Shared dataclasses (FieldDefinition, etc.)
│   ├── document_source.py            # PluginDocumentSource base class
│   ├── live_source.py                # PluginLiveSource base class
│   ├── action.py                     # PluginActionHandler base class
│   ├── oauth.py                      # OAuthMixin for OAuth-based plugins
│   ├── loader.py                     # Plugin discovery and registration
│   └── validator.py                  # Config validation utilities
│
├── builtin_plugins/                  # Built-in plugins (shipped with app)
│   ├── document_sources/
│   │   ├── google_drive.py
│   │   ├── gmail.py
│   │   ├── notion.py
│   │   └── ...
│   ├── live_sources/
│   │   ├── google_routes.py
│   │   ├── weather.py
│   │   └── ...
│   └── actions/
│       ├── email.py
│       ├── calendar.py
│       └── ...
│
└── tests/
    └── plugins/
        ├── test_plugin_loader.py
        ├── test_document_source_base.py
        ├── test_live_source_base.py
        ├── test_action_base.py
        └── fixtures/
            ├── mock_document_source.py
            ├── mock_live_source.py
            └── mock_action.py
```

### Database Schema Changes

#### New Table: `plugin_configs`

Stores configuration for plugin instances (replaces sparse columns approach).

```sql
CREATE TABLE plugin_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plugin_type VARCHAR(20) NOT NULL,      -- 'document_source', 'live_source', 'action'
    source_type VARCHAR(100) NOT NULL,      -- Plugin's source_type identifier
    name VARCHAR(100) NOT NULL UNIQUE,      -- User-friendly name
    enabled BOOLEAN DEFAULT TRUE,
    config_json TEXT,                        -- Plugin-specific config as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_plugin_configs_type ON plugin_configs(plugin_type, source_type);
```

#### Migration Strategy

- Keep existing `document_stores` and `live_data_sources` tables
- Add `plugin_config_id` foreign key to link to new table
- Gradually migrate built-in sources to plugin architecture
- Eventually deprecate sparse columns

---

## Phase 1: Plugin Infrastructure (Week 1-2)

### 1.1 Common Dataclasses

**File: `plugin_base/common.py`**

```python
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum

class FieldType(Enum):
    """Supported field types for admin UI."""
    TEXT = "text"
    PASSWORD = "password"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTISELECT = "multiselect"
    TEXTAREA = "textarea"
    JSON = "json"
    OAUTH_ACCOUNT = "oauth_account"
    FOLDER_PICKER = "folder_picker"
    CALENDAR_PICKER = "calendar_picker"
    CHANNEL_PICKER = "channel_picker"

@dataclass
class SelectOption:
    """Option for select/multiselect fields."""
    value: str
    label: str

@dataclass
class FieldDefinition:
    """
    Defines a configuration field for the admin UI.
    
    Used by all plugin types to declare their configuration requirements.
    The admin UI dynamically renders forms based on these definitions.
    """
    name: str
    label: str
    field_type: FieldType | str  # FieldType enum or string for flexibility
    required: bool = False
    default: Any = None
    help_text: str = ""
    placeholder: str = ""
    
    # For select/multiselect fields
    options: list[SelectOption | dict] = field(default_factory=list)
    
    # For oauth_account, folder_picker, etc.
    picker_options: dict = field(default_factory=dict)
    
    # Conditional visibility: {"field_name": "expected_value"}
    depends_on: dict = field(default_factory=dict)
    
    # Validation
    min_value: Optional[int | float] = None
    max_value: Optional[int | float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {
            "name": self.name,
            "label": self.label,
            "field_type": self.field_type.value if isinstance(self.field_type, FieldType) else self.field_type,
            "required": self.required,
            "help_text": self.help_text,
            "placeholder": self.placeholder,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.options:
            result["options"] = [
                o if isinstance(o, dict) else {"value": o.value, "label": o.label}
                for o in self.options
            ]
        if self.picker_options:
            result["picker_options"] = self.picker_options
        if self.depends_on:
            result["depends_on"] = self.depends_on
        # Add validation constraints if set
        for attr in ["min_value", "max_value", "min_length", "max_length", "pattern"]:
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val
        return result

@dataclass
class ValidationError:
    """Validation error for a specific field."""
    field: str
    message: str

@dataclass
class ValidationResult:
    """Result of config validation."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    
    @property
    def error_message(self) -> str:
        """Combined error message."""
        return "; ".join(f"{e.field}: {e.message}" for e in self.errors)
```

### 1.2 Plugin Loader

**File: `plugin_base/loader.py`**

```python
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Type, TypeVar, Generic

from plugin_base.document_source import PluginDocumentSource
from plugin_base.live_source import PluginLiveSource
from plugin_base.action import PluginActionHandler

logger = logging.getLogger(__name__)

T = TypeVar('T')

class PluginRegistry(Generic[T]):
    """Generic registry for plugins of a specific type."""
    
    def __init__(self, base_class: Type[T], plugin_type_name: str):
        self.base_class = base_class
        self.plugin_type_name = plugin_type_name
        self._plugins: Dict[str, Type[T]] = {}
    
    def register(self, plugin_class: Type[T]) -> None:
        """Register a plugin class."""
        source_type = getattr(plugin_class, 'source_type', None) or getattr(plugin_class, 'action_type', None)
        if not source_type:
            raise ValueError(f"Plugin {plugin_class.__name__} missing source_type/action_type")
        
        if source_type in self._plugins:
            logger.warning(f"Duplicate {self.plugin_type_name} '{source_type}', overwriting")
        
        self._plugins[source_type] = plugin_class
        logger.info(f"Registered {self.plugin_type_name}: {source_type} ({plugin_class.__name__})")
    
    def get(self, source_type: str) -> Type[T] | None:
        """Get a plugin class by source_type."""
        return self._plugins.get(source_type)
    
    def get_all(self) -> Dict[str, Type[T]]:
        """Get all registered plugins."""
        return self._plugins.copy()
    
    def get_all_metadata(self) -> list[dict]:
        """Get metadata for all plugins (for admin UI)."""
        result = []
        for source_type, cls in self._plugins.items():
            result.append({
                "source_type": source_type,
                "display_name": getattr(cls, 'display_name', source_type),
                "description": getattr(cls, 'description', ''),
                "category": getattr(cls, 'category', 'other'),
                "icon": getattr(cls, 'icon', ''),
                "fields": [f.to_dict() for f in cls.get_config_fields()],
            })
        return result

# Global registries
document_source_registry = PluginRegistry(PluginDocumentSource, "document_source")
live_source_registry = PluginRegistry(PluginLiveSource, "live_source")
action_registry = PluginRegistry(PluginActionHandler, "action")

def _load_plugins_from_directory(directory: Path, registry: PluginRegistry, base_class: type) -> int:
    """Load all plugins from a directory."""
    if not directory.exists():
        return 0
    
    count = 0
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all plugin subclasses in the module
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) 
                        and issubclass(obj, base_class) 
                        and obj is not base_class
                        and not getattr(obj, '_abstract', False)):
                        registry.register(obj)
                        count += 1
                        
        except Exception as e:
            logger.error(f"Failed to load plugin {py_file}: {e}", exc_info=True)
    
    return count

def discover_plugins(
    plugins_dir: Path = Path("plugins"),
    builtin_dir: Path = Path("builtin_plugins")
) -> dict:
    """
    Discover and register all plugins.
    
    Loads from both builtin_plugins/ (shipped with app) and plugins/ (user plugins).
    User plugins can override builtin plugins with the same source_type.
    
    Returns:
        Dict with counts: {"document_sources": N, "live_sources": N, "actions": N}
    """
    counts = {"document_sources": 0, "live_sources": 0, "actions": 0}
    
    # Load builtin plugins first
    for subdir, registry, base_class, key in [
        ("document_sources", document_source_registry, PluginDocumentSource, "document_sources"),
        ("live_sources", live_source_registry, PluginLiveSource, "live_sources"),
        ("actions", action_registry, PluginActionHandler, "actions"),
    ]:
        counts[key] += _load_plugins_from_directory(
            builtin_dir / subdir, registry, base_class
        )
    
    # Load user plugins (can override builtins)
    for subdir, registry, base_class, key in [
        ("document_sources", document_source_registry, PluginDocumentSource, "document_sources"),
        ("live_sources", live_source_registry, PluginLiveSource, "live_sources"),
        ("actions", action_registry, PluginActionHandler, "actions"),
    ]:
        counts[key] += _load_plugins_from_directory(
            plugins_dir / subdir, registry, base_class
        )
    
    logger.info(
        f"Plugin discovery complete: {counts['document_sources']} document sources, "
        f"{counts['live_sources']} live sources, {counts['actions']} actions"
    )
    
    return counts

def get_document_source_plugin(source_type: str) -> Type[PluginDocumentSource] | None:
    """Get a document source plugin by type."""
    return document_source_registry.get(source_type)

def get_live_source_plugin(source_type: str) -> Type[PluginLiveSource] | None:
    """Get a live source plugin by type."""
    return live_source_registry.get(source_type)

def get_action_plugin(action_type: str) -> Type[PluginActionHandler] | None:
    """Get an action plugin by type."""
    return action_registry.get(action_type)
```

### 1.3 Checklist - Phase 1

- [x] **1.1.1** Create `plugin_base/` directory structure
- [x] **1.1.2** Implement `plugin_base/common.py` with FieldDefinition, ValidationResult
- [x] **1.1.3** Write unit tests for common dataclasses (40 tests)
- [x] **1.2.1** Implement `plugin_base/loader.py` with PluginRegistry
- [x] **1.2.2** Write unit tests for plugin loader (24 tests with mock plugins)
- [x] **1.2.3** Add plugin discovery call to app startup (`proxy.py`)
- [x] **1.3.1** Create `plugins/` directory with README.md
- [x] **1.3.2** Create `builtin_plugins/` directory structure
- [x] **1.3.3** Add database migration for `plugin_configs` table (auto-created via SQLAlchemy)
- [x] **1.3.4** Implement CRUD functions in `db/plugin_configs.py`

**Phase 1 Completed: 2026-01-19**

Additional work completed:
- Created stub base classes (`document_source.py`, `live_source.py`, `action.py`) to support loader testing
- Added mock plugin fixtures for testing (`tests/plugins/fixtures/`)
- Fixed datetime deprecation warning in `live_source.py`
- Added `PluginConfig` model to `db/models.py`
- Updated `db/__init__.py` with all new exports

---

## Phase 2: Action Plugins (Week 3-4)

Actions are highest value - enable new capabilities without touching core code.

### 2.1 Action Base Class

**File: `plugin_base/action.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from plugin_base.common import FieldDefinition, ValidationResult, ValidationError

class ActionRisk(Enum):
    """
    Risk level determines approval requirements.
    
    - READ_ONLY: No side effects, never needs approval
    - LOW: Minor side effects, can be pre-approved
    - MEDIUM: Visible side effects, confirmation recommended
    - HIGH: Significant side effects, always confirm
    - DESTRUCTIVE: Irreversible, cannot be automated
    """
    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"

@dataclass
class ActionDefinition:
    """
    Defines an action that can be invoked by the LLM.
    
    Each plugin declares its available actions with parameters,
    risk levels, and examples for the LLM prompt.
    """
    name: str  # e.g., "create", "send", "delete"
    description: str  # Human-readable, included in LLM prompt
    risk: ActionRisk
    params: list[FieldDefinition]  # Parameters the LLM must provide
    examples: list[dict] = field(default_factory=list)  # Example param values
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "risk": self.risk.value,
            "params": [p.to_dict() for p in self.params],
            "examples": self.examples,
        }

@dataclass
class ActionResult:
    """Result from action execution."""
    success: bool
    message: str  # Human-readable result message
    data: Any = None  # Structured result data (for chaining)
    error: Optional[str] = None
    
@dataclass
class ActionContext:
    """Context passed to execute()."""
    session_key: Optional[str] = None
    user_tags: list[str] = field(default_factory=list)
    smart_alias_name: str = ""
    conversation_id: Optional[str] = None
    # Extensible - add more context as needed

class PluginActionHandler(ABC):
    """
    Base class for action handler plugins.
    
    Subclasses define:
    - action_type: Unique identifier (e.g., "email", "slack", "todoist")
    - display_name: Human-readable name for admin UI
    - get_config_fields(): Configuration required (API keys, OAuth, etc.)
    - get_actions(): Available actions with parameters
    - execute(): Action execution logic
    
    Example usage in LLM response:
    ```xml
    <smart_action type="todoist" action="create">
    {"content": "Review PR", "due_string": "tomorrow"}
    </smart_action>
    ```
    """
    
    # --- Required class attributes (override in subclass) ---
    action_type: str  # Unique identifier
    display_name: str  # Shown in admin UI
    description: str  # Help text
    
    # --- Optional class attributes ---
    icon: str = "⚡"
    category: str = "other"  # For grouping in UI: "communication", "productivity", "home", etc.
    
    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """
        Define configuration fields for the admin UI.
        
        These are set once when configuring the plugin, not per-request.
        Examples: API keys, OAuth accounts, default settings.
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_actions(cls) -> list[ActionDefinition]:
        """
        Define available actions and their parameters.
        
        These are exposed to the LLM via system prompt injection.
        """
        pass
    
    @classmethod
    def get_action(cls, action_name: str) -> ActionDefinition | None:
        """Get a specific action definition by name."""
        return next((a for a in cls.get_actions() if a.name == action_name), None)
    
    @classmethod
    def get_llm_instructions(cls) -> str:
        """
        Generate instructions for LLM system prompt.
        
        Override for custom formatting. Default format:
        
        ## Email Actions
        
        ### email:draft_new
        Create a new email draft.
        Parameters: account (required), to (required), subject (required), body (required)
        
        Example:
        ```xml
        <smart_action type="email" action="draft_new">
        {"account": "user@gmail.com", "to": ["recipient@example.com"], "subject": "Hello", "body": "..."}
        </smart_action>
        ```
        """
        actions = cls.get_actions()
        if not actions:
            return ""
        
        lines = [f"## {cls.display_name}", ""]
        
        for action in actions:
            lines.append(f"### {cls.action_type}:{action.name}")
            lines.append(action.description)
            
            # Parameter list
            param_parts = []
            for p in action.params:
                req = "required" if p.required else "optional"
                param_parts.append(f"{p.name} ({req})")
            if param_parts:
                lines.append(f"Parameters: {', '.join(param_parts)}")
            
            # Example
            if action.examples:
                lines.append("")
                lines.append("Example:")
                lines.append("```xml")
                lines.append(f'<smart_action type="{cls.action_type}" action="{action.name}">')
                # Pretty-print first example
                import json
                lines.append(json.dumps(action.examples[0], indent=2))
                lines.append("</smart_action>")
                lines.append("```")
            
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """
        Validate plugin configuration.
        
        Override for custom validation logic.
        """
        errors = []
        for field_def in cls.get_config_fields():
            value = config.get(field_def.name)
            
            # Required check
            if field_def.required and not value:
                errors.append(ValidationError(field_def.name, "This field is required"))
                continue
            
            if value is None:
                continue
            
            # Type-specific validation
            if field_def.min_length and len(str(value)) < field_def.min_length:
                errors.append(ValidationError(field_def.name, f"Minimum length is {field_def.min_length}"))
            if field_def.max_length and len(str(value)) > field_def.max_length:
                errors.append(ValidationError(field_def.name, f"Maximum length is {field_def.max_length}"))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize with validated configuration.
        
        Args:
            config: Dict matching get_config_fields() definitions
        """
        pass
    
    @abstractmethod
    def execute(self, action: str, params: dict, context: ActionContext) -> ActionResult:
        """
        Execute an action.
        
        Args:
            action: Action name (from get_actions())
            params: Parameters provided by LLM
            context: Execution context (session, user, etc.)
        
        Returns:
            ActionResult with success/failure and message
        """
        pass
    
    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """
        Validate action parameters before execution.
        
        Override for custom validation.
        """
        action_def = self.get_action(action)
        if not action_def:
            return ValidationResult(
                valid=False, 
                errors=[ValidationError("action", f"Unknown action: {action}")]
            )
        
        errors = []
        for param_def in action_def.params:
            value = params.get(param_def.name)
            
            if param_def.required and value is None:
                errors.append(ValidationError(param_def.name, "Required parameter missing"))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    def get_approval_summary(self, action: str, params: dict) -> str:
        """
        Generate human-readable summary for approval UI.
        
        Override for action-specific summaries.
        """
        action_def = self.get_action(action)
        if action_def and action_def.risk == ActionRisk.DESTRUCTIVE:
            return f"⚠️ DESTRUCTIVE: {self.action_type}:{action}"
        return f"{self.action_type}:{action} with {len(params)} parameters"
    
    def get_action_risk(self, action: str) -> ActionRisk:
        """Get risk level for an action."""
        action_def = self.get_action(action)
        return action_def.risk if action_def else ActionRisk.HIGH
    
    def is_available(self) -> bool:
        """Check if plugin is properly configured and available."""
        return True
    
    def test_connection(self) -> tuple[bool, str]:
        """Test the plugin configuration."""
        return True, "OK"
```

### 2.2 OAuth Mixin

**File: `plugin_base/oauth.py`**

```python
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

class OAuthMixin:
    """
    Mixin for plugins that use OAuth authentication.
    
    Provides common token refresh and authenticated request methods.
    Requires the plugin to set:
    - self.oauth_account_id: ID of the stored OAuth token
    - self.oauth_provider: Provider name (e.g., "google", "slack")
    
    Usage:
        class MyPlugin(OAuthMixin, PluginActionHandler):
            def __init__(self, config: dict):
                self.oauth_account_id = config["oauth_account_id"]
                self.oauth_provider = "google"
                self._init_oauth_client()
    """
    
    oauth_account_id: int
    oauth_provider: str
    _oauth_client: Optional[httpx.Client] = None
    _access_token: Optional[str] = None
    
    def _init_oauth_client(self) -> None:
        """Initialize the OAuth HTTP client."""
        self._oauth_client = httpx.Client(timeout=30)
        self._refresh_token_if_needed()
    
    def _refresh_token_if_needed(self) -> bool:
        """
        Check token expiry and refresh if needed.
        
        Returns True if we have a valid token.
        """
        from db.oauth_tokens import get_oauth_token, update_oauth_token_data
        
        token_record = get_oauth_token(self.oauth_account_id)
        if not token_record:
            logger.error(f"OAuth token not found: {self.oauth_account_id}")
            return False
        
        token_data = token_record.token_data
        
        # Check if token is expired
        import time
        expires_at = token_data.get("expires_at", 0)
        if expires_at and time.time() > expires_at - 300:  # 5 min buffer
            # Refresh the token
            refreshed = self._do_token_refresh(token_data)
            if refreshed:
                update_oauth_token_data(self.oauth_account_id, refreshed)
                token_data = refreshed
            else:
                return False
        
        self._access_token = token_data.get("access_token")
        return bool(self._access_token)
    
    def _do_token_refresh(self, token_data: dict) -> Optional[dict]:
        """
        Perform OAuth token refresh.
        
        Override for provider-specific refresh logic.
        """
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            logger.error("No refresh token available")
            return None
        
        # Default implementation for Google-style OAuth
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")
        token_url = token_data.get("token_url", "https://oauth2.googleapis.com/token")
        
        try:
            response = httpx.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=30
            )
            response.raise_for_status()
            new_token = response.json()
            
            # Merge with existing data (preserve refresh_token if not returned)
            result = token_data.copy()
            result["access_token"] = new_token["access_token"]
            if "refresh_token" in new_token:
                result["refresh_token"] = new_token["refresh_token"]
            if "expires_in" in new_token:
                import time
                result["expires_at"] = time.time() + new_token["expires_in"]
            
            return result
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    def _get_auth_headers(self) -> dict:
        """Get authorization headers for requests."""
        self._refresh_token_if_needed()
        return {"Authorization": f"Bearer {self._access_token}"}
    
    def oauth_get(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated GET request."""
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.get(url, headers=headers, **kwargs)
    
    def oauth_post(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated POST request."""
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.post(url, headers=headers, **kwargs)
    
    def oauth_patch(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated PATCH request."""
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.patch(url, headers=headers, **kwargs)
    
    def oauth_delete(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated DELETE request."""
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())
        return self._oauth_client.delete(url, headers=headers, **kwargs)
```

### 2.3 Example Action Plugin

**File: `builtin_plugins/actions/todoist.py`**

```python
"""
Todoist action plugin - create and manage tasks.

This serves as the reference implementation for action plugins.
"""

import httpx
from plugin_base.action import (
    PluginActionHandler,
    ActionDefinition,
    ActionRisk,
    ActionResult,
    ActionContext,
)
from plugin_base.common import FieldDefinition, FieldType

class TodoistActionHandler(PluginActionHandler):
    """Create and manage Todoist tasks."""
    
    action_type = "todoist"
    display_name = "Todoist"
    description = "Create, complete, and manage tasks in Todoist"
    icon = "✅"
    category = "productivity"
    
    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="api_token",
                label="API Token",
                field_type=FieldType.PASSWORD,
                required=True,
                help_text="From Todoist Settings > Integrations > Developer > API token"
            ),
            FieldDefinition(
                name="default_project",
                label="Default Project",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Project name for new tasks (leave empty for Inbox)"
            ),
        ]
    
    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="create",
                description="Create a new task in Todoist",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(name="content", label="Task title", field_type=FieldType.TEXT, required=True),
                    FieldDefinition(name="description", label="Description", field_type=FieldType.TEXTAREA, required=False),
                    FieldDefinition(name="due_string", label="Due date (natural language)", field_type=FieldType.TEXT, required=False),
                    FieldDefinition(name="priority", label="Priority (1=low, 4=urgent)", field_type=FieldType.INTEGER, required=False, default=1),
                    FieldDefinition(name="project", label="Project name", field_type=FieldType.TEXT, required=False),
                    FieldDefinition(name="labels", label="Labels (comma-separated)", field_type=FieldType.TEXT, required=False),
                ],
                examples=[
                    {"content": "Review quarterly report", "due_string": "tomorrow 2pm", "priority": 2},
                    {"content": "Call dentist", "due_string": "next monday", "labels": "health,calls"},
                ]
            ),
            ActionDefinition(
                name="complete",
                description="Mark a task as complete",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(name="task_id", label="Task ID", field_type=FieldType.TEXT, required=True),
                ],
                examples=[{"task_id": "123456789"}]
            ),
            ActionDefinition(
                name="update",
                description="Update an existing task",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(name="task_id", label="Task ID", field_type=FieldType.TEXT, required=True),
                    FieldDefinition(name="content", label="New title", field_type=FieldType.TEXT, required=False),
                    FieldDefinition(name="description", label="New description", field_type=FieldType.TEXTAREA, required=False),
                    FieldDefinition(name="due_string", label="New due date", field_type=FieldType.TEXT, required=False),
                    FieldDefinition(name="priority", label="New priority", field_type=FieldType.INTEGER, required=False),
                ],
                examples=[{"task_id": "123456789", "due_string": "next friday", "priority": 3}]
            ),
            ActionDefinition(
                name="delete",
                description="Delete a task permanently",
                risk=ActionRisk.DESTRUCTIVE,
                params=[
                    FieldDefinition(name="task_id", label="Task ID", field_type=FieldType.TEXT, required=True),
                ],
                examples=[{"task_id": "123456789"}]
            ),
        ]
    
    def __init__(self, config: dict):
        self.api_token = config["api_token"]
        self.default_project = config.get("default_project")
        self.client = httpx.Client(
            base_url="https://api.todoist.com/rest/v2",
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=15
        )
    
    def execute(self, action: str, params: dict, context: ActionContext) -> ActionResult:
        try:
            if action == "create":
                return self._create_task(params)
            elif action == "complete":
                return self._complete_task(params)
            elif action == "update":
                return self._update_task(params)
            elif action == "delete":
                return self._delete_task(params)
            else:
                return ActionResult(success=False, message="", error=f"Unknown action: {action}")
        except httpx.HTTPStatusError as e:
            return ActionResult(success=False, message="", error=f"API error: {e.response.status_code}")
        except Exception as e:
            return ActionResult(success=False, message="", error=str(e))
    
    def _create_task(self, params: dict) -> ActionResult:
        data = {"content": params["content"]}
        
        if params.get("description"):
            data["description"] = params["description"]
        if params.get("due_string"):
            data["due_string"] = params["due_string"]
        if params.get("priority"):
            data["priority"] = int(params["priority"])
        if params.get("labels"):
            data["labels"] = [l.strip() for l in params["labels"].split(",")]
        
        # Resolve project name to ID if provided
        project_name = params.get("project") or self.default_project
        if project_name:
            project_id = self._resolve_project_id(project_name)
            if project_id:
                data["project_id"] = project_id
        
        response = self.client.post("/tasks", json=data)
        response.raise_for_status()
        task = response.json()
        
        return ActionResult(
            success=True,
            message=f"Created task: {task['content']}",
            data={"task_id": task["id"], "url": task.get("url")}
        )
    
    def _complete_task(self, params: dict) -> ActionResult:
        response = self.client.post(f"/tasks/{params['task_id']}/close")
        response.raise_for_status()
        
        return ActionResult(
            success=True,
            message=f"Completed task {params['task_id']}"
        )
    
    def _update_task(self, params: dict) -> ActionResult:
        task_id = params.pop("task_id")
        data = {k: v for k, v in params.items() if v is not None}
        
        if not data:
            return ActionResult(success=False, message="", error="No fields to update")
        
        response = self.client.post(f"/tasks/{task_id}", json=data)
        response.raise_for_status()
        task = response.json()
        
        return ActionResult(
            success=True,
            message=f"Updated task: {task['content']}",
            data={"task_id": task["id"]}
        )
    
    def _delete_task(self, params: dict) -> ActionResult:
        response = self.client.delete(f"/tasks/{params['task_id']}")
        response.raise_for_status()
        
        return ActionResult(
            success=True,
            message=f"Deleted task {params['task_id']}"
        )
    
    def _resolve_project_id(self, project_name: str) -> str | None:
        """Resolve project name to ID."""
        try:
            response = self.client.get("/projects")
            response.raise_for_status()
            projects = response.json()
            
            for project in projects:
                if project["name"].lower() == project_name.lower():
                    return project["id"]
            return None
        except Exception:
            return None
    
    def get_approval_summary(self, action: str, params: dict) -> str:
        if action == "create":
            due = f" (due: {params['due_string']})" if params.get("due_string") else ""
            return f"Create Todoist task: \"{params.get('content', '?')}\"{due}"
        elif action == "complete":
            return f"Mark Todoist task {params.get('task_id')} as complete"
        elif action == "delete":
            return f"⚠️ DELETE Todoist task {params.get('task_id')} (permanent)"
        return super().get_approval_summary(action, params)
    
    def test_connection(self) -> tuple[bool, str]:
        try:
            response = self.client.get("/projects")
            response.raise_for_status()
            projects = response.json()
            return True, f"Connected. Found {len(projects)} projects."
        except Exception as e:
            return False, str(e)
```

### 2.4 Checklist - Phase 2

- [x] **2.1.1** Implement `plugin_base/action.py` with ActionRisk, ActionDefinition, ActionResult, ActionContext (done in Phase 1)
- [x] **2.1.2** Implement PluginActionHandler base class with all methods (done in Phase 1)
- [x] **2.1.3** Write unit tests for ActionHandler validation (39 tests in test_action.py)
- [x] **2.1.4** Write unit tests for LLM instruction generation (included in test_action.py)
- [x] **2.2.1** Implement `plugin_base/oauth.py` OAuthMixin
- [x] **2.2.2** Write unit tests for OAuth token refresh logic (28 tests in test_oauth.py)
- [x] **2.3.1** Create `builtin_plugins/actions/` directory (done in Phase 1)
- [x] **2.3.2** Implement Todoist action plugin (reference implementation)
- [x] **2.3.3** Write integration tests for Todoist plugin (35 tests in test_todoist.py)
- [x] **2.4.1** Update action executor to use plugin registry (PluginActionAdapter in actions/loader.py)
- [x] **2.4.2** Update action parser to discover actions from plugins (parser is action-agnostic)
- [x] **2.4.3** Update system prompt injection to use plugin LLM instructions (PluginActionAdapter.get_system_prompt_instructions)
- [x] **2.5.1** Add admin API endpoint: `GET /api/plugins/actions`
- [x] **2.5.2** Add admin API endpoint: `GET /api/plugins/actions/{type}` (with fields)
- [x] **2.5.3** Add admin API endpoint: `POST /api/plugins/configs/{id}/test`
- [x] **2.5.4** Add admin API endpoints for plugin config CRUD
- [x] **2.6.1** Migrate existing notification action handler to plugin (43 tests)
- [x] **2.6.2** Migrate existing calendar action handler to plugin (56 tests)
- [x] **2.6.3** Migrate existing email action handler to plugin (53 tests)
- [x] **2.6.4** Migrate existing schedule action handler to plugin (56 tests)

**Phase 2 Completed: 2026-01-19**

Test counts (374 total):
- test_common.py: 40 tests
- test_loader.py: 24 tests
- test_action.py: 39 tests
- test_oauth.py: 28 tests
- test_todoist.py: 35 tests
- test_notification.py: 43 tests
- test_calendar.py: 56 tests
- test_email.py: 53 tests
- test_schedule.py: 56 tests

### Migrated Plugins

| Plugin | File | Actions | Tests |
|--------|------|---------|-------|
| Notification | `builtin_plugins/actions/notification.py` | send | 43 |
| Calendar | `builtin_plugins/actions/calendar.py` | create, update, delete | 56 |
| Email | `builtin_plugins/actions/email.py` | draft_new, draft_reply, draft_forward, send_new, send_reply, send_forward, label, archive, mark_read, mark_unread | 53 |
| Schedule | `builtin_plugins/actions/schedule.py` | prompt, cancel | 56 |
| Todoist | `builtin_plugins/actions/todoist.py` | create, complete, update, delete, list | 35 |

All plugins use:
- `OAuthMixin` for Google OAuth (calendar, email, schedule)
- `PluginActionHandler` base class
- Dynamic config fields via `get_config_fields()`
- Action definitions with risk levels via `get_actions()`
- LLM instructions via `get_llm_instructions()`
- Parameter validation via `validate_action_params()`
- Connection testing via `test_connection()`

---

## Phase 3: Live Source Plugins (Week 5-6)

### 3.1 Live Source Base Class

**File: `plugin_base/live_source.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from plugin_base.common import FieldDefinition, ValidationResult

@dataclass
class ParamDefinition:
    """
    Defines a parameter the designator can pass at query time.
    
    Unlike FieldDefinition (static config), these are dynamic per-request.
    """
    name: str
    description: str  # Included in designator prompt
    param_type: str  # "string", "integer", "number", "boolean", "datetime"
    required: bool = False
    default: Any = None
    examples: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "param_type": self.param_type,
            "required": self.required,
            "default": self.default,
            "examples": self.examples,
        }

@dataclass
class LiveDataResult:
    """Result from fetch()."""
    success: bool
    data: Any = None  # Raw data (for caching)
    formatted: str = ""  # Formatted string for LLM context
    error: Optional[str] = None
    cache_ttl: int = 300  # Seconds, 0 = don't cache, -1 = cache forever
    timestamp: datetime = field(default_factory=datetime.utcnow)

class PluginLiveSource(ABC):
    """
    Base class for live data source plugins.
    
    Live sources fetch real-time data at request time (unlike document sources
    which index content for RAG). The designator selects relevant sources and
    provides parameters based on the user's query.
    
    Subclasses define:
    - source_type: Unique identifier
    - data_type: Category (weather, finance, sports, etc.)
    - best_for: Description for designator
    - get_config_fields(): Static configuration (API keys, etc.)
    - get_param_definitions(): Dynamic parameters for designator
    - fetch(): Data fetching logic
    """
    
    # --- Required class attributes ---
    source_type: str
    display_name: str
    description: str
    data_type: str  # Category for grouping
    best_for: str  # Description for designator prompt
    
    # --- Optional ---
    icon: str = "🔌"
    default_cache_ttl: int = 300
    
    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI (API keys, settings, etc.)."""
        pass
    
    @classmethod
    @abstractmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        """Parameters the designator can pass at query time."""
        pass
    
    @classmethod
    def get_designator_hint(cls) -> str:
        """
        Generate hint for designator prompt.
        
        Override for custom formatting.
        """
        params = cls.get_param_definitions()
        
        param_parts = []
        for p in params:
            req = "" if p.required else "optional, "
            examples = f" e.g., {', '.join(p.examples)}" if p.examples else ""
            param_parts.append(f"{p.name} ({req}{p.description}{examples})")
        
        param_str = "; ".join(param_parts) if param_parts else "no parameters"
        return f"Parameters: {param_str}. {cls.best_for}"
    
    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """Validate plugin configuration."""
        from plugin_base.common import ValidationError
        errors = []
        for field_def in cls.get_config_fields():
            if field_def.required and not config.get(field_def.name):
                errors.append(ValidationError(field_def.name, "Required"))
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with validated config."""
        pass
    
    @abstractmethod
    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live data based on designator-provided params.
        
        Args:
            params: Dict from designator, matches get_param_definitions()
        
        Returns:
            LiveDataResult with formatted string for context injection
        """
        pass
    
    def validate_params(self, params: dict) -> ValidationResult:
        """Validate query parameters."""
        from plugin_base.common import ValidationError
        errors = []
        for param_def in self.get_param_definitions():
            if param_def.required and params.get(param_def.name) is None:
                errors.append(ValidationError(param_def.name, "Required parameter"))
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    def is_available(self) -> bool:
        """Check if source is configured and available."""
        return True
    
    def test_connection(self) -> tuple[bool, str]:
        """Test the connection with a sample query."""
        try:
            result = self.fetch({})
            if result.success:
                preview = result.formatted[:100] + "..." if len(result.formatted) > 100 else result.formatted
                return True, f"OK: {preview}"
            return False, result.error or "Unknown error"
        except Exception as e:
            return False, str(e)
```

### 3.2 Checklist - Phase 3

- [x] **3.1.1** Implement `plugin_base/live_source.py` with ParamDefinition, LiveDataResult (done in Phase 1)
- [x] **3.1.2** Implement PluginLiveSource base class (done in Phase 1)
- [x] **3.1.3** Write unit tests for live source validation (`tests/plugins/test_live_source.py`)
- [x] **3.1.4** Write unit tests for designator hint generation (included in test_live_source.py)
- [x] **3.2.1** Create `builtin_plugins/live_sources/` directory (done in Phase 1)
- [x] **3.2.2** Migrate Weather providers to plugins (`open_meteo.py`, `accuweather.py`)
- [x] **3.2.3** Migrate Stocks provider to plugin (`stocks.py`)
- [x] **3.2.4** Migrate Transport provider to plugin (`transport.py`)
- [x] **3.2.5** Migrate GoogleMaps provider to plugin (`google_maps.py`)
- [x] **3.2.6** Migrate Routes provider to plugin (`routes.py`)
- [x] **3.3.1** Update smart_enricher.py to use plugin registry for live sources
- [x] **3.3.2** Update designator prompt generation to use plugin hints
- [x] **3.3.3** Update live data fetching to instantiate plugins (`PluginLiveSourceAdapter` in `live/sources.py`)
- [x] **3.4.1** Add admin API endpoint: `GET /api/plugins/live-sources`
- [x] **3.4.2** Add admin API endpoint: `GET /api/plugins/live-sources/{source_type}` (with fields, params, designator_hint)
- [x] **3.4.3** Add admin UI for live source plugin configuration (dynamic form rendering from plugin fields)
- [x] **3.4.4** Auto-seed database entries for registered live source plugins on startup
- [ ] **3.5.1** Write integration tests with mocked APIs

### Migrated Live Source Plugins (Current State)

All live source plugins have been migrated to "smart" versions that combine multiple APIs and support natural language inputs.

| Plugin | File | Source Type | Description |
|--------|------|-------------|-------------|
| Smart Weather | `smart_weather.py` | `open_meteo_enhanced` | Open-Meteo + AccuWeather with natural language locations |
| Smart Sports | `smart_sports.py` | `sportapi7_enhanced` | Sports data with team name resolution |
| Smart Stocks | `smart_stocks.py` | `finnhub_enhanced` | Finnhub + Alpha Vantage with company name resolution |
| Smart Health | `smart_health.py` | `oura_withings` | Oura + Withings unified health data |
| Smart Transport | `smart_transport.py` | `transportapi_enhanced` | UK trains with station name resolution |
| Smart Places | `smart_places.py` | `google_places_enhanced` | Google Places with location geocoding |
| Smart Routes | `routes.py` | `google_routes_enhanced` | Google Routes with natural language locations/times |
| Smart News | `smart_news.py` | `realtime_news` | Real-Time News Data API via RapidAPI |
| Smart Amazon | `smart_amazon.py` | `realtime_amazon` | Amazon product info via RapidAPI |

### 3.3 Smart Providers (Phase 3b)

**Strategy:** Build "smart" versions of live sources directly as plugins, rather than migrating basic implementations and enhancing later. Smart providers handle complexity internally:

- **Natural language inputs** - Accept location names, team names, company names instead of IDs/coordinates
- **Automatic lookups** - Geocoding, entity resolution, ID lookups with caching
- **Natural time parsing** - "tomorrow 9am", "next Saturday", "in 2 hours"
- **Context-ready output** - Pre-formatted text optimized for LLM consumption
- **Intelligent caching** - Cache lookups (long TTL) separately from live data (short TTL)

The `routes.py` plugin serves as the reference implementation for this pattern.

#### 3.3.1 Smart Weather Plugin (Unified Open-Meteo + AccuWeather)

**File:** `builtin_plugins/live_sources/smart_weather.py`

Unified weather interface combining Open-Meteo (free, always available) and AccuWeather (optional, enhanced).
Accept natural language locations, handle geocoding internally, provide comprehensive weather context.

**Parameters (natural language):**
- `location` - "London", "Paris, France", "10 Downing Street" (geocoded internally)
- `when` - "now", "tomorrow", "this weekend", "next 3 days" (optional)

**Data Sources:**
- **Open-Meteo** (free, no API key): Always available, forecasts up to 16 days
- **AccuWeather** (optional, API key): Enhanced current conditions with feels-like, UV index, detailed precipitation
  - Daily forecasts: 1, 5, 10, 15 days (free tier: 1 and 5 day; premium: longer)
  - Hourly forecasts: 1, 12, 24, 72 hours
  - Current conditions with extended details

**Features:**
- Geocode location names to coordinates (30-day cache)
- AccuWeather location key resolution (30-day cache)
- Current conditions from AccuWeather (preferred) or Open-Meteo (fallback)
- Combined daily forecasts with AccuWeather condition descriptions
- Hourly forecasts when querying "today" or "now"
- Graceful fallback if AccuWeather premium endpoints unavailable
- Context-formatted output optimized for LLM consumption

**Example output:**
```
**Current weather in London, England, United Kingdom:**
- Light rain, 8°C (feels like 3°C)
- Humidity: 88%
- Wind: 27 km/h SSE
_via AccuWeather_

**Forecast for this weekend:**
- **Saturday (2026-01-24)**: Partly sunny, partly cloudy at night, 5°C to 9°C
- **Sunday (2026-01-25)**: Foggy, 3°C to 8°C

_Data: AccuWeather + Open-Meteo_
```

**Checklist:**
- [x] **3.3.1.1** Create `smart_weather.py` plugin with natural language location support
- [x] **3.3.1.2** Implement geocoding with 30-day cache (reuse from routes.py)
- [x] **3.3.1.3** Implement natural time parsing for forecast periods
- [x] **3.3.1.4** Format output as context-ready weather summary
- [x] **3.3.1.5** Combine Open-Meteo (free) + AccuWeather (optional) into unified plugin
- [x] **3.3.1.6** Add AccuWeather daily forecasts (1/5/10/15 day) and hourly forecasts (1/12/24/72 hour)
- [x] **3.3.1.7** Remove standalone AccuWeather plugin (merged into smart_weather.py)
- [ ] **3.3.1.8** Write unit tests with mocked API responses

#### 3.3.2 Smart Sports Plugin

**File:** `builtin_plugins/live_sources/smart_sports.py`

Accept team/league names in natural language, resolve to IDs internally, provide match/fixture context.

**Parameters (natural language):**
- `team` - "Arsenal", "Manchester United", "Lakers" (resolved to team ID)
- `league` - "Premier League", "NBA", "Champions League" (optional, helps disambiguation)
- `query_type` - "next_match", "recent_results", "standings", "live_scores" (optional, default: contextual)

**Features:**
- Team name → ID resolution with caching (entity cache, 90-day TTL)
- League detection from team name when unambiguous
- Smart query type: if match is today/tomorrow, include live scores; otherwise show upcoming
- Context-formatted output: "Arsenal's next match: vs Chelsea, Saturday 3pm at Emirates Stadium. Arsenal are 3rd in the Premier League with 45 points."
- Comprehensive TOURNAMENT_IDS mapping for UK leagues: Premier League (17), Championship (18), League One (19), League Two (20), FA Cup (29), EFL Cup (21)

**Checklist:**
- [x] **3.3.2.1** Create `smart_sports.py` plugin structure
- [x] **3.3.2.2** Implement team name → ID resolution with entity cache
- [x] **3.3.2.3** Implement league detection and disambiguation
- [x] **3.3.2.4** Add next match / recent results / standings / live scores queries
- [x] **3.3.2.5** Format output as context-ready sports summary
- [x] **3.3.2.6** Add comprehensive TOURNAMENT_IDS with UK football leagues
- [ ] **3.3.2.7** Write unit tests with mocked API responses

#### 3.3.3 Smart Stocks Plugin (Unified Finnhub + Alpha Vantage)

**File:** `builtin_plugins/live_sources/smart_stocks.py`

Unified finance interface combining Finnhub (US stocks, 500 calls/min free) and Alpha Vantage (UK stocks, 25 calls/day free).

**Parameters (natural language):**
- `query` - "Apple stock", "AAPL", "Tesla share price", "my portfolio", "Tesco", "TSCO.LON" (flexible input)
- `period` - "today", "this week", "YTD", "1 year" (optional, for performance context)

**Data Sources:**
- **Finnhub** (free tier: 500 calls/min): US stocks and international markets
- **Alpha Vantage** (free tier: 25 calls/day): UK stocks with .LON suffix, better for LSE

**Features:**
- Company name → ticker resolution with hardcoded mappings for common stocks
- Automatic API routing: UK stocks (.LON suffix) → Alpha Vantage, others → Finnhub
- Comprehensive UK stock mappings: Tesco, Lloyds, BP, Barclays, HSBC, Shell, etc.
- Fund web lookup fallback for mutual funds (L&G, Vanguard funds)
- Uses configured search/scraper providers for fund price lookups
- Portfolio support (configured tickers, show aggregate performance)
- Context-formatted output optimized for LLM consumption

**Example output:**
```
**Stock Prices:**
- Apple (AAPL): $178.50 (+2.3%)
- Tesco (TSCO.LON): £2.85 (+0.7%)
- L&G Global Technology: £3.45 (from web lookup)
```

**Checklist:**
- [x] **3.3.3.1** Create unified `smart_stocks.py` with Finnhub + Alpha Vantage
- [x] **3.3.3.2** Implement company name → ticker resolution with hardcoded mappings
- [x] **3.3.3.3** Add automatic API routing (UK → Alpha Vantage, US → Finnhub)
- [x] **3.3.3.4** Add comprehensive UK stock symbol mappings
- [x] **3.3.3.5** Implement fund web lookup using configured search/scraper providers
- [x] **3.3.3.6** Format output as context-ready financial summary
- [x] **3.3.3.7** Remove standalone Finnhub plugin (merged)
- [ ] **3.3.3.8** Write unit tests with mocked API responses

#### 3.3.4 Smart Health Plugin

**File:** `builtin_plugins/live_sources/smart_health.py`

Unified health data interface supporting Oura Ring and Withings devices with natural language queries.

**Parameters (natural language):**
- `query_type` - "sleep", "readiness", "activity", "weight", "body_composition", "heart_rate", "hrv"
- `period` - "today", "yesterday", "this week", "last 7 days" (optional)

**Features:**
- Works with either or both Oura and Withings (auto-detects configured providers)
- Automatic routing: sleep queries go to both, weight to Withings, readiness to Oura
- Combines data from multiple sources when both are configured
- Natural language period parsing
- Context-formatted health summaries

**Checklist:**
- [x] **3.3.4.1** Create `smart_health.py` plugin with unified interface
- [x] **3.3.4.2** Implement Oura API integration (sleep, readiness, activity, HRV)
- [x] **3.3.4.3** Implement Withings API integration (weight, body composition, sleep)
- [x] **3.3.4.4** Add query routing based on data type and available providers
- [x] **3.3.4.5** Format output as context-ready health summary
- [ ] **3.3.4.6** Write unit tests with mocked API responses

#### 3.3.5 Smart Transport Plugin

**File:** `builtin_plugins/live_sources/smart_transport.py`

UK public transport interface using TransportAPI with natural language support.

**Parameters (natural language):**
- `query_type` - "journey", "departures", "service", "bus"
- `origin` - Station, stop, or location name (e.g., "Harpenden", "Kings Cross")
- `destination` - For journeys or departure filtering (e.g., "trains to Luton")
- `time` - Natural language time ("tomorrow 9am", "in 30 minutes")
- `arrive_by` - "true" if time is arrival time (default: departure time)
- `service_id` - Specific train service for status queries

**Features:**
- Multi-modal journey planning via `/public_journey.json` endpoint
- Train departures with optional destination filtering ("calling_at" parameter)
- Service status and delay checking
- Bus departures from nearby stops
- Station name resolution with 30-day caching
- Natural language time parsing
- Hardcoded lookup table for 60+ common UK stations

**Checklist:**
- [x] **3.3.5.1** Create `smart_transport.py` plugin with TransportAPI integration
- [x] **3.3.5.2** Implement station/stop name resolution with caching
- [x] **3.3.5.3** Add journey planning endpoint integration
- [x] **3.3.5.4** Add departure board queries for trains and buses
- [x] **3.3.5.5** Add service status queries
- [x] **3.3.5.6** Format output for LLM context with delay indicators
- [ ] **3.3.5.7** Write unit tests with mocked API responses

#### 3.3.6 Smart Places Plugin (Google Places Enhanced)

**File:** `builtin_plugins/live_sources/smart_places.py`

Google Places with natural language location support and automatic geocoding.

**Parameters (natural language):**
- `query` - "Italian restaurants", "coffee shops", "pharmacies" (what to search for)
- `location` - "Soho", "near Tower Bridge", "SW1A 1AA" (natural language location/postcode)
- `open_now` - Filter to currently open places (optional)
- `min_rating` - Minimum star rating (optional)

**Features:**
- Natural language location support ("in Soho", "near Tower Bridge", "central London")
- Geocodes location names to coordinates automatically (30-day cache)
- Supports UK postcodes and addresses
- Place search with filtering (open now, rating)
- Context-formatted output with ratings, prices, addresses

**Example output:**
```
**Italian restaurants near Tower Bridge:**
1. Cecconi's City of London - ⭐ 4.3 (£££) - 8 Bury Court, EC3A 5AT
2. Obicà Mozzarella Bar - ⭐ 4.2 (££) - 11 St Mary Axe, EC3A 8BF
3. Fiume - ⭐ 4.4 (£££) - Battersea Power Station, SW11 8AL
```

**Checklist:**
- [x] **3.3.6.1** Create `smart_places.py` plugin with natural language location support
- [x] **3.3.6.2** Implement geocoding with 30-day cache
- [x] **3.3.6.3** Support postcodes and address geocoding
- [x] **3.3.6.4** Add place search with open_now and rating filters
- [x] **3.3.6.5** Format output as context-ready place listing
- [x] **3.3.6.6** Remove standalone Google Maps plugin (merged)
- [ ] **3.3.6.7** Write unit tests with mocked API responses

#### 3.3.7 Smart News Plugin (RapidAPI)

**File:** `builtin_plugins/live_sources/smart_news.py`

Real-time news via RapidAPI's Real-Time News Data API.

**Parameters (natural language):**
- `query` - Search term ("Apple earnings", "UK politics", "climate change")
- `country` - Country filter (GB, US, etc.)
- `limit` - Number of articles to return (default: 10)

**Features:**
- Configurable default country (GB by default)
- Searches across major news sources
- Returns headlines, descriptions, sources, and publication times
- Context-formatted output with clickable links

**Configuration:**
- `rapidapi_key` - RapidAPI key (can be set via `RAPIDAPI_KEY` env var)
- `default_country` - Default country for news (GB, US, DE, FR, etc.)

**Checklist:**
- [x] **3.3.7.1** Create `smart_news.py` plugin with RapidAPI integration
- [x] **3.3.7.2** Implement configurable default country
- [x] **3.3.7.3** Format output as context-ready news summary
- [x] **3.3.7.4** Add env var support for API key
- [ ] **3.3.7.5** Write unit tests with mocked API responses

#### 3.3.8 Smart Amazon Plugin (RapidAPI)

**File:** `builtin_plugins/live_sources/smart_amazon.py`

Amazon product information via RapidAPI's Real-Time Amazon Data API.

**Parameters (natural language):**
- `query` - Product name ("OnePlus Watch 3", "MacBook Pro M3", "wireless earbuds")
- `country` - Amazon store (GB, US, DE, etc.)
- `limit` - Number of products to return (default: 5)

**Features:**
- Natural language product searches
- Country code normalization (UK → GB automatically)
- Product details including price, ratings, reviews
- "About this product" bullet points and specifications
- Context-formatted output optimized for price/comparison queries

**Configuration:**
- `rapidapi_key` - RapidAPI key (can be set via `RAPIDAPI_KEY` env var)
- `default_country` - Default Amazon store (GB by default)

**Designator hint:** "BEST FOR: 'How much does X cost?', product prices, Amazon product lookups, reviews, comparisons."

**Checklist:**
- [x] **3.3.8.1** Create `smart_amazon.py` plugin with RapidAPI integration
- [x] **3.3.8.2** Implement country code normalization (UK→GB, domain→ISO)
- [x] **3.3.8.3** Add comprehensive product detail formatting
- [x] **3.3.8.4** Add env var support for API key
- [x] **3.3.8.5** Fix config loading from `auth_config_json` in PluginLiveSourceAdapter
- [ ] **3.3.8.6** Write unit tests with mocked API responses

#### Smart Provider Pattern (Reference)

All smart providers should follow this pattern from `routes.py`:

```python
class SmartWeatherLiveSource(PluginLiveSource):
    """
    Smart Weather Provider - accepts natural language locations.
    
    Unlike basic weather providers requiring coordinates,
    this provider accepts location names and handles:
    - Geocoding location names to coordinates (with 30-day caching)
    - Natural time expressions for forecast periods
    - Comprehensive weather summary formatting
    """
    
    source_type = "smart_weather"
    display_name = "Smart Weather"
    description = "Weather with natural language location support"
    data_type = "weather"
    best_for = "Weather forecasts, current conditions, weather alerts for any location by name"
    
    # Separate cache TTLs
    GEOCODE_CACHE_TTL = 86400 * 30  # 30 days for geocoding
    WEATHER_CACHE_TTL = 1800        # 30 minutes for weather data
    
    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="location",
                description="Location name (city, address, landmark)",
                param_type="string",
                required=True,
                examples=["London", "Paris, France", "Central Park, NYC"]
            ),
            ParamDefinition(
                name="when",
                description="Time period for forecast",
                param_type="string",
                required=False,
                default="now",
                examples=["now", "tomorrow", "this weekend", "next 3 days"]
            ),
        ]
    
    def fetch(self, params: dict) -> LiveDataResult:
        location = params.get("location")
        when = params.get("when", "now")
        
        # 1. Geocode location (cached)
        coords = self._geocode_location(location)
        if not coords:
            return LiveDataResult(success=False, error=f"Could not find location: {location}")
        
        # 2. Parse time period
        forecast_days = self._parse_forecast_period(when)
        
        # 3. Fetch weather data
        weather = self._fetch_weather(coords, forecast_days)
        
        # 4. Format for LLM context
        formatted = self._format_weather_context(location, weather, when)
        
        return LiveDataResult(
            success=True,
            data=weather,
            formatted=formatted,
            cache_ttl=self.WEATHER_CACHE_TTL
        )
```

---

## Phase 4: Document Source Plugins (Week 7-8)

### 4.1 Document Source Base Class

**File: `plugin_base/document_source.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional, Any

from plugin_base.common import FieldDefinition, ValidationResult

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

class PluginDocumentSource(ABC):
    """
    Base class for document source plugins.
    
    Document sources enumerate and fetch documents for RAG indexing.
    Content is chunked, embedded, and stored in ChromaDB.
    
    Subclasses define:
    - source_type: Unique identifier (e.g., "notion", "gdrive")
    - get_config_fields(): Configuration requirements
    - list_documents(): Enumerate available documents
    - read_document(): Fetch document content
    
    The indexer calls list_documents() to discover content, then
    read_document() for each document to get content for embedding.
    """
    
    # --- Required class attributes ---
    source_type: str
    display_name: str
    description: str
    category: str  # "oauth", "api_key", "local", "crawler"
    
    # --- Optional ---
    icon: str = "📄"
    supports_incremental: bool = True  # Can detect changed documents
    
    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration fields for admin UI."""
        pass
    
    @classmethod
    def validate_config(cls, config: dict) -> ValidationResult:
        """Validate configuration before saving."""
        from plugin_base.common import ValidationError
        errors = []
        for field_def in cls.get_config_fields():
            if field_def.required and not config.get(field_def.name):
                errors.append(ValidationError(field_def.name, "Required"))
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
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
            docs = list(self.list_documents())
            return True, f"Connected. Found {len(docs)} documents."
        except Exception as e:
            return False, str(e)
```

### 4.2 Unified Smart Sources (New Architecture)

**Key Insight:** For API-backed sources like Gmail, Calendar, Drive, Slack, and GitHub, users currently need to configure:
1. A **Document Store** for RAG (indexed historical content)
2. A **Live Source** for real-time API queries

But users don't think this way - they just ask "What emails did I get from John?" and expect the system to figure out whether to search the index or query the API.

**Solution: Unified Smart Source**

A new plugin type that combines document indexing AND live querying in a single source:

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Gmail Source                        │
│                                                              │
│  Config: OAuth account, labels to index, sync schedule       │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────┐            │
│  │  Document Side  │         │   Live Side     │            │
│  │                 │         │                 │            │
│  │ list_documents()│         │ fetch(params)   │            │
│  │ read_document() │         │                 │            │
│  │                 │         │ - recent emails │            │
│  │ → Indexed into  │         │ - search by     │            │
│  │   ChromaDB      │         │   sender/date   │            │
│  │ → Used for RAG  │         │ - specific msg  │            │
│  └─────────────────┘         └─────────────────┘            │
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │           Smart Query Router                 │            │
│  │                                              │            │
│  │  Analyzes query characteristics:             │            │
│  │  - Time range (recent vs historical)         │            │
│  │  - Specificity (search vs lookup)            │            │
│  │  - Freshness requirements                    │            │
│  │                                              │            │
│  │  Routes to: RAG | Live API | Both + Merge    │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Single configuration** - One OAuth connection, one set of settings
- **Intelligent routing** - Designator doesn't choose RAG vs Live; the smart source decides
- **Better results** - Can combine indexed history with fresh API data
- **Simpler mental model** - Users configure "Gmail" not "Gmail Docs + Gmail Live"
- **Deduplication** - Results from RAG and Live are merged and deduplicated

#### 4.2.1 PluginUnifiedSource Base Class

**File: `plugin_base/unified_source.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional, Any

from plugin_base.common import FieldDefinition, ValidationResult
from plugin_base.document_source import DocumentInfo, DocumentContent
from plugin_base.live_source import ParamDefinition, LiveDataResult

@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""
    use_rag: bool = True           # Should search indexed documents
    use_live: bool = False         # Should query live API
    rag_query: str = ""            # Query to send to RAG
    live_params: dict = field(default_factory=dict)  # Params for live fetch
    merge_strategy: str = "dedupe" # How to combine: "dedupe", "rag_first", "live_first"
    reason: str = ""               # Explanation for routing decision

class PluginUnifiedSource(ABC):
    """
    Base class for unified smart sources.
    
    Combines document indexing (for RAG) and live querying (for real-time data)
    into a single plugin. The smart source decides whether to use RAG, live API,
    or both based on query characteristics.
    
    Subclasses implement:
    - Document side: list_documents(), read_document() for indexing
    - Live side: fetch() for real-time queries
    - Router: analyze_query() to decide RAG vs Live vs Both
    
    The system calls:
    - list_documents()/read_document() during scheduled indexing
    - smart_query() at request time, which uses analyze_query() to route
    """
    
    # --- Required class attributes ---
    source_type: str
    display_name: str
    description: str
    category: str  # "google", "microsoft", "api", "local"
    icon: str = "📦"
    
    # --- Capabilities ---
    supports_rag: bool = True      # Can index documents
    supports_live: bool = True     # Can query live API
    supports_actions: bool = False # Can perform actions (send, create, etc.)
    
    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Unified configuration for both document and live sides."""
        pass
    
    @classmethod
    @abstractmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can pass for live queries."""
        pass
    
    @classmethod
    def get_designator_hint(cls) -> str:
        """Hint for designator prompt."""
        params = cls.get_live_params()
        param_parts = []
        for p in params:
            req = "" if p.required else "optional, "
            param_parts.append(f"{p.name} ({req}{p.description})")
        param_str = "; ".join(param_parts) if param_parts else "no parameters"
        
        capabilities = []
        if cls.supports_rag:
            capabilities.append("historical search via RAG")
        if cls.supports_live:
            capabilities.append("real-time API queries")
        cap_str = " and ".join(capabilities)
        
        return f"Unified source with {cap_str}. Parameters: {param_str}"
    
    # --- Document Side (for indexing) ---
    
    @abstractmethod
    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate documents for indexing."""
        pass
    
    @abstractmethod
    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Fetch document content for indexing."""
        pass
    
    # --- Live Side (for real-time queries) ---
    
    @abstractmethod
    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live data based on parameters."""
        pass
    
    # --- Smart Router ---
    
    @abstractmethod
    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze a query to determine routing.
        
        Args:
            query: The user's natural language query
            params: Parameters from the designator
        
        Returns:
            QueryAnalysis indicating whether to use RAG, Live, or both
        
        Example logic:
        - "emails from last hour" → Live only (too recent for index)
        - "emails from John in 2023" → RAG only (historical)
        - "latest email from John" → Both, prefer Live, dedupe
        """
        pass
    
    def smart_query(self, query: str, params: dict, rag_results: list[str] = None) -> LiveDataResult:
        """
        Execute a smart query using the appropriate path(s).
        
        This is called by the enricher at request time.
        
        Args:
            query: The user's natural language query
            params: Parameters from the designator
            rag_results: Pre-fetched RAG results (if RAG was already searched)
        
        Returns:
            LiveDataResult with merged/formatted results
        """
        analysis = self.analyze_query(query, params)
        
        results = []
        
        # Get RAG results if needed
        if analysis.use_rag and rag_results:
            results.extend(rag_results)
        
        # Get Live results if needed
        if analysis.use_live:
            live_result = self.fetch(analysis.live_params or params)
            if live_result.success and live_result.formatted:
                results.append(live_result.formatted)
        
        # Merge and format
        if not results:
            return LiveDataResult(success=False, error="No results from RAG or Live")
        
        merged = self._merge_results(results, analysis.merge_strategy)
        return LiveDataResult(success=True, formatted=merged)
    
    def _merge_results(self, results: list[str], strategy: str) -> str:
        """Merge results from RAG and Live."""
        if strategy == "live_first":
            return "\n\n".join(results[::-1])
        # Default: dedupe or rag_first
        return "\n\n".join(results)
```

#### 4.2.2 Unified Sources to Implement

| Source | RAG Content | Live Queries | Actions |
|--------|-------------|--------------|---------|
| **Smart Gmail** | Indexed emails (by label, date range) | Recent emails, search by sender/subject, specific message | Send, reply, label, archive |
| **Smart Calendar** | Past events (for context) | Upcoming events, free/busy, specific date | Create, update, delete events |
| **Smart Drive** | Indexed documents (by folder) | Search files, recent changes | (future: create, share) |
| **Smart Slack** | Channel history | Recent messages, search, specific thread | Send message |
| **Smart GitHub** | Issues, PRs, code (by repo) | Open issues, recent commits, PR status | Create issue, comment |
| **Smart Contacts** | Contact details | Search by name, lookup specific | (future: create, update) |

#### 4.2.3 Example: Smart Gmail

```python
class SmartGmailSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Gmail source - RAG for history, Live for recent/specific.
    
    Single configuration provides:
    - Document indexing: Emails indexed by label/date for RAG
    - Live queries: Recent emails, search, specific message lookup
    - Actions: Send, reply, forward, label, archive (via action plugin)
    """
    
    source_type = "smart_gmail"
    display_name = "Smart Gmail"
    description = "Unified Gmail with historical search and real-time queries"
    category = "google"
    icon = "📧"
    
    supports_rag = True
    supports_live = True
    supports_actions = True  # Links to email action plugin
    
    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "google", "scopes": ["gmail"]}
            ),
            FieldDefinition(
                name="index_labels",
                label="Labels to Index",
                field_type=FieldType.MULTISELECT,
                help_text="Which labels to index for RAG (empty = all)"
            ),
            FieldDefinition(
                name="index_days",
                label="Days to Index",
                field_type=FieldType.INTEGER,
                default=90,
                help_text="How many days of history to index"
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 * * * *", "label": "Hourly"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                ]
            ),
        ]
    
    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="query",
                description="Search query (sender, subject, content)",
                param_type="string",
                required=False,
                examples=["from:john@example.com", "subject:invoice", "project update"]
            ),
            ParamDefinition(
                name="time_range",
                description="Time range for search",
                param_type="string",
                required=False,
                default="recent",
                examples=["last hour", "today", "this week", "recent"]
            ),
            ParamDefinition(
                name="limit",
                description="Maximum emails to return",
                param_type="integer",
                required=False,
                default=10
            ),
        ]
    
    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Decide whether to use RAG, Live API, or both."""
        time_range = params.get("time_range", "").lower()
        
        # Very recent queries → Live only (not yet indexed)
        if time_range in ["last hour", "last 30 minutes", "just now"]:
            return QueryAnalysis(
                use_rag=False,
                use_live=True,
                live_params=params,
                reason="Very recent time range - using live API"
            )
        
        # Historical queries → RAG only
        if any(word in query.lower() for word in ["last year", "2023", "2022", "months ago"]):
            return QueryAnalysis(
                use_rag=True,
                use_live=False,
                rag_query=query,
                reason="Historical query - using RAG index"
            )
        
        # "Latest" or "recent" → Both, prefer live
        if any(word in query.lower() for word in ["latest", "recent", "new"]):
            return QueryAnalysis(
                use_rag=True,
                use_live=True,
                rag_query=query,
                live_params=params,
                merge_strategy="live_first",
                reason="Recent query - checking both, preferring live"
            )
        
        # Default: RAG with live fallback
        return QueryAnalysis(
            use_rag=True,
            use_live=True,
            rag_query=query,
            live_params=params,
            merge_strategy="dedupe",
            reason="General query - using both sources"
        )
    
    # Document side implementation...
    def list_documents(self) -> Iterator[DocumentInfo]:
        # List emails from configured labels within index_days
        pass
    
    def read_document(self, uri: str) -> Optional[DocumentContent]:
        # Fetch email content by message ID
        pass
    
    # Live side implementation...
    def fetch(self, params: dict) -> LiveDataResult:
        # Query Gmail API for recent/specific emails
        pass
```

### 4.3 Revised Phase 4 Checklist

**Phase 4a: Unified Source Infrastructure** ✅ **Complete (2026-01-20)**
- [x] **4.3.1** Create `plugin_base/unified_source.py` with PluginUnifiedSource base class
- [x] **4.3.2** Add QueryAnalysis dataclass for routing decisions (QueryRouting, MergeStrategy, UnifiedResult)
- [x] **4.3.3** Update plugin loader to discover unified sources (unified_source_registry)
- [x] **4.3.4** Write unit tests for unified source base class (46 tests)
- [x] **4.3.5** Add admin API endpoints for unified sources
- [ ] **4.3.6** Update smart_enricher to detect and use unified sources
- [ ] **4.3.7** Update indexer to work with unified sources (document side)

**Files created:**
- `plugin_base/unified_source.py` - Base class with QueryRouting, MergeStrategy, QueryAnalysis, UnifiedResult
- `tests/plugins/test_unified_source.py` - 46 unit tests
- `builtin_plugins/unified_sources/` - Directory for unified source plugins
- `plugins/unified_sources/` - Directory for user unified source plugins

**Admin API endpoints:**
- `GET /api/unified-sources/plugins` - List all registered unified source plugins
- `GET /api/unified-sources/plugins/<source_type>` - Get plugin details
- `POST /api/unified-sources/plugins/<source_type>/validate` - Validate config
- `POST /api/unified-sources/plugins/<source_type>/test` - Test plugin config

**Phase 4b: Google & Productivity Unified Sources** ✅ **Complete (2026-01-20)**
- [x] **4.4.1** Implement Gmail unified source (`builtin_plugins/unified_sources/gmail.py`)
- [x] **4.4.2** Implement Google Calendar unified source (`builtin_plugins/unified_sources/gcalendar.py`)
- [x] **4.4.3** Implement Google Drive unified source (`builtin_plugins/unified_sources/gdrive.py`)
- [x] **4.4.4** Implement Google Contacts unified source (`builtin_plugins/unified_sources/gcontacts.py`)
- [x] **4.4.5** Implement Google Tasks unified source (`builtin_plugins/unified_sources/gtasks.py`)
- [x] **4.4.6** Implement Todoist unified source (`builtin_plugins/unified_sources/todoist.py`)
- [ ] **4.4.7** Write integration tests for unified sources

**Unified Sources Created:**

| Source | File | RAG | Live | Actions | Description |
|--------|------|-----|------|---------|-------------|
| Gmail | `gmail.py` | Emails by label/date | Recent, unread, search | Email actions | Historical + real-time email |
| Google Calendar | `gcalendar.py` | Events by date range | Today, upcoming, week | Calendar actions | Historical + schedule |
| Google Drive | `gdrive.py` | Files by folder | Recent, search, shared | - | Content search + file listing |
| Google Contacts | `gcontacts.py` | All contacts | Search, lookup | - | Semantic + exact search |
| Google Tasks | `gtasks.py` | Tasks by list | Pending, due today, overdue | Task actions | Historical + current state |
| Todoist | `todoist.py` | Tasks by project | Pending, due today, overdue | Todoist actions | Historical + current state |

**Intelligent Query Routing:**
Each unified source implements `analyze_query()` to determine optimal data path:
- Current state queries (today, pending, upcoming) → Live only
- Historical queries (last year, 2023) → RAG only
- Search queries → Both with merge/dedupe
- Default → Source-specific optimal routing

---

### Complete Source Inventory

**All Document Sources (existing in `mcp/sources.py`):**

| Source | Type | Status | Unified Plugin |
|--------|------|--------|----------------|
| **Google** |
| Gmail | `mcp:gmail` | Exists | ✅ `gmail.py` |
| Google Calendar | `mcp:gcalendar` | Exists | ✅ `gcalendar.py` |
| Google Drive | `mcp:gdrive` | Exists | ✅ `gdrive.py` |
| Google Tasks | `mcp:gtasks` | Exists | ✅ `gtasks.py` |
| Google Contacts | `mcp:gcontacts` | Exists | ✅ `gcontacts.py` |
| **Microsoft** |
| OneDrive | `mcp:onedrive` | Exists | ⬜ Planned |
| Outlook Mail | `mcp:outlook` | Exists | ⬜ Planned |
| Outlook Calendar | `mcp:outlook_calendar` | ⚠️ Missing | ⬜ Planned (new) |
| OneNote | `mcp:onenote` | Exists | ⬜ Planned |
| Teams | `mcp:teams` | Exists | ⬜ Planned |
| **Third-Party** |
| Slack | `slack` | Exists | ⬜ Planned |
| GitHub | `mcp:github` | Exists | ⬜ Planned |
| Notion | `notion` | Exists | ⬜ Planned |
| Todoist | `todoist` | Exists | ✅ `todoist.py` |
| **RAG-Only (no live needed)** |
| Local Filesystem | `local` | Exists | ⬜ Plugin migration |
| Website Crawler | `website` | Exists | ⬜ Plugin migration |
| Paperless-ngx | `paperless` | Exists | ⬜ Plugin migration |
| Nextcloud | `nextcloud` | Exists | ⬜ Plugin migration |
| Web Search | `websearch` | Exists | ⬜ Plugin migration |

**All Live Sources (existing in `live/sources.py`):**

| Source | Type | Status | Notes |
|--------|------|--------|-------|
| Stocks | `finnhub_enhanced` | Smart plugin | Finnhub + Alpha Vantage |
| Weather | `open_meteo_enhanced` | Smart plugin | Open-Meteo + AccuWeather |
| Sports | `sportapi7_enhanced` | Smart plugin | SportAPI7 |
| Health | `oura_withings` | Smart plugin | Oura + Withings |
| Transport | `transportapi_enhanced` | Smart plugin | UK trains |
| Places | `google_places_enhanced` | Smart plugin | Google Places |
| Routes | `google_routes_enhanced` | Smart plugin | Google Routes |
| Google Calendar | `google_calendar` | Legacy | → Use unified `gcalendar` |
| Google Tasks | `google_tasks` | Legacy | → Use unified `gtasks` |
| Gmail | `gmail` | Legacy | → Use unified `gmail` |

**Planned Live Sources:**

| Source | Type | Status | Notes |
|--------|------|--------|-------|
| News | `news` | ⬜ Planned | News headlines, articles (NewsAPI, GNews, or similar) |
| Amazon | `amazon` | ⬜ Planned | Product info, pricing (Amazon Product API or scraping) |

---

**Phase 4c: Microsoft Unified Sources**
- [ ] **4.5.1** Implement Outlook Mail unified source
- [ ] **4.5.2** Implement Outlook Calendar unified source (NEW - no doc source yet)
- [ ] **4.5.3** Implement OneDrive unified source
- [ ] **4.5.4** Implement OneNote unified source
- [ ] **4.5.5** Implement Teams unified source
- [ ] **4.5.6** Write integration tests for Microsoft sources

**Existing Microsoft Document Sources (in `mcp/sources.py`):**
- `OneDriveDocumentSource` - Files from OneDrive
- `OutlookMailDocumentSource` - Emails from Outlook/Microsoft 365
- `OneNoteDocumentSource` - Notes from OneNote
- `TeamsDocumentSource` - Messages from Microsoft Teams
- ⚠️ `OutlookCalendarDocumentSource` - **Does not exist yet** (needs to be created)

**Phase 4d: Third-Party Unified Sources**
- [ ] **4.6.1** Implement Slack unified source
- [ ] **4.6.2** Implement GitHub unified source
- [ ] **4.6.3** Implement Notion unified source
- [ ] **4.6.4** Write integration tests for third-party sources

**Existing Third-Party Document Sources:**
- `SlackDocumentSource` - Channel messages
- `GitHubDocumentSource` - Issues, PRs, code
- `NotionDocumentSource` - Pages and databases

**Phase 4e: Simple Document Sources (non-unified)**

These sources don't need live querying - RAG-only is appropriate:
- [ ] **4.7.1** Migrate Local filesystem source to plugin
- [ ] **4.7.2** Migrate Website crawler source to plugin
- [ ] **4.7.3** Migrate Paperless source to plugin
- [ ] **4.7.4** Migrate Nextcloud source to plugin
- [ ] **4.7.5** Migrate WebSearch source to plugin

**Existing RAG-only Document Sources:**
- `LocalDocumentSource` - Local filesystem folders
- `WebsiteDocumentSource` - Crawled websites
- `PaperlessDocumentSource` - Paperless-ngx documents
- `NextcloudDocumentSource` - Nextcloud files via WebDAV
- `WebSearchDocumentSource` - Search results indexed for RAG
- `TodoistDocumentSource` - Todoist tasks (now superseded by unified source)

**Phase 4f: Admin Integration**
- [x] **4.8.1** Add admin API endpoints for unified sources (done in 4a)
- [ ] **4.8.2** Update admin UI to show unified source configuration
- [ ] **4.8.3** Update admin UI to show RAG + Live status together
- [ ] **4.8.4** Add "Test Connection" for both document and live sides

---

## Phase 5: Admin UI (Week 9-10)

### 5.1 Dynamic Form Generation

The admin UI should render forms dynamically based on plugin field definitions.

**API Endpoints:**

```
GET /api/plugins
  Returns: {document_sources: [...], live_sources: [...], actions: [...]}

GET /api/plugins/{type}/{source_type}/fields
  Returns: {fields: [...], display_name: "...", description: "..."}

POST /api/plugins/{type}/{source_type}/validate
  Body: {config: {...}}
  Returns: {valid: bool, errors: [...]}

POST /api/plugins/{type}/{source_type}/test
  Body: {config: {...}}
  Returns: {success: bool, message: "..."}
```

**Alpine.js Component Pattern:**

```javascript
// Generic plugin config form component
Alpine.data('pluginConfigForm', (pluginType, sourceType) => ({
    fields: [],
    config: {},
    errors: {},
    loading: true,
    
    async init() {
        // Load field definitions from API
        const response = await fetch(`/api/plugins/${pluginType}/${sourceType}/fields`);
        const data = await response.json();
        this.fields = data.fields;
        
        // Set defaults
        for (const field of this.fields) {
            if (field.default !== undefined) {
                this.config[field.name] = field.default;
            }
        }
        
        this.loading = false;
    },
    
    shouldShowField(field) {
        if (!field.depends_on) return true;
        for (const [key, value] of Object.entries(field.depends_on)) {
            if (this.config[key] !== value) return false;
        }
        return true;
    },
    
    async validate() {
        const response = await fetch(`/api/plugins/${pluginType}/${sourceType}/validate`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({config: this.config})
        });
        const data = await response.json();
        this.errors = {};
        for (const error of data.errors || []) {
            this.errors[error.field] = error.message;
        }
        return data.valid;
    },
    
    async testConnection() {
        const response = await fetch(`/api/plugins/${pluginType}/${sourceType}/test`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({config: this.config})
        });
        return await response.json();
    }
}));
```

### 5.2 Checklist - Phase 5

- [ ] **5.1.1** Add `/api/plugins` endpoint returning all plugin metadata
- [ ] **5.1.2** Add `/api/plugins/{type}/{source_type}/fields` endpoint
- [ ] **5.1.3** Add `/api/plugins/{type}/{source_type}/validate` endpoint
- [ ] **5.1.4** Add `/api/plugins/{type}/{source_type}/test` endpoint
- [ ] **5.2.1** Create Alpine.js `pluginConfigForm` component
- [ ] **5.2.2** Create field renderer templates (text, password, select, etc.)
- [ ] **5.2.3** Create OAuth account picker component
- [ ] **5.2.4** Create folder/calendar picker components (async loading)
- [ ] **5.3.1** Update Document Stores page to use dynamic forms
- [ ] **5.3.2** Update Live Data Sources page to use dynamic forms
- [ ] **5.3.3** Update Smart Aliases page for action configuration
- [ ] **5.4.1** Add Plugins management page in Settings
- [ ] **5.4.2** Show plugin status (loaded, errors, version)
- [ ] **5.4.3** Add plugin documentation viewer

---

## Testing Strategy

### Unit Tests

Each plugin base class should have comprehensive unit tests:

```python
# tests/plugins/test_action_base.py

import pytest
from plugin_base.action import PluginActionHandler, ActionDefinition, ActionRisk, ActionResult
from plugin_base.common import FieldDefinition, FieldType

class MockActionHandler(PluginActionHandler):
    """Mock plugin for testing."""
    action_type = "mock"
    display_name = "Mock Actions"
    description = "Test plugin"
    
    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(name="api_key", label="API Key", field_type=FieldType.PASSWORD, required=True),
        ]
    
    @classmethod
    def get_actions(cls):
        return [
            ActionDefinition(
                name="test",
                description="Test action",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(name="message", label="Message", field_type=FieldType.TEXT, required=True),
                ],
            ),
        ]
    
    def __init__(self, config):
        self.api_key = config["api_key"]
    
    def execute(self, action, params, context):
        if action == "test":
            return ActionResult(success=True, message=f"Got: {params['message']}")
        return ActionResult(success=False, error="Unknown action")

class TestActionHandler:
    def test_get_config_fields(self):
        fields = MockActionHandler.get_config_fields()
        assert len(fields) == 1
        assert fields[0].name == "api_key"
        assert fields[0].required is True
    
    def test_get_actions(self):
        actions = MockActionHandler.get_actions()
        assert len(actions) == 1
        assert actions[0].name == "test"
        assert actions[0].risk == ActionRisk.LOW
    
    def test_validate_config_missing_required(self):
        result = MockActionHandler.validate_config({})
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "api_key"
    
    def test_validate_config_success(self):
        result = MockActionHandler.validate_config({"api_key": "secret"})
        assert result.valid is True
    
    def test_execute_success(self):
        handler = MockActionHandler({"api_key": "secret"})
        result = handler.execute("test", {"message": "hello"}, None)
        assert result.success is True
        assert "hello" in result.message
    
    def test_llm_instructions(self):
        instructions = MockActionHandler.get_llm_instructions()
        assert "mock:test" in instructions
        assert "Test action" in instructions
        assert "message (required)" in instructions
```

### Integration Tests

Test full plugin lifecycle with mocked external APIs:

```python
# tests/plugins/test_todoist_integration.py

import pytest
import httpx
from unittest.mock import patch, MagicMock

from builtin_plugins.actions.todoist import TodoistActionHandler
from plugin_base.action import ActionContext

@pytest.fixture
def todoist_handler():
    return TodoistActionHandler({"api_token": "test-token"})

class TestTodoistIntegration:
    def test_create_task(self, todoist_handler):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "123",
            "content": "Test task",
            "url": "https://todoist.com/task/123"
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(todoist_handler.client, 'post', return_value=mock_response):
            result = todoist_handler.execute(
                "create",
                {"content": "Test task", "due_string": "tomorrow"},
                ActionContext()
            )
        
        assert result.success is True
        assert result.data["task_id"] == "123"
    
    def test_complete_task(self, todoist_handler):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(todoist_handler.client, 'post', return_value=mock_response):
            result = todoist_handler.execute(
                "complete",
                {"task_id": "123"},
                ActionContext()
            )
        
        assert result.success is True
```

### Test Fixtures

```python
# tests/plugins/fixtures/mock_document_source.py

from plugin_base.document_source import PluginDocumentSource, DocumentInfo, DocumentContent
from plugin_base.common import FieldDefinition, FieldType

class MockDocumentSource(PluginDocumentSource):
    """Mock document source for testing."""
    
    source_type = "mock_docs"
    display_name = "Mock Documents"
    description = "Test document source"
    category = "other"
    
    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(name="doc_count", label="Document Count", field_type=FieldType.INTEGER, default=5),
        ]
    
    def __init__(self, config):
        self.doc_count = config.get("doc_count", 5)
    
    def list_documents(self):
        for i in range(self.doc_count):
            yield DocumentInfo(
                uri=f"doc_{i}",
                title=f"Document {i}",
                mime_type="text/plain"
            )
    
    def read_document(self, uri):
        if uri.startswith("doc_"):
            return DocumentContent(
                content=f"Content of {uri}",
                mime_type="text/plain"
            )
        return None
```

---

## Migration Guide

### Migrating Existing Code

When migrating existing built-in sources to plugins:

1. **Create plugin class** in `builtin_plugins/` matching the base class interface
2. **Extract config fields** from sparse database columns to `get_config_fields()`
3. **Move business logic** to plugin methods
4. **Update factory function** to check plugin registry first
5. **Add database migration** to move config to JSON column
6. **Deprecate old code paths** with logging warnings
7. **Remove old code** after transition period

### Example: Migrating Google Drive

**Before (in `mcp/sources.py`):**
```python
class GoogleDriveDocumentSource(DocumentSource):
    def __init__(self, store: DocumentStore):
        self.google_account_id = store.google_account_id
        self.folder_id = store.gdrive_folder_id
        # ...
```

**After (in `builtin_plugins/document_sources/google_drive.py`):**
```python
class GoogleDrivePlugin(OAuthMixin, PluginDocumentSource):
    source_type = "gdrive"
    display_name = "Google Drive"
    # ...
    
    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(name="oauth_account_id", ...),
            FieldDefinition(name="folder_id", ...),
        ]
```

**Factory update:**
```python
def get_document_source(source_type: str, config: dict):
    # Check plugin registry first
    plugin_cls = get_document_source_plugin(source_type)
    if plugin_cls:
        return plugin_cls(config)
    
    # Fall back to legacy sources (deprecated)
    logger.warning(f"Using legacy source for {source_type}, please migrate to plugin")
    # ... old code ...
```

---

## Security Considerations

### Plugin Trust

Plugins are **trusted code** - they run with full application permissions:

- No sandboxing or isolation
- Full database access
- Full network access
- Can import any Python module

**Mitigations:**
- Only administrators can add plugins (via file system)
- Plugin directory not writable via admin UI
- Plugins should be code-reviewed before deployment
- Consider signing plugins in future versions

### OAuth Security

- OAuth tokens stored encrypted in database
- Refresh tokens never exposed to plugins directly
- OAuthMixin handles token lifecycle
- Scopes requested per-plugin, not blanket permissions

### Action Approval

- ActionRisk enum determines approval requirements
- DESTRUCTIVE actions always require confirmation
- Per-alias allow-lists restrict which actions can be used
- Full audit log of all action executions

---

## Future Enhancements

### Plugin Marketplace (v3.0+)

- Remote plugin repository
- Version management
- Automatic updates
- User ratings/reviews

### Hot Reloading (v3.0+)

- File watcher for plugin directory
- Reload plugins without restart
- Graceful handling of in-flight requests

### Plugin SDK

- CLI tool for scaffolding new plugins
- Type stubs for IDE support
- Documentation generator
- Test harness

---

## Summary

| Phase | Scope | Status | Effort |
|-------|-------|--------|--------|
| 1 | Infrastructure (loader, common, DB) | ✅ Complete | 2 weeks |
| 2 | Action plugins | ✅ Complete | 2 weeks |
| 3 | Live source plugins (basic migration) | ✅ Complete | 2 weeks |
| 3b | Smart Providers | ✅ Complete | 1-2 weeks |
| 4a | Unified source infrastructure | ✅ Complete | 1 week |
| 4b | Smart Google sources (Gmail, Calendar, Drive, Contacts) | 🔴 Not started | 2 weeks |
| 4c | Smart third-party sources (Slack, GitHub, Notion) | 🔴 Not started | 1-2 weeks |
| 4d | Simple document sources (local, web, Paperless, Nextcloud) | 🔴 Not started | 1 week |
| 4e | Admin integration for unified sources | 🟡 Partial (API done) | 1 week |
| 5 | Admin UI (dynamic forms, plugin management) | 🟡 Partial | 2 weeks |
| **Total** | | | **15-17 weeks** |

### Phase 3b Smart Providers - Completed

| Plugin | File | Source Type | Combines | Status |
|--------|------|-------------|----------|--------|
| Smart Weather | `smart_weather.py` | `open_meteo_enhanced` | Open-Meteo + AccuWeather | ✅ Complete |
| Smart Sports | `smart_sports.py` | `sportapi7_enhanced` | SportAPI7 | ✅ Complete |
| Smart Stocks | `smart_stocks.py` | `finnhub_enhanced` | Finnhub + Alpha Vantage | ✅ Complete |
| Smart Health | `smart_health.py` | `oura_withings` | Oura + Withings | ✅ Complete |
| Smart Transport | `smart_transport.py` | `transportapi_enhanced` | TransportAPI | ✅ Complete |
| Smart Places | `smart_places.py` | `google_places_enhanced` | Google Places | ✅ Complete |
| Smart Routes | `routes.py` | `google_routes_enhanced` | Google Routes | ✅ Complete |

**Legacy plugins removed:** `open_meteo.py`, `accuweather.py`, `stocks.py`, `google_maps.py`, `transport.py`

### Key Files

| File | Purpose |
|------|---------|
| `plugin_base/common.py` | Shared dataclasses (FieldDefinition, ValidationResult) |
| `plugin_base/loader.py` | Discovery and registration |
| `plugin_base/action.py` | Action handler base class |
| `plugin_base/live_source.py` | Live source base class |
| `plugin_base/document_source.py` | Document source base class (simple indexing only) |
| `plugin_base/unified_source.py` | **NEW:** Unified source base class (RAG + Live + Actions) |
| `plugin_base/oauth.py` | OAuth mixin for token refresh |
| `db/plugin_configs.py` | Plugin config CRUD |
| `builtin_plugins/unified/` | **NEW:** Smart unified sources (Gmail, Calendar, etc.) |

### Success Criteria

- [ ] New action plugins can be added without modifying core code
- [ ] New live sources can be added without modifying core code
- [ ] New document sources can be added without modifying core code
- [ ] Admin UI dynamically renders forms for any plugin
- [ ] All existing built-in sources work unchanged during migration
- [ ] Plugin development guide enables third-party contributions
