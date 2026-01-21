# LLM Relay Plugin Development Guide

This guide covers how to create plugins for LLM Relay. Plugins allow you to extend the system with new data sources, live data feeds, and actions without modifying core code.

## Table of Contents

1. [Overview](#overview)
2. [Plugin Types](#plugin-types)
3. [Getting Started](#getting-started)
4. [Live Source Plugins](#live-source-plugins)
5. [Action Plugins](#action-plugins)
6. [Unified Source Plugins](#unified-source-plugins)
7. [Common Patterns](#common-patterns)
8. [Configuration Fields](#configuration-fields)
9. [OAuth Integration](#oauth-integration)
10. [Testing Your Plugin](#testing-your-plugin)
11. [Deployment](#deployment)

---

## Overview

LLM Relay uses a plugin architecture that allows you to:

- **Live Sources**: Fetch real-time data at request time (weather, stocks, sports, etc.)
- **Actions**: Allow LLMs to perform side effects (send emails, create events, etc.)
- **Unified Sources**: Combine document indexing (RAG) with live data in a single plugin

Plugins are Python files placed in specific directories. They're discovered automatically at startup.

### Directory Structure

```
llm-relay/
├── builtin_plugins/          # Built-in plugins (shipped with app)
│   ├── live_sources/
│   ├── actions/
│   └── unified_sources/
├── plugins/                   # User plugins (gitignored, override builtins)
│   ├── live_sources/
│   ├── actions/
│   └── unified_sources/
└── plugin_base/              # Base classes and utilities
    ├── common.py             # FieldDefinition, ValidationResult
    ├── live_source.py        # PluginLiveSource base class
    ├── action.py             # PluginActionHandler base class
    ├── unified_source.py     # PluginUnifiedSource base class
    └── oauth.py              # OAuthMixin for OAuth-based plugins
```

### Plugin Discovery

- Plugins are loaded at app startup
- `builtin_plugins/` is loaded first, then `plugins/`
- User plugins can override builtins with the same `source_type`
- Files starting with `_` are ignored

---

## Plugin Types

| Type | Purpose | Base Class | Example |
|------|---------|------------|---------|
| Live Source | Real-time data fetching | `PluginLiveSource` | Weather, stocks, sports |
| Action | LLM-triggered side effects | `PluginActionHandler` | Email, calendar, tasks |
| Unified Source | RAG indexing + live data | `PluginUnifiedSource` | Gmail, Calendar, Tasks |

---

## Getting Started

### Minimal Live Source Plugin

```python
"""
Example: Simple quote of the day plugin.
File: plugins/live_sources/quote_of_day.py
"""

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.live_source import (
    LiveDataResult,
    ParamDefinition,
    PluginLiveSource,
)

class QuoteOfDayLiveSource(PluginLiveSource):
    """Fetches an inspirational quote."""
    
    # Required class attributes
    source_type = "quote_of_day"
    display_name = "Quote of the Day"
    description = "Fetches inspirational quotes"
    category = "lifestyle"
    data_type = "quotes"
    best_for = "Inspirational quotes, motivation, daily wisdom"
    icon = "💬"
    
    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="name",
                label="Source Name",
                field_type=FieldType.TEXT,
                required=True,
                default="Quote of the Day",
                help_text="Display name for this source",
            ),
        ]
    
    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="category",
                description="Quote category (inspire, funny, love)",
                param_type="string",
                required=False,
                examples=["inspire", "funny", "love"],
            ),
        ]
    
    def __init__(self, config: dict):
        self.name = config.get("name", "Quote of the Day")
    
    def fetch(self, params: dict) -> LiveDataResult:
        # Your API logic here
        quote = "The only way to do great work is to love what you do."
        author = "Steve Jobs"
        
        return LiveDataResult(
            success=True,
            data={"quote": quote, "author": author},
            formatted=f'"{quote}"\n— {author}',
            cache_ttl=3600,  # Cache for 1 hour
        )
    
    def test_connection(self) -> tuple[bool, str]:
        return True, "OK"
```

---

## Live Source Plugins

Live sources fetch real-time data when a Smart Alias with live data is triggered. The designator LLM selects relevant sources and provides parameters.

### Required Class Attributes

```python
class MyLiveSource(PluginLiveSource):
    source_type = "my_source"      # Unique identifier (used in DB)
    display_name = "My Source"      # Shown in admin UI
    description = "What this does"  # Help text
    category = "finance"            # Category for grouping
    data_type = "finance"           # Data type
    best_for = "..."                # Included in designator prompt
    icon = "📊"                     # Emoji for UI
```

### Required Methods

#### `get_config_fields() -> list[FieldDefinition]`

Define configuration fields for the admin UI. **Always include a `name` field**.

```python
@classmethod
def get_config_fields(cls) -> list[FieldDefinition]:
    return [
        FieldDefinition(
            name="name",
            label="Source Name",
            field_type=FieldType.TEXT,
            required=True,
            default="My Source",
            help_text="Display name for this source",
        ),
        FieldDefinition(
            name="api_key",
            label="API Key",
            field_type=FieldType.PASSWORD,
            required=True,
            env_var="MY_API_KEY",  # Auto-fill from environment
            help_text="Your API key from ...",
        ),
    ]
```

#### `get_param_definitions() -> list[ParamDefinition]`

Define parameters the designator can pass at query time.

```python
@classmethod
def get_param_definitions(cls) -> list[ParamDefinition]:
    return [
        ParamDefinition(
            name="symbol",
            description="Stock symbol to look up",
            param_type="string",
            required=True,
            examples=["AAPL", "GOOGL", "MSFT"],
        ),
        ParamDefinition(
            name="period",
            description="Time period for historical data",
            param_type="string",
            required=False,
            default="1D",
            examples=["1D", "1W", "1M", "1Y"],
        ),
    ]
```

#### `__init__(self, config: dict)`

Initialize with validated configuration.

```python
def __init__(self, config: dict):
    self.api_key = config["api_key"]
    self.default_market = config.get("default_market", "US")
    self._client = httpx.Client(timeout=15)
```

#### `fetch(self, params: dict) -> LiveDataResult`

Fetch data based on designator-provided parameters.

```python
def fetch(self, params: dict) -> LiveDataResult:
    symbol = params.get("symbol")
    if not symbol:
        return LiveDataResult(success=False, error="Symbol required")
    
    try:
        response = self._client.get(f"{API_URL}/quote/{symbol}")
        response.raise_for_status()
        data = response.json()
        
        return LiveDataResult(
            success=True,
            data=data,
            formatted=f"**{symbol}**: ${data['price']:.2f}",
            cache_ttl=60,
        )
    except Exception as e:
        return LiveDataResult(success=False, error=str(e))
```

### Optional Methods

#### `test_connection() -> tuple[bool, str]`

Test the plugin configuration. Called from admin UI.

```python
def test_connection(self) -> tuple[bool, str]:
    try:
        response = self._client.get(f"{API_URL}/ping")
        response.raise_for_status()
        return True, "Connected successfully"
    except Exception as e:
        return False, str(e)
```

#### `is_available() -> bool`

Check if the plugin is properly configured.

```python
def is_available(self) -> bool:
    return bool(self.api_key)
```

### LiveDataResult

```python
@dataclass
class LiveDataResult:
    success: bool
    data: Any = None           # Raw data (for caching)
    formatted: str = ""        # Formatted string for LLM context
    error: Optional[str] = None
    cache_ttl: int = 300       # Seconds, 0 = don't cache, -1 = forever
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

---

## Action Plugins

Action plugins allow LLMs to perform side effects via structured action blocks in responses.

### Action Block Format

```xml
<smart_action type="email" action="draft_new">
{"to": ["user@example.com"], "subject": "Hello", "body": "..."}
</smart_action>
```

### Required Class Attributes

```python
class MyActionHandler(PluginActionHandler):
    action_type = "myaction"       # Unique identifier
    display_name = "My Action"     # Shown in admin UI
    description = "What this does"
    icon = "⚡"
    category = "productivity"      # For grouping
```

### Required Methods

#### `get_config_fields() -> list[FieldDefinition]`

Static configuration (API keys, OAuth accounts, etc.)

#### `get_actions() -> list[ActionDefinition]`

Define available actions with parameters and risk levels.

```python
@classmethod
def get_actions(cls) -> list[ActionDefinition]:
    return [
        ActionDefinition(
            name="create",
            description="Create a new task",
            risk=ActionRisk.LOW,
            params=[
                FieldDefinition(
                    name="title",
                    label="Task title",
                    field_type=FieldType.TEXT,
                    required=True,
                ),
                FieldDefinition(
                    name="due_date",
                    label="Due date",
                    field_type=FieldType.TEXT,
                    required=False,
                ),
            ],
            examples=[
                {"title": "Review PR", "due_date": "tomorrow"},
            ],
        ),
        ActionDefinition(
            name="delete",
            description="Delete a task permanently",
            risk=ActionRisk.DESTRUCTIVE,
            params=[
                FieldDefinition(
                    name="task_id",
                    label="Task ID",
                    field_type=FieldType.TEXT,
                    required=True,
                ),
            ],
        ),
    ]
```

#### `execute(self, action: str, params: dict, context: ActionContext) -> ActionResult`

Execute an action.

```python
def execute(self, action: str, params: dict, context: ActionContext) -> ActionResult:
    if action == "create":
        return self._create_task(params)
    elif action == "delete":
        return self._delete_task(params)
    return ActionResult(success=False, message="", error=f"Unknown action: {action}")

def _create_task(self, params: dict) -> ActionResult:
    try:
        response = self._client.post("/tasks", json=params)
        response.raise_for_status()
        task = response.json()
        return ActionResult(
            success=True,
            message=f"Created task: {task['title']}",
            data={"task_id": task["id"]},
        )
    except Exception as e:
        return ActionResult(success=False, message="", error=str(e))
```

### Risk Levels

```python
class ActionRisk(Enum):
    READ_ONLY = "read_only"      # No side effects
    LOW = "low"                  # Minor side effects
    MEDIUM = "medium"            # Visible side effects
    HIGH = "high"                # Significant side effects
    DESTRUCTIVE = "destructive"  # Irreversible
```

### ActionResult

```python
@dataclass
class ActionResult:
    success: bool
    message: str        # Human-readable result
    data: Any = None    # Structured result data
    error: Optional[str] = None
```

---

## Unified Source Plugins

Unified sources combine document indexing (for RAG) with live data fetching in a single plugin.

### Base Class

```python
from plugin_base.unified_source import PluginUnifiedSource

class MyUnifiedSource(PluginUnifiedSource):
    source_type = "my_unified"
    display_name = "My Unified Source"
    
    # Document source capabilities
    supports_documents = True
    
    # Live source capabilities
    supports_live = True
```

See existing unified sources in `builtin_plugins/unified_sources/` for examples.

---

## Common Patterns

### Caching

Use module-level caches with TTL:

```python
_cache: dict[str, tuple[Any, float]] = {}

def _get_cached(cache: dict, key: str, ttl: int) -> Optional[Any]:
    import time
    if key in cache:
        value, timestamp = cache[key]
        if time.time() - timestamp < ttl:
            return value
    return None

def _set_cached(cache: dict, key: str, value: Any) -> None:
    import time
    cache[key] = (value, time.time())
```

### Name Resolution

For "smart" plugins that accept natural language inputs:

```python
# Hardcoded mappings for common names
KNOWN_MAPPINGS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
}

def _resolve_symbol(self, company: str) -> Optional[str]:
    # Check known mappings FIRST
    company_lower = company.lower().strip()
    if company_lower in self.KNOWN_MAPPINGS:
        return self.KNOWN_MAPPINGS[company_lower]
    
    # Then check if already a valid symbol
    if company.isupper() and len(company) <= 5:
        return company
    
    # Finally, try API lookup
    return self._search_api(company)
```

### Natural Language Date Parsing

```python
from datetime import datetime, timedelta

def _parse_date(self, date_str: str) -> Optional[str]:
    date_lower = date_str.lower().strip()
    today = datetime.now()
    
    if date_lower in ("today", "now"):
        return today.strftime("%Y-%m-%d")
    elif date_lower == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_lower == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_lower == "last week":
        return (today - timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Try parsing as YYYY-MM-DD
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None
```

### Error Handling

Always return structured results, never raise exceptions from `fetch()` or `execute()`:

```python
def fetch(self, params: dict) -> LiveDataResult:
    try:
        # Your logic here
        return LiveDataResult(success=True, ...)
    except httpx.HTTPStatusError as e:
        return LiveDataResult(
            success=False,
            error=f"API error: {e.response.status_code}"
        )
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return LiveDataResult(success=False, error=str(e))
```

---

## Configuration Fields

### FieldDefinition

```python
@dataclass
class FieldDefinition:
    name: str                    # Field identifier
    label: str                   # Display label
    field_type: FieldType        # See below
    required: bool = False
    default: Any = None
    help_text: str = ""
    placeholder: str = ""
    env_var: str = ""            # Auto-fill from environment variable
    
    # For select/multiselect
    options: list[SelectOption] = field(default_factory=list)
    
    # Conditional visibility
    depends_on: dict = field(default_factory=dict)
    
    # Validation
    min_value: Optional[int | float] = None
    max_value: Optional[int | float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex
```

### Field Types

```python
class FieldType(Enum):
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
```

### Environment Variable Auto-Fill

Use `env_var` to auto-populate from environment variables:

```python
FieldDefinition(
    name="api_key",
    label="API Key",
    field_type=FieldType.PASSWORD,
    required=True,
    env_var="MY_SERVICE_API_KEY",  # Will show as pre-filled if env var exists
    help_text="Leave empty to use MY_SERVICE_API_KEY environment variable",
)
```

### Select Options

```python
FieldDefinition(
    name="region",
    label="Region",
    field_type=FieldType.SELECT,
    options=[
        SelectOption(value="us", label="United States"),
        SelectOption(value="eu", label="Europe"),
        SelectOption(value="asia", label="Asia Pacific"),
    ],
    default="us",
)
```

### Conditional Fields

Show/hide fields based on other field values:

```python
FieldDefinition(
    name="custom_url",
    label="Custom API URL",
    field_type=FieldType.TEXT,
    depends_on={"use_custom_endpoint": True},  # Only show if use_custom_endpoint is True
)
```

---

## OAuth Integration

For plugins that need OAuth authentication, use the `OAuthMixin`:

```python
from plugin_base.oauth import OAuthMixin

class MyOAuthPlugin(OAuthMixin, PluginActionHandler):
    def __init__(self, config: dict):
        self.oauth_account_id = config["oauth_account_id"]
        self.oauth_provider = "google"  # or "microsoft", etc.
        self._init_oauth_client()
    
    def _do_something(self):
        # Use oauth_get, oauth_post, etc. for authenticated requests
        response = self.oauth_get("https://api.example.com/data")
        return response.json()
```

### OAuth Config Field

```python
FieldDefinition(
    name="oauth_account_id",
    label="Google Account",
    field_type=FieldType.OAUTH_ACCOUNT,
    required=True,
    picker_options={"provider": "google"},
    help_text="Select a connected Google account",
)
```

---

## Testing Your Plugin

### Unit Tests

```python
import unittest
from plugins.live_sources.my_plugin import MyLiveSource

class TestMyPlugin(unittest.TestCase):
    def test_fetch_success(self):
        source = MyLiveSource({"name": "Test", "api_key": "test"})
        result = source.fetch({"query": "test"})
        self.assertTrue(result.success)
    
    def test_fetch_missing_param(self):
        source = MyLiveSource({"name": "Test", "api_key": "test"})
        result = source.fetch({})
        self.assertFalse(result.success)
```

### Manual Testing

```python
# In Python shell or script
from plugins.live_sources.my_plugin import MyLiveSource

source = MyLiveSource({"name": "Test", "api_key": "your-key"})

# Test connection
ok, msg = source.test_connection()
print(f"Connection: {ok} - {msg}")

# Test fetch
result = source.fetch({"query": "test"})
print(f"Success: {result.success}")
print(f"Data: {result.formatted}")
```

### Run in Container

```bash
docker exec llm-relay-dev python3 -c "
from builtin_plugins.live_sources.my_plugin import MyLiveSource
source = MyLiveSource({'name': 'Test'})
ok, msg = source.test_connection()
print(f'Connection: {ok} - {msg}')
"
```

---

## Deployment

### Development

1. Create your plugin file in `plugins/live_sources/` (or appropriate subdirectory)
2. Restart the dev container:
   ```bash
   docker compose -f docker-compose.dev.yml up -d --build
   ```
3. Check logs for registration:
   ```bash
   docker logs llm-relay-dev 2>&1 | grep -i "your_source_type"
   ```

### Production

1. Add plugin file to `builtin_plugins/` (or mount `plugins/` directory)
2. Rebuild and deploy:
   ```bash
   docker compose up -d --build
   ```

### Plugin Auto-Seeding

Live source plugins are automatically seeded in the database on startup if they don't exist. The plugin will appear in the admin UI under Data Sources > Live Data.

---

## Examples

See these built-in plugins for reference implementations:

### Live Sources
- `builtin_plugins/live_sources/smart_weather.py` - Weather with natural language locations
- `builtin_plugins/live_sources/smart_stocks.py` - Stocks with company name resolution
- `builtin_plugins/live_sources/smart_exchange.py` - Currency exchange rates
- `builtin_plugins/live_sources/smart_sports.py` - Sports data with team name resolution

### Actions
- `builtin_plugins/actions/email.py` - Gmail integration with drafts and sending
- `builtin_plugins/actions/calendar.py` - Google Calendar events
- `builtin_plugins/actions/todoist.py` - Task management

### Unified Sources
- `builtin_plugins/unified_sources/gmail.py` - Gmail with RAG indexing + live search
- `builtin_plugins/unified_sources/gcalendar.py` - Calendar with indexing + live events
