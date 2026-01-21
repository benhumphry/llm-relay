# LLM Relay Plugins

This directory contains user-defined plugins that extend LLM Relay's functionality.

## Plugin Types

There are three types of plugins:

### Document Sources (`document_sources/`)

Document sources enumerate and fetch documents for RAG indexing. Use these to add support for new document storage systems.

### Live Sources (`live_sources/`)

Live sources fetch real-time data at request time. Use these to add support for new APIs that provide dynamic data (weather, stocks, etc.).

### Actions (`actions/`)

Action handlers allow the LLM to perform side effects. Use these to add new capabilities like sending messages, creating tasks, or controlling smart home devices.

## Creating a Plugin

### Document Source Example

Create a file in `document_sources/` (e.g., `my_wiki.py`):

```python
from plugin_base import (
    PluginDocumentSource,
    DocumentInfo,
    DocumentContent,
    FieldDefinition,
    FieldType,
)

class MyWikiDocumentSource(PluginDocumentSource):
    source_type = "my_wiki"  # Unique identifier
    display_name = "My Wiki"
    description = "Indexes pages from My Wiki"
    category = "api_key"
    icon = "📚"
    
    # Mark as non-abstract to enable registration
    _abstract = False
    
    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(
                name="api_key",
                label="API Key",
                field_type=FieldType.PASSWORD,
                required=True,
                help_text="Your My Wiki API key",
            ),
            FieldDefinition(
                name="base_url",
                label="Base URL",
                field_type=FieldType.TEXT,
                required=True,
                placeholder="https://wiki.example.com",
            ),
        ]
    
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
    
    def list_documents(self):
        # Fetch document list from your wiki
        # Yield DocumentInfo for each document
        for page in self._fetch_pages():
            yield DocumentInfo(
                uri=page["id"],
                title=page["title"],
                modified_at=page.get("updated_at"),
            )
    
    def read_document(self, uri):
        # Fetch document content
        content = self._fetch_page_content(uri)
        return DocumentContent(content=content)
    
    def _fetch_pages(self):
        # Your implementation here
        pass
    
    def _fetch_page_content(self, page_id):
        # Your implementation here
        pass
```

### Live Source Example

Create a file in `live_sources/` (e.g., `my_weather.py`):

```python
from plugin_base import (
    PluginLiveSource,
    ParamDefinition,
    LiveDataResult,
    FieldDefinition,
    FieldType,
)

class MyWeatherLiveSource(PluginLiveSource):
    source_type = "my_weather"
    display_name = "My Weather Service"
    description = "Current weather data"
    data_type = "weather"
    best_for = "Getting current weather conditions and forecasts"
    icon = "🌤️"
    
    _abstract = False
    
    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(
                name="api_key",
                label="API Key",
                field_type=FieldType.PASSWORD,
                required=True,
            ),
        ]
    
    @classmethod
    def get_param_definitions(cls):
        return [
            ParamDefinition(
                name="location",
                description="City name or coordinates",
                param_type="string",
                required=True,
                examples=["London", "New York", "37.7749,-122.4194"],
            ),
        ]
    
    def __init__(self, config):
        self.api_key = config["api_key"]
    
    def fetch(self, params):
        location = params.get("location")
        
        # Fetch weather data
        data = self._get_weather(location)
        
        return LiveDataResult(
            success=True,
            data=data,
            formatted=f"Weather in {location}: {data['temp']}°C, {data['conditions']}",
            cache_ttl=300,  # Cache for 5 minutes
        )
    
    def _get_weather(self, location):
        # Your implementation here
        pass
```

### Action Handler Example

Create a file in `actions/` (e.g., `my_tasks.py`):

```python
from plugin_base import (
    PluginActionHandler,
    ActionDefinition,
    ActionRisk,
    ActionResult,
    ActionContext,
    FieldDefinition,
    FieldType,
)

class MyTasksActionHandler(PluginActionHandler):
    action_type = "my_tasks"
    display_name = "My Tasks"
    description = "Create and manage tasks in My Tasks app"
    icon = "✅"
    category = "productivity"
    
    _abstract = False
    
    @classmethod
    def get_config_fields(cls):
        return [
            FieldDefinition(
                name="api_token",
                label="API Token",
                field_type=FieldType.PASSWORD,
                required=True,
            ),
        ]
    
    @classmethod
    def get_actions(cls):
        return [
            ActionDefinition(
                name="create",
                description="Create a new task",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="title",
                        label="Title",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="due_date",
                        label="Due Date",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                ],
                examples=[{"title": "Review PR", "due_date": "tomorrow"}],
            ),
            ActionDefinition(
                name="complete",
                description="Mark a task as complete",
                risk=ActionRisk.LOW,
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
    
    def __init__(self, config):
        self.api_token = config["api_token"]
    
    def execute(self, action, params, context):
        if action == "create":
            # Create task
            task_id = self._create_task(params["title"], params.get("due_date"))
            return ActionResult(
                success=True,
                message=f"Created task: {params['title']}",
                data={"task_id": task_id},
            )
        elif action == "complete":
            # Complete task
            self._complete_task(params["task_id"])
            return ActionResult(
                success=True,
                message=f"Completed task {params['task_id']}",
            )
        else:
            return ActionResult(
                success=False,
                message="",
                error=f"Unknown action: {action}",
            )
    
    def _create_task(self, title, due_date):
        # Your implementation here
        pass
    
    def _complete_task(self, task_id):
        # Your implementation here
        pass
```

## Field Types

The following field types are available for configuration:

| Type | Description |
|------|-------------|
| `TEXT` | Single-line text input |
| `PASSWORD` | Masked text input (for API keys) |
| `TEXTAREA` | Multi-line text input |
| `INTEGER` | Whole number input |
| `NUMBER` | Decimal number input |
| `BOOLEAN` | Checkbox |
| `SELECT` | Dropdown selection |
| `MULTISELECT` | Multi-select dropdown |
| `JSON` | JSON editor |
| `OAUTH_ACCOUNT` | OAuth account picker |
| `FOLDER_PICKER` | Folder selection (for cloud storage) |
| `CALENDAR_PICKER` | Calendar selection |
| `CHANNEL_PICKER` | Channel selection (for messaging) |

## Action Risk Levels

When defining actions, specify the appropriate risk level:

| Risk | Description |
|------|-------------|
| `READ_ONLY` | No side effects, never needs approval |
| `LOW` | Minor side effects, can be pre-approved |
| `MEDIUM` | Visible side effects, confirmation recommended |
| `HIGH` | Significant side effects, always confirm |
| `DESTRUCTIVE` | Irreversible, cannot be automated |

## OAuth Support

For plugins that need OAuth authentication, use the `OAuthMixin`:

```python
from plugin_base.oauth import OAuthMixin

class MyOAuthPlugin(OAuthMixin, PluginActionHandler):
    # ...
    
    def __init__(self, config):
        self.oauth_account_id = config["oauth_account_id"]
        self.oauth_provider = "google"
        self._init_oauth_client()
    
    def execute(self, action, params, context):
        # Use self.oauth_get(), self.oauth_post(), etc.
        response = self.oauth_get("https://api.example.com/data")
        # ...
```

## Plugin Loading

Plugins are automatically discovered at startup from:
1. `builtin_plugins/` - Shipped with the app
2. `plugins/` - User plugins (this directory)

User plugins with the same `source_type` or `action_type` as a builtin plugin will override the builtin.

A restart is required after adding or modifying plugins.

## Testing Your Plugin

You can test your plugin by:

1. Adding it to this directory
2. Restarting LLM Relay
3. Checking the startup logs for registration
4. Configuring it via the Admin UI

For more detailed testing, see the test fixtures in `tests/plugins/fixtures/`.
