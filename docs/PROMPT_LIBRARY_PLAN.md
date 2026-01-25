# Prompt & Keyword Library Plan

## Problem

Currently, prompts, instructions, and keyword synonyms are scattered throughout the codebase:

1. **Designator prompts** - hardcoded in `routing/smart_enricher.py`
2. **Context injection instructions** - hardcoded in `routing/smart_enricher.py`
3. **Date/time synonyms** - duplicated in each live source plugin (`_parse_date_param`)
4. **Action instructions** - hardcoded in `actions/executor.py`
5. **Query type keywords** - duplicated in each plugin's `fetch()` method

This causes:
- Inconsistency (e.g., "this weekend" works in sports but not transport)
- Difficulty tuning prompts without code changes
- No path to internationalization
- Repeated code for common patterns

## Solution: Structured Prompt Library with Jinja2

### Why Jinja2?

- **Conditionals**: `{% if task_accounts %}...{% endif %}`
- **Loops**: `{% for source in sources %}...{% endfor %}`
- **Filters**: `{{ current_time | format_datetime }}`, `{{ accounts | join(', ') }}`
- **Inheritance**: Base templates with blocks for customization
- **Well-known**: Standard Python templating, widely understood
- **Safe**: Auto-escaping, sandboxed execution available

### Directory Structure

```
prompts/
├── __init__.py              # PromptLibrary class, convenience functions
├── loader.py                # YAML loader with Jinja2 rendering
├── filters.py               # Custom Jinja2 filters (format_datetime, etc.)
├── defaults/                # Built-in prompts (shipped with app)
│   ├── designators/
│   │   ├── router.yaml
│   │   ├── rag.yaml
│   │   ├── web.yaml
│   │   └── live.yaml
│   ├── context/
│   │   ├── injection.yaml   # Context injection instructions
│   │   └── priorities.yaml  # Data freshness priorities
│   ├── actions/
│   │   └── instructions.yaml
│   └── keywords/
│       ├── dates.yaml       # Date/time synonyms
│       ├── query_types.yaml # Query type mappings
│       └── sports.yaml      # Sport-specific keywords
├── overrides/               # User overrides (gitignored)
│   └── ...                  # Same structure as defaults/
└── locales/                 # Future: translations
    ├── en/
    ├── de/
    └── ...
```

### Example: dates.yaml (Keywords)

```yaml
# Date keyword mappings
# Keys are canonical values, values are lists of synonyms

today:
  - today
  - now
  - tonight

tomorrow:
  - tomorrow
  - next day

yesterday:
  - yesterday

weekend:
  - weekend
  - this weekend
  - the weekend
  - sat and sun
  - saturday and sunday

saturday:
  - saturday
  - this saturday
  - sat

sunday:
  - sunday
  - this sunday
  - sun

next_week:
  - next week
  - following week

# Pattern-based (regex) - handler name maps to Python function
patterns:
  - pattern: "in (\\d+) days?"
    handler: relative_days
  - pattern: "(\\d{4}-\\d{2}-\\d{2})"
    handler: iso_date
  - pattern: "(monday|tuesday|wednesday|thursday|friday)"
    handler: next_weekday
```

### Example: designators/live.yaml (Jinja2 Template)

```yaml
name: live
description: Live data source selector and parameter extractor

# Variables this template expects (for documentation/validation)
variables:
  - name: current_time
    type: datetime
    description: Current date/time for context
  - name: user_memory
    type: string
    optional: true
    description: User context from memory
  - name: task_accounts
    type: list[dict]
    description: Available task accounts with name, provider, project info
  - name: email_accounts
    type: list[dict]
    description: Available email accounts
  - name: calendar_accounts
    type: list[dict]
    description: Available calendar accounts
  - name: notes_accounts
    type: list[dict]
    description: Available notes accounts
  - name: live_sources
    type: list[dict]
    description: All available live data sources
  - name: mcp_tools
    type: list[dict]
    optional: true
    description: MCP API tools if configured
  - name: query
    type: string
    description: User's query

system_prompt: |
  You are a live data parameter extractor. Select relevant live data sources 
  and extract specific parameters needed for the user's query.

user_template: |
  Current time: {{ current_time | format_datetime('%A, %B %d, %Y at %H:%M') }}
  
  {% if user_memory %}
  USER CONTEXT:
  {{ user_memory }}
  
  {% endif %}
  === AVAILABLE ACCOUNTS ===
  
  {% if task_accounts %}
  TASK ACCOUNTS (use for "what tasks do I have?" queries):
  {% for account in task_accounts %}
  - "{{ account.name }}" ({{ account.provider }}){% if account.project %} - Project: {{ account.project }}{% endif %}
  
  {% endfor %}
  {% endif %}
  
  {% if email_accounts %}
  EMAIL ACCOUNTS (use for email queries):
  {% for account in email_accounts %}
  - "{{ account.name }}" ({{ account.provider }})
  {% endfor %}
  
  {% endif %}
  
  {% if calendar_accounts %}
  CALENDAR ACCOUNTS (use for calendar/event queries):
  {% for account in calendar_accounts %}
  - "{{ account.name }}" ({{ account.provider }})
  {% endfor %}
  
  {% endif %}
  
  {% if notes_accounts %}
  NOTES ACCOUNTS (use for "my notes", "what notes" queries):
  {% for account in notes_accounts %}
  - "{{ account.name }}" ({{ account.provider }})
  {% endfor %}
  
  {% endif %}
  
  === LIVE DATA SOURCES ===
  {% for source in live_sources %}
  - {{ source.name }}{% if source.data_type %} [{{ source.data_type }}]{% endif %}
  
    {% if source.best_for %}Best for: {{ source.best_for }}{% endif %}
  
    {% if source.param_hints %}{{ source.param_hints }}{% endif %}
  
  {% endfor %}
  
  {% if mcp_tools %}
  === MCP API TOOLS ===
  IMPORTANT: Most MCP APIs require ID lookups. Use agentic mode for multi-step queries:
    {"source_name": [{"agentic": true, "goal": "describe what you need"}]}
  
  {% for tool in mcp_tools %}
  - {{ tool.source_name }}.{{ tool.tool_name }}: {{ tool.description }}
    {% if tool.params %}Required: {{ tool.params | join(', ') }}{% endif %}
  
  {% endfor %}
  {% endif %}
  
  USER QUERY: {{ query }}

response_format:
  type: json
  description: |
    Return a JSON object mapping source names to parameter arrays.
    Example: {"personal-todoist": [{"action": "pending"}], "weather": [{"location": "London"}]}

# Examples shown to the model
examples:
  - description: List all user's tasks
    input: "what tasks do I have?"
    output: |
      {% raw %}{"personal-todoist": [{"action": "pending"}], "work-todoist": [{"action": "pending"}], "my-tasks": [{"action": "list"}]}{% endraw %}
  
  - description: List tasks from specific source
    input: "tasks in bh-ltd-todoist"
    output: |
      {% raw %}{"bh-ltd-todoist": [{"action": "pending"}]}{% endraw %}
  
  - description: Weather query
    input: "what's the weather in Paris?"
    output: |
      {% raw %}{"weather": [{"location": "Paris"}]}{% endraw %}
  
  - description: No relevant sources
    input: "tell me a joke"
    output: "{}"

rules:
  - Keys in JSON MUST be source names from the lists above
  - Do NOT invent keys like "follow_ups", "suggestions", etc.
  - For generic task queries, include ALL task accounts
  - For generic email queries, include ALL email accounts
  - Return empty {} ONLY if no live data source is relevant
```

### Example: designators/router.yaml

```yaml
name: router
description: Model selection designator

variables:
  - name: query
    type: string
  - name: token_count
    type: int
    description: Estimated tokens in conversation
  - name: has_images
    type: bool
  - name: candidates
    type: list[dict]
    description: Available models with capabilities
  - name: intelligence
    type: string
    optional: true
    description: Model intelligence/comparison info
  - name: user_memory
    type: string
    optional: true

system_prompt: |
  You are a model routing expert. Select the best LLM for the query based on 
  the candidates' strengths, weaknesses, and the query characteristics.

user_template: |
  Select the best model for this query.
  
  Query: {{ query }}
  Estimated tokens: {{ token_count }}
  {% if has_images %}Contains images: Yes{% endif %}
  
  CANDIDATES:
  {% for model in candidates %}
  - {{ model.id }}
    {% if model.strengths %}Strengths: {{ model.strengths }}{% endif %}
    {% if model.weaknesses %}Weaknesses: {{ model.weaknesses }}{% endif %}
    {% if model.context_window %}Context: {{ model.context_window | format_number }}{% endif %}
    {% if model.supports_vision %}Vision: Yes{% endif %}
  
  {% endfor %}
  
  {% if intelligence %}
  MODEL INTELLIGENCE:
  {{ intelligence }}
  {% endif %}
  
  {% if user_memory %}
  USER PREFERENCES:
  {{ user_memory }}
  {% endif %}

response_format:
  type: json
  schema:
    selected_model:
      type: string
      description: The model ID to use
```

### Example: context/priorities.yaml

```yaml
name: priorities
description: Data freshness and context injection priorities

freshness_instruction: |
  DATA FRESHNESS PRIORITY:
  - Live Data: Real-time information fetched just now (stock prices, weather, 
    sports scores, etc.) - ALWAYS use this for current values
  - Web Context: Recently searched web content - use for current events
  - Document Context: User's stored documents - may contain historical data; 
    check dates carefully
  
  When the same information appears in multiple sources, prefer the most 
  recent/live source.

section_headers:
  live: "Live Data (REAL-TIME - fetched just now, use for current values)"
  rag: "Document Context (user's stored documents - check dates for freshness)"
  web: "Web Context (recently searched)"
  live_history: "Previous Live Data (earlier in this session)"

# Template for the full context injection block
context_template: |
  {% if freshness_note %}
  {{ freshness_instruction }}
  
  {% endif %}
  {% if live_context %}
  === {{ section_headers.live }} ===
  {{ live_context }}
  
  {% endif %}
  {% if web_context %}
  === {{ section_headers.web }} ===
  {{ web_context }}
  
  {% endif %}
  {% if rag_context %}
  === {{ section_headers.rag }} ===
  {{ rag_context }}
  
  {% endif %}
  {% if live_history %}
  === {{ section_headers.live_history }} ===
  {{ live_history }}
  {% endif %}
```

### Example: actions/instructions.yaml

```yaml
name: action_instructions
description: Instructions for LLM on how to use action blocks

variables:
  - name: allowed_actions
    type: list[str]
    description: Actions enabled for this Smart Alias (e.g., ["email:*", "calendar:create"])
  - name: available_accounts
    type: dict
    description: Available accounts by category (tasks, email, calendar, notes)
  - name: action_handlers
    type: list[dict]
    description: Registered action handlers with their actions

preamble: |
  You can perform actions by including action blocks in your response.
  Action blocks are executed after your response is generated.

format_instructions: |
  ACTION BLOCK FORMAT:
  ```
  [action_type:verb]
  param1: value1
  param2: value2
  [/action_type:verb]
  ```

action_template: |
  {{ preamble }}
  
  {{ format_instructions }}
  
  AVAILABLE ACTIONS:
  {% for handler in action_handlers %}
  {% if handler.name in allowed_action_types %}
  
  {{ handler.name | upper }}:
  {% for action in handler.actions %}
  - {{ handler.name }}:{{ action.name }} - {{ action.description }}
    {% if action.params %}
    Parameters:
    {% for param in action.params %}
      {{ param.name }}{% if param.required %} (required){% endif %}: {{ param.help_text }}
    {% endfor %}
    {% endif %}
  {% endfor %}
  {% endif %}
  {% endfor %}
  
  {% if available_accounts.tasks %}
  TASK ACCOUNTS:
  {% for account in available_accounts.tasks %}
  - {{ account.name }} ({{ account.provider }})
  {% endfor %}
  {% endif %}
  
  {% if available_accounts.email %}
  EMAIL ACCOUNTS:
  {% for account in available_accounts.email %}
  - {{ account.name }} ({{ account.provider }})
  {% endfor %}
  {% endif %}
  
  RULES:
  - Only use actions from the AVAILABLE ACTIONS list above
  - Always confirm destructive actions with the user first
  - Include all required parameters
  - Use the exact account names shown above
```

## Implementation

### Phase 1: Core Library

**File: `prompts/__init__.py`**

```python
from prompts.loader import PromptLibrary

# Global singleton
_library: PromptLibrary | None = None

def get_library() -> PromptLibrary:
    """Get the global PromptLibrary instance."""
    global _library
    if _library is None:
        _library = PromptLibrary()
    return _library

def render(category: str, name: str, template_key: str = "user_template", **variables) -> str:
    """Render a prompt template with variables."""
    return get_library().render(category, name, template_key, **variables)

def get_config(category: str, name: str) -> dict:
    """Get raw config dict for a prompt (for accessing non-template fields)."""
    return get_library().get_config(category, name)

def get_keywords(category: str, name: str) -> dict[str, list[str]]:
    """Get keyword mappings."""
    return get_library().get_keywords(category, name)

def match_keyword(category: str, name: str, value: str) -> str | None:
    """Match a value to its canonical keyword."""
    return get_library().match_keyword(category, name, value)

def reload():
    """Reload all prompts from disk (for hot-reload)."""
    global _library
    _library = PromptLibrary()
```

**File: `prompts/loader.py`**

```python
import re
import logging
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound

from prompts.filters import register_filters

logger = logging.getLogger(__name__)


class YAMLTemplateLoader(BaseLoader):
    """Custom Jinja2 loader that loads templates from YAML files."""
    
    def __init__(self, library: "PromptLibrary"):
        self.library = library
    
    def get_source(self, environment, template):
        # Template name format: "category/name:template_key"
        try:
            parts = template.rsplit(":", 1)
            if len(parts) == 2:
                path, key = parts
            else:
                path, key = parts[0], "user_template"
            
            category, name = path.split("/", 1)
            config = self.library.get_config(category, name)
            source = config.get(key, "")
            
            return source, template, lambda: True
        except Exception as e:
            raise TemplateNotFound(template)


class PromptLibrary:
    """
    Manages prompt templates and keyword mappings.
    
    Loads from defaults/, with overrides/ taking precedence.
    Templates are rendered using Jinja2.
    """
    
    def __init__(self, 
                 defaults_dir: str = "prompts/defaults",
                 overrides_dir: str = "prompts/overrides",
                 locale: str = "en"):
        self.defaults_dir = Path(defaults_dir)
        self.overrides_dir = Path(overrides_dir)
        self.locale = locale
        self._cache: dict[str, dict] = {}
        
        # Set up Jinja2 environment
        self._env = Environment(
            loader=YAMLTemplateLoader(self),
            autoescape=False,  # We're generating prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )
        register_filters(self._env)
        
        # Load all configs
        self._load_all()
    
    def _load_all(self):
        """Load all YAML files from defaults and overrides."""
        # Load defaults first
        if self.defaults_dir.exists():
            self._load_directory(self.defaults_dir)
        
        # Then load overrides (will replace defaults)
        if self.overrides_dir.exists():
            self._load_directory(self.overrides_dir)
        
        logger.info(f"Loaded {len(self._cache)} prompt configs")
    
    def _load_directory(self, base_dir: Path):
        """Recursively load YAML files from a directory."""
        for yaml_file in base_dir.rglob("*.yaml"):
            # Get relative path as cache key
            rel_path = yaml_file.relative_to(base_dir)
            # Remove .yaml extension and convert to category/name format
            key = str(rel_path.with_suffix("")).replace("\\", "/")
            
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                self._cache[key] = config
                logger.debug(f"Loaded prompt config: {key}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
    
    def get_config(self, category: str, name: str) -> dict:
        """Get raw config dict for a prompt."""
        key = f"{category}/{name}"
        return self._cache.get(key, {})
    
    def render(self, category: str, name: str, template_key: str = "user_template", **variables) -> str:
        """
        Render a Jinja2 template from a prompt config.
        
        Args:
            category: Prompt category (e.g., "designators", "context")
            name: Prompt name (e.g., "live", "router")
            template_key: Key in the YAML for the template (default: "user_template")
            **variables: Variables to pass to the template
        
        Returns:
            Rendered template string
        """
        config = self.get_config(category, name)
        template_str = config.get(template_key, "")
        
        if not template_str:
            logger.warning(f"No template '{template_key}' found in {category}/{name}")
            return ""
        
        try:
            template = self._env.from_string(template_str)
            
            # Include other config values as variables (e.g., section_headers)
            context = {**config, **variables}
            
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render {category}/{name}:{template_key}: {e}")
            return f"[Template error: {e}]"
    
    def get_keywords(self, category: str, name: str) -> dict[str, list[str]]:
        """Get keyword mappings from a keywords config."""
        config = self.get_config(category, name)
        # Filter out non-keyword entries (patterns, metadata)
        return {
            k: v for k, v in config.items() 
            if isinstance(v, list) and k != "patterns"
        }
    
    def match_keyword(self, category: str, name: str, value: str) -> tuple[str | None, dict | None]:
        """
        Match input to canonical keyword using synonyms.
        
        Returns:
            (canonical_keyword, match_groups) or (None, None) if no match
        """
        config = self.get_config(category, name)
        value_lower = value.lower().strip()
        
        # Check direct synonyms first
        for canonical, synonyms in config.items():
            if canonical in ("patterns", "name", "description"):
                continue
            if isinstance(synonyms, list):
                if value_lower in [s.lower() for s in synonyms]:
                    return canonical, None
        
        # Check regex patterns
        patterns = config.get("patterns", [])
        for p in patterns:
            match = re.match(p["pattern"], value_lower, re.IGNORECASE)
            if match:
                return p["handler"], match.groupdict() or dict(enumerate(match.groups()))
        
        return None, None
```

**File: `prompts/filters.py`**

```python
"""Custom Jinja2 filters for prompt templates."""

from datetime import datetime, date
from typing import Any


def format_datetime(value: datetime | date | str, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime object or ISO string."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())
    return value.strftime(fmt)


def format_number(value: int | float, sep: str = ",") -> str:
    """Format a number with thousand separators."""
    return f"{value:,}".replace(",", sep)


def truncate(value: str, length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length."""
    if len(value) <= length:
        return value
    return value[:length - len(suffix)] + suffix


def indent(value: str, spaces: int = 2, first: bool = False) -> str:
    """Indent all lines in a string."""
    prefix = " " * spaces
    lines = value.split("\n")
    if first:
        return "\n".join(prefix + line for line in lines)
    return lines[0] + "\n" + "\n".join(prefix + line for line in lines[1:])


def register_filters(env):
    """Register all custom filters with a Jinja2 environment."""
    env.filters["format_datetime"] = format_datetime
    env.filters["format_number"] = format_number
    env.filters["truncate"] = truncate
    env.filters["indent"] = indent
```

### Phase 2: Integration

1. **Update `_parse_date_param` in plugins**:
   ```python
   from prompts import match_keyword
   
   def _parse_date_param(self, date_param: str) -> list[str]:
       canonical, groups = match_keyword("keywords", "dates", date_param)
       
       if canonical == "today":
           return [today.strftime("%Y-%m-%d")]
       elif canonical == "weekend":
           return self._get_weekend_dates()
       elif canonical == "relative_days":
           days = int(groups[1])  # From regex capture
           return [(today + timedelta(days=days)).strftime("%Y-%m-%d")]
       # ... etc
   ```

2. **Update designator prompts** in `smart_enricher.py`:
   ```python
   from prompts import render, get_config
   
   def _designate_live(self, query, live_source_info, ...):
       prompt = render("designators", "live",
           current_time=datetime.now(),
           user_memory=memory,
           task_accounts=task_accounts,
           email_accounts=email_accounts,
           calendar_accounts=calendar_accounts,
           notes_accounts=notes_accounts,
           live_sources=live_source_info,
           mcp_tools=mcp_tool_info,
           query=query,
       )
       
       system_prompt = get_config("designators", "live").get("system_prompt", "")
       # ... use prompts
   ```

3. **Update context injection**:
   ```python
   from prompts import render, get_config
   
   def _inject_context(self, ...):
       priorities = get_config("context", "priorities")
       
       context_block = render("context", "priorities", "context_template",
           freshness_note=True,
           live_context=live_context,
           web_context=web_context,
           rag_context=rag_context,
           live_history=live_history,
       )
   ```

4. **Update action instructions**:
   ```python
   from prompts import render
   
   def get_action_instructions(self, allowed_actions, available_accounts, handlers):
       return render("actions", "instructions", "action_template",
           allowed_actions=allowed_actions,
           allowed_action_types=[a.split(":")[0] for a in allowed_actions],
           available_accounts=available_accounts,
           action_handlers=handlers,
       )
   ```

### Phase 3: Admin UI (Future)

Add admin page for editing prompts:
- List all prompts by category
- Edit prompt text with syntax highlighting (CodeMirror/Monaco with YAML + Jinja2)
- Preview with sample variables
- Validate Jinja2 syntax on save
- Save to overrides directory
- Hot-reload button (calls `prompts.reload()`)

### Phase 4: Localization (Future)

- Add locale selection in settings
- Load prompts from `locales/{locale}/` first, fall back to defaults
- Community-contributed translations
- Variables remain in English (they're code identifiers)

## Benefits

1. **Consistency** - Keywords work the same across all plugins
2. **Maintainability** - Edit YAML instead of Python code
3. **Tunability** - Tweak prompts without deployment
4. **Extensibility** - Users can add custom keywords/prompts
5. **Testability** - Prompts can be unit tested separately
6. **Internationalization** - Clear path to translations
7. **Flexibility** - Jinja2 allows complex logic in templates when needed

## Migration Path

1. Create `prompts/` directory structure
2. Add Jinja2 dependency (already in requirements.txt for Flask)
3. Create `PromptLibrary` class and filters
4. Extract current hardcoded prompts to YAML files
5. Update one component at a time to use library
6. Add admin UI for editing

## Files to Migrate

| Current Location | Target |
|------------------|--------|
| `smart_enricher.py:_designate_routing()` | `designators/router.yaml` |
| `smart_enricher.py:_designate_rag()` | `designators/rag.yaml` |
| `smart_enricher.py:_designate_web()` | `designators/web.yaml` |
| `smart_enricher.py:_designate_live()` | `designators/live.yaml` |
| `smart_enricher.py:_inject_context()` | `context/injection.yaml` |
| `smart_enricher.py` (context headers) | `context/priorities.yaml` |
| `smart_sports.py:_parse_date_param()` | `keywords/dates.yaml` |
| `smart_transport.py:_parse_date_param()` | `keywords/dates.yaml` |
| `actions/executor.py` instructions | `actions/instructions.yaml` |
| Various plugin `best_for` strings | Could remain in code or move to `keywords/sources.yaml` |

## Testing

```python
# tests/prompts/test_loader.py

def test_render_simple_template():
    from prompts import render
    result = render("designators", "live", query="what's the weather?", ...)
    assert "what's the weather?" in result

def test_keyword_matching():
    from prompts import match_keyword
    canonical, _ = match_keyword("keywords", "dates", "this weekend")
    assert canonical == "weekend"

def test_regex_pattern_matching():
    from prompts import match_keyword
    canonical, groups = match_keyword("keywords", "dates", "in 3 days")
    assert canonical == "relative_days"
    assert groups[1] == "3"

def test_override_takes_precedence():
    # Create override file, verify it's used instead of default
    ...
```
