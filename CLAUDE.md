# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Relay - An intelligent LLM gateway with smart routing and context enrichment. Unifies multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek, Groq, xAI, and more) behind Ollama and OpenAI-compatible APIs with AI-powered model routing, document RAG, real-time web search, semantic caching, and usage tracking.

## Tech Stack

- **Python 3.12+** with Flask
- **SQLAlchemy 2.0** for database (SQLite default, PostgreSQL recommended)
- **Gunicorn** for production serving
- **Alpine.js** for admin UI interactivity
- **Docker** for deployment
- **ChromaDB** for vector storage (semantic caching, RAG)

## Project Structure

```
proxy.py              # Main Flask app with all API endpoints
VERSION               # Single source of truth for version number
providers/            # LLM provider implementations
  base.py             # Abstract base class for providers
  registry.py         # Model resolution and provider discovery
  loader.py           # Provider initialization
db/                   # Database layer
  models.py           # SQLAlchemy models
  connection.py       # DB connection and migrations
  seed.py             # LiteLLM pricing data sync
  smart_aliases.py    # Smart Alias CRUD operations
  document_stores.py  # Document Store CRUD operations
  redirects.py        # Redirect CRUD operations
tracking/             # Usage tracking
  usage_tracker.py    # Async request logging and statistics
  tag_extractor.py    # Tag extraction and @relay command parsing
routing/              # Smart features
  smart_alias.py      # Unified routing + enrichment + caching engine
  smart_router.py     # LLM-based intelligent routing (used by SmartAliasEngine)
  smart_enricher.py   # RAG + Web enrichment engine (used by SmartAliasEngine)
  smart_cache.py      # Semantic response caching
augmentation/         # Web search and scraping
  search/             # Search providers (SearXNG, Perplexity, Jina)
  scraper.py          # URL content scraping (builtin + Jina)
context/              # ChromaDB integration
  chroma.py           # ChromaDB client wrapper
  model_intelligence.py # Web-gathered model assessments
admin/                # Admin dashboard
  app.py              # Admin API endpoints and pages
  templates/          # Jinja2 templates with Alpine.js
rag/                  # Document processing for RAG
  embeddings.py       # Embedding providers (local, Ollama, OpenAI)
  indexer.py          # Document indexing with Docling
  retriever.py        # ChromaDB semantic search with reranking
  reranker.py         # Cross-encoder reranking (local + Jina)
  model_cache.py      # GPU model cache with TTL-based eviction
mcp/                  # Document sources (direct APIs)
  sources.py          # Document source classes for RAG indexing
plugin_base/          # Plugin framework (v2.0)
  common.py           # Shared dataclasses (FieldDefinition, ValidationResult)
  loader.py           # Plugin discovery and registration
  document_source.py  # PluginDocumentSource base class
  live_source.py      # PluginLiveSource base class
  action.py           # PluginActionHandler base class
  oauth.py            # OAuthMixin for OAuth-based plugins
builtin_plugins/      # Built-in plugins (shipped with app)
  document_sources/   # Document source plugins
  live_sources/       # Live data source plugins
  actions/            # Action handler plugins
plugins/              # User plugins (gitignored, can override builtins)
  document_sources/
  live_sources/
  actions/
tests/                # Unit tests
  plugins/            # Plugin system tests
docs/                 # Documentation
  guides/             # Feature guides
```

## Admin UI Navigation

```
Dashboard
Routing                         # Core features - first priority
├── Smart Aliases               # Unified routing + enrichment + caching
└── Redirects                   # Transparent model mappings
Data Sources
├── Document Stores             # Indexed documents for RAG
├── Websites                    # Crawled websites for RAG
├── RAG Config                  # Global embedding/vision settings
└── Web Config                  # Search and scraping settings
Model Management
├── Providers & Models
├── Local Servers               # Ollama instances
└── Statistics                  # Usage analytics
System
├── Request Log
├── Alerts
└── Settings
```

## Key Concepts

### Model Resolution (providers/registry.py)

Resolution order:
1. **Redirects** - Transparent model name mapping (supports wildcards)
2. **Smart Tags** - If request tags match an alias with `is_smart_tag=True`
3. **Smart Alias by name** - Unified routing + enrichment + caching
4. **Provider prefix** - e.g., `openai/gpt-4o`
5. **Provider search** - Search all providers for model
6. **Default fallback** - Configured default model

### Smart Aliases (db/smart_aliases.py, routing/smart_alias.py)

**The unified feature** that combines routing, enrichment, and caching into one concept. A Smart Alias can enable any combination of features via checkboxes:

| Feature | Description |
|---------|-------------|
| **Routing** (`use_routing`) | Designator LLM picks best candidate model per request |
| **RAG** (`use_rag`) | Inject context from indexed Document Stores |
| **Web** (`use_web`) | Real-time web search and scraping |
| **Cache** (`use_cache`) | Semantic response caching (disabled if Web enabled) |
| **Memory** (`use_memory`) | Persistent memory across sessions (explicit user facts only) |
| **Smart Tag** (`is_smart_tag`) | Trigger by request tag instead of model name |
| **Passthrough** (`passthrough_model`) | Honor original model when triggered as Smart Tag |
| **Context Priority** (`context_priority`) | When both RAG+Web enabled: balanced/prefer_rag/prefer_web |
| **Smart Source Selection** (`use_smart_source_selection`) | Designator allocates token budget across RAG stores + web |

**Processing pipeline (SmartAliasEngine):**
1. Cache check (if enabled)
2. Routing (if enabled) - picks target model
3. Enrichment (if RAG or Web enabled) - injects context
4. Memory injection (if enabled) - adds persistent user context
5. Forward to target model
6. Cache store (if enabled)
7. Memory update (if enabled) - extracts explicit user facts from query

**Key fields:**
- `name` - Alias name (used as model name in requests)
- `target_model` - Default/fallback model
- `designator_model` - Fast model for routing/web query generation
- `candidates` - List of candidate models for routing
- `document_stores` - Linked stores for RAG
- `context_priority` - Token allocation for hybrid enrichment (balanced/prefer_rag/prefer_web)
- `tags` - Metadata tags for usage tracking

### Smart Tags

Tag-based alias triggering. When `is_smart_tag=True`:
- Requests tagged with the alias name trigger the alias
- Works with any tagging method (header, bearer, model suffix, @relay command)
- If `passthrough_model=True`, honors the original requested model

### Tagging Methods

Tags are extracted from requests for tracking and Smart Tag triggering:

| Method | Example | Priority |
|--------|---------|----------|
| Header | `X-Proxy-Tag: alice,project-x` | 1 |
| Model suffix | `claude-sonnet@alice` | 2 |
| Bearer token | `Authorization: Bearer alice` | 3 |
| Query command | `@relay[tag:alice]` in message | Merged |

### @relay Commands (tracking/tag_extractor.py)

In-content commands that are extracted and stripped before forwarding:

```
@relay[tag:cached]           # Add tag
@relay[tag:one,two,three]    # Multiple tags
```

Format designed for extensibility:
```
@relay[command:value]
@relay[command:key=value,key2="quoted value"]
```

### Redirects (db/redirects.py)

Transparent model name mappings checked first in resolution:
- Support wildcard patterns (e.g., `openrouter/anthropic/*` -> `anthropic/*`)
- Don't appear in logs as aliases
- Use cases: model upgrades, provider switching

### Document Stores (db/document_stores.py)

Indexed document collections for RAG. Documents are parsed, chunked, and embedded into ChromaDB.

**Document sources (mcp/sources.py):**
- `local` - Docker-mounted folder
- `website` - Crawled websites (trafilatura-based crawler)
- `mcp:gdrive` - Google Drive via OAuth
- `mcp:gmail` - Gmail via OAuth  
- `mcp:gcalendar` - Google Calendar via OAuth
- `mcp:gtasks` - Google Tasks via OAuth
- `mcp:gcontacts` - Google Contacts via OAuth (People API)
- `paperless` - Paperless-ngx via REST API
- `notion` - Notion via direct REST API
- `nextcloud` - Nextcloud via WebDAV
- `mcp:github` - GitHub via REST API
- `slack` - Slack via Bot OAuth

**Key settings:**
- `embedding_provider` / `embedding_model` - How to embed chunks
- `vision_provider` / `vision_model` - For PDF/image parsing
- `chunk_size` / `chunk_overlap` - Chunking parameters
- `index_schedule` - Cron for scheduled re-indexing

### Model Intelligence (context/model_intelligence.py)

Web-gathered comparative assessments for Smart Alias routing. When enabled:
1. Searches for model reviews and comparisons
2. Summarizes into relative strengths/weaknesses
3. Cached in ChromaDB with TTL expiry

### Semantic Caching (routing/smart_cache.py)

ChromaDB-based response caching. Cache settings available on Smart Aliases:
- `cache_similarity_threshold` - Match threshold (0.0-1.0, default 0.95)
- `cache_match_last_message_only` - Ignore conversation history
- `cache_match_system_prompt` - Include system prompt in cache key
- `cache_ttl_hours` - Entry expiration

**Note:** Caching is not permitted when `use_web=True` (real-time data).

### GPU Model Cache (rag/model_cache.py)

Local GPU models (embeddings, reranker, Docling) are managed by a unified cache with TTL-based eviction:

- Models are loaded once and reused across requests
- Automatically unloaded after inactivity period (default 5 minutes)
- Background cleanup thread checks for expired models every 60 seconds
- Prevents GPU memory leaks from repeated model loading

**Configuration (environment variables):**
- `GPU_MODEL_TTL_SECONDS` - Time before unused models are unloaded (default: 300)
- `GPU_MODEL_CLEANUP_INTERVAL` - How often to check for expired models (default: 60)

**Admin API:**
- `GET /api/gpu-cache` - View cached models and GPU memory usage
- `POST /api/gpu-cache/clear` - Manually clear all cached models

## Common Tasks

### Running locally
```bash
pip install -r requirements.txt
python proxy.py
```

### Database migrations
Migrations run automatically on startup via `db/connection.py:run_migrations()`.

### Development workflow
```bash
# Build and deploy to dev instance
docker compose -f docker-compose.dev.yml up -d --build

# View dev logs
docker logs -f llm-relay-dev
```

## Code Patterns

### Adding a new provider
1. Create `providers/new_provider.py` extending `LLMProvider` from `base.py`
2. Implement required methods: `chat()`, `chat_stream()`, `list_models()`
3. Register in `providers/loader.py`

### Adding a new document source
1. Create class in `mcp/sources.py` extending `DocumentSource`
2. Implement `list_documents()` and `read_document()` methods
3. Add to `get_document_source()` factory function
4. Add UI form fields in `admin/templates/document_stores.html`

### Database changes
1. Add/modify models in `db/models.py`
2. For NEW tables: SQLAlchemy's `create_all()` creates them automatically
3. For EXISTING tables: Add migration in `db/connection.py:_run_migrations()`
4. Add CRUD functions in appropriate `db/*.py` file
5. Export from `db/__init__.py`

### Adding admin endpoints
1. Add route in `admin/app.py`
2. Use `@require_auth` for page routes, `@require_auth_api` for API routes
3. Create template in `admin/templates/`

### Alpine.js select dropdown restoration (IMPORTANT)
When editing forms with select dropdowns populated by async data (e.g., Google calendars, folders, labels), the saved value won't display correctly after loading options. Alpine's `x-model` doesn't sync when options are rendered via `x-for` after the value is set.

**Fix:** After loading options, set the value via DOM and dispatch a `change` event:
```javascript
setTimeout(() => {
    if (savedValue) {
        const el = document.querySelector('select[x-model="form.fieldName"]');
        if (el) {
            el.value = savedValue;
            el.dispatchEvent(new Event("change"));
        }
    }
}, 100);
```

This pattern is used in `document_stores.html`, `rag_config.html`, and `web_sources.html`.

## API Endpoints

**Ollama API** (port 11434):
- `GET /api/tags` — List models
- `POST /api/chat` — Chat completion
- `POST /api/generate` — Text generation

**OpenAI API** (port 11434):
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completion
- `POST /v1/completions` — Text completion

**Anthropic API** (port 11434):
- `POST /v1/messages` — Anthropic-format chat completion (supports streaming with named SSE events)

Configure Claude Code to use LLM Relay:
```bash
export ANTHROPIC_BASE_URL=http://your-relay-host:11434
```

## Important Notes

- All model names are normalized to lowercase
- Streaming responses use Server-Sent Events
- Admin UI uses Alpine.js - no build step required
- Cost tracking uses provider-specific pricing from `db/seed.py`
- Edit `VERSION` file to update version
- Node.js is NOT required - all document sources use direct REST APIs

## Dev vs Production Ports
- Production: 11434 (API), 8080 (Admin)
- Development: 11435 (API), 8081 (Admin)

---

## Recent Development Session (2026-01-15/16)

### Live Data Sources - MCP/RapidAPI Integration

**Completed:**
1. **Agentic tool loop for MCP providers** (`live/sources.py`)
   - `fetch_agentic()` method enables multi-step API lookups
   - LLM (designator) orchestrates tool calls when API requires multiple steps (e.g., lookup ID → fetch data)
   - Auto-fallback: When no `tool_name` specified, MCP sources automatically use agentic mode

2. **Global MCP tools cache** (`live/sources.py`)
   - Tool listings cached globally by API host (1 hour TTL)
   - Persists across provider instances - eliminates ~4 second delay on subsequent requests
   - Cache key: `_mcp_tools_cache[api_host] = (tools_list, timestamp)`

3. **Fixed spurious API calls**
   - Live sources only queried when designator explicitly selects them via `live_params`
   - No more fallback querying of all sources with raw query
   - MCP sources require explicit selection (too expensive for blanket queries)

4. **Designator model passthrough** (`routing/smart_enricher.py:1273`)
   - `designator_model` now passed to MCP providers for agentic fallback
   - Uses Smart Alias's configured designator instead of hardcoded Groq default

### Smart Source Selection Changes

**Completed:**
1. **Removed baseline token allocation**
   - Previously: 50% baseline to all stores, 50% priority allocation
   - Now: 100% designator-controlled allocation
   - Stores with 0 allocation are not queried at all
   - More focused RAG retrieval, better performance

### Scheduler Fix

**Bug fixed:** `_schedule_all_rags()` was calling non-existent `get_rags_with_schedule()` function, causing ImportError that was misreported as "APScheduler not installed".

**Fix:** Removed the dead code path. Scheduler now works correctly for document stores.

### Key Files Modified This Session
- `live/sources.py` - Agentic loop, global tools cache, auto-fallback
- `routing/smart_enricher.py` - Designator allocation, live params handling
- `rag/indexer.py` - Scheduler fix
- `docs/PLANNED_FEATURES.md` - Added webhooks and expanded Smart Actions

### Current State
- Dev container (`llm-relay-dev`) has all changes deployed
- Production (`llm-relay`) is on older version (1.5.1)
- Gmail store (ID 31) set to hourly indexing schedule

### Known Issues to Watch
- Model "thinking" leakage observed once with `gemini-3-pro-preview` - may be transient model behavior, not relay bug

### Roadmap Updates (docs/PLANNED_FEATURES.md)
Added to v2.0:
- **Webhooks for Document Stores** - Real-time indexing triggered by external events

Added to v2.1 (Smart Actions):
- Expanded action categories: email, calendar, scheduled prompts, documents, communication, tasks
- **Scheduled Prompts** - Proactive agent capability (cron-based prompt execution)
- Plugin architecture with `ActionHandler` base class for extensibility

---

## Development Session (2026-01-16) - Google Tasks/Contacts & OAuth Fix

### New Document Sources Added

**Google Tasks (`mcp:gtasks`)**
- Indexes tasks from Google Tasks API
- Optional filter by task list (`gtasks_tasklist_id`)
- Documents include: title, notes, due date, status, completion time
- API endpoint: `GET /api/oauth/google/<account_id>/tasklists`

**Google Contacts (`mcp:gcontacts`)**
- Indexes contacts from Google People API
- Optional filter by label (`gcontacts_group_id`) - uses contactGroups endpoint
- Documents include: names, emails, phones, organizations, addresses, notes, birthdays
- API endpoint: `GET /api/oauth/google/<account_id>/contactgroups`
- UI shows "Label" (what users see in Google Contacts) not "Contact Group"

### Files Modified
- `mcp/sources.py` - Added `GoogleTasksDocumentSource` and `GoogleContactsDocumentSource` classes
- `db/models.py` - Added columns: `gtasks_tasklist_id`, `gtasks_tasklist_name`, `gcontacts_group_id`, `gcontacts_group_name`
- `db/document_stores.py` - Updated CRUD operations for new fields
- `db/connection.py` - Added migrations for new columns
- `rag/indexer.py` - Pass new parameters to `get_document_source()`
- `admin/app.py` - Added OAuth scopes (`tasks.readonly`, `contacts.readonly`) and API endpoints
- `admin/templates/document_stores.html` - Added UI for task list and label selection

### OAuth Refresh Token Bug Fix

**Problem:** Users had to repeatedly reconnect Google accounts because refresh tokens were being lost.

**Root cause:** When re-authorizing an existing account, Google may not return a new `refresh_token` (even with `prompt=consent`). The `store_oauth_token()` function was blindly overwriting the stored token data, losing the existing refresh_token.

**Fix in `db/oauth_tokens.py`:**
```python
# In store_oauth_token(), when updating existing token:
if "refresh_token" not in token_data and "refresh_token" in existing_data:
    logger.info(f"Preserving existing refresh_token for {provider}/{account_email}")
    token_data["refresh_token"] = existing_data["refresh_token"]
```

This ensures the original refresh_token is preserved when Google doesn't return a new one.

### Live Data Cache System (from earlier in session)

**New files:**
- `db/live_cache.py` - CRUD operations for API response caching
- `db/models.py` - Added `LiveDataCache` and `LiveEntityCache` models

**Features:**
- Data cache: Stores MCP API responses with TTL-based expiration
- Entity cache: Stores name→ID mappings (e.g., "Apple" → "AAPL") with 90-day TTL
- Historical data detection: Past dates cached forever (TTL=0)
- Admin UI in Settings page with stats and clear buttons

### Current State
- Dev container (`llm-relay-dev`) has all changes deployed
- 4 new database migrations ran successfully for gtasks/gcontacts columns
- OAuth fix deployed - existing accounts should now maintain their refresh tokens

---

## Development Session (2026-01-19) - Smart Actions, Scheduled Prompts, Bug Fixes

### Smart Actions Completed

Full implementation of the Smart Actions system with multiple action handlers:

**Email Actions (`actions/handlers/email.py`):**
- `draft_new`, `draft_reply`, `draft_forward` - Create drafts
- `send_new`, `send_reply`, `send_forward` - Send immediately
- `mark_read`, `mark_unread` - Toggle read status
- `archive` - Archive messages
- `label` - Add/remove labels

**Calendar Actions (`actions/handlers/calendar.py`):**
- `create` - Create calendar events
- `update` - Modify existing events
- `delete` - Delete events

**Notification Actions (`actions/handlers/notification.py`):**
- `push` - Send push notifications via configured webhook URLs

**Schedule Actions (`actions/handlers/schedule.py`):**
- `prompt` - Schedule recurring or one-time prompts
- `cancel` - Cancel scheduled prompts
- Supports natural time parsing: "06:30", "tomorrow at 9am", "in 30 minutes"
- Supports recurrence: "daily", "weekly", "weekdays", "monthly"

### Scheduled Prompts System

Calendar-triggered prompt execution system:

**New files:**
- `db/scheduled_prompts.py` - CRUD operations for scheduled prompt executions
- `scheduling/prompt_scheduler.py` - Background scheduler that:
  - Polls calendars every 5 minutes for events matching scheduled prompts
  - Executes due prompts every 30 seconds
  - Creates execution records for calendar events
  - Executes prompts through the Smart Alias pipeline

**Smart Alias fields added:**
- `use_scheduled_prompts` - Enable scheduled prompts feature
- `scheduled_prompts_account_id` - Google account for calendar access
- `scheduled_prompts_calendar_id` - Calendar to monitor for prompt events
- `scheduled_prompts_lookahead_minutes` - How far ahead to look for events (default: 15)
- `scheduled_prompts_store_response` - Store LLM responses in event description

### Bug Fixes This Session

1. **Designator tracking not logging costs to tags/clients**
   - Problem: `log_designator_usage()` was called with empty tag string
   - Fix: Pass `tag` parameter to all 4 call sites in proxy.py
   - Files: `proxy.py` lines 319, 1190, 1696, 2106, 2661

2. **Gmail live source `account_email` undefined**
   - Problem: Variable `account_email` used but never defined
   - Fix: Changed to `self._get_account_email()`
   - File: `live/sources.py` line 3729

3. **Gmail mark_read action returning 403**
   - Problem: OAuth tokens missing `gmail.modify` scope
   - Fix: Added `gmail.modify` scope to both `gmail` and `workspace` scope sets
   - File: `admin/app.py` GOOGLE_SCOPES

4. **Calendar actions need full access**
   - Added `calendar` full scope for delete operations
   - File: `admin/app.py` GOOGLE_SCOPES

5. **OAuth token credentials being wiped on refresh**
   - Problem: `update_oauth_token_data()` was overwriting entire token data
   - Fix: Merge with existing data to preserve client_id, client_secret, refresh_token
   - File: `db/oauth_tokens.py`

6. **Withings token refresh "Expired nonce" error**
   - Problem: OAuth refresh was incorrectly including signature/nonce
   - Fix: Removed signature/nonce from token refresh (only needed for API calls)
   - File: `live/sources.py`

### OAuth Scopes Updated

Added write permissions to Google OAuth scopes:

```python
GOOGLE_SCOPES = {
    "gmail": [
        # ... existing ...
        "https://www.googleapis.com/auth/gmail.modify",  # Mark read/unread, archive, label
    ],
    "calendar": [
        # ... existing ...
        "https://www.googleapis.com/auth/calendar",  # Full access including delete
    ],
    "workspace": [
        # ... includes all above ...
    ],
}
```

**Note:** Users must re-authenticate Google accounts in admin UI to get new scopes.

### Key Files Created This Session
- `actions/` - Entire Smart Actions module
  - `base.py` - ActionHandler abstract base class
  - `executor.py` - Action execution engine
  - `parser.py` - Smart action block parser
  - `registry.py` - Handler registration
  - `loader.py` - Handler discovery and loading
  - `handlers/email.py` - Email actions
  - `handlers/calendar.py` - Calendar actions
  - `handlers/notification.py` - Notification actions
  - `handlers/schedule.py` - Scheduled prompt actions
- `db/scheduled_prompts.py` - Scheduled prompts CRUD
- `scheduling/prompt_scheduler.py` - Background prompt scheduler

### Current State
- Dev container (`llm-relay-dev`) has all changes deployed
- Branch: `beta-calendar-picker`
- All Smart Actions working (email, calendar, notifications, scheduled prompts)
- Gmail modify actions require re-authentication to get new scopes

### Known Issues
- Users need to re-authenticate Google accounts to get `gmail.modify` and `calendar` scopes

---

## Development Session (2026-01-19) - Smart Routes Provider

### Routes Smart Provider (`live/sources.py`)

New high-level `RoutesSmartProvider` class that simplifies journey planning:

**Features:**
- Accepts natural language locations ("London", "Loftus Road Stadium")
- Automatic geocoding with 30-day caching
- Natural time parsing ("tomorrow 9am", "Saturday 3pm", "in 2 hours")
- Supports both `arrival_time` (arrive BY) and `departure_time` (leave AT)
- Multiple travel modes: drive, walk, bicycle, transit
- Transit mode uses Google Routes API with full public transport connections
- Traffic-aware routing for driving
- Alternative routes returned
- Turn-by-turn directions formatted for context injection

**Parameters:**
```python
{
    "origin": "Harpenden",
    "destination": "Loftus Road Stadium",
    "arrival_time": "Saturday 3pm",  # OR departure_time, OR neither
    "mode": "transit"  # drive|walk|bicycle|transit
}
```

**Auto-creation:**
- Source auto-created when `GOOGLE_MAPS_API_KEY` is set
- Registered as `builtin_routes` source type
- Designator hint guides usage: arrival_time for events, departure_time for leaving

**Key fixes during development:**
1. `routingPreference: "TRAFFIC_AWARE"` only valid for DRIVE mode (caused 400 error for transit)
2. Transit routes should use `arrivalTime` not `departureTime` for event-based queries
3. Separated arrival_time and departure_time as distinct parameters

### Files Modified
- `live/sources.py` - Added `RoutesSmartProvider` class (~600 lines)
- `db/live_data_sources.py` - Auto-create `routes` source when API key present
- `routing/smart_enricher.py` - Added param hints for routes and transport providers
- `admin/templates/live_data_sources.html` - Updated type labels

### Designator Guidance Updates

Updated param hints to guide the designator:
- **routes**: "PREFER THIS for all journey planning" with arrival_time/departure_time options
- **transport**: "UK train departures ONLY - for full journey planning use routes with mode=transit"
- **google-maps**: "For directions/routes, use routes source instead"

### Current State
- Dev container has all changes deployed
- Routes provider tested with driving and transit modes
- Geocoding caching working (30-day TTL)
- Arrival time correctly used for transit (calculates backwards from event time)

---

## Development Session (2026-01-19) - Plugin System Phase 1

### Plugin System Infrastructure

Implemented Phase 1 of the plugin system as documented in `docs/PLUGIN_SYSTEM_PLAN.md`. This creates a modular architecture for extending LLM Relay without modifying core code.

**New directories:**
- `plugin_base/` - Core plugin framework
- `builtin_plugins/` - Built-in plugins shipped with app
- `plugins/` - User plugins (can override builtins)
- `tests/plugins/` - Unit tests for plugin system

**Key files created:**
- `plugin_base/common.py` - Shared dataclasses (FieldDefinition, FieldType, ValidationResult, etc.)
- `plugin_base/loader.py` - Plugin discovery and registration (PluginRegistry)
- `plugin_base/document_source.py` - PluginDocumentSource base class
- `plugin_base/live_source.py` - PluginLiveSource base class
- `plugin_base/action.py` - PluginActionHandler base class
- `db/plugin_configs.py` - CRUD operations for plugin configurations
- `db/models.py` - Added PluginConfig model
- `plugins/README.md` - Plugin development guide

**Plugin types:**
1. **Document Sources** - Enumerate and fetch documents for RAG indexing
2. **Live Sources** - Fetch real-time data at request time (weather, stocks, etc.)
3. **Actions** - Allow LLM to perform side effects (email, calendar, tasks)

**Plugin discovery:**
- Runs at app startup via `discover_plugins()` in `proxy.py`
- Scans `builtin_plugins/` first, then `plugins/`
- User plugins can override builtins with same `source_type`
- 64 unit tests covering all functionality

**Database:**
- New `plugin_configs` table stores plugin instance configurations
- Config stored as JSON, allowing plugins to define custom fields
- Auto-created via SQLAlchemy's `create_all()`

### Key Design Decisions

1. **Stub base classes created early** - Needed for loader testing even though full implementation is in later phases
2. **No hot reloading** - Restart required for new plugins (simpler, more reliable)
3. **Plugins are trusted code** - No sandboxing (admin-installed only)
4. **Field definitions drive UI** - Plugins declare config fields, admin UI renders dynamically

### Test Coverage

```
tests/plugins/test_common.py - 40 tests
tests/plugins/test_loader.py - 24 tests
tests/plugins/fixtures/ - Mock plugins for testing
```

### Current State
- Phase 1 complete (infrastructure)
- Plugin discovery working (shows "0 document sources, 0 live sources, 0 actions" until builtins migrated)
- Next: Phase 2 (Action Plugins) - migrate existing action handlers to plugin architecture

### Files Modified
- `proxy.py` - Added plugin discovery at startup
- `db/models.py` - Added PluginConfig model
- `db/__init__.py` - Added plugin_configs exports

---

## Development Session (2026-01-21) - Smart Plugins (News, Amazon) & Bug Fixes

### New Smart Live Source Plugins

**Smart News (`builtin_plugins/live_sources/smart_news.py`)**
- Real-Time News Data API via RapidAPI
- Smart features:
  - Auto-detects query type: headlines, topic, search, local news
  - Fetches headlines then enriches with full story coverage
  - Geo-based local news support
  - Combines multiple sources for comprehensive coverage
- Parameters: `query`, `country`, `time_range`, `full_coverage`
- Default country: GB (configurable)

**Smart Amazon (`builtin_plugins/live_sources/smart_amazon.py`)**
- Real-Time Amazon Data API via RapidAPI
- Smart features:
  - Natural language product search
  - Auto-resolves product names to ASINs with 24-hour caching
  - Combines search + details + reviews in one query
  - Handles comparisons, bestsellers, deals queries
- Parameters: `query`, `type`, `country`, `include_reviews`
- Default country: GB (configurable)
- Output includes: price, rating, availability, about product, specifications, reviews

### Bug Fixes

1. **Amazon API 400 errors - country format**
   - Problem: API expects 2-letter codes (`GB`) not domains (`amazon.co.uk`)
   - Fix: Updated `COUNTRIES` mapping to use ISO codes
   - Added `_normalize_country_code()` to handle legacy values and UK→GB alias

2. **Plugin config not loading from auth_config_json**
   - Problem: `PluginLiveSourceAdapter._build_config_from_source()` only checked `config_json`
   - Fix: Added fallback to read from `auth_config_json` where plugin configs are stored
   - File: `live/sources.py`

3. **Sports plugin missing EFL Championship**
   - Problem: `TOURNAMENT_IDS` only had top leagues
   - Fix: Added comprehensive tournament list including:
     - English: Championship, League One, League Two, FA Cup, EFL Cup
     - European: Eredivisie, Conference League
     - Scottish Premiership, World Cup, Euros, MLS
   - File: `builtin_plugins/live_sources/smart_sports.py`

4. **Notification handler showing OAuth connect button**
   - Problem: Default `requires_oauth=True` in base class
   - Fix: Added `requires_oauth = False` to NotificationActionHandler
   - File: `actions/handlers/notification.py`

### Admin UI Changes

**Live Data Sources page:**
- Fields with configured env vars are now hidden (e.g., RAPIDAPI_KEY)
- Added `/api/live-data-sources/env-var-status` endpoint
- Select dropdowns now properly restore saved values

**Smart Actions page:**
- Removed info banner about Smart Aliases
- Removed usage examples section (meant for LLMs, not humans)
- Notification handler no longer shows "Connect" dropdown

**Navigation:**
- Moved RAG Config and Web Config from Data Sources to System section

### Plugin System Enhancements

- `FieldDefinition.env_var` property - hide field when env var is set
- Plugin `best_for` descriptions improved for better designator selection
- Amazon: "BEST FOR: 'How much does X cost?', product prices..."

### Key Files Modified
- `builtin_plugins/live_sources/smart_news.py` - New plugin
- `builtin_plugins/live_sources/smart_amazon.py` - New plugin
- `builtin_plugins/live_sources/smart_sports.py` - Added tournament IDs
- `live/sources.py` - Fixed PluginLiveSourceAdapter config loading
- `actions/handlers/notification.py` - Set requires_oauth=False
- `admin/templates/smart_actions.html` - Removed banner and examples
- `admin/templates/live_data_sources.html` - Env var field hiding
- `admin/templates/base.html` - Moved nav links
- `admin/app.py` - Added env-var-status endpoint
- `plugin_base/common.py` - Added env_var to FieldDefinition

---

## Development Session (2026-01-25) - Live Document Lookup for File-Based Sources

### Feature Overview

Added live document lookup capability to file-based unified sources (local filesystem, Paperless, Nextcloud). This enables fetching full document content by filename when users ask "show me the invoice" or "what's in that document".

**Problem solved:** RAG retrieves chunks semantically, but when users ask for specific file content, only chunks were returned. Now the system can fetch and return full document content.

### Implementation

**Unified sources updated with `supports_live = True`:**
- `LocalFilesystemUnifiedSource` - Fuzzy filename matching, full file content fetch
- `PaperlessUnifiedSource` - Title matching, API-based document fetch
- `NextcloudUnifiedSource` - Filename matching, WebDAV-based file fetch

**New capabilities:**
- `action='lookup'` with `filename`/`title` param - Fetch full document content
- `action='list'` - Enumerate files in the source
- `action='search'` (Paperless only) - Search by content via API

**Document resolution flow:**
1. User asks for specific file ("show me INVOICE-2022.docx")
2. Designator selects unified source with `action='lookup', filename='...'`
3. `_resolve_document()` uses fuzzy matching (rapidfuzz) against indexed metadata
4. `read_document()` fetches full content
5. Text extraction handles PDF, DOCX, XLSX, PPTX formats

### Key Architecture Patterns

**Unified Source Live Params:**
- `param_type` is a string ("string", "integer", "boolean"), NOT an enum
- Import `ParamDefinition` from `plugin_base.live_source`, not `ParamType`

**Plugin Registry:**
- `discover_plugins()` returns counts: `{'unified_sources': 19, ...}`
- Access plugins via `unified_source_registry.get(source_type)` or `unified_source_registry.get_all()`
- `get_unified_source_plugin(source_type)` is the public API for lookups

**Document Source Base Class (`mcp/sources.py`):**
- Added `supports_live_lookup` property (default: True)
- Added `resolve_document(query, indexed_metadata)` method with fuzzy matching
- Fallback to exact substring match when rapidfuzz unavailable

**Fuzzy Matching (rapidfuzz):**
- Use `fuzz.partial_ratio` scorer, NOT `fuzz.token_set_ratio`
- `partial_ratio` finds substrings: "20241024" matches "scanned_20241024-0726.pdf"
- `token_set_ratio` requires full token matches, fails on partial date/number queries
- Score cutoff of 70 works well for filename matching

### Files Created/Modified

**Modified:**
- `mcp/sources.py` - Added `resolve_document()` to DocumentSource base class
- `builtin_plugins/unified_sources/local_filesystem.py` - Added live lookup support
- `builtin_plugins/unified_sources/paperless.py` - Added live lookup support
- `builtin_plugins/unified_sources/nextcloud.py` - Added live lookup support
- `rag/indexer.py` - Fixed DOCX processing bug (missing `doc = Document(...)` line)
- `requirements.txt` - Added `rapidfuzz>=3.0.0`

**New docs:**
- `docs/LIVE_DOCUMENT_LOOKUP_PLAN.md` - Implementation plan

### Bug Fix

**DOCX processing in `_process_content()` (`rag/indexer.py:1634`):**
```python
# Before (broken):
from docx import Document
text = "\n".join(para.text for para in doc.paragraphs)  # doc undefined!

# After (fixed):
from docx import Document
doc = Document(io.BytesIO(content.binary))
text = "\n".join(para.text for para in doc.paragraphs)
```

### Smart Source Selection Dependency

Live document lookup requires **Smart Source Selection** to be enabled on the Smart Alias. Without it:
- No designator call occurs
- `live_params` is never populated
- Unified sources won't be queried for live data

With Smart Source Selection enabled, unified sources from document stores are automatically included in the designator prompt (even when "Live Data Sources" toggle is off), allowing the designator to route document lookup requests appropriately.

### Two-Pass Retrieval

Two-pass retrieval combines semantic RAG search with live document fetch for full content retrieval:

**Flow:**
1. User asks for document content ("show me the invoice from October")
2. `analyze_query()` returns `routing=QueryRouting.TWO_PASS` with `two_pass_fetch_full=True`
3. **Pass 1 (RAG)**: Semantic search with `include_metadata=True` returns chunk content + `source_uri`
4. **Pass 2 (Live)**: For each unique `source_uri`, call `fetch()` with `_two_pass_uri` param
5. Results merged: RAG chunks provide context, live fetch provides full document content

**Key components:**
- `QueryRouting.TWO_PASS` in `plugin_base/unified_source.py`
- `QueryAnalysis.two_pass_fetch_full` - enables pass 2 document fetch
- `rag_search_fn(query, limit, include_metadata=True)` - returns `list[dict]` with `source_uri`, `source_file`, `content`, `score`
- `_two_pass_uri` param in `fetch()` - direct URI for document resolution (skips fuzzy matching)

**URI formats by source:**
- Local filesystem: `file:///path/to/file.pdf`
- Paperless: `paperless://{doc_id}`
- Nextcloud: `nextcloud://{relative_path}`

**When to use:**
- User asks for full content of specific documents
- Semantic search can identify relevant documents but chunks are insufficient
- Need to combine RAG context with complete document content
