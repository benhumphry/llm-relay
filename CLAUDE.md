# CLAUDE.md

This file provides guidance for Claude Code when working on this codebase.

## Project Overview

LLM Relay - A self-hosted proxy that unifies multiple LLM providers (Anthropic, OpenAI, Google, Perplexity, OpenRouter, Ollama) behind Ollama and OpenAI-compatible APIs with usage tracking and cost attribution.

## Tech Stack

- **Python 3.12+** with Flask
- **SQLAlchemy 2.0** for database (SQLite default, PostgreSQL optional)
- **Gunicorn** for production serving
- **Alpine.js** for admin UI interactivity
- **Docker** for deployment

## Project Structure

```
proxy.py              # Main Flask app with all API endpoints
VERSION               # Single source of truth for version number
version.py            # Module to read VERSION file
providers/            # LLM provider implementations
  base.py             # Abstract base class for providers
  registry.py         # Model resolution and provider discovery
  loader.py           # Provider initialization
  *_provider.py       # Individual provider implementations
db/                   # Database layer
  models.py           # SQLAlchemy models
  connection.py       # DB connection and migrations
  aliases.py          # Alias CRUD operations
  smart_routers.py    # Smart router CRUD operations
  smart_caches.py     # Smart cache CRUD operations
  smart_augmentors.py # Smart augmentor CRUD operations
  sync_descriptions.py # Model description sync from providers/OpenRouter
  seed.py             # LiteLLM pricing data sync
tracking/             # Usage tracking
  usage_tracker.py    # Async request logging and statistics
  tag_extractor.py    # Tag parsing from API keys
  ip_resolver.py      # Client IP detection
routing/              # Smart routing
  smart_router.py     # Designator-based model selection
  smart_cache.py      # Semantic response caching with ChromaDB
  smart_augmentor.py  # Web search/scrape context augmentation
augmentation/         # Web search and scraping
  search/             # Search provider implementations
    base.py           # SearchProvider abstract base class
    searxng.py        # SearXNG search provider
    perplexity.py     # Perplexity search provider
  scraper.py          # URL content scraping
context/              # ChromaDB integration
  chroma.py           # ChromaDB client wrapper and collection management
admin/                # Admin dashboard
  app.py              # Admin API endpoints and pages
  templates/          # Jinja2 templates with Alpine.js
config/               # Configuration
  override_loader.py  # Model override YAML loading
```

## Key Concepts

### Model Resolution (providers/registry.py)
Resolution order: Smart Router -> Smart Cache -> Smart Augmentor -> Alias -> Provider prefix -> Provider search -> Default fallback

### ResolvedModel
Dataclass returned by `registry.resolve_model()` containing provider, model_id, and metadata about how it was resolved (alias, router, default fallback).

### Usage Tracking (tracking/usage_tracker.py)
Async background service that batches request logs and updates daily statistics. Tracks tokens, costs, and attributes to tags extracted from API keys.

### Aliases (db/aliases.py)
Simple name -> target_model mappings for user-friendly model names.

### Smart Routers (routing/smart_router.py)
Use a designator LLM to intelligently route requests to the best candidate model based on query content. The designator receives candidate model info including descriptions to help make informed routing decisions.

### Smart Caches (routing/smart_cache.py)
Semantic response caching using ChromaDB. Caches LLM responses and returns them for semantically similar queries, reducing token usage and costs. Key settings:
- `similarity_threshold` - How similar queries must be (0.0-1.0, default 0.95)
- `match_last_message_only` - Only match last user message, ignores conversation history (useful for OpenWebUI)
- `match_system_prompt` - Whether to include system prompt in cache key
- `min_cached_tokens` / `max_cached_tokens` - Filter responses by length (filters out titles, short follow-ups)
- `cache_ttl_hours` - How long cached entries remain valid

Requires ChromaDB (set `CHROMA_URL` environment variable).

### Smart Augmentors (routing/smart_augmentor.py)
Context augmentation using web search and URL scraping. A designator LLM analyzes each query and decides:
- `direct` - pass through unchanged (simple questions, coding, creative tasks)
- `search:query` - search the web and inject results (current events, recent data)
- `scrape:url1,url2` - fetch specific URLs mentioned by the user
- `search+scrape:query` - search then scrape top results for comprehensive research

Key settings:
- `designator_model` - Fast/cheap model to decide augmentation (e.g., "openai/gpt-4o-mini")
- `target_model` - Model to forward augmented requests to
- `search_provider` - Which search provider to use ("searxng", "perplexity")
- `max_search_results` / `max_scrape_urls` - Limits on fetched content
- `max_context_tokens` - Maximum tokens for injected context
- `purpose` - Context for the designator (e.g., "research assistant for current events")

Requires a search provider (set `SEARXNG_URL` for SearXNG, or `PERPLEXITY_API_KEY` for Perplexity).

### Model Descriptions (db/sync_descriptions.py)
Fetches model descriptions from provider APIs (Google) and OpenRouter's public API. Descriptions help smart routers make better decisions since LLM designators have training cutoffs and don't know about newer models.

### System Alerts (admin/app.py)
Query-based alerts that detect problematic models from RequestLog errors (404s, auth failures). Computed on-demand rather than stored.

## Common Tasks

### Running locally
```bash
pip install -r requirements.txt
python proxy.py
```

### Running tests
No test suite currently exists.

### Database migrations
Migrations run automatically on startup via `db/connection.py:run_migrations()`.

## Code Patterns

### Adding a new provider
1. Create `providers/new_provider.py` extending `LLMProvider` from `base.py`
2. Implement required methods: `chat()`, `chat_stream()`, `list_models()`
3. Register in `providers/loader.py`

### Adding admin endpoints
1. Add route in `admin/app.py`
2. Use `@require_auth` for page routes, `@require_auth_api` for API routes
3. Create template in `admin/templates/` following existing patterns

### Database changes
1. Add/modify models in `db/models.py`
2. Add migration logic in `db/connection.py:run_migrations()`
3. Add CRUD functions in appropriate `db/*.py` file
4. Export from `db/__init__.py`

## Important Notes

- All model names are normalized to lowercase
- The proxy supports both Ollama API (`/api/chat`, `/api/generate`) and OpenAI API (`/v1/chat/completions`)
- Streaming responses use Server-Sent Events
- Admin UI uses Alpine.js for reactivity - no build step required
- Cost tracking uses provider-specific pricing from `db/seed.py`

### Versioning
- Edit `VERSION` file to update the version number - it propagates to:
  - Startup log in `proxy.py`
  - Admin UI settings page (via `version` template variable)
  - Docker image (copied at build time)
