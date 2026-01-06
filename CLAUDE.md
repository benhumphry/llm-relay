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
tracking/             # Usage tracking
  usage_tracker.py    # Async request logging and statistics
  tag_extractor.py    # Tag parsing from API keys
  ip_resolver.py      # Client IP detection
routing/              # Smart routing
  smart_router.py     # Designator-based model selection
admin/                # Admin dashboard
  app.py              # Admin API endpoints and pages
  templates/          # Jinja2 templates with Alpine.js
config/               # Configuration
  override_loader.py  # Model override YAML loading
```

## Key Concepts

### Model Resolution (providers/registry.py)
Resolution order: Smart Router -> Alias -> Provider prefix -> Provider search -> Default fallback

### ResolvedModel
Dataclass returned by `registry.resolve_model()` containing provider, model_id, and metadata about how it was resolved (alias, router, default fallback).

### Usage Tracking (tracking/usage_tracker.py)
Async background service that batches request logs and updates daily statistics. Tracks tokens, costs, and attributes to tags extracted from API keys.

### Aliases (db/aliases.py)
Simple name -> target_model mappings for user-friendly model names.

### Smart Routers (routing/smart_router.py)
Use a designator LLM to intelligently route requests to the best candidate model based on query content.

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
