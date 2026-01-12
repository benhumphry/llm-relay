# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Relay - A self-hosted proxy that unifies multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek, Groq, xAI, and more) behind Ollama and OpenAI-compatible APIs with usage tracking, cost attribution, and intelligent features.

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
mcp/                  # Document sources (direct APIs)
  sources.py          # Document source classes for RAG indexing
docs/                 # Documentation
  guides/             # Feature guides
```

## Admin UI Navigation

```
Dashboard
Model Management
├── Providers & Models
└── Local Servers (Ollama instances)
Data Sources
├── Document Stores (indexed documents for RAG)
└── RAG Config / Web Config (global settings)
Routing
├── Smart Aliases (unified routing + enrichment + caching)
└── Redirects (transparent model mappings)
Usage
├── Statistics
└── Request Log
System
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
| **Smart Tag** (`is_smart_tag`) | Trigger by request tag instead of model name |
| **Passthrough** (`passthrough_model`) | Honor original model when triggered as Smart Tag |
| **Context Priority** (`context_priority`) | When both RAG+Web enabled: balanced/prefer_rag/prefer_web |

**Processing pipeline (SmartAliasEngine):**
1. Cache check (if enabled)
2. Routing (if enabled) - picks target model
3. Enrichment (if RAG or Web enabled) - injects context
4. Forward to target model
5. Cache store (if enabled)

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
- `mcp:gdrive` - Google Drive via OAuth
- `mcp:gmail` - Gmail via OAuth  
- `mcp:gcalendar` - Google Calendar via OAuth
- `paperless` - Paperless-ngx via REST API
- `notion` - Notion via direct REST API
- `mcp:github` - GitHub via REST API

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

## API Endpoints

**Ollama API** (port 11434):
- `GET /api/tags` — List models
- `POST /api/chat` — Chat completion
- `POST /api/generate` — Text generation

**OpenAI API** (port 11434):
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completion
- `POST /v1/completions` — Text completion

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
