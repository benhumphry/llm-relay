# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
providers/            # LLM provider implementations
  base.py             # Abstract base class for providers
  registry.py         # Model resolution and provider discovery
  loader.py           # Provider initialization
db/                   # Database layer
  models.py           # SQLAlchemy models
  connection.py       # DB connection and migrations
  seed.py             # LiteLLM pricing data sync
tracking/             # Usage tracking
  usage_tracker.py    # Async request logging and statistics
routing/              # Smart features (router, cache, augmentor)
augmentation/         # Web search and scraping
  search/             # Search provider implementations (SearXNG, Perplexity)
  scraper.py          # URL content scraping
context/              # ChromaDB integration
  chroma.py           # ChromaDB client wrapper
  model_intelligence.py # Web-gathered model assessments
admin/                # Admin dashboard
  app.py              # Admin API endpoints and pages
  templates/          # Jinja2 templates with Alpine.js
rag/                  # Smart RAG (document retrieval)
  embeddings.py       # Embedding providers (local, Ollama, OpenAI)
  indexer.py          # Document indexing with Docling
  retriever.py        # ChromaDB semantic search
```

## Key Concepts

### Model Resolution (providers/registry.py)
Resolution order: Redirect -> Smart Router -> Smart Cache -> Smart Augmentor -> Smart RAG -> Alias -> Provider prefix -> Provider search -> Default fallback

### ResolvedModel
Dataclass returned by `registry.resolve_model()` containing provider, model_id, and metadata about how it was resolved (alias, router, default fallback).

### Usage Tracking (tracking/usage_tracker.py)
Async background service that batches request logs and updates daily statistics. Tracks tokens, costs, and attributes to tags extracted from API keys.

### Redirects (db/redirects.py)
Transparent model name mappings checked first in resolution. Unlike aliases, redirects:
- Are checked before smart routers (first in resolution order)
- Support wildcard patterns (e.g., `openrouter/anthropic/*` -> `anthropic/*`)
- Are transparent to the caller (no alias tracking in logs)

Use cases: model upgrades (`gpt-4` -> `gpt-5`), provider switches (`openrouter/anthropic/*` -> `anthropic/*`).

### Aliases (db/aliases.py)
Simple name -> target_model mappings for user-friendly model names.

### Smart Routers (routing/smart_router.py)
Use a designator LLM to intelligently route requests to the best candidate model based on query content. The designator receives candidate model info including descriptions to help make informed routing decisions.

Key settings:
- `use_model_intelligence` - Enhance designator with web-gathered model assessments (strengths, weaknesses, best use cases). Requires ChromaDB.

### Model Intelligence (context/model_intelligence.py)
Web-gathered comparative assessments of LLM models for Smart Router designators. When enabled on a router:
1. Searches for individual model reviews/benchmarks
2. Searches for direct model comparisons ("Claude vs GPT-4o")
3. Summarizes into RELATIVE strengths/weaknesses between the specific candidates
4. Cached in ChromaDB with TTL-based expiry

Key settings (on SmartRouter):
- `use_model_intelligence` - Enable intelligence-enhanced routing
- `intelligence_model` - Model to summarize search results (required when enabled)
- `search_provider` - Which search provider to use for gathering intelligence

The comparative approach means assessments highlight when to choose Model A over Model B specifically, rather than generic model descriptions.

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

### Smart RAGs (routing/smart_rag.py)
Document-based context augmentation using RAG (Retrieval-Augmented Generation). Index local document folders and retrieve relevant context to enhance LLM requests.

How it works:
1. Documents are parsed using Docling (PDF, DOCX, PPTX, HTML, Markdown, images)
2. Text is chunked and embedded into ChromaDB (per-RAG collection)
3. When a request comes in, the query is embedded and similar chunks retrieved
4. Retrieved context is injected into the system prompt
5. Request is forwarded to the configured target model

Key settings:
- `source_path` - Docker-mounted folder containing documents (e.g., `/data/documents`)
- `target_model` - Model to forward augmented requests to
- `embedding_provider` - "local" (bundled sentence-transformers), "ollama", or any configured provider
- `embedding_model` - Model name for the embedding provider
- `vision_provider` - "local" (Docling default), "ollama", or any configured provider for PDF parsing
- `vision_model` - Vision model for document understanding (e.g., "granite3.2-vision:latest")
- `chunk_size` / `chunk_overlap` - Document chunking parameters
- `max_results` - Maximum chunks to retrieve
- `similarity_threshold` - Minimum similarity score for retrieval (0.0-1.0)
- `max_context_tokens` - Maximum tokens for injected context
- `index_schedule` - Cron expression for scheduled re-indexing

Embedding providers:
- **local** (default) - Bundled `BAAI/bge-small-en-v1.5` model (~130MB), no external deps
- **ollama** - Uses Ollama's `/api/embeddings` endpoint (nomic-embed-text, mxbai-embed-large, etc.)
- **Any configured provider** - Uses provider's embedding API (costs logged to request log)

Vision providers (for PDF parsing):
- **local** (default) - Bundled Docling models, runs on CPU/GPU
- **ollama** - Offload to Ollama vision model (e.g., granite3.2-vision) for faster processing
- **Any configured provider** - Use OpenAI, Anthropic, etc. for document understanding

Requires ChromaDB (set `CHROMA_URL` environment variable).

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
2. For NEW tables: SQLAlchemy's `create_all()` creates them automatically
3. For EXISTING tables (adding columns): Add migration in `db/connection.py:_run_migrations()`
4. Add CRUD functions in appropriate `db/*.py` file (follow `smart_caches.py` pattern)
5. Export from `db/__init__.py`

## API Endpoints

The proxy exposes two compatible APIs:

**Ollama API** (port 11434):
- `GET /api/tags` — List models
- `POST /api/chat` — Chat completion
- `POST /api/generate` — Text generation

**OpenAI API** (port 11434):
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completion

## Important Notes

- All model names are normalized to lowercase
- Streaming responses use Server-Sent Events
- Admin UI uses Alpine.js for reactivity - no build step required
- Cost tracking uses provider-specific pricing from `db/seed.py`
- Edit `VERSION` file to update version (propagates to logs, admin UI, Docker image)

## Development Workflow

### Docker Compose Files
- `docker-compose.yml` - Production config, pulls from `ghcr.io/benhumphry/llm-relay:latest`
- `docker-compose.dev.yml` - Development config, builds locally with hot reload

### Building and Testing Changes
```bash
# Build and deploy to dev instance (does NOT affect production)
docker compose -f docker-compose.dev.yml up -d --build

# View dev logs
docker logs -f llm-relay-dev

# Production uses the ghcr.io image - do NOT use `docker compose up -d --build` 
# on docker-compose.yml as it will rebuild and restart production
```

### Dev vs Production Ports
- Production (`llm-relay-llm-relay-1`): ports 11434, 8080
- Development (`llm-relay-dev`): ports 11435, 8081
