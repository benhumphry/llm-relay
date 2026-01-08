<p align="center">
  <img src="logo.png" alt="LLM Relay" width="128" height="128">
</p>

<h1 align="center">LLM Relay</h1>

<p align="center">
<strong>A lightweight, self-hosted LLM Proxy with Smart Features</strong><br>
Unified API for cloud and local LLMs with cost tracking, intelligent routing, semantic caching, web augmentation, and document RAG.
</p>

<p align="center">
  <img src="screenshot-dashboard.png" alt="LLM Relay Dashboard" width="800">
</p>

---

## What is LLM Relay?

A single self-hosted proxy that puts all your LLM providers behind one API:

- **One endpoint, all models** — Claude, GPT, Gemini, Llama, and 700+ others
- **Accurate cost tracking** — Token-level tracking with cache and reasoning tokens
- **Flexible attribution** — Tag requests by user, project, or team
- **Works with any client** — Ollama and OpenAI API compatible
- **Smart Features** — Intelligent routing, semantic caching, web search augmentation, and document RAG

## Quick Start

```bash
git clone https://github.com/benhumphry/llm-relay.git
cd llm-relay
cp .env.example .env
# Add your API keys to .env

docker compose up -d
```

**That's it.** Your proxy is running:
- API: http://localhost:11434
- Admin UI: http://localhost:8080

## Use It

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="my-team")

# Use any provider through one client
client.chat.completions.create(model="claude-sonnet", messages=[...])
client.chat.completions.create(model="gpt-4o", messages=[...])
client.chat.completions.create(model="gemini-2.5-pro", messages=[...])
```

Or with curl:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Works with Open WebUI, Cursor, Continue, and any Ollama or OpenAI-compatible client.

## Features

### Providers

15 built-in providers, 700+ models:

| Provider | Set Environment Variable |
|----------|--------------------------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| xAI | `XAI_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| DeepInfra | `DEEPINFRA_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| SambaNova | `SAMBANOVA_API_KEY` |
| Cohere | `COHERE_API_KEY` |

Plus: connect local Ollama instances and add custom OpenAI-compatible providers through the Admin UI.

<p align="center">
  <img src="screenshot-providers.png" alt="Providers and models management" width="800">
</p>

### Cost Tracking

Every request is logged with input tokens, output tokens, reasoning tokens, cache tokens, and calculated cost. View breakdowns by provider, model, tag, or client in the Admin UI.

Pricing syncs from LiteLLM and handles provider quirks automatically (Gemini tiered pricing, Perplexity per-request fees, Anthropic cache multipliers, etc).

### Tagging

Attribute costs to users, projects, or teams:

```bash
# Via bearer token
curl -H "Authorization: Bearer alice,project-x" ...

# Via header  
curl -H "X-Proxy-Tag: alice,project-x" ...

# Via model suffix
curl -d '{"model": "claude-sonnet@alice"}' ...
```

### Aliases

Create friendly names for models:

| Alias | Target |
|-------|--------|
| `claude` | `anthropic/claude-sonnet-4-20250514` |
| `gpt` | `openai/gpt-4o` |
| `fast` | `groq/llama-3.3-70b-versatile` |

### Redirects

Transparent model name mappings with wildcard support. Unlike aliases, redirects are checked first in resolution and don't appear in logs:

| Source Pattern | Target |
|----------------|--------|
| `gpt-4` | `gpt-4o` |
| `openrouter/anthropic/*` | `anthropic/*` |

Use cases: seamless model upgrades, provider switching without client changes.

### Smart Routers

Let an LLM pick the best model for each request. Configure candidate models, and a fast designator model routes requests based on query content.

**Model Intelligence** (optional): Enable web-gathered comparative assessments so the designator knows each model's relative strengths and weaknesses. The system searches for model reviews and direct comparisons, then summarizes into actionable routing guidance.

<p align="center">
  <img src="screenshot-routers.png" alt="Smart router configuration" width="800">
</p>

### Smart Caches

Semantic response caching using ChromaDB. Returns cached responses for semantically similar queries, reducing token usage and costs.

- Configurable similarity threshold (default 95%)
- TTL-based expiration
- Token length filters (skip caching short responses)
- Option to match only last message (ignores conversation history)

Requires ChromaDB (`CHROMA_URL` environment variable).

### Smart Augmentors

Context augmentation via web search and URL scraping. A fast designator LLM analyzes each query and decides how to augment:

- **direct** — Pass through unchanged (coding, creative tasks)
- **search** — Search the web and inject results (current events, recent data)
- **scrape** — Fetch specific URLs mentioned by the user
- **search+scrape** — Search then scrape top results for comprehensive research

Requires a search provider (SearXNG or Perplexity).

### Smart RAGs

Document-based context augmentation using RAG (Retrieval-Augmented Generation). Index local document folders and automatically retrieve relevant context for each query.

- **Multiple formats** — PDF, DOCX, PPTX, HTML, Markdown, images (via Docling)
- **Flexible embeddings** — Local (bundled), Ollama, or OpenAI
- **Semantic search** — ChromaDB vector storage with configurable similarity threshold
- **Scheduled indexing** — Cron-based re-indexing for updated documents

Mount your document folders into the container, create a Smart RAG pointing to the path, and requests to that model name automatically include relevant document context.

Requires ChromaDB (`CHROMA_URL` environment variable).

### Admin UI

Clean web interface for:
- Provider and model management
- Ollama instance management (pull/delete models)
- Usage analytics with charts and filters
- Settings, pricing sync, data export

## Installation

### Docker Compose (Recommended)

```yaml
services:
  llm-relay:
    image: ghcr.io/benhumphry/llm-relay:latest
    ports:
      - "11434:11434"
      - "8080:8080"
    volumes:
      - llm-relay-data:/data
    env_file:
      - .env

volumes:
  llm-relay-data:
```

### Production & Advanced Setup

For production deployments, PostgreSQL is recommended. For Smart Cache, Smart Augmentor, and Model Intelligence features, ChromaDB is required.

See **[INSTALLATION.md](INSTALLATION.md)** for:
- PostgreSQL setup
- ChromaDB integration (vector storage for smart features)
- SearXNG integration (web search for Smart Augmentor)
- Full environment variable reference
- Docker Swarm deployment
- Troubleshooting guide

### Quick Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 11434 | API server port |
| `ADMIN_PORT` | 8080 | Admin UI port |
| `ADMIN_PASSWORD` | (random) | Admin UI password |
| `DATABASE_URL` | SQLite | PostgreSQL URL for production |
| `CHROMA_URL` | (none) | ChromaDB URL (enables Smart Cache, Model Intelligence) |
| `SEARXNG_URL` | (none) | SearXNG URL (enables Smart Augmentor search) |

### Without Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python proxy.py
```

## API Endpoints

### Ollama API
- `GET /api/tags` — List models
- `POST /api/chat` — Chat completion
- `POST /api/generate` — Text generation

### OpenAI API
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completion

## License

MIT
