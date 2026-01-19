<p align="center">
  <img src="logo.png" alt="LLM Relay" width="128" height="128">
</p>

<h1 align="center">LLM Relay</h1>

<p align="center">
<strong>Intelligent LLM Gateway with Smart Routing and Context Enrichment</strong><br>
A self-hosted proxy that unifies all your LLM providers behind one API, with AI-powered routing, document RAG, real-time web search, and semantic caching.
</p>

<p align="center">
  <img src="screenshot-dashboard.png" alt="LLM Relay Dashboard" width="800">
</p>

---

## What is LLM Relay?

An intelligent gateway that makes your LLMs smarter and easier to use:

- **Smart Routing** — Let AI pick the best model for each request from your available providers
- **Document RAG** — Automatically inject relevant context from your indexed documents
- **Real-time Web Search** — Enrich requests with current information from the web
- **Semantic Caching** — Cache and reuse responses for similar queries to reduce costs
- **One endpoint, all models** — Claude, GPT, Gemini, Llama, and 700+ others through one API
- **Accurate cost tracking** — Token-level tracking with cache and reasoning tokens
- **Works with any client** — Ollama and OpenAI API compatible

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

### Smart Aliases

The core feature for intelligent model handling. Create a Smart Alias and enable any combination of capabilities:

| Feature | Description |
|---------|-------------|
| **Routing** | Designator LLM analyzes each request and picks the best model |
| **RAG** | Inject relevant context from your indexed documents |
| **Web** | Real-time web search and scraping for current information |
| **Cache** | Semantic response caching to reduce costs |
| **Memory** | Persistent memory that remembers explicit user facts across sessions |
| **Smart Source Selection** | Designator allocates token budget across RAG stores and web |
| **Smart Tag** | Trigger by request tag instead of model name |

**Example configurations:**

| Alias Name | Features | Use Case |
|------------|----------|----------|
| `smart` | Routing | Let AI pick between Claude, GPT, Gemini per request |
| `research` | RAG + Web | Answer questions with docs and current info |
| `docs` | Smart Tag + RAG | Tag any request to add document context |
| `cached-claude` | Cache | Reduce costs for repeated queries |
| `assistant` | Memory | Maintain context across conversation sessions |

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

# Via in-content command (stripped before sending to LLM)
curl -d '{"messages": [{"role": "user", "content": "@relay[tag:alice] Hello!"}]}' ...
```

### Redirects

Transparent model name mappings with wildcard support:

| Source Pattern | Target |
|----------------|--------|
| `gpt-4` | `gpt-4o` |
| `openrouter/anthropic/*` | `anthropic/*` |

Use cases: seamless model upgrades, provider switching without client changes.

### Data Sources

Index documents and websites for RAG retrieval:

**Document Stores:**
- **Multiple formats** — PDF, DOCX, PPTX, HTML, Markdown, images (with OCR)
- **Multiple sources** — Local files, Google Drive, Gmail, Calendar, Notion, GitHub, Paperless, Nextcloud
- **Flexible embeddings** — Local (bundled), Ollama, or cloud providers
- **Scheduled indexing** — Cron-based re-indexing for updated documents

**Websites:**
- **Crawl and index** — Automatically crawl websites and extract content
- **Configurable depth** — Control how many levels of links to follow
- **URL patterns** — Include/exclude URLs matching specific patterns

### Admin UI

Clean web interface for:
- Provider and model management
- Smart Alias configuration
- Document Store management
- Usage analytics with charts and filters
- Settings and data export

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

### Production Setup

For production deployments, PostgreSQL is recommended. For Smart Alias features (RAG, Cache, Model Intelligence), ChromaDB is required.

See **[INSTALLATION.md](INSTALLATION.md)** for:
- PostgreSQL setup
- ChromaDB integration
- SearXNG integration (web search)
- Full environment variable reference
- Docker Swarm deployment

### Quick Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 11434 | API server port |
| `ADMIN_PORT` | 8080 | Admin UI port |
| `ADMIN_PASSWORD` | (random) | Admin UI password |
| `DATABASE_URL` | SQLite | PostgreSQL URL for production |
| `CHROMA_URL` | (none) | ChromaDB URL (enables Cache, RAG) |
| `SEARXNG_URL` | (none) | SearXNG URL (enables Web search) |
| `JINA_API_KEY` | (none) | Jina API key (search, scraping, reranking) |

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
- `POST /v1/completions` — Text completion

## Documentation

- [Getting Started](docs/guides/getting-started.md) - First-time setup walkthrough
- [Smart Aliases](docs/guides/smart-aliases.md) - Unified routing, caching, and enrichment

## License

MIT
