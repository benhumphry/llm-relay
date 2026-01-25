<p align="center">
  <img src="logo.png" alt="LLM Relay" width="128" height="128">
</p>

<h1 align="center">LLM Relay</h1>

<p align="center">
<strong>Intelligent LLM Gateway with Smart Routing, RAG, and Real-Time Data</strong><br>
A self-hosted proxy that unifies all your LLM providers behind one API, with AI-powered routing, document RAG, live data sources, web search, and semantic caching.
</p>

<p align="center">
  <img src="screenshot-dashboard.png" alt="LLM Relay Dashboard" width="800">
</p>

---

## What is LLM Relay?

An intelligent gateway that makes your LLMs smarter and easier to use:

- **Smart Routing** — Let AI pick the best model for each request from your available providers
- **Document RAG** — Automatically inject relevant context from your indexed documents
- **Live Data Sources** — Real-time data from weather, stocks, sports, news, and custom APIs
- **Smart Actions** — Let LLMs send emails, create calendar events, and perform tasks
- **Real-time Web Search** — Enrich requests with current information from the web
- **Semantic Caching** — Cache and reuse responses for similar queries to reduce costs
- **One endpoint, all models** — Claude, GPT, Gemini, Llama, and 700+ others through one API
- **Accurate cost tracking** — Token-level tracking with cache and reasoning tokens
- **Works with any client** — Ollama, OpenAI, and Anthropic API compatible

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

# Use a Smart Alias with RAG and live data
client.chat.completions.create(model="research-assistant", messages=[...])
```

Or with curl:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Works with Open WebUI, Cursor, Continue, Claude Code, and any Ollama or OpenAI-compatible client.

## Core Features

### Smart Aliases

The central feature for intelligent LLM handling. A Smart Alias is a virtual model that can combine any of these capabilities:

| Feature | Description |
|---------|-------------|
| **Routing** | Designator LLM analyzes each request and picks the best model from candidates |
| **RAG** | Inject relevant context from your indexed Document Stores |
| **Live Data** | Query real-time data sources (weather, stocks, sports, news, APIs) |
| **Web Search** | Real-time web search and page scraping for current information |
| **Smart Actions** | Let LLMs perform actions (send emails, create events, manage tasks) |
| **Cache** | Semantic response caching to reduce costs |
| **Memory** | Persistent memory that remembers explicit user facts across sessions |
| **Smart Source Selection** | Designator allocates token budget across RAG stores, web, and live sources |

**Example configurations:**

| Alias Name | Features | Use Case |
|------------|----------|----------|
| `smart` | Routing | Let AI pick between Claude, GPT, Gemini per request |
| `research` | RAG + Web + Live Data | Answer questions with docs, current info, and live data |
| `assistant` | RAG + Actions + Memory | Personal assistant that remembers you and can take actions |
| `docs` | Smart Tag + RAG | Tag any request to add document context |
| `cached-claude` | Cache | Reduce costs for repeated queries |

### Document RAG

Index your documents for semantic search and context injection:

**Supported Sources:**
- Local filesystem (Docker-mounted folders)
- Google Drive, Gmail, Calendar, Tasks, Contacts
- Notion databases and pages
- GitHub repositories
- Paperless-ngx document management
- Nextcloud files
- Slack channels
- Microsoft 365 (OneDrive, Outlook, Calendar, OneNote, Teams)

**Supported Formats:**
- Documents: PDF, DOCX, PPTX, XLSX, HTML, Markdown, plain text
- Images: PNG, JPG, TIFF with OCR support
- Code: Most programming languages

**Features:**
- Configurable embeddings (local, Ollama, or cloud providers)
- Cross-encoder reranking for improved relevance
- Scheduled indexing with cron expressions
- Incremental indexing (only changed documents)
- Two-pass retrieval: semantic search finds documents, then fetches full content

### Live Data Sources

Real-time data that's fetched at request time:

**Built-in Sources:**
- **Weather** — Current conditions and forecasts via Open-Meteo
- **Stocks** — Real-time prices, charts, and financials via Yahoo Finance
- **Sports** — Live scores, fixtures, and standings via API-Football
- **News** — Headlines and articles via News API
- **Routes** — Journey planning with traffic via Google Routes API
- **Transport** — UK train times via National Rail

**Plugin Sources:**
- **Amazon** — Product search, prices, and reviews
- **Custom APIs** — Connect any RapidAPI or MCP-compatible API

The designator LLM automatically decides which sources to query based on the user's question.

### Smart Actions

Let your LLMs perform real-world tasks:

**Email Actions (Gmail/Outlook):**
- Draft or send new emails
- Reply to and forward messages
- Archive, label, mark read/unread

**Calendar Actions (Google/Outlook):**
- Create, update, and delete events
- Check availability

**Task Actions (Todoist, Google Tasks):**
- Create and complete tasks
- Set due dates and priorities

**Notification Actions:**
- Send push notifications via webhooks

Actions are sandboxed - you configure which actions each Smart Alias can perform.

### Web Search

Real-time web search and content extraction:

- **SearXNG integration** — Self-hosted meta-search engine
- **Perplexity search** — Alternative search provider
- **Jina Reader** — Clean content extraction from URLs
- **Smart query generation** — Designator rewrites queries for better results

### Providers

15+ built-in providers, 700+ models:

| Provider | Environment Variable |
|----------|---------------------|
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

Plus: connect local Ollama instances and custom OpenAI-compatible providers through the Admin UI.

### Cost Tracking

Every request is logged with:
- Input tokens, output tokens, reasoning tokens
- Cache read/write tokens (Anthropic, Gemini, DeepSeek)
- Calculated cost in USD

View breakdowns by provider, model, tag, client, or time period in the Admin UI.

Pricing syncs from LiteLLM and handles provider quirks automatically.

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

## API Compatibility

### Ollama API (port 11434)
- `GET /api/tags` — List models
- `POST /api/chat` — Chat completion
- `POST /api/generate` — Text generation

### OpenAI API (port 11434)
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completion
- `POST /v1/completions` — Text completion

### Anthropic API (port 11434)
- `POST /v1/messages` — Chat completion with native Anthropic format

Configure Claude Code to use LLM Relay:
```bash
export ANTHROPIC_BASE_URL=http://your-relay-host:11434
```

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
      - /path/to/your/docs:/documents:ro  # Optional: mount local docs
    env_file:
      - .env

volumes:
  llm-relay-data:
```

### Production Setup

For production deployments:
- **PostgreSQL** — Recommended for the database (SQLite works for testing)
- **ChromaDB** — Required for RAG, caching, and semantic features

See **[INSTALLATION.md](INSTALLATION.md)** for complete setup including:
- PostgreSQL and ChromaDB configuration
- SearXNG integration for web search
- Google OAuth setup for Drive/Gmail/Calendar
- Full environment variable reference
- Docker Swarm deployment

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 11434 | API server port |
| `ADMIN_PORT` | 8080 | Admin UI port |
| `ADMIN_PASSWORD` | (random) | Admin UI password |
| `DATABASE_URL` | SQLite | PostgreSQL URL for production |
| `CHROMA_URL` | (none) | ChromaDB URL (enables RAG, Cache) |
| `SEARXNG_URL` | (none) | SearXNG URL (enables Web search) |
| `JINA_API_KEY` | (none) | Jina API key (search, scraping, reranking) |
| `GOOGLE_CLIENT_ID` | (none) | Google OAuth (Drive, Gmail, Calendar) |
| `RAPIDAPI_KEY` | (none) | RapidAPI key (news, Amazon, sports) |

### Without Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python proxy.py
```

## Admin UI

Web interface at port 8080 for:

**Routing**
- Smart Aliases — Configure routing, RAG, live data, actions, caching
- Redirects — Model name mappings

**Data Sources**
- Document Stores — Configure and index document sources
- Live Data Sources — Enable and configure real-time data
- Websites — Crawl and index websites

**Model Management**
- Providers & Models — API keys and model availability
- Local Servers — Connect Ollama instances

**Analytics**
- Statistics — Usage charts and cost breakdowns
- Request Log — Detailed request history

**System**
- Settings — Global configuration
- Alerts — System notifications

## Plugin System

LLM Relay supports plugins for extending functionality:

- **Document Sources** — Add new document indexing sources
- **Live Sources** — Add new real-time data providers
- **Action Handlers** — Add new actions LLMs can perform

Plugins are Python classes that implement a base interface. See `plugins/README.md` for development guide.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                               │
│    (Open WebUI, Cursor, Continue, Claude Code, curl, etc.)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Relay                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Routing   │  │     RAG     │  │  Live Data  │         │
│  │  (Designator│  │  (ChromaDB) │  │  (APIs)     │         │
│  │   selects)  │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Actions   │  │  Web Search │  │   Caching   │         │
│  │  (Email,    │  │  (SearXNG)  │  │  (Semantic) │         │
│  │   Calendar) │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Providers                            │
│   Anthropic  OpenAI  Gemini  DeepSeek  Ollama  ...         │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT
