<p align="center">
  <img src="logo.png" alt="LLM Proxy" width="128" height="128">
</p>

<h1 align="center">Multi-Provider LLM Proxy</h1>

<p align="center">
<strong>A self-hosted alternative to OpenRouter</strong><br>
Unified API for cloud and local LLMs with comprehensive cost tracking and flexible attribution.
</p>

<p align="center">
<a href="#quick-start">Quick Start</a> •
<a href="#features">Features</a> •
<a href="#cost-tracking">Cost Tracking</a> •
<a href="#tagging">Tagging</a> •
<a href="#admin-ui">Admin UI</a> •
<a href="#providers">Providers</a>
</p>

---

## Why This Proxy?

**The problem:** You use multiple LLM providers—Claude for writing, GPT for coding, Gemini for long context, local Ollama for experimentation. Each has different APIs, pricing models, and billing dashboards. Tracking costs across teams and projects is painful.

**The solution:** A single self-hosted proxy that:

- **Unifies all providers** behind Ollama and OpenAI-compatible APIs
- **Tracks every token** with accurate cost calculation per provider
- **Attributes costs** to users, projects, or teams via flexible tagging
- **Works with any client** that supports Ollama or OpenAI SDKs

Think of it as running your own OpenRouter—but self-hosted, with granular cost tracking, and support for local models.

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/benhumphry/ollama-llm-proxy.git
cd ollama-llm-proxy
cp .env.example .env
# Edit .env with your API keys

# 2. Run
docker compose up -d

# 3. Use
curl http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Access:**
- API Server: http://localhost:11434
- Admin UI: http://localhost:8080

## Features

### Unified LLM Interface

One API endpoint, all your models:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="my-team")

# Use Claude
client.chat.completions.create(model="claude-sonnet", messages=[...])

# Use GPT
client.chat.completions.create(model="gpt-5.2", messages=[...])

# Use Gemini
client.chat.completions.create(model="gemini-2.5-pro", messages=[...])

# Use local Ollama
client.chat.completions.create(model="llama3.2", messages=[...])
```

Both **Ollama API** (`/api/chat`) and **OpenAI API** (`/v1/chat/completions`) formats are supported. Works with Open WebUI, Cursor, Continue, and any Ollama or OpenAI-compatible client.

### 10+ Providers, 400+ Models

| Provider | Models | Highlights |
|----------|--------|------------|
| **Anthropic** | Claude 4.5, 4, 3.5 | Cache tokens, vision, extended thinking |
| **OpenAI** | GPT-5.x, 4.x, o1/o3/o4 | Reasoning models, vision |
| **Google** | Gemini 3, 2.5, 2.0 | 2M context, grounding, thinking |
| **Perplexity** | Sonar, Deep Research | Real-time web search |
| **Groq** | Llama 4, Qwen, Compound | Ultra-fast inference |
| **DeepSeek** | V3.2, R1 | Advanced reasoning |
| **Mistral** | Large, Codestral, Pixtral | European provider |
| **xAI** | Grok 4.1, 4, 3 | 2M context, real-time knowledge |
| **OpenRouter** | 400+ models | Meta-provider with dynamic pricing |
| **Ollama** | Any local model | Automatic discovery, multiple instances |

Add custom OpenAI-compatible or Anthropic-compatible providers through the Admin UI.

### Accurate Cost Tracking

The proxy calculates costs with the same precision as your provider bills:

**Token-level tracking:**
- Input and output tokens
- Reasoning tokens (o1, o3, DeepSeek-R1)
- Cache read/write tokens (Anthropic, OpenAI, xAI, Groq)

**Provider-specific pricing:**
- **Static rates** from pre-configured pricing (updated with releases)
- **Dynamic pricing** from OpenRouter (actual cost per request)
- **Tiered pricing** for Gemini (rates change above 200k tokens)
- **Complex pricing** for Perplexity (per-request fees + search costs)
- **Cache multipliers** per model (e.g., Anthropic cache reads at 10% of input cost)

**Model overrides:** Customize pricing for any model through the Admin UI.

### Flexible Cost Attribution

Tag requests to track costs by user, project, team, or any dimension:

```bash
# Via bearer token (best for SDKs)
curl -H "Authorization: Bearer alice" ...

# Via header
curl -H "X-Proxy-Tag: project-alpha" ...

# Via model suffix
curl -d '{"model": "claude-sonnet@alice"}' ...

# Multiple tags for multi-dimensional tracking
curl -H "Authorization: Bearer alice,project-alpha,q1-2025" ...
```

Tags appear in the usage dashboard with breakdowns by cost, tokens, and request count.

### Web Admin UI

A polished interface for configuration and analytics:

- **Dashboard** — Provider status, connection testing
- **Providers** — Add/remove providers, configure Ollama instances
- **Models** — Browse all models, override pricing, enable/disable
- **Aliases** — Create shortcuts (e.g., `claude` → `claude-sonnet-4-5-20250929`)
- **Usage** — Charts, breakdowns by tag/provider/model, request logs
- **Settings** — Default model, admin password, data retention

### Additional Capabilities

- **Streaming** — NDJSON (Ollama) and SSE (OpenAI) formats
- **Vision** — Pass images to any vision-capable model
- **Reasoning models** — Automatic parameter handling for o1/o3/o4/DeepSeek-R1
- **Local Ollama** — Connect multiple instances, auto-discover models
- **Docker secrets** — `*_FILE` suffix for secure key injection
- **PostgreSQL** — Optional external database for production

## Cost Tracking

### How It Works

Every request is logged with:

| Field | Description |
|-------|-------------|
| `input_tokens` | Prompt tokens |
| `output_tokens` | Response tokens |
| `reasoning_tokens` | Thinking tokens (reasoning models) |
| `cached_input_tokens` | Cache hits (OpenAI, xAI, Groq) |
| `cache_read_tokens` | Cache reads (Anthropic) |
| `cache_creation_tokens` | Cache writes (Anthropic) |
| `cost` | Calculated or provider-reported cost |
| `tag` | Attribution tag(s) |

### Cost Calculation

The proxy uses model-specific rates and multipliers:

```yaml
# Example: Anthropic Claude Sonnet
claude-sonnet-4-5-20250929:
  input_cost: 3.00           # $ per million tokens
  output_cost: 15.00
  cache_read_multiplier: 0.1  # Cache reads at 10% of input cost
  cache_write_multiplier: 1.25 # Cache writes at 125% of input cost
```

For a request with 10,000 input tokens (5,000 cached) and 2,000 output tokens:

```
Regular input:  5,000 × $3.00/1M = $0.015
Cached input:   5,000 × $3.00/1M × 0.1 = $0.0015
Output:         2,000 × $15.00/1M = $0.030
Total:          $0.0465
```

### Provider-Specific Handling

| Provider | Method |
|----------|--------|
| **OpenRouter** | Extracts actual cost from API response |
| **Gemini** | Tiered calculation (rates double above 200k tokens) |
| **Perplexity** | Per-request fees + token costs + citation costs |
| **Anthropic** | Separate cache read/write multipliers |
| **Others** | Standard token × rate calculation |

### Usage Dashboard

View costs in the Admin UI:

- **Summary** — Total requests, tokens, and cost for the period
- **Time series** — Cost trends over days/weeks/months
- **Breakdowns** — By tag, provider, model, or client
- **Recent requests** — Individual request logs with cost

Filter by date range, tag, provider, or model. Click any row to drill down.

## Tagging

Tags enable cost attribution across any dimension you need.

### Methods

| Method | Example | Priority |
|--------|---------|----------|
| Bearer token | `Authorization: Bearer alice` | 1 (highest) |
| Header | `X-Proxy-Tag: alice` | 2 |
| Model suffix | `model: claude@alice` | 3 |
| Default tag | Configured in Settings | 4 (lowest) |

### Multiple Tags

Assign multiple tags to track across dimensions:

```bash
curl -H "Authorization: Bearer alice,project-x,q1-2025" \
  -d '{"model": "claude-sonnet", "messages": [...]}' \
  http://localhost:11434/v1/chat/completions
```

Each tag gets its own entry in the usage breakdown, so you can analyze costs by user AND by project AND by quarter.

### SDK Usage

```python
from openai import OpenAI

# Tag via api_key parameter
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="alice,project-x"  # Your tags
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Admin UI

Access the Admin UI at port 8080 (password protected).

### Dashboard

Overview of all providers with:
- Connection status indicators
- Quick test buttons to verify API connectivity
- Model and alias counts

### Providers

Manage provider connections:
- View system providers (from configuration)
- Add custom Ollama instances
- Add OpenAI-compatible endpoints
- Add Anthropic-compatible endpoints
- Test connections
- Enable/disable providers

### Models

Browse and configure models:
- **System models** — Pre-configured, update with releases
- **Dynamic models** — Auto-discovered from Ollama instances
- **Custom models** — Create your own definitions

**Override system models** without editing files:
- Input/output cost
- Context length
- Capabilities
- Description

### Aliases

Create shortcuts for model names:
- `claude` → `claude-sonnet-4-5-20250929`
- `gpt` → `gpt-5.2`
- `fast` → `llama-3.1-8b-instant`

### Usage

Analytics dashboard with:
- Summary cards (requests, tokens, cost)
- Time series charts
- Breakdowns by tag, provider, model, client
- Recent request logs
- Export capabilities

### Settings

Configure:
- Default model for unknown requests
- Default tag for untagged requests
- Admin password
- DNS resolution for client hostnames
- Usage tracking toggle
- Data retention period

## Providers

### Built-in Providers

Set the corresponding environment variable to enable:

| Provider | Environment Variable |
|----------|---------------------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| Groq | `GROQ_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

### Adding Ollama Instances

Connect to local or remote Ollama servers:

1. Go to **Providers** → **Add Provider**
2. Select **Ollama Compatible**
3. Enter base URL (e.g., `http://192.168.1.100:11434`)
4. Models are automatically discovered

### Adding Custom Providers

Add any OpenAI-compatible or Anthropic-compatible endpoint:

1. Go to **Providers** → **Add Provider**
2. Select provider type
3. Enter base URL and API key environment variable
4. Add models manually or let the proxy discover them

## Installation

### Docker Compose (Recommended)

```yaml
services:
  llm-proxy:
    image: ghcr.io/benhumphry/ollama-llm-proxy:latest
    ports:
      - "11434:11434"  # API
      - "8080:8080"    # Admin UI
    volumes:
      - llm-proxy-data:/data
    env_file:
      - .env
    restart: unless-stopped

volumes:
  llm-proxy-data:
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 11434 | API server port |
| `ADMIN_PORT` | 8080 | Admin UI port |
| `ADMIN_PASSWORD` | (random) | Admin UI password |
| `ADMIN_ENABLED` | true | Enable Admin UI |
| `DATABASE_URL` | SQLite | PostgreSQL connection URL |

All API keys support `_FILE` suffix for Docker secrets.

### PostgreSQL (Production)

For multi-instance deployments:

```yaml
environment:
  - DATABASE_URL=postgresql://user:password@postgres:5432/llm_proxy
```

### Running Without Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python proxy.py
```

## API Reference

### Ollama-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tags` | GET | List available models |
| `/api/chat` | POST | Chat completion |
| `/api/generate` | POST | Text generation |
| `/api/show` | POST | Get model details |
| `/api/ps` | GET | List running models |

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |

## Examples

### Python with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="my-team"  # Tag for cost attribution
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### Open WebUI

```yaml
environment:
  - OLLAMA_BASE_URL=http://llm-proxy:11434
```

All models from all providers appear in the model selector.

### curl

```bash
# Ollama format
curl http://localhost:11434/api/chat \
  -d '{"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello"}]}'

# OpenAI format with tag
curl http://localhost:11434/v1/chat/completions \
  -H "Authorization: Bearer alice" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello"}]}'

# Streaming
curl http://localhost:11434/api/chat \
  -d '{"model": "gemini-2.5-pro", "stream": true, "messages": [...]}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                               │
│   (Open WebUI, Cursor, Python SDK, curl, any Ollama client) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Proxy (:11434)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Ollama API  │  │ OpenAI API  │  │ Cost Tracking       │  │
│  │ /api/chat   │  │ /v1/chat    │  │ Tokens, Tags, Rates │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│   Anthropic   │   │    OpenAI     │   │  Local Ollama     │
│   Claude      │   │    GPT        │   │  Llama, Mistral   │
└───────────────┘   └───────────────┘   └───────────────────┘
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│    Gemini     │   │  Perplexity   │   │   OpenRouter      │
│    Google     │   │  Sonar        │   │   400+ models     │
└───────────────┘   └───────────────┘   └───────────────────┘
```

**Admin UI** runs on port 8080, sharing the same database for configuration and usage data.

## Comparison with OpenRouter

| Feature | This Proxy | OpenRouter |
|---------|-----------|------------|
| Self-hosted | Yes | No |
| Local models | Yes (Ollama) | No |
| Multiple Ollama instances | Yes | No |
| Cost tracking granularity | Token-level with cache/reasoning | Basic |
| Tag-based attribution | Multi-tag, 4 methods | Limited |
| Custom providers | Full UI support | No |
| Model overrides | Per-model via UI | No |
| Data ownership | 100% yours | SaaS |
| Pricing | Your API costs only | Markup on API costs |

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Credits

Originally forked from [psenger/ollama-claude-proxy](https://github.com/psenger/ollama-claude-proxy).

## License

MIT License
