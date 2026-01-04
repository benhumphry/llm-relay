<p align="center">
  <img src="logo.png" alt="LLM Proxy" width="128" height="128">
</p>

<h1 align="center">Multi-Provider LLM Proxy</h1>

<p align="center">
A self-hosted proxy that presents multiple LLM providers via both Ollama and OpenAI-compatible API interfaces. This allows any Ollama or OpenAI-compatible application to use models from multiple providers seamlessly.
</p>

**Built-in providers:** Anthropic Claude, OpenAI GPT, Google Gemini, Perplexity, Groq, DeepSeek, Mistral, xAI Grok, OpenRouter

**v2.2 Features:** Dynamic pricing from OpenRouter, model cost overrides, enhanced usage filtering

## Features

- **10+ providers supported** - Anthropic, OpenAI, Gemini, Perplexity, Groq, DeepSeek, Mistral, xAI, OpenRouter, and local Ollama instances
- **Web Admin UI** - Manage providers, models, and aliases through a polished web interface
- **Usage tracking** - Monitor requests, tokens, and costs with charts and breakdowns
- **Dynamic pricing** - OpenRouter reports actual costs per request; other providers use configured rates
- **Tag-based attribution** - Track usage by user/project via headers, bearer tokens, or model suffix
- **Local Ollama support** - Connect to local or remote Ollama instances with automatic model discovery
- **Custom providers** - Add OpenAI-compatible or Anthropic-compatible providers via the UI
- **Model overrides** - Customize pricing, capabilities, and descriptions for any model
- **Full Ollama API compatibility** - Works with any application that supports Ollama (including Open WebUI)
- **OpenAI API compatibility** - Also exposes `/v1/*` endpoints for OpenAI SDK compatibility
- **Reasoning model support** - Automatic parameter handling for o1, o3, DeepSeek-R1 models
- **Vision support** - Pass images via base64 encoding (Ollama and OpenAI formats)
- **Streaming responses** - Real-time streaming (NDJSON for Ollama, SSE for OpenAI)
- **Docker ready** - Simple deployment with persistent configuration

## Quick Start

### Docker Compose (Recommended)

1. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. Run:
   ```bash
   docker compose up -d
   ```

3. Access:
   - **API Server:** http://localhost:11434
   - **Admin UI:** http://localhost:8080

That's it! The proxy is ready to use. Configure additional providers and models through the Admin UI.

### Test the API

```bash
# List available models
curl http://localhost:11434/api/tags

# Chat with Claude
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Chat with GPT-4o
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Installation

### Docker Compose

The recommended way to run the proxy:

```yaml
# docker-compose.yml
services:
  llm-proxy:
    image: ghcr.io/your-username/ollama-llm-proxy:latest
    ports:
      - "11434:11434"  # API
      - "8080:8080"    # Admin UI
    volumes:
      - llm-proxy-data:/data
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      # Add other API keys as needed
    restart: unless-stopped

volumes:
  llm-proxy-data:
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | No | | Anthropic API key |
| `OPENAI_API_KEY` | No | | OpenAI API key |
| `GOOGLE_API_KEY` | No | | Google API key |
| `PERPLEXITY_API_KEY` | No | | Perplexity API key |
| `GROQ_API_KEY` | No | | Groq API key |
| `DEEPSEEK_API_KEY` | No | | DeepSeek API key |
| `MISTRAL_API_KEY` | No | | Mistral API key |
| `XAI_API_KEY` | No | | xAI API key |
| `OPENROUTER_API_KEY` | No | | OpenRouter API key |
| `PORT` | No | 11434 | API server port |
| `ADMIN_PORT` | No | 8080 | Admin UI port |
| `ADMIN_PASSWORD` | No | (random) | Admin UI password |
| `ADMIN_ENABLED` | No | true | Set to "false" to disable Admin UI |

All API keys support `_FILE` suffix for Docker secrets (e.g., `ANTHROPIC_API_KEY_FILE`).

**First Run:** If `ADMIN_PASSWORD` is not set, a random password is generated and logged:
```
============================================================
ADMIN PASSWORD NOT SET - Generated random password:
  abc123xyz789...
============================================================
```

### Running Without Docker

```bash
# Clone the repository
git clone https://github.com/your-username/ollama-llm-proxy.git
cd ollama-llm-proxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Run the proxy
python proxy.py
```

## Usage Tracking & Tags

The proxy tracks all requests with token counts, response times, and costs. Usage can be attributed to different users or projects using **tags**.

### Tagging Requests

There are four ways to tag requests (in priority order):

#### 1. Bearer Token (Best for API clients)

Use the Authorization header with your tag as the bearer token:

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Authorization: Bearer alice" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

This works seamlessly with OpenAI SDK clients:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="alice"  # Your tag becomes the "API key"
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### 2. X-Proxy-Tag Header

Set the `X-Proxy-Tag` header explicitly:

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "X-Proxy-Tag: alice" \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

#### 3. Model Suffix

Append `@tag` to the model name:

```bash
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet@alice", "messages": [{"role": "user", "content": "Hello!"}]}'
```

#### 4. Default Tag

Requests without any tag use the default tag (configurable in Settings, defaults to "default").

### Multiple Tags

You can assign multiple tags to a single request using comma-separated values:

```bash
# Via header
curl -X POST http://localhost:11434/api/chat \
  -H "X-Proxy-Tag: alice,project-x,testing" \
  -d '{"model": "claude-sonnet", "messages": [...]}'

# Via bearer token
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Authorization: Bearer alice,project-x" \
  -d '{"model": "claude-sonnet", "messages": [...]}'

# Via model suffix
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet@alice,project-x", "messages": [...]}'
```

Multiple tags allow you to track usage across different dimensions simultaneously (e.g., by user AND by project).

### Cost Tracking

The proxy tracks costs in two ways:

1. **Dynamic pricing (OpenRouter)** - OpenRouter returns actual cost per request in its API response. This is shown as the exact cost in the Recent Requests view.

2. **Estimated pricing (all other providers)** - Costs are calculated based on token counts and configured rates (cost per million tokens). Rates are pre-configured for all major providers and can be overridden via the Admin UI.

View usage breakdowns, costs, and trends in the **Usage** page of the Admin UI.

## Admin UI

The Admin UI at port 8080 provides complete management of the proxy:

### Dashboard
- Overview of all providers and their status
- Quick test buttons to verify API connectivity
- Model and alias counts per provider

### Providers
- View all configured providers (system and custom)
- Add custom providers (Ollama, OpenAI-compatible, Anthropic-compatible)
- Test provider connections
- Enable/disable providers

### Models
- Browse all available models across providers
- **System models** - Pre-configured, update automatically
- **Dynamic models** - Discovered from Ollama instances
- **Custom models** - Add your own model definitions
- **Override system models** - Customize pricing, capabilities, context length
- Enable/disable specific models

### Aliases
- Create shortcuts for model names
- Manage system and custom aliases
- Example: `claude` → `claude-sonnet-4-5-20250929`

### Usage
- **Summary cards** - Total requests, tokens, and cost
- **Time series charts** - Visualize usage trends over time
- **Breakdowns** - View usage by tag, provider, model, or client
- **Filtering** - Filter by tag, provider, model, or date range
- **Drill-down** - Click any breakdown row to filter further
- **Recent requests** - Detailed request logs with cost, tokens, and timing
- **Per-request cost** - OpenRouter shows actual cost; others show "-"

### Settings
- Set default model for unknown requests
- Change admin password
- Configure default tag for untagged requests
- Enable/disable usage tracking
- Configure DNS resolution for client hostnames
- Set data retention period

## Adding Providers

### Via Admin UI (Recommended)

1. Go to **Providers** page
2. Click **Add Provider**
3. Select type:
   - **Ollama Compatible** - Local or remote Ollama instance
   - **OpenAI Compatible** - Any OpenAI-compatible API
   - **Anthropic Compatible** - Any Anthropic-compatible API
4. Enter the base URL and API key environment variable
5. Click **Add Provider**

### Adding a Local Ollama Instance

1. Go to **Providers** → **Add Provider**
2. Select **Ollama Compatible**
3. Enter:
   - **Provider ID:** `my-ollama` (any unique name)
   - **Base URL:** `http://192.168.1.100:11434` (your Ollama server)
4. Click **Add Provider**

Models from your Ollama instance will be automatically discovered and appear in the models list.

## Architecture

The proxy runs two servers on separate ports:

| Server | Default Port | Purpose |
|--------|-------------|---------|
| **API Server** | 11434 | Ollama and OpenAI compatible endpoints |
| **Admin UI** | 8080 | Web interface (password protected) |

Data is persisted in a Docker volume at `/data`, ensuring your custom providers, models, and settings survive container restarts.

## API Endpoints

### Ollama API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/tags` | GET | List available models |
| `/api/chat` | POST | Chat completion |
| `/api/generate` | POST | Text generation |
| `/api/show` | POST | Get model details |
| `/api/ps` | GET | List running models |

### OpenAI API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |

## Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Basic usage
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

```python
# With usage tracking tag
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="alice"  # Tag for usage tracking
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Open WebUI

Configure the Ollama connection in your Open WebUI setup:

```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama-llm-proxy:11434
```

All models from all providers appear in the model selector.

To track usage per user, configure Open WebUI to pass user info in headers (if supported), or use different proxy instances per user.

### curl

```bash
# Ollama format
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# OpenAI format
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'

# With tag via bearer token
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Authorization: Bearer alice" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# With multiple tags
curl -X POST http://localhost:11434/api/chat \
  -H "X-Proxy-Tag: alice,project-alpha" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Streaming

```bash
# Ollama streaming (NDJSON)
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "stream": true, "messages": [{"role": "user", "content": "Write a poem"}]}'

# OpenAI streaming (SSE)
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "stream": true, "messages": [{"role": "user", "content": "Write a poem"}]}'
```

## Supported Models

### Anthropic Claude
`claude-opus`, `claude-sonnet`, `claude-haiku` and version-specific variants

### OpenAI GPT
`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini`, `o3`, `o3-mini` and more

### Google Gemini
`gemini`, `gemini-pro`, `gemini-flash`, `gemini-2.5-pro`, `gemini-2.5-flash` and variants

### Perplexity
`sonar`, `sonar-pro` (online search-augmented models)

### Groq
`llama`, `llama-70b`, `llama-8b`, `qwen`, `compound` (fast inference)

### DeepSeek
`deepseek`, `deepseek-v3`, `deepseek-r1` (reasoning model)

### Mistral
`mistral`, `mistral-large`, `mistral-small`, `codestral`, `ministral`

### xAI Grok
`grok`, `grok-3`, `grok-2`, `grok-vision`

### OpenRouter
Access 400+ models through a single API. OpenRouter provides **dynamic pricing** - actual costs are returned per request and tracked in the usage dashboard.

### Local Ollama
Any models installed on connected Ollama instances are automatically discovered.

Use the Admin UI to see all available models and their aliases.

## Model Overrides

You can override properties of any system model without editing configuration files:

1. Go to **Models** page
2. Find the model and click the **Override** button
3. Customize:
   - **Input/Output Cost** - Override pricing (per million tokens)
   - **Context Length** - Override context window size
   - **Capabilities** - Override capability tags
   - **Description** - Override model description

Overrides are stored in the database and persist across updates. Click **Clear Override** to restore original values.

## Docker Swarm

For production deployments with Docker Swarm:

```bash
# Create secrets
echo "sk-ant-..." | docker secret create anthropic_api_key -
echo "sk-..." | docker secret create openai_api_key -

# Deploy
docker stack deploy -c docker-compose.swarm.yml llm-proxy
```

## Development

```bash
# Clone and setup
git clone https://github.com/your-username/ollama-llm-proxy.git
cd ollama-llm-proxy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Run
python proxy.py
```

## Changelog

### v2.2.1
- **OpenRouter dynamic pricing** - Actual costs from API responses
- **Per-request cost tracking** - Cost column in Recent Requests
- **Model overrides** - Customize system model properties via UI
- **Updated provider pricing** - Current rates for Groq, Mistral, xAI, Perplexity, Gemini
- **Database migrations** - Automatic schema updates for existing installations
- **Bug fixes** - Favicon loading, SQLite date handling in filtered queries

### v2.1.0
- Usage tracking and statistics dashboard
- Tag-based attribution for requests
- Cost estimation based on model pricing
- Time series charts and breakdowns
- Client hostname resolution

### v2.0.0
- Complete rewrite with Admin UI
- Support for 10+ providers
- Custom providers via UI
- Model and alias management

## Credits

This project is forked from [psenger/ollama-claude-proxy](https://github.com/psenger/ollama-claude-proxy).

## License

MIT License
