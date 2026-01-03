<p align="center">
  <img src="logo.png" alt="LLM Proxy" width="128" height="128">
</p>

<h1 align="center">Multi-Provider LLM Proxy</h1>

<p align="center">
A self-hosted proxy that presents multiple LLM providers via both Ollama and OpenAI-compatible API interfaces. This allows any Ollama or OpenAI-compatible application to use models from multiple providers seamlessly.
</p>

**Built-in providers:** Anthropic Claude, OpenAI GPT, Google Gemini, Perplexity, Groq, DeepSeek, Mistral, xAI Grok, OpenRouter

**v2.1 Features:** Usage tracking & stats, cost estimation, tag-based attribution

## Features

- **10+ providers supported** - Anthropic, OpenAI, Gemini, Perplexity, Groq, DeepSeek, Mistral, xAI, OpenRouter, and local Ollama instances
- **Web Admin UI** - Manage providers, models, and aliases through a polished web interface
- **Usage tracking** - Monitor requests, tokens, and estimated costs with charts and breakdowns
- **Tag-based attribution** - Track usage by user/project with `X-Proxy-Tag` header or `model@tag` syntax
- **Local Ollama support** - Connect to local or remote Ollama instances with automatic model discovery
- **Custom providers** - Add OpenAI-compatible or Anthropic-compatible providers via the UI
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
- Enable/disable specific models

### Aliases
- Create shortcuts for model names
- Manage system and custom aliases
- Example: `claude` → `claude-sonnet-4-5-20250929`

### Usage
- **Request tracking** - Monitor all proxy requests with tokens, response time, and cost
- **Cost estimation** - Automatic cost calculation based on model pricing
- **Breakdowns** - View usage by tag, provider, or model
- **Time series charts** - Visualize usage trends over time
- **Tag-based attribution** - Track usage per user/project/team
- **Recent requests** - View detailed request logs with client IP and hostname

### Settings
- Set default model for unknown requests
- Change admin password
- Configure proxy behavior
- Enable/disable usage tracking
- Configure DNS resolution for client hostnames

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

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | No | | Anthropic API key |
| `OPENAI_API_KEY` | No | | OpenAI API key |
| `GOOGLE_API_KEY` | No | | Google API key |
| `PERPLEXITY_API_KEY` | | | Perplexity API key |
| `GROQ_API_KEY` | | | Groq API key |
| `DEEPSEEK_API_KEY` | | | DeepSeek API key |
| `MISTRAL_API_KEY` | | | Mistral API key |
| `XAI_API_KEY` | | | xAI API key |
| `OPENROUTER_API_KEY` | | | OpenRouter API key |
| `PORT` | | 11434 | API server port |
| `ADMIN_PORT` | | 8080 | Admin UI port |
| `ADMIN_PASSWORD` | | (random) | Admin UI password |
| `ADMIN_ENABLED` | | true | Set to "false" to disable Admin UI |

All API keys support `_FILE` suffix for Docker secrets (e.g., `ANTHROPIC_API_KEY_FILE`).

**First Run:** If `ADMIN_PASSWORD` is not set, a random password is generated and logged:
```
============================================================
ADMIN PASSWORD NOT SET - Generated random password:
  abc123xyz789...
============================================================
```

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

## Usage Tracking

The proxy tracks all requests with token counts, response times, and estimated costs. Usage can be attributed to different users or projects using tags.

### Tagging Requests

There are three ways to tag requests (in priority order):

1. **HTTP Header** - Set `X-Proxy-Tag` header:
   ```bash
   curl -X POST http://localhost:11434/api/chat \
     -H "X-Proxy-Tag: alice" \
     -d '{"model": "claude-sonnet", "messages": [...]}'
   ```

2. **Model suffix** - Append `@tag` to the model name:
   ```bash
   curl -X POST http://localhost:11434/api/chat \
     -d '{"model": "claude-sonnet@alice", "messages": [...]}'
   ```

3. **Default tag** - Requests without a tag use the default (configurable in Settings)

### Cost Tracking

Model costs are configured per-million tokens and automatically calculated for each request. Built-in costs are provided for all major providers. Custom models can have costs set when created.

View usage breakdowns, costs, and trends in the **Usage** page of the Admin UI.

## Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="claude-sonnet",  # or "gpt-4o", "gemini", etc.
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Open WebUI

Configure the Ollama connection:
```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama-llm-proxy:11434
```

All models from all providers appear in the model selector.

### curl

```bash
# Ollama format
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# OpenAI format
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Supported Models

### Anthropic Claude
`claude-opus`, `claude-sonnet`, `claude-haiku` and version-specific variants

### OpenAI GPT
`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini`, `o3`, `o3-mini` and more

### Google Gemini
`gemini`, `gemini-pro`, `gemini-flash` and version variants

### Perplexity
`sonar`, `sonar-pro`, `sonar-reasoning`

### Groq
`llama`, `llama-70b`, `llama-8b`, `qwen`, `compound`

### DeepSeek
`deepseek`, `deepseek-v3`, `deepseek-r1`

### Mistral
`mistral`, `mistral-large`, `mistral-small`, `codestral`

### xAI Grok
`grok`, `grok-4`, `grok-3`, `grok-vision`

### OpenRouter
`or-claude`, `or-gpt`, `or-gemini`, `or-llama` (400+ models available)

### Local Ollama
Any models installed on connected Ollama instances are automatically discovered.

Use the Admin UI to see all available models and their aliases.

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
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python proxy.py
```

## Credits

This project is forked from [psenger/ollama-claude-proxy](https://github.com/psenger/ollama-claude-proxy).

## License

MIT License
