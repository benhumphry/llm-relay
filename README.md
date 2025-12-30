# Multi-Provider LLM Proxy

A self-hosted proxy that presents multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini, Perplexity) via both Ollama and OpenAI-compatible API interfaces. This allows any Ollama or OpenAI-compatible application to use models from multiple providers seamlessly.

## Features

- **Multi-provider support** - Anthropic Claude, OpenAI GPT, Google Gemini, Perplexity (and easily extensible)
- **Full Ollama API compatibility** - Works with any application that supports Ollama
- **OpenAI API compatibility** - Also exposes `/v1/*` endpoints for OpenAI SDK compatibility
- **Vision support** - Pass images via base64 encoding (Ollama and OpenAI formats)
- **Streaming responses** - Real-time streaming (NDJSON for Ollama, SSE for OpenAI)
- **Docker Swarm ready** - Production-ready containerisation
- **Extensible architecture** - Add new providers with ~50 lines of code

## Supported Providers & Models

### Anthropic Claude

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `claude-opus`, `claude-4.5-opus` | claude-opus-4-5-20251101 | Complex analysis, OCR, detailed reasoning |
| `claude-sonnet`, `claude-4.5-sonnet` | claude-sonnet-4-5-20250929 | Balanced performance |
| `claude-haiku`, `claude-4.5-haiku` | claude-haiku-4-5-20251001 | Fast tasks, tagging, classification |
| `claude-4-opus`, `claude-4-sonnet` | claude-opus-4-*, claude-sonnet-4-* | Previous generation |
| `claude-3.5-sonnet`, `claude-3.5-haiku` | claude-3-5-* | Legacy models |

### OpenAI GPT

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `gpt-4o`, `openai-gpt-4o` | gpt-4o | Most capable multimodal |
| `gpt-4o-mini` | gpt-4o-mini | Fast and affordable |
| `gpt-4-turbo`, `gpt-4` | gpt-4-turbo, gpt-4 | High capability |
| `gpt-3.5-turbo`, `chatgpt` | gpt-3.5-turbo | Fast and cost-effective |
| `o1`, `o1-mini`, `o1-pro` | o1, o1-mini, o1-pro | Advanced reasoning |

### Google Gemini

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `gemini`, `gemini-flash` | gemini-2.5-flash | Fast and versatile |
| `gemini-pro` | gemini-2.5-pro | Most capable Gemini |
| `gemini-2.0-flash` | gemini-2.0-flash | Previous generation |
| `gemini-1.5-pro`, `gemini-1.5-flash` | gemini-1.5-* | Legacy models |

### Perplexity

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `perplexity`, `pplx`, `sonar` | sonar | Search-augmented |
| `sonar-pro` | sonar-pro | Advanced search |
| `sonar-reasoning` | sonar-reasoning | Search with chain-of-thought |

## Quick Start

### Using Docker Compose

1. Set your API keys (at least one required):
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."
   export GOOGLE_API_KEY="AIza..."
   export PERPLEXITY_API_KEY="pplx-..."
   ```

2. Run with Docker Compose:
   ```bash
   docker compose up -d
   ```

3. Test:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Docker Swarm Deployment

For Docker Swarm, use `docker-compose.swarm.yml` which includes Docker secrets support, resource limits, and Traefik labels.

1. Create Docker secrets:
   ```bash
   echo "sk-ant-..." | docker secret create anthropic_api_key -
   echo "sk-..." | docker secret create openai_api_key -
   ```

2. Deploy:
   ```bash
   docker stack deploy -c docker-compose.swarm.yml llm-proxy
   ```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | At least one | Anthropic API key for Claude models |
| `OPENAI_API_KEY` | provider | OpenAI API key for GPT models |
| `GOOGLE_API_KEY` | required | Google API key for Gemini models |
| `PERPLEXITY_API_KEY` | | Perplexity API key for Sonar models |
| `PORT` | No | Server port (default: 11434) |
| `HOST` | No | Bind address (default: 0.0.0.0) |
| `DEBUG` | No | Enable debug logging (default: false) |

All API keys also support `_FILE` suffix for Docker secrets (e.g., `ANTHROPIC_API_KEY_FILE`).

## API Endpoints

### Ollama API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/tags` | GET | List available models from all providers |
| `/api/show` | POST | Get model details |
| `/api/chat` | POST | Chat completion (main endpoint) |
| `/api/generate` | POST | Text generation |
| `/api/pull` | POST | Returns success (no download needed) |
| `/api/version` | GET | Version information |

### OpenAI-Compatible API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |
| `/v1/completions` | POST | Text completions |

## Usage Examples

### Ollama API

```bash
# Use Claude
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "claude-sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Use GPT-4o
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'

# Use Gemini
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "gemini", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

# Use any provider's model
response = client.chat.completions.create(
    model="claude-sonnet",  # or "gpt-4o", "gemini", etc.
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Model Naming

Models can be referenced in multiple ways:

1. **Simple alias**: `claude-sonnet`, `gpt-4o`, `gemini`
2. **Provider-prefixed**: `anthropic-claude-sonnet`, `openai-gpt-4o`, `gemini-gemini-2.5-flash`
3. **Full model ID**: `claude-sonnet-4-5-20250929`, `gpt-4o`

## Adding New Providers

The proxy uses an extensible provider architecture. To add a new provider:

1. Create `providers/newprovider_provider.py`:

```python
from .base import OpenAICompatibleProvider, ModelInfo

class NewProvider(OpenAICompatibleProvider):
    name = "newprovider"
    base_url = "https://api.newprovider.com/v1"
    api_key_env = "NEWPROVIDER_API_KEY"
    
    models = {
        "model-name": ModelInfo(
            family="model-family",
            description="Model description",
            context_length=128000,
            capabilities=["vision", "coding"],
        ),
    }
    
    aliases = {
        "newprovider": "model-name",
    }
```

2. Register in `providers/__init__.py`:

```python
from .newprovider_provider import NewProvider
registry.register(NewProvider())
```

## Integration Examples

### Paperless-GPT

```yaml
environment:
  - OLLAMA_HOST=http://llm-proxy:11434
  - PAPERLESS_GPT_OCR_MODEL=claude-opus
  - PAPERLESS_GPT_TAGGING_MODEL=claude-haiku
```

### Open WebUI

```yaml
environment:
  - OLLAMA_BASE_URL=http://llm-proxy:11434
```

## Limitations

1. **No embeddings** - Use a dedicated embedding service
2. **API costs** - Monitor usage at each provider's console
3. **Rate limits** - Subject to each provider's rate limits

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

## Licence

MIT License
