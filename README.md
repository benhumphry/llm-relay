# Multi-Provider LLM Proxy

A self-hosted proxy that presents multiple LLM providers via both Ollama and OpenAI-compatible API interfaces. This allows any Ollama or OpenAI-compatible application to use models from multiple providers seamlessly.

**Supported providers:** Anthropic Claude, OpenAI GPT, Google Gemini, Perplexity, Groq, DeepSeek, Mistral, xAI Grok, OpenRouter (and easily extensible via YAML config).

## Features

- **9 providers built-in** - Anthropic, OpenAI, Gemini, Perplexity, Groq, DeepSeek, Mistral, xAI, OpenRouter
- **Full Ollama API compatibility** - Works with any application that supports Ollama (including Open WebUI)
- **OpenAI API compatibility** - Also exposes `/v1/*` endpoints for OpenAI SDK compatibility
- **Reasoning model support** - Automatic parameter handling for GPT-5, o1, o3 models
- **Vision support** - Pass images via base64 encoding (Ollama and OpenAI formats)
- **Streaming responses** - Real-time streaming (NDJSON for Ollama, SSE for OpenAI)
- **Docker Swarm ready** - Production-ready containerisation
- **Config-based models** - Add/customize models via YAML without code changes
- **Extensible architecture** - Add new providers with just YAML config

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
| `openai`, `gpt-5.2` | gpt-5.2 | Latest flagship (Dec 2025) |
| `gpt-5.2-pro` | gpt-5.2-pro | Research-grade, slower but smarter* |
| `gpt-5.1`, `gpt-5.1-codex` | gpt-5.1, gpt-5.1-codex | Previous flagship, codex variants |
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | gpt-5-* | Original GPT-5 family |
| `gpt-5-pro` | gpt-5-pro | Research-grade GPT-5* |
| `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | gpt-4.1-* | 1M context, great for coding |
| `gpt-4o`, `gpt-4o-mini` | gpt-4o, gpt-4o-mini | Multimodal with vision |
| `o4-mini` | o4-mini | Latest compact reasoning* |
| `o3`, `o3-pro`, `o3-mini` | o3, o3-pro, o3-mini | Advanced reasoning* |
| `o1`, `o1-pro`, `o1-mini` | o1, o1-pro, o1-mini | Reasoning models* |

*\*Pro/reasoning models don't support temperature, top_p, or system prompts. The proxy automatically handles this.*

### Google Gemini

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `gemini`, `gemini-flash` | gemini-3-flash | Latest fast and versatile |
| `gemini-pro` | gemini-3-pro | Most capable Gemini |
| `gemini-2.5-flash`, `gemini-2.5-pro` | gemini-2.5-* | Previous generation |
| `gemini-2.0-flash` | gemini-2.0-flash | Older generation |
| `gemini-1.5-pro`, `gemini-1.5-flash` | gemini-1.5-* | Legacy models |

### Perplexity

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `perplexity`, `pplx`, `sonar` | sonar | Search-augmented |
| `sonar-pro` | sonar-pro | Advanced search |
| `sonar-reasoning` | sonar-reasoning | Search with chain-of-thought |

### Groq (Ultra-fast inference)

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `groq`, `llama`, `llama-70b` | llama-3.3-70b-versatile | Best open-source model |
| `llama-4`, `llama-4-maverick` | meta-llama/llama-4-maverick-17b-128e-instruct | Latest Llama |
| `llama-8b` | llama-3.1-8b-instant | Ultra-fast small model |
| `qwen`, `qwen-32b` | qwen/qwen3-32b | Strong multilingual |
| `kimi`, `moonshot` | moonshotai/kimi-k2-instruct-0905 | Extended 262K context |
| `compound` | groq/compound | Agentic with tools |

### DeepSeek (Best value)

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `deepseek`, `deepseek-v3` | deepseek-chat | 671B MoE, excellent value |
| `deepseek-r1`, `r1` | deepseek-reasoner | Reasoning model (rivals o1)* |

*\*DeepSeek-R1 doesn't support temperature/top_p. The proxy handles this automatically.*

### Mistral

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `mistral`, `mistral-large` | mistral-large-latest | Top-tier reasoning |
| `mistral-medium` | mistral-medium-latest | Balanced performance |
| `mistral-small` | mistral-small-latest | Fast and efficient |
| `codestral`, `code` | codestral-latest | Code specialist, 80+ langs |
| `magistral` | magistral-medium-latest | Advanced reasoning |
| `pixtral` | pixtral-large-latest | Vision specialist |

### xAI Grok

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `grok`, `xai` | grok-4-1-fast | Latest Grok, 2M context, best tool-calling |
| `grok-4` | grok-4 | Frontier reasoning model |
| `grok-3`, `grok-3-mini` | grok-3-beta, grok-3-mini-beta | Previous generation |
| `grok-2-vision` | grok-2-vision-1212 | Vision capabilities |
| `grok-code` | grok-code-fast-1 | Optimized for agentic coding |

### OpenRouter (400+ models)

| Model Name | Model ID | Use Case |
|------------|----------|----------|
| `or-claude` | anthropic/claude-sonnet-4 | Claude via OpenRouter |
| `or-gpt` | openai/gpt-4o | GPT-4o via OpenRouter |
| `or-gemini` | google/gemini-2.5-flash | Gemini via OpenRouter |
| `or-llama` | meta-llama/llama-3.3-70b-instruct | Llama via OpenRouter |
| `or-deepseek` | deepseek/deepseek-chat | DeepSeek via OpenRouter |
| `llama-405b` | meta-llama/llama-3.1-405b-instruct | Largest open model |

*OpenRouter provides access to 400+ models from all major providers via a single API.*

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
| `GROQ_API_KEY` | | Groq API key for fast inference |
| `DEEPSEEK_API_KEY` | | DeepSeek API key |
| `MISTRAL_API_KEY` | | Mistral API key |
| `XAI_API_KEY` | | xAI API key for Grok models |
| `OPENROUTER_API_KEY` | | OpenRouter API key (access 400+ models) |
| `PORT` | No | Server port (default: 11434) |
| `HOST` | No | Bind address (default: 0.0.0.0) |
| `DEBUG` | No | Enable debug logging (default: false) |
| `CONFIG_DIR` | No | Custom config directory path |

All API keys also support `_FILE` suffix for Docker secrets (e.g., `ANTHROPIC_API_KEY_FILE`).

## API Endpoints

### Ollama API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/tags` | GET | List available models from all providers |
| `/api/ps` | GET | List running models (all models shown as available) |
| `/api/show` | POST | Get model details |
| `/api/chat` | POST | Chat completion (main endpoint) |
| `/api/generate` | POST | Text generation |
| `/api/pull` | POST | Returns success (no download needed) |
| `/api/version` | GET | Version information |
| `/api/embeddings` | POST | Not supported (returns 501) |

### OpenAI-Compatible API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |
| `/api/chat/completions` | POST | Alias for `/v1/chat/completions` (Open WebUI compatibility) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Not supported (returns 501) |

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

## Customizing Models

All model definitions are stored in YAML configuration files, making it easy to add new models, customize aliases, or add entirely new providers without modifying code.

### Config Directory Structure

```
config/
├── providers.yml          # Provider definitions
└── models/
    ├── anthropic.yml      # Anthropic models and aliases
    ├── openai.yml         # OpenAI models and aliases
    ├── gemini.yml         # Gemini models and aliases
    └── perplexity.yml     # Perplexity models and aliases
```

### Adding a New Model

Edit the provider's model config file. For example, to add a new OpenAI model:

```yaml
# config/models/openai.yml
models:
  gpt-6:  # New model!
    family: gpt-6
    description: "GPT-6 - Next generation model"
    context_length: 256000
    capabilities: [vision, reasoning, coding]
    # For reasoning models that don't support temperature/top_p:
    unsupported_params: *reasoning_unsupported
    supports_system_prompt: false
    use_max_completion_tokens: true

aliases:
  gpt6: gpt-6
```

### Adding a New Provider (No Code Required!)

Any OpenAI-compatible provider can be added with just 2 YAML files - no Python code needed.

1. Add the provider to `config/providers.yml`:

```yaml
providers:
  groq:
    type: openai-compatible
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
```

2. Create `config/models/groq.yml`:

```yaml
models:
  llama-3.3-70b-versatile:
    family: llama-3.3
    description: "Llama 3.3 70B on Groq"
    context_length: 128000
    capabilities: [coding, analysis]

aliases:
  groq: llama-3.3-70b-versatile
  llama: llama-3.3-70b-versatile
```

3. Set `GROQ_API_KEY` environment variable and restart. That's it!

### Custom Config with Docker

The proxy includes default config files in the image. To customize, mount your own config directory:

**Option 1: Override entire config directory**

```yaml
# docker-compose.yml
services:
  ollama-llm-proxy:
    image: ollama-llm-proxy
    volumes:
      - ./my-config:/app/config
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

Your `my-config/` directory should contain:
```
my-config/
├── providers.yml
└── models/
    ├── anthropic.yml
    ├── openai.yml
    └── ...
```

**Option 2: Use CONFIG_DIR environment variable**

```yaml
services:
  ollama-llm-proxy:
    image: ollama-llm-proxy
    volumes:
      - ./custom-config:/custom-config
    environment:
      - CONFIG_DIR=/custom-config
```

**Option 3: Extend default config (Docker Swarm)**

Copy the default config, modify it, and mount:

```bash
# Copy defaults from running container
docker cp ollama-llm-proxy:/app/config ./my-config

# Edit as needed
vim my-config/models/openai.yml

# Redeploy with volume mount
docker stack deploy -c docker-compose.swarm.yml llm-proxy
```

**Docker Compose with custom config:**

```yaml
version: '3.8'
services:
  ollama-llm-proxy:
    image: ollama-llm-proxy:latest
    ports:
      - "11434:11434"
    volumes:
      - ./config:/app/config:ro  # Read-only mount
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}  # Custom provider
    restart: unless-stopped
```

### Model Config Options

| Field | Required | Description |
|-------|----------|-------------|
| `family` | Yes | Model family name |
| `description` | Yes | Human-readable description |
| `context_length` | Yes | Max context window size |
| `capabilities` | No | List: vision, coding, reasoning, fast, etc. |
| `unsupported_params` | No | Parameters to filter (for reasoning models) |
| `supports_system_prompt` | No | Default: true |
| `use_max_completion_tokens` | No | Use max_completion_tokens instead of max_tokens |

## Integration Examples

### Paperless-GPT

```yaml
environment:
  - OLLAMA_HOST=http://llm-proxy:11434
  - PAPERLESS_GPT_OCR_MODEL=claude-opus
  - PAPERLESS_GPT_TAGGING_MODEL=claude-haiku
```

### Open WebUI

Open WebUI works seamlessly with this proxy. Configure the Ollama connection:

```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama-llm-proxy:11434
```

All models from all configured providers will appear in Open WebUI's model selector. The proxy handles:
- Model listing with proper `:latest` tags
- Running models status via `/api/ps`
- Automatic parameter filtering for reasoning models (GPT-5, o1, o3)

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
