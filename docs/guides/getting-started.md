# Getting Started with LLM Relay

This guide walks you through setting up LLM Relay and making your first requests.

## Prerequisites

- Docker and Docker Compose
- At least one LLM provider API key (Anthropic, OpenAI, Google, etc.)

## Installation

### 1. Clone and Configure

```bash
git clone https://github.com/benhumphry/llm-relay.git
cd llm-relay
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required: At least one provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Optional: Admin UI password (random if not set)
ADMIN_PASSWORD=your-secure-password
```

### 2. Start the Services

```bash
docker compose up -d
```

This starts:
- **API Server**: http://localhost:11434 (Ollama and OpenAI compatible)
- **Admin UI**: http://localhost:8080

### 3. Verify It's Running

```bash
curl http://localhost:11434/api/tags
```

You should see a list of available models from your configured providers.

## Making Your First Request

### Using curl (Ollama API)

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using curl (OpenAI API)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="any-string"  # Used for tagging, not authentication
)

response = client.chat.completions.create(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Model Names

LLM Relay uses a simple naming convention:

```
provider/model-name
```

Examples:
- `anthropic/claude-sonnet-4-5`
- `openai/gpt-4o`
- `gemini/gemini-2.5-pro`
- `groq/llama-3.3-70b-versatile`

You can also use just the model name without the provider prefix - LLM Relay will search across providers to find a match.

## Creating an Alias

Aliases let you create friendly names for models. In the Admin UI:

1. Go to **Aliases** in the sidebar
2. Click **New Alias**
3. Enter:
   - **Name**: `claude` (your short name)
   - **Target**: `anthropic/claude-sonnet-4-5` (the full model)
4. Click **Save**

Now you can use `claude` instead of `anthropic/claude-sonnet-4-5`:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "claude", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Tagging Requests

Tags let you attribute costs to users, projects, or teams. Add tags via:

### Bearer Token

```bash
curl -H "Authorization: Bearer alice,project-x" \
  http://localhost:11434/api/chat \
  -d '{"model": "claude", "messages": [...]}'
```

### Header

```bash
curl -H "X-Proxy-Tag: alice,project-x" \
  http://localhost:11434/api/chat \
  -d '{"model": "claude", "messages": [...]}'
```

### Model Suffix

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "claude@alice,project-x", "messages": [...]}'
```

View tagged usage in the Admin UI under **Dashboard** or **Usage**.

## Connecting Clients

### Open WebUI

1. Go to Settings > Connections
2. Set Ollama URL to `http://llm-relay:11434` (or your host)
3. Models will appear automatically

### Cursor / Continue

Configure as an OpenAI-compatible endpoint:
- Base URL: `http://localhost:11434/v1`
- API Key: Any string (used for tagging)

### Any Ollama-Compatible Client

Point the Ollama URL to your LLM Relay instance.

## Enabling Smart Features

LLM Relay's smart features require additional services:

### ChromaDB (for Cache, RAG, Web, Model Intelligence)

Add to your `docker-compose.yml`:

```yaml
services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma

volumes:
  chroma-data:
```

Add to `.env`:

```bash
CHROMA_URL=http://chroma:8000
```

### SearXNG (for Web enrichment)

Add to your `docker-compose.yml`:

```yaml
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng
```

Add to `.env`:

```bash
SEARXNG_URL=http://searxng:8080
```

See [INSTALLATION.md](../../INSTALLATION.md) for complete production setup.

## Next Steps

- [Smart Aliases](smart-aliases.md) - Unified routing, caching, RAG, web, and memory features
