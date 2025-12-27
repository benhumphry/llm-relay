# Ollama API Proxy for Anthropic Claude

A self-hosted proxy that presents Anthropic Claude models via the Ollama API interface. This allows any Ollama-compatible application to use Claude models seamlessly.

## Features

- **Full Ollama API compatibility** - Works with any application that supports Ollama
- **All Claude model families** - Supports Claude 4.5, 4, and 3.5 models
- **Vision support** - Pass images via base64 encoding (same as Ollama)
- **Streaming responses** - Real-time streaming just like local Ollama
- **Docker Swarm ready** - Production-ready containerisation
- **Lightweight** - Minimal resource footprint (~128MB RAM)

## Supported Models

### Claude 4.5 Family (Latest - Recommended)

| Ollama Model Name | Anthropic Model | Use Case |
|-------------------|-----------------|----------|
| `claude-opus`, `claude-4.5-opus` | claude-opus-4-5-20251101 | Complex analysis, OCR, detailed reasoning |
| `claude-sonnet`, `claude-4.5-sonnet` | claude-sonnet-4-5-20250929 | Balanced performance |
| `claude-haiku`, `claude-4.5-haiku` | claude-haiku-4-5-20251001 | Fast tasks, tagging, classification |

### Claude 4 Family

| Ollama Model Name | Anthropic Model |
|-------------------|-----------------|
| `claude-4-opus` | claude-opus-4-20250514 |
| `claude-4-sonnet` | claude-sonnet-4-20250514 |

### Claude 3.5 Family (Legacy)

| Ollama Model Name | Anthropic Model |
|-------------------|-----------------|
| `claude-3.5-sonnet` | claude-3-5-sonnet-20241022 |
| `claude-3.5-haiku` | claude-3-5-haiku-20241022 |

## Quick Start

### Using Docker Compose

1. Clone or copy the files to your server
2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
3. Build and run:
   ```bash
   docker compose up -d
   ```
4. Test:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Docker Swarm Deployment

1. Build the image on your manager node:
   ```bash
   docker build -t ollama-claude-proxy:latest .
   ```

2. Create a Docker secret for the API key:
   ```bash
   echo "sk-ant-..." | docker secret create anthropic_api_key -
   ```

3. Deploy the stack:
   ```bash
   docker stack deploy -c docker-compose.yml ollama-claude
   ```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Your Anthropic API key |
| `PORT` | No | 11434 | Server port |
| `HOST` | No | 0.0.0.0 | Bind address |
| `DEBUG` | No | false | Enable debug logging |

## API Endpoints

The proxy implements the following Ollama API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/tags` | GET | List available models |
| `/api/show` | POST | Get model details |
| `/api/chat` | POST | Chat completion (main endpoint) |
| `/api/generate` | POST | Text generation |
| `/api/pull` | POST | Returns success (no download needed) |
| `/api/version` | GET | Version information |
| `/api/embeddings` | POST | Returns error (not supported by Claude) |

## Usage Examples

### Basic Chat Request

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### With System Prompt

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus",
    "messages": [
      {"role": "system", "content": "You are an expert document analyst."},
      {"role": "user", "content": "Analyze this document for key themes."}
    ],
    "stream": false
  }'
```

### With Image (Vision)

```bash
# Base64 encode your image first
IMAGE_BASE64=$(base64 -w0 document.jpg)

curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"claude-opus\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"What text is in this image?\",
        \"images\": [\"$IMAGE_BASE64\"]
      }
    ],
    \"stream\": false
  }"
```

## Integration with Paperless-GPT

Set the following environment variables in your Paperless-GPT configuration:

```yaml
environment:
  - OLLAMA_HOST=http://ollama-claude-proxy:11434
  # For OCR tasks (high accuracy)
  - PAPERLESS_GPT_OCR_MODEL=claude-opus
  # For tagging tasks (fast)
  - PAPERLESS_GPT_TAGGING_MODEL=claude-haiku
```

## Integration with Open WebUI

Point Open WebUI to use the proxy as its Ollama backend:

```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama-claude-proxy:11434
```

## Limitations

1. **No embeddings** - Claude doesn't provide embedding vectors. Use a local model like `nomic-embed-text` with actual Ollama for embeddings.

2. **API costs** - Unlike local models, Claude API usage incurs costs. Monitor your usage at console.anthropic.com.

3. **Rate limits** - Subject to Anthropic API rate limits. The proxy doesn't implement its own rate limiting.

4. **No model downloading** - The `/api/pull` endpoint always returns success as there's nothing to download.

## Troubleshooting

### "ANTHROPIC_API_KEY environment variable is required"

Ensure the API key is set:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Connection refused

Check the service is running:
```bash
docker logs ollama-claude-proxy
curl http://localhost:11434/
```

### Model not found

The proxy will default to `claude-sonnet-4.5` for unknown model names. Check the supported models list above.

### Timeout errors

For long-running requests (like OCR on large documents), you may need to increase timeouts in your client application.

## Development

### Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python proxy.py
```

### Adding new models

Edit the `MODEL_MAPPINGS` dictionary in `proxy.py`:

```python
MODEL_MAPPINGS = {
    # Add your new model alias
    'my-custom-alias': 'claude-sonnet-4-5-20250929',
    ...
}
```

## Licence

MIT License - Use freely, but note that Anthropic API usage is subject to their terms of service.
