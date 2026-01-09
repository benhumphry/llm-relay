# Smart Caches

Smart Caches provide semantic response caching. When a query is semantically similar to a previous one, the cached response is returned instantly without calling the LLM.

## How It Works

1. **Request arrives** with model name set to your Smart Cache
2. **Query is embedded** using the configured embedding model
3. **Semantic search** finds similar previous queries in ChromaDB
4. **If match found** (above similarity threshold): return cached response
5. **If no match**: forward to target model, cache the response

```
Client Request → Smart Cache → Semantic Search
                     ↓              ↓
              Cache Hit?      ChromaDB
                 ↓ Yes           ↓ No
           Return Cached    Forward to Target
                              ↓
                         Cache Response
```

## Prerequisites

- **ChromaDB**: Required for vector storage (`CHROMA_URL` environment variable)

## Creating a Smart Cache

In the Admin UI:

1. Go to **Smart Caches** in the sidebar
2. Click **New Cache**
3. Configure:

| Field | Description |
|-------|-------------|
| **Name** | The model name clients will use (e.g., `cached-claude`) |
| **Target Model** | The underlying model to call on cache miss |
| **Similarity Threshold** | How similar queries must be (0.0-1.0, default 0.95) |
| **TTL (seconds)** | How long to keep cached responses (0 = forever) |
| **Min Response Tokens** | Don't cache responses shorter than this |
| **Max Response Tokens** | Don't cache responses longer than this |
| **Match Last Message Only** | Ignore conversation history for matching |

4. Click **Save**

## Example Configuration

**Cache Name**: `cached-claude`

**Target Model**: `anthropic/claude-sonnet-4-5`

**Similarity Threshold**: `0.95` (95% similar)

**TTL**: `86400` (24 hours)

**Min Response Tokens**: `50` (skip short responses)

Now use it:

```bash
# First request - calls Claude, caches response
curl http://localhost:11434/api/chat \
  -d '{"model": "cached-claude", "messages": [{"role": "user", "content": "What is Python?"}]}'

# Similar request - returns cached response instantly
curl http://localhost:11434/api/chat \
  -d '{"model": "cached-claude", "messages": [{"role": "user", "content": "Explain Python to me"}]}'
```

## Configuration Options

### Similarity Threshold

Controls how similar a query must be to return a cached response.

| Value | Behavior |
|-------|----------|
| `0.99` | Nearly identical queries only |
| `0.95` | Very similar queries (recommended) |
| `0.90` | Moderately similar queries |
| `0.80` | Loosely similar queries (may return wrong answers) |

Start with 0.95 and adjust based on your use case.

### TTL (Time to Live)

How long cached responses remain valid, in seconds.

| Value | Duration |
|-------|----------|
| `0` | Cache forever |
| `3600` | 1 hour |
| `86400` | 24 hours |
| `604800` | 1 week |

Consider your data freshness requirements. For factual queries about current events, use shorter TTLs.

### Token Filters

**Min Response Tokens**: Skip caching very short responses that may be unhelpful or error messages.

**Max Response Tokens**: Skip caching very long responses to save storage space.

Recommended starting values:
- Min: 50 tokens
- Max: 4000 tokens (or leave at 0 for no limit)

### Match Last Message Only

When enabled, only the last user message is used for similarity matching. The conversation history is ignored.

**Use cases**:
- Open WebUI and other chat interfaces where each turn is independent
- FAQ-style queries where context doesn't matter
- Reducing false negatives from conversation context

**When to disable**:
- Multi-turn conversations where context matters
- Queries that depend on previous messages

## Embedding Configuration

By default, Smart Cache uses a local embedding model (bundled with LLM Relay). You can configure alternative embedding providers:

| Provider | Configuration |
|----------|---------------|
| **Local** | `local` - Uses bundled sentence-transformers |
| **Ollama** | `ollama:<instance>` - Uses your Ollama instance |
| **OpenAI** | `openai` - Uses text-embedding-3-small |

For Ollama, you'll need to pull an embedding model first:

```bash
# On your Ollama instance
ollama pull nomic-embed-text
```

Then configure the cache with:
- Embedding Provider: `ollama:your-instance`
- Embedding Model: `nomic-embed-text`

## Viewing Cache Performance

### Response Headers

Cache hits include headers:

```
X-LLM-Relay-Cache: hit
X-LLM-Relay-Cache-Similarity: 0.97
```

Cache misses:

```
X-LLM-Relay-Cache: miss
```

### Admin UI

The Smart Cache detail page shows:
- Total requests
- Cache hit rate
- Cache entries
- Storage usage

## Best Practices

1. **Start conservative** - Use 0.95+ similarity threshold initially
2. **Monitor hit rate** - Aim for 20-40% hit rate for good ROI
3. **Use appropriate TTL** - Match your data freshness needs
4. **Enable match_last_message_only** for chat UIs - Improves hit rate significantly
5. **Filter short responses** - Avoid caching error messages

## Cache Invalidation

Currently, cache entries expire based on TTL. To manually clear the cache:

1. Go to **Smart Caches** in the Admin UI
2. Click on the cache
3. Click **Clear Cache**

This removes all cached entries for that cache.

## Troubleshooting

### Low hit rate

- Lower the similarity threshold (try 0.90)
- Enable "Match Last Message Only" for chat interfaces
- Check if queries are too varied for caching

### Wrong cached responses

- Raise the similarity threshold (try 0.98)
- Reduce TTL for time-sensitive content
- Disable "Match Last Message Only" if context matters

### High storage usage

- Reduce TTL
- Set max_response_tokens to limit cached content
- Clear cache periodically

### Cache not working

- Verify ChromaDB is running (`CHROMA_URL` is set)
- Check container logs for embedding errors
- Ensure target model is accessible
