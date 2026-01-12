# Smart Aliases Guide

Smart Aliases are the unified feature for intelligent model handling in LLM Relay. A single Smart Alias can combine routing, enrichment (RAG + Web), and caching — enable only what you need.

## Overview

| Feature | What it does |
|---------|--------------|
| **Simple Alias** | Friendly name for a model (e.g., `fast` → `groq/llama-3.3-70b`) |
| **Routing** | Designator LLM picks the best candidate model per request |
| **RAG** | Inject context from indexed Document Stores |
| **Web** | Real-time web search and scraping |
| **Cache** | Semantic response caching |
| **Smart Tag** | Trigger by request tag instead of model name |

## Creating a Smart Alias

1. Go to **Routing → Smart Aliases** in the Admin UI
2. Click **Add Smart Alias**
3. Enter a **Name** (this becomes the model name in requests)
4. Select a **Target Model** (default/fallback)
5. Enable features as needed
6. Save

## Feature Configurations

### Simple Alias (No Features)

Just a friendly name for a model:

| Setting | Value |
|---------|-------|
| Name | `claude` |
| Target Model | `anthropic/claude-sonnet-4-20250514` |
| Features | None enabled |

Use as: `model="claude"`

### Smart Routing

Let an LLM pick the best model for each request:

| Setting | Value |
|---------|-------|
| Name | `smart` |
| Target Model | `anthropic/claude-sonnet-4-20250514` (fallback) |
| **Routing** | ✓ Enabled |
| Designator | `openai/gpt-4o-mini` (fast, cheap) |
| Candidates | Claude, GPT-4o, Gemini with descriptions |
| Strategy | Per Request |

The designator receives candidate descriptions and the user's query, then picks the best model.

**Model Intelligence** (optional): Enable to enhance the designator with web-gathered model comparisons. Requires ChromaDB.

### RAG (Document Retrieval)

Inject relevant context from your indexed documents:

| Setting | Value |
|---------|-------|
| Name | `docs` |
| Target Model | `anthropic/claude-sonnet-4-20250514` |
| **RAG** | ✓ Enabled |
| Document Stores | Select your indexed stores |
| Max Chunks | 5 |
| Similarity Threshold | 0.7 |

Requires: ChromaDB, at least one indexed Document Store.

### Web Access

Add real-time web search and scraping:

| Setting | Value |
|---------|-------|
| Name | `web` |
| Target Model | `anthropic/claude-sonnet-4-20250514` |
| **Web** | ✓ Enabled |
| Query Optimizer | `openai/gpt-4o-mini` (optional) |
| Max Search Results | 5 |
| Max URLs to Scrape | 3 |

Requires: Search provider (SearXNG, Perplexity, or Jina).

**Note:** Caching is automatically disabled when Web is enabled (real-time data shouldn't be cached).

### Response Caching

Cache responses for semantically similar queries:

| Setting | Value |
|---------|-------|
| Name | `cached-claude` |
| Target Model | `anthropic/claude-sonnet-4-20250514` |
| **Cache** | ✓ Enabled |
| Similarity Threshold | 0.95 |
| TTL | 168 hours |
| Min/Max Tokens | 50 / 4000 |

Requires: ChromaDB.

**Match Last Message Only**: Enable for OpenWebUI-style clients where conversation history varies but the last question is what matters.

### Combined Features

Enable multiple features together:

| Name | Features | Use Case |
|------|----------|----------|
| `research` | Routing + RAG + Web | Best model + all context |
| `cached-docs` | RAG + Cache | Document Q&A with caching |
| `smart-cached` | Routing + Cache | Best model with caching |

## Smart Tags

Smart Tags let you trigger an alias by tagging the request instead of using the alias as the model name.

### Setup

1. Create a Smart Alias with desired features
2. Enable **Smart Tag** checkbox
3. Optionally enable **Passthrough Model**

### Usage

Tag your request with the alias name:

```bash
# Via header
curl -H "X-Proxy-Tag: docs" \
  -d '{"model": "gpt-4o", "messages": [...]}'

# Via bearer token
curl -H "Authorization: Bearer docs" \
  -d '{"model": "gpt-4o", "messages": [...]}'

# Via @relay command (stripped before sending to LLM)
curl -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "@relay[tag:docs] What does our policy say?"}]}'
```

### Passthrough Model

When enabled, the Smart Tag uses the original requested model instead of the alias's target:

| Passthrough | Request Model | Actual Model |
|-------------|---------------|--------------|
| Off | `gpt-4o` | Alias target (e.g., `claude-sonnet`) |
| On | `gpt-4o` | `gpt-4o` (with alias features applied) |

This is useful for applying enrichment (RAG, Web) to any model the user chooses.

## @relay Commands

Add tags directly in your message content:

```
@relay[tag:cached] What is the capital of France?
@relay[tag:research,analytics] Analyze this data...
```

Commands are:
- Extracted before processing
- Stripped from content sent to LLM
- Merged with other tags (header, bearer, etc.)
- Can trigger Smart Tags

## Advanced Settings

### System Prompt

Add a custom system prompt that's prepended to every request through the alias. Useful for:
- Setting agent personas
- Adding context about available tools
- Defining response formats

### Reranking

When RAG or Web is enabled, retrieved content is reranked by relevance:
- **Local** (default): Bundled cross-encoder model
- **Jina**: Jina Reranker API (requires `JINA_API_KEY`)

### Statistics

Each Smart Alias tracks:
- Total requests
- Routing decisions (if routing enabled)
- Context injections (if RAG/Web enabled)
- Cache hits and tokens saved (if caching enabled)

Reset statistics from the expanded row in the aliases table.

## Requirements

| Feature | Requires |
|---------|----------|
| Simple Alias | Nothing |
| Routing | Nothing (ChromaDB for Model Intelligence) |
| RAG | ChromaDB + Document Stores |
| Web | Search provider + ChromaDB |
| Cache | ChromaDB |

## Tips

1. **Start simple** — Create a basic alias first, add features incrementally
2. **Use fast designators** — For routing/web, use cheap models like `gpt-4o-mini`
3. **Tune thresholds** — Lower similarity thresholds for more matches, higher for precision
4. **Check statistics** — Monitor cache hit rates and routing decisions
5. **Combine with Smart Tags** — Apply features to any model without changing client code
