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
| **Memory** | Persistent memory that remembers explicit user facts across sessions |
| **Smart Source Selection** | Designator allocates token budget across RAG stores and web |
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

### Persistent Memory

Remember explicit user facts across sessions:

| Setting | Value |
|---------|-------|
| Name | `assistant` |
| Target Model | `anthropic/claude-sonnet-4-20250514` |
| **Memory** | ✓ Enabled |

**How it works:**
1. After each response, the system analyzes the user's query (not the response)
2. Only **explicitly stated facts** about the user are extracted and stored
3. Memory is injected into future requests as context
4. Memory persists across sessions until manually cleared

**Important:** Memory only captures what users directly say about themselves, not inferences. For example:
- "I work at Acme Corp" → Stored
- Asking about "Acme Corp products" → **Not** stored (asking about ≠ interest in)

View and clear memory from the expanded alias row in the Admin UI.

### Smart Source Selection

When using RAG with multiple Document Stores, let the designator allocate token budget:

| Setting | Value |
|---------|-------|
| Name | `research` |
| Target Model | `anthropic/claude-sonnet-4-20250514` |
| **RAG** | ✓ Enabled |
| Document Stores | Multiple stores selected |
| **Smart Source Selection** | ✓ Enabled |
| Designator | `openai/gpt-4o-mini` |

**How it works:**
1. Each Document Store has intelligence metadata (themes, best_for, summary)
2. The designator analyzes the query and store intelligence
3. Budget is split: 50% baseline to all sources, 50% priority-allocated to most relevant
4. All sources return results, but relevant ones get more token budget
5. Final results are reranked by relevance

**Benefits:**
- Ensures all sources contribute (no blind spots)
- Prioritizes most relevant sources for the query
- Works with Web as an additional "source"

Generate store intelligence from **Data Sources → Document Stores** by expanding a store and clicking "Generate Intelligence".

### Combined Features

Enable multiple features together:

| Name | Features | Use Case |
|------|----------|----------|
| `research` | Routing + RAG + Web | Best model + all context |
| `cached-docs` | RAG + Cache | Document Q&A with caching |
| `smart-cached` | Routing + Cache | Best model with caching |
| `assistant` | Memory + RAG | Personal assistant with docs |
| `smart-research` | RAG + Web + Smart Source Selection | Multi-source research |

### Context Priority (Hybrid RAG + Web)

When both RAG and Web are enabled, use **Context Priority** to control how token budget is allocated and which source takes precedence:

| Priority | RAG Tokens | Web Tokens | Best For |
|----------|------------|------------|----------|
| **Balanced** (default) | 50% | 50% | General hybrid queries |
| **Prefer Documents** | 70% | 30% | Questions about your indexed docs with web backup |
| **Prefer Web** | 30% | 70% | Current events with document context |

**Behavior:**
- Token allocation determines how much context each source can contribute
- Prioritized source appears first in the injected context
- LLM receives a hint about which source to prefer when answering
- Setting is only shown when both RAG and Web are enabled
- Ignored if either feature is later disabled

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

### Multiple Tags

You can specify multiple tags in a single request:

```
@relay[tag:tracking,docs] What does our policy say?
```

Or via header/suffix:
```
model@tracking,docs
```

**Behavior:**
- All tags are recorded in the request log for tracking/attribution
- Tags are checked in order for Smart Tag matches
- First matching Smart Tag wins and triggers that alias
- Non-Smart-Tag values pass through harmlessly (just logged)

Example: `@relay[tag:analytics,docs]` where only `docs` is a Smart Tag:
1. `analytics` - not a Smart Tag, logged for tracking
2. `docs` - is a Smart Tag, triggers the `docs` alias
3. Both tags appear in the request log

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
| Memory | Nothing (stores in database) |
| Smart Source Selection | RAG enabled + Document Store intelligence |

## Tips

1. **Start simple** — Create a basic alias first, add features incrementally
2. **Use fast designators** — For routing/web, use cheap models like `gpt-4o-mini`
3. **Tune thresholds** — Lower similarity thresholds for more matches, higher for precision
4. **Check statistics** — Monitor cache hit rates and routing decisions
5. **Combine with Smart Tags** — Apply features to any model without changing client code
