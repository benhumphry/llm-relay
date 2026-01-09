# LLM Relay Roadmap

## Overview

This document covers the near-term v1.x enhancements and the v2.0 feature set.

---

# Completed in v1.5

## Re-ranking for RAG and Augmentor

Cross-encoder reranking is now always-on for both Smart RAG and Smart Augmentors.

- **Default model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~48MB, runs locally)
- **Optional**: Jina Reranker API (requires `JINA_API_KEY`)

### How It Works

| Feature | Flow |
|---------|------|
| **Smart RAG** | Query → Embed → Top-n chunks → Rerank → Top-k |
| **Smart Augmentor** | Search → Get URLs with titles/snippets → Rerank → Scrape top URLs |

## Jina Integration

Three Jina services are now supported:

| Service | Endpoint | API Key Required |
|---------|----------|------------------|
| **Jina Scraper** | `r.jina.ai` | No (optional for rate limits) |
| **Jina Search** | `s.jina.ai` | Yes |
| **Jina Reranker** | `api.jina.ai/v1/rerank` | Yes |

## Global Web Settings

Search provider and scraper provider are now configured globally in Settings, not per-augmentor.

## Documentation Guides

Created comprehensive guides in `docs/guides/`:
- `getting-started.md` - First-time setup walkthrough
- `smart-routers.md` - Intelligent model routing
- `smart-caches.md` - Semantic response caching
- `smart-augmentors.md` - Web search augmentation
- `smart-rags.md` - Document RAG setup

---

# v1.x Enhancements (Near-Term)

## v1.6: MCP Integration for Document Sources

**Goal**: Replace file upload/WebDAV with MCP (Model Context Protocol) for flexible document source integration.

### Why MCP?
- Connect to any document source via MCP servers
- Growing ecosystem: Google Drive, Notion, Confluence, S3, databases
- Implement once, support many sources
- Future-proof architecture

### Planned Sources via MCP
| Source | MCP Server | Priority |
|--------|------------|----------|
| Google Drive | `mcp-gdrive` | High |
| Notion | `mcp-notion` | High |
| S3/MinIO | `mcp-s3` | Medium |
| Confluence | `mcp-confluence` | Medium |
| Local Files | Built-in | Already supported |

## v1.7: Smart Tags & Inline Tags

**Goal**: Tag-based routing and augmentation with inline query syntax.

### Smart Tags
Define tags that trigger specific routing or augmentation:

| Tag Name | Target Model | Augmentor | RAG | System Prefix |
|----------|--------------|-----------|-----|---------------|
| `augment` | `claude-sonnet-4-5` | web-search | - | - |
| `research` | `smart-research-router` | web-search | docs-rag | - |
| `fast` | `groq:llama-3.3-70b` | - | - | - |
| `creative` | `claude-sonnet-4-5` | - | - | "Be creative..." |

### Inline Tag Syntax
Add tags directly in queries:

```
What's the weather in London? #tag:augmentor #tag:alice
```

The system:
1. Extracts `#tag:*` patterns from message
2. Removes them from query sent to LLM
3. Applies as tracking tags
4. Triggers Smart Tags if configured

### Use Cases
- Per-message control in conversations
- Works with any client (Open WebUI, Cursor, etc.)
- Combines with existing tagging methods (bearer, header, model suffix)

---

## Additional Search Providers

| Provider | Description | Status |
|----------|-------------|--------|
| `searxng` | Self-hosted metasearch | Implemented |
| `perplexity` | Perplexity API | Implemented |
| `jina` | Jina Search API | Implemented (v1.5) |
| `tavily` | AI-focused search API | Planned |
| `brave` | Privacy-focused search | Planned |

---

## Image & Audio API Endpoints

**Goal**: Add image generation and audio transcription endpoints.

### New Endpoints

| Endpoint | Description | Providers |
|----------|-------------|-----------|
| `POST /v1/images/generations` | Image generation | OpenAI (DALL-E), local (SD via ComfyUI) |
| `POST /v1/audio/transcriptions` | Speech-to-text | OpenAI (Whisper), Groq |
| `POST /v1/audio/speech` | Text-to-speech | OpenAI, ElevenLabs |

---

# v2.0 Features (Major Release)

## 1. Smart Pipe Studio

**Goal**: Visual pipeline builder for chaining Smart components.

### Concept
Users create custom "pipes" by connecting components in a visual flow editor:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Web Search  │───▶│    Router    │───▶│    Cache     │
│  Augmentor   │    │  (4 models)  │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
                           │
                    ┌──────┴──────┐
              ┌─────┴─────┐  ┌────┴─────┐
              │  Claude   │  │  GPT-4o  │
              │  (cached) │  │          │
              └───────────┘  └──────────┘
```

### Node Types
| Type | Description |
|------|-------------|
| `augmentor` | Web search/scrape injection |
| `rag` | Document context injection |
| `router` | LLM-based model selection |
| `cache` | Semantic cache lookup/store |
| `model` | Terminal node - actual LLM call |
| `transform` | Custom prompt transformation |
| `gate` | Conditional branch (yes/no question) |

### Gate Node (Conditional Branching)
The `gate` node enables conditional routing based on a designator LLM's yes/no decision:

```
                    ┌─────────────────┐
                    │      Gate       │
                    │ "Needs recent   │
          ┌─────────│  information?"  │─────────┐
          │ YES     └─────────────────┘    NO   │
          ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│   Augmentor     │                   │     Cache       │
│  (web search)   │                   │                 │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        ▼
               ┌─────────────────┐
               │     Router      │
               │  (pick model)   │
               └─────────────────┘
```

---

## 2. Smart Query Studio

**Goal**: Built-in chat interface for testing and using models.

### Features
- **Chat Interface**: OpenWebUI-style conversation UI
- **Model Selector**: All available models, aliases, and smart features
- **Conversation History**: Persist and resume conversations
- **Blind A/B Testing**: Compare models without knowing which is which
- **Pipeline Testing**: Test Smart Pipes before deploying

---

## 3. Model Sync Subscription Service

**Goal**: High-quality, regularly updated model metadata as a subscription.

### What We Provide
1. **Model Intelligence Database** - Regularly refreshed assessments for 100+ models
2. **Pricing Data** - Scraped from provider pricing pages, more accurate than LiteLLM
3. **Model Availability** - Deprecation notices, new model announcements

---

# Implementation Priority

## Phase 1: v1.5 (Completed)
- [x] Re-ranking for RAG and Augmentor (cross-encoder model)
- [x] Jina scraper provider
- [x] Jina search provider
- [x] Jina reranker provider
- [x] Global web settings (search + scraper)
- [x] Documentation guides

## Phase 2: v1.6
- [ ] MCP integration for document sources
- [ ] Additional search providers (Tavily, Brave)

## Phase 3: v1.7
- [ ] Smart Tags
- [ ] Inline tag syntax (`#tag:name`)
- [ ] Image generation endpoint
- [ ] Audio transcription endpoint

## Phase 4: v2.0
- [ ] Smart Pipe Studio (visual pipeline builder)
- [ ] Smart Query Studio (chat interface)
- [ ] Model Sync subscription service
