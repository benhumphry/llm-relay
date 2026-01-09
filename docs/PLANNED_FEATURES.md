# LLM Relay Roadmap

## Overview

This document covers the near-term v1.x enhancements and the v2.0 feature set.

---

# v1.x Enhancements (Near-Term)

## 1. Additional Document Sources for Smart RAG

**Goal**: Extend Smart RAG beyond local Docker-mounted folders to support cloud storage and document management systems.

### Document Source Types

| Source | Description | Priority |
|--------|-------------|----------|
| **File Upload** | Direct upload via Admin UI, stored in `/data/uploads` | High |
| **WebDAV/Nextcloud** | Connect to WebDAV-compatible storage | Medium |
| **Google Drive** | OAuth-based Google Drive integration | Medium |
| **Paperless-ngx** | API integration with Paperless document management | Low |

### Implementation Phases
1. **Upload Source** (simplest, immediate value)
2. **WebDAV Source** (covers Nextcloud, ownCloud, many NAS devices)
3. **Google Drive** (requires OAuth flow, more complex)
4. **Paperless** (niche but useful for document management users)

---

## 2. Additional Scraping Providers

**Goal**: Better web scraping for JavaScript-heavy sites.

| Provider | Description | Status |
|----------|-------------|--------|
| `builtin` | httpx + BeautifulSoup | Implemented |
| `jina` | Jina Reader API (handles JS, free tier) | Planned |
| `firecrawl` | Self-hostable, structured extraction | Planned |

---

## 3. Additional Search Providers

**Goal**: More options beyond SearXNG and Perplexity.

| Provider | Description |
|----------|-------------|
| Tavily | AI-focused search API |
| Brave Search | Privacy-focused, good API |
| Google Custom Search | Official Google API |

---

## 4. Documentation & Guides

**Goal**: Comprehensive documentation for users to get the most out of LLM Relay.

### Documentation Structure

```
docs/
  README.md              # Overview and quick links
  INSTALLATION.md        # Existing, enhanced
  
  guides/
    getting-started.md   # First-time setup walkthrough
    smart-routers.md     # How to configure intelligent routing
    smart-caches.md      # Semantic caching best practices
    smart-augmentors.md  # Web search augmentation guide
    smart-rags.md        # Document RAG setup and tuning
    smart-pipes.md       # (v2) Pipeline builder guide
    
  reference/
    api.md               # Full API reference (Ollama + OpenAI)
    models.md            # Recommended models for each role
    providers.md         # Provider setup and configuration
    environment.md       # All environment variables
    
  use-cases/
    research-assistant.md    # Web-augmented research workflow
    document-qa.md           # RAG for internal docs
    cost-optimization.md     # Using caches and routing to reduce costs
    multi-provider.md        # Unified access to multiple providers
    openwebui-setup.md       # Integration with Open WebUI
```

### Recommended Models Guide

| Role | Recommended Models | Notes |
|------|-------------------|-------|
| **Designator (routing/gates)** | `openai/gpt-4o-mini`, `groq/llama-3.3-70b` | Fast, cheap, good at classification |
| **Summarizer (intelligence)** | `anthropic/claude-sonnet-4`, `openai/gpt-4o` | Good at synthesis |
| **Embeddings** | `local` (bundled), `ollama/nomic-embed-text` | Local is free, Ollama for better quality |
| **Vision (PDF parsing)** | `ollama/granite3.2-vision`, `ollama/granite-docling` | Granite-docling optimized for documents |
| **General chat** | User preference | Claude, GPT-4o, Gemini all excellent |
| **Coding** | `anthropic/claude-sonnet-4`, `deepseek/deepseek-chat` | Strong coding capabilities |
| **Fast/cheap** | `groq/llama-3.3-70b`, `cerebras/llama-3.3-70b` | Sub-second responses |

---

## 5. Re-ranking for RAG and Augmentor

**Goal**: Improve retrieval quality by re-ranking results with a cross-encoder model.

### Problem
- Embedding-based retrieval (bi-encoder) is fast but can miss semantic nuances
- Web search results are ranked by the search engine, not by relevance to the specific query
- Top-k results may not be the most relevant

### Solution
Add an optional re-ranking step using a cross-encoder model. Fetch more results than needed, then re-rank to get the most relevant top-k.

### Where It Applies

| Feature | Current Flow | With Re-ranking |
|---------|--------------|-----------------|
| **Smart RAG** | Query → Embed → Top-k chunks | Query → Embed → Top-n chunks → Re-rank → Top-k |
| **Smart Augmentor** | Search → Scrape top results | Search → Re-rank URLs → Scrape top results |

### Model Options
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 130MB | Medium | Better |
| `BAAI/bge-reranker-base` | 270MB | Slower | Best |

---

## 6. Image & Audio API Endpoints

**Goal**: Add image generation and audio transcription endpoints - valuable for routing between self-hosted and cloud services.

### New Endpoints

| Endpoint | Description | Providers |
|----------|-------------|-----------|
| `POST /v1/images/generations` | Image generation | OpenAI (DALL-E), local (Stable Diffusion via Ollama/ComfyUI) |
| `POST /v1/audio/transcriptions` | Speech-to-text | OpenAI (Whisper), local (Whisper via Ollama), Groq |
| `POST /v1/audio/speech` | Text-to-speech | OpenAI, ElevenLabs, local |

### Why This Matters
- **Cost optimization**: Route to local Stable Diffusion for bulk image generation, OpenAI for quality
- **Unified API**: Same endpoint works with cloud or self-hosted models
- **Smart routing potential**: Could route based on quality requirements, speed, cost

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

### Visual Editor (Admin UI)
- Drag-and-drop node placement
- Connection drawing between nodes
- Node configuration panels
- Pipeline validation (must have entry, must reach terminal)
- Live preview/test

---

## 2. Smart Query Studio

**Goal**: Built-in chat interface for testing and using models.

### Features
- **Chat Interface**: OpenWebUI-style conversation UI
- **Model Selector**: All available models, aliases, and smart features
- **Conversation History**: Persist and resume conversations
- **Blind A/B Testing**: Compare models without knowing which is which
- **Pipeline Testing**: Test Smart Pipes before deploying

### Blind Test Flow
1. User selects two models to compare
2. System randomly assigns A and B
3. User sends message, both models respond (labeled A/B)
4. User votes for preferred response
5. Results revealed, statistics tracked

### Aggregate Statistics
Track win rates across all blind tests to build a preference database:
- Feed back into Model Intelligence for Smart Routers
- Show "Community rankings" in Query Studio
- Help users decide which models to use for different tasks
- Provide real-world preference data beyond benchmarks

---

## 3. Model Sync Subscription Service

**Goal**: High-quality, regularly updated model metadata as a subscription.

### What We Provide
1. **Model Intelligence Database**
   - Regularly refreshed assessments for 100+ models
   - Comparative analysis between popular model pairs
   - Benchmark scores, pricing, capabilities

2. **Pricing Data**
   - Scraped from provider pricing pages
   - More accurate than LiteLLM (handles edge cases)
   - Updated daily

3. **Model Availability**
   - Which models are currently available per provider
   - Deprecation notices
   - New model announcements

### Pricing Tiers
| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | LiteLLM pricing only (current behavior) |
| Basic | $5/mo | Daily pricing sync, weekly intelligence |
| Pro | $15/mo | Daily everything, priority updates, API access |

### Implementation
- Separate standalone project/repo
- FastAPI service with PostgreSQL
- Stripe for subscriptions
- LLM Relay syncs on startup + daily

---

# Implementation Priority

## Phase 1: v1.5 (Near-term)
1. File upload document source for RAG
2. Jina scraper provider
3. Re-ranking for RAG and Augmentor (cross-encoder model)
4. Documentation: Getting started, Smart features guides

## Phase 2: v1.6
1. WebDAV document source
2. Tavily/Brave search providers
3. Image generation endpoint (`/v1/images/generations`)
4. Audio transcription endpoint (`/v1/audio/transcriptions`)
5. Documentation: Reference docs, use case guides

## Phase 3: v2.0
1. Smart Pipe Studio (visual pipeline builder)
2. Smart Query Studio (chat interface)
3. Model Sync subscription service (separate project)
