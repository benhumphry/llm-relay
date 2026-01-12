# LLM Relay Roadmap

## Overview

This document covers completed features and the roadmap for future releases.

---

# Completed Features

## v1.5: Reranking & Jina Integration

- **Cross-encoder reranking** for RAG and Web enrichment
- **Jina integration**: Search, Scraper, and Reranker APIs
- **Global web settings** (search + scraper configuration)
- **Documentation guides**

## v1.6: Smart Aliases Unification

Consolidated separate features into unified **Smart Aliases**:

| Old Feature | New Location |
|-------------|--------------|
| Aliases | Smart Alias (simple mode) |
| Smart Routers | Smart Alias with Routing enabled |
| Smart Enrichers | Smart Alias with RAG/Web enabled |
| Smart Caches | Smart Alias with Cache enabled |

**Benefits:**
- Single UI for all smart features
- Mix and match: routing + RAG + caching in one alias
- Cleaner codebase with single processing pipeline

## v1.7: Smart Tags & Query Tags

- **Smart Tags**: Trigger aliases by request tag instead of model name
- **Passthrough Model**: Honor original model when triggered as Smart Tag
- **@relay commands**: In-content tag syntax (`@relay[tag:name]`)
- **Extensible format**: `@relay[command:key=value]` for future commands

## Document Sources (v1.6+)

Direct API integrations (no Node.js required):
- Google Drive, Gmail, Calendar (via OAuth)
- Notion (via REST API)
- GitHub (via REST API)
- Paperless-ngx (via REST API)
- Local filesystem

---

# Planned Features

## v1.8: Additional Integrations

### Search Providers

| Provider | Description | Status |
|----------|-------------|--------|
| `searxng` | Self-hosted metasearch | ✓ Done |
| `perplexity` | Perplexity API | ✓ Done |
| `jina` | Jina Search API | ✓ Done |
| `tavily` | AI-focused search API | Planned |
| `brave` | Privacy-focused search | Planned |

### Media Endpoints

| Endpoint | Description | Providers |
|----------|-------------|-----------|
| `POST /v1/images/generations` | Image generation | OpenAI, Stability |
| `POST /v1/audio/transcriptions` | Speech-to-text | OpenAI, Groq |
| `POST /v1/audio/speech` | Text-to-speech | OpenAI, ElevenLabs |

---

## v2.0: Major Release

### Smart Pipe Studio

Visual pipeline builder for chaining components:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Web Search  │───▶│    Router    │───▶│    Cache     │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Node types:**
- `augmentor` - Web search/scrape
- `rag` - Document context
- `router` - Model selection
- `cache` - Response caching
- `gate` - Conditional branching
- `transform` - Prompt transformation

### Smart Query Studio

Built-in chat interface for testing:
- OpenWebUI-style conversation UI
- Model selector with all aliases
- Blind A/B testing
- Pipeline testing

### Model Sync Service

Subscription for high-quality model metadata:
- Regularly refreshed model intelligence
- Accurate pricing data
- Deprecation notices

---

## v2.1: Smart Actions

Bidirectional Google Integration - let LLMs take actions with user approval.

### Concept

LLM outputs structured action blocks:

```xml
<smart_action type="calendar_create">
{"summary": "Meeting with John", "start": "2026-01-14T14:00:00"}
</smart_action>
```

Proxy detects actions and presents approval UI to user.

### Supported Actions

| Type | Description |
|------|-------------|
| `calendar_create` | Create calendar event |
| `calendar_update` | Modify existing event |
| `email_reply` | Reply to email thread |
| `email_compose` | New email |
| `drive_create` | Create document |

### Security

- Explicit user approval required
- OAuth write scopes
- Action audit log
- Rate limiting
- Per-alias allow-lists

---

# Implementation Priority

## Completed
- [x] Cross-encoder reranking
- [x] Jina integration (search, scraper, reranker)
- [x] Smart Aliases unification
- [x] Smart Tags
- [x] @relay query commands
- [x] Document sources (Notion, GitHub, Google, Paperless)

## Next Up (v1.8)
- [ ] Additional search providers (Tavily, Brave)
- [ ] Image generation endpoint
- [ ] Audio transcription endpoint

## Future (v2.0+)
- [ ] Smart Pipe Studio
- [ ] Smart Query Studio
- [ ] Model Sync service
- [ ] Smart Actions (bidirectional Google)
