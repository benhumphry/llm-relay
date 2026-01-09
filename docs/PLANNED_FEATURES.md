# Planned Features

This document outlines planned features for LLM Relay.

---

## Implemented Features

### v1.4.x: Smart RAG (Document Context)

Document-based context augmentation using RAG (Retrieval-Augmented Generation). Index local document folders and retrieve relevant context to enhance LLM requests.

**How it works:**
1. User creates a Smart RAG pointing at a Docker-mounted folder
2. Documents are parsed via **Docling** and indexed into ChromaDB (per-RAG collection)
3. User requests model by RAG name (e.g., "docs-assistant")
4. Query is embedded, similar chunks retrieved from ChromaDB
5. Retrieved context injected into system prompt
6. Request forwarded to configured target model

**Key features:**
- Multiple document formats via Docling (PDF, DOCX, PPTX, HTML, Markdown, images)
- Flexible embedding providers (local bundled, Ollama, or any configured provider)
- Vision model offloading for PDF parsing (local Docling, Ollama, or cloud providers)
- Scheduled re-indexing with cron expressions
- Real-time indexing progress with cancel support
- Semantic search with configurable similarity threshold

### v1.2.x: Smart Cache, Smart Augmentor, Model Intelligence

**Smart Cache** — Semantic response caching using ChromaDB. Returns cached responses for semantically similar queries, reducing token usage and costs.

**Smart Augmentor** — Context augmentation via web search and URL scraping. Always performs search+scrape for comprehensive web context. A designator LLM generates optimized search queries for each request.

**Model Intelligence** — Web-gathered comparative assessments for Smart Router candidates. Searches for model reviews and direct comparisons, then summarizes into relative strengths/weaknesses for informed routing decisions.

**Redirects** — Transparent model name mappings with wildcard support. Checked first in resolution order, don't appear in logs.

---

## Architecture Overview: ChromaDB as Unified Vector Store

ChromaDB serves as the unified backend for all context features, providing semantic search across multiple data sources.

```
┌─────────────────────────────────────────────────────────────────┐
│                          ChromaDB                                │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ Collection:      │  │ Collection:      │  │ Collection:    │ │
│  │ "response_cache" │  │ "model_intel"    │  │ "smartrag_{id}"│ │
│  │                  │  │                  │  │                │ │
│  │ - Cached LLM     │  │ - Model          │  │ - User docs    │ │
│  │   responses      │  │   assessments    │  │ - Chunked text │ │
│  │ - TTL expiry     │  │ - Comparisons    │  │                │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
            │                      │                    │
            ▼                      ▼                    ▼
    ┌───────────────┐      ┌───────────────┐    ┌───────────────┐
    │ Smart Cache   │      │ Smart Router  │    │ Smart RAG     │
    │ (responses)   │      │ (intelligence)│    │ (documents)   │
    └───────────────┘      └───────────────┘    └───────────────┘
```

### Environment Variables
```bash
CHROMA_URL=http://localhost:8000  # ChromaDB server URL
CHROMA_COLLECTION_PREFIX=llmrelay_  # Namespace for collections
SEARXNG_URL=http://localhost:8080  # SearXNG for web search
```

---

## Implementation Roadmap

### Completed
- [x] Phase 1: ChromaDB Foundation
- [x] Phase 1.5: Smart Cache
- [x] Phase 2: Smart Augmentor (search + scrape)
- [x] Phase 2.5: Model Intelligence
- [x] Phase 3: Smart RAG (document context)

### Planned (v2.x)
- [ ] Smart Context (unified multi-source retrieval)
- [ ] Additional scraping providers (Jina, Firecrawl)
- [ ] Additional search providers (Tavily, Brave)
- [ ] Re-ranking models for improved retrieval
- [ ] Custom context templates
- [ ] User feedback loop for routing optimization

---

## Future Enhancements

### Smart Context (Unified Multi-Source)

**Status:** Concept  
**Priority:** Medium

A unified context provider that can pull from multiple sources (web search, documents, cached results) based on the query.

**How It Works:**
1. User configures which sources to include (web search, specific RAG collections)
2. Designator analyzes query and decides which sources are relevant
3. Retrieves from multiple ChromaDB collections in parallel
4. Merges and ranks results
5. Injects combined context

### Web Scraping Providers

Tiered approach for different site types:

| Provider | Description | Status |
|----------|-------------|--------|
| `builtin` | httpx + BeautifulSoup | Implemented |
| `jina` | Jina Reader API (free, handles JS) | Planned |
| `firecrawl` | Self-hostable, structured extraction | Planned |

### Additional Search Providers
- Tavily
- Brave Search
- Google Custom Search

### Re-ranking Models
Use a re-ranker to improve retrieval quality before context injection.

### User Feedback Loop
Store routing outcomes and user feedback to learn which models actually perform well for different query types.

---

## Docker Compose (Full Stack)

```yaml
services:
  llm-relay:
    image: ghcr.io/benhumphry/llm-relay:latest
    ports:
      - "11434:11434"
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://...
      - CHROMA_URL=http://chroma:8000
      - SEARXNG_URL=http://searxng:8080
    volumes:
      # Mount folders for Smart RAG
      - /path/to/docs:/data/documents:ro
    depends_on:
      - postgres
      - chroma
      - searxng

  postgres:
    image: postgres:16-alpine
    volumes:
      - postgres-data:/var/lib/postgresql/data

  chroma:
    image: chromadb/chroma:latest
    volumes:
      - chroma-data:/chroma/chroma

  searxng:
    image: searxng/searxng:latest
    volumes:
      - ./searxng:/etc/searxng

volumes:
  postgres-data:
  chroma-data:
```

See `docker-compose.full.yml` for complete configuration.
