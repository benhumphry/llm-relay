# Planned Features

This document outlines planned features for LLM Relay.

---

## Implemented Features (v1.2.0)

The following features from earlier plans have been implemented:

### Smart Cache
Semantic response caching using ChromaDB. Returns cached responses for semantically similar queries, reducing token usage and costs.

**Key features:**
- Semantic similarity matching (configurable threshold 0.90-0.99)
- TTL-based expiry
- System prompt matching (optional)
- Token length filters (min/max cached tokens)
- Match last message only option (for OpenWebUI compatibility)
- Cache hit statistics

### Smart Augmentor
Context augmentation via web search and URL scraping. A designator LLM analyzes queries and decides how to augment:
- `direct` - pass through unchanged
- `search:query` - search the web, inject results
- `scrape:url1,url2` - fetch specific URLs
- `search+scrape:query` - search then scrape top results

**Key features:**
- Extensible search providers (SearXNG, Perplexity)
- Built-in web scraper (httpx + BeautifulSoup)
- Configurable context token limits
- Admin UI for management

### Model Intelligence
Web-gathered comparative assessments for Smart Router candidates. When enabled on a router:
1. Searches for individual model reviews/benchmarks
2. Searches for direct model comparisons ("Model A vs Model B")
3. Summarizes into RELATIVE strengths/weaknesses
4. Cached in ChromaDB with TTL expiry

**Key features:**
- Comparative analysis (not just generic descriptions)
- Per-router configuration (search provider, summarizer model)
- Refresh button in Admin UI
- Intelligence injected into designator prompts

### Redirects
Transparent model name mappings with wildcard support. Unlike aliases:
- Checked first in resolution order
- Support wildcard patterns (`openrouter/anthropic/*` -> `anthropic/*`)
- Don't appear in logs (transparent)

---

## Architecture Overview: ChromaDB as Unified Vector Store

ChromaDB serves as the unified backend for all context features, providing semantic search across multiple data sources.

```
┌─────────────────────────────────────────────────────────────────┐
│                          ChromaDB                                │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ Collection:      │  │ Collection:      │  │ Collection:    │ │
│  │ "response_cache" │  │ "model_intel"    │  │ "docs_{id}"    │ │
│  │                  │  │                  │  │                │ │
│  │ - Cached LLM     │  │ - Model          │  │ - User docs    │ │
│  │   responses      │  │   assessments    │  │   (future)     │ │
│  │ - TTL expiry     │  │ - Comparisons    │  │                │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
            │                      │                    │
            ▼                      ▼                    ▼
    ┌───────────────┐      ┌───────────────┐    ┌───────────────┐
    │ Smart Cache   │      │ Smart Router  │    │ Smart RAG     │
    │ (responses)   │      │ (intelligence)│    │ (future)      │
    └───────────────┘      └───────────────┘    └───────────────┘
```

### Environment Variables
```bash
CHROMA_URL=http://localhost:8000  # ChromaDB server URL
CHROMA_COLLECTION_PREFIX=llmrelay_  # Namespace for collections
SEARXNG_URL=http://localhost:8080  # SearXNG for web search
```

---

## Smart RAG (Document Context)

**Status:** Planned  
**Priority:** Medium

### Overview
Retrieval-Augmented Generation that indexes local document folders (Docker-mapped) and retrieves relevant context to enhance LLM requests.

### How It Works
1. User creates a Smart RAG pointing at a Docker-mounted folder
2. Documents are parsed via **Docling** and indexed into ChromaDB (per-RAG collection)
3. User requests model by RAG name (e.g., "docs-assistant")
4. Query is embedded, similar chunks retrieved from ChromaDB
5. Retrieved context injected into system prompt
6. Request forwarded to configured target model

### Key Technologies

| Component | Library | Purpose |
|-----------|---------|---------|
| Document Parsing | **Docling** | PDF, DOCX, PPTX, XLSX, HTML, images, audio |
| RAG Orchestration | **LlamaIndex** | Chunking, embedding, retrieval pipeline |
| Vector Store | **ChromaDB** | Already used for Smart Cache, Model Intelligence |
| Embeddings | Local or Ollama | Bundled `BAAI/bge-small-en-v1.5` or Ollama models |

### Document Types (via Docling)
- **Documents:** PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc
- **Images:** PNG, JPG, TIFF, BMP (with OCR)
- **Audio:** WAV, MP3 (transcription)

### Embedding Options

| Provider | Model | Notes |
|----------|-------|-------|
| **Local** (default) | `BAAI/bge-small-en-v1.5` | ~130MB, bundled, no external deps |
| **Ollama** | `granite3.2-vision`, `nomic-embed-text`, `mxbai-embed-large` | Uses existing Ollama instance |
| **OpenAI** | `text-embedding-3-small` | API costs, high quality |

### Database Schema
```python
class SmartRAG(Base):
    __tablename__ = "smart_rags"
    
    id: int
    name: str  # unique, e.g., "docs-assistant"
    
    # Document source (Docker-mapped folder)
    source_path: str  # e.g., "/data/documents"
    
    # Target model
    target_model: str  # "provider_id/model_id"
    
    # Embedding config
    embedding_provider: str  # "local" | "ollama" | "openai"
    embedding_model: str | None  # e.g., "granite3.2-vision:latest"
    ollama_url: str | None  # Override for Ollama instance
    
    # Indexing config
    index_schedule: str | None  # Cron expression, e.g., "0 2 * * *"
    last_indexed: datetime | None
    index_status: str  # "pending" | "indexing" | "ready" | "error"
    index_error: str | None
    
    # Chunking config
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval config
    max_results: int = 5
    similarity_threshold: float = 0.7
    max_context_tokens: int = 4000
    
    # ChromaDB collection (auto-generated)
    collection_name: str  # "smartrag_{id}"
    
    # Statistics
    document_count: int = 0
    chunk_count: int = 0
    
    tags_json: str
    enabled: bool
```

### Scheduling Options (Admin UI Presets)
- Never (manual only)
- Every hour / 6 hours
- Daily at 2 AM
- Weekly on Sunday at 2 AM
- Custom cron expression

Uses **APScheduler** for background scheduling.

### Docker Volume Mapping
```yaml
services:
  llm-relay:
    volumes:
      # Map local/network folders for Smart RAG
      - /path/to/docs:/data/documents:ro
      - /network/share:/data/shared:ro
```

### Implementation Phases
1. Database layer (model, CRUD, migration)
2. Embedding provider abstraction (local, Ollama, OpenAI)
3. Document processor (Docling wrapper)
4. Indexer service (APScheduler + ChromaDB)
5. Retriever (query + context formatting)
6. RAG engine & registry integration
7. Admin UI

### Dependencies
```
docling>=2.0.0
llama-index-core
llama-index-vector-stores-chroma
llama-index-embeddings-huggingface
llama-index-embeddings-ollama  # optional
apscheduler>=3.10.0
```

---

## Smart Context (Future: Unified)

**Status:** Concept  
**Priority:** Low (after Smart RAG)

### Overview
A unified context provider that can pull from multiple sources (web search, documents, cached results) based on the query.

### How It Works
1. User configures which sources to include (web search, specific RAG collections)
2. Designator analyzes query and decides which sources are relevant
3. Retrieves from multiple ChromaDB collections in parallel
4. Merges and ranks results
5. Injects combined context

---

## Future Enhancements

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

### Custom Context Templates
Allow users to define how context is formatted and injected (beyond default system prompt injection).

### Re-ranking Models
Use a re-ranker to improve retrieval quality before context injection.

### User Feedback Loop
Store routing outcomes and user feedback to learn which models actually perform well for different query types.

---

## Implementation Roadmap

### Completed
- [x] Phase 1: ChromaDB Foundation
- [x] Phase 1.5: Smart Cache
- [x] Phase 2: Smart Augmentor (search + scrape)
- [x] Phase 2.5: Model Intelligence

### In Progress
- [ ] Additional scraping providers (Jina, Firecrawl)

### Planned
- [ ] Phase 3: Smart RAG (document context)
- [ ] Phase 4: Smart Context (unified multi-source)
- [ ] Additional search providers
- [ ] Custom context templates

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
