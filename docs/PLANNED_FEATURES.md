# Planned Features

This document outlines planned features for LLM Relay.

---

## Architecture Overview: ChromaDB as Unified Vector Store

ChromaDB serves as the unified backend for all context augmentation features, providing semantic search across multiple data sources.

```
┌─────────────────────────────────────────────────────────────────┐
│                          ChromaDB                                │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ Collection:      │  │ Collection:      │  │ Collection:    │ │
│  │ "web_cache"      │  │ "docs_{rag_id}"  │  │ "docs_{...}"   │ │
│  │                  │  │                  │  │                │ │
│  │ - Search results │  │ - User docs      │  │ - Other RAG    │ │
│  │ - Scraped pages  │  │ - PDFs, DOCX     │  │   collections  │ │
│  │ - TTL expiry     │  │ - Nextcloud/etc  │  │                │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
            │                      │
            ▼                      ▼
    ┌───────────────┐      ┌───────────────┐
    │ Smart         │      │ Smart         │
    │ Augmentor     │      │ RAG           │
    │ (web search)  │      │ (documents)   │
    └───────────────┘      └───────────────┘
            │                      │
            └──────────┬───────────┘
                       ▼
              ┌───────────────┐
              │ Smart Context │  (Future: unified)
              │ (web + docs)  │
              └───────────────┘
```

### Why ChromaDB?
- Open source, runs in Docker
- Handles embeddings automatically (sentence-transformers)
- Semantic search with metadata filtering
- Persistent storage
- Good Python client
- Single dependency for all context features

### Environment Variables
```bash
CHROMA_URL=http://localhost:8000  # ChromaDB server URL
CHROMA_COLLECTION_PREFIX=llmrelay_  # Namespace for collections
```

---

## Model Intelligence (Smart Router Enhancement)

**Status:** Planned  
**Priority:** Medium  
**Dependencies:** ChromaDB, Search Provider (SearXNG or similar)

### Overview
Enhance Smart Router designator decisions by providing balanced, real-world model assessments instead of relying on static marketing descriptions. Uses web search to gather recent model comparisons, benchmarks, and user feedback, then caches summarized intelligence in ChromaDB.

### The Problem
Current smart router descriptions are often:
- Overly positive marketing-speak ("excels at", "best-in-class")
- Static and quickly outdated
- Lacking comparative context
- Missing real-world performance insights

### The Solution
Periodically search for model reviews/benchmarks and cache balanced summaries:

```
# Current (static, marketing-focused):
"claude-sonnet-4-20250514: Claude Sonnet excels at complex analysis and coding tasks 
              with exceptional instruction-following capabilities."

# With Model Intelligence (balanced, comparative):
"claude-sonnet-4-20250514: Strong at code generation and analysis. More verbose than GPT-4o.
              Better at following complex multi-step instructions. Higher cost
              but more reliable for professional/technical work. Can be slower
              for simple queries where GPT-4o-mini would suffice."
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Admin: Refresh Model Intelligence            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ For each candidate model in Smart Routers:                      │
│   1. Search: "{model_name} review benchmark comparison 2024"    │
│   2. Aggregate top results                                      │
│   3. Summarize via LLM: "Create balanced assessment..."         │
│   4. Store in ChromaDB collection: "model_intelligence"         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Smart Router Request                         │
│                                                                 │
│ Designator prompt now includes:                                 │
│ - Static descriptions (fallback)                                │
│ - Model intelligence from ChromaDB (if available)               │
│ - More balanced view → better routing decisions                 │
└─────────────────────────────────────────────────────────────────┘
```

### ChromaDB Integration
```python
# Collection: "model_intelligence"
{
    "id": "anthropic/claude-sonnet-4-20250514",
    "content": "claude-sonnet-4-20250514 balanced assessment for semantic search",
    "metadata": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "intelligence": "Strong at code and analysis. More verbose than...",
        "strengths": ["code generation", "instruction following", "long context"],
        "weaknesses": ["verbosity", "cost", "speed on simple tasks"],
        "best_for": ["professional work", "complex analysis", "coding"],
        "avoid_for": ["simple queries", "cost-sensitive tasks"],
        "sources": ["url1", "url2", "url3"],
        "search_queries": ["claude sonnet review 2024", ...],
        "generated_at": "2024-01-07T12:00:00Z",
        "expires_at": "2024-01-14T12:00:00Z",  # Refresh weekly
    }
}
```

### Designator Prompt Enhancement
```python
# Current designator prompt
DESIGNATOR_PROMPT = """
You are a model router. Select the best model for this query.

CANDIDATES:
{candidates_with_descriptions}

Query: {query}
"""

# Enhanced with Model Intelligence
DESIGNATOR_PROMPT_WITH_INTELLIGENCE = """
You are a model router. Select the best model for this query.

CANDIDATES:
{for each candidate}
- {model_name}: {static_description}
  Intelligence: {model_intelligence_from_chromadb}
{end for}

Query: {query}
"""
```

### Admin UI Features
- "Refresh Model Intelligence" button (manual trigger)
- Last refresh timestamp
- View cached intelligence per model
- Enable/disable per Smart Router
- Configure refresh frequency (if auto-refresh enabled)

### Database Changes
```python
# Add to SmartRouter model
class SmartRouter(Base):
    # ... existing fields ...
    
    # Model Intelligence (optional enhancement)
    use_model_intelligence: bool = False  # Opt-in per router
```

### Settings
```python
# Global settings for Model Intelligence
MODEL_INTELLIGENCE_ENABLED = True  # Master switch
MODEL_INTELLIGENCE_SEARCH_PROVIDER = "searxng"  # Which search to use
MODEL_INTELLIGENCE_SUMMARIZER_MODEL = "openai/gpt-4o-mini"  # LLM for summaries
MODEL_INTELLIGENCE_TTL_DAYS = 7  # How often to refresh
MODEL_INTELLIGENCE_MAX_SOURCES = 5  # Sources per model
```

### Implementation Phases
1. Add `use_model_intelligence` field to SmartRouter model
2. Create "model_intelligence" ChromaDB collection wrapper
3. Implement intelligence gathering:
   - Search provider integration (reuse from Smart Augmentor)
   - Result aggregation
   - LLM summarization
4. Modify SmartRouterEngine to include intelligence in designator prompt
5. Admin UI: refresh button, view intelligence, per-router toggle
6. Optional: Background job for auto-refresh

### Optional: User Feedback Loop (Future)
Store routing outcomes and user feedback to learn which models actually perform well:
```python
# Collection: "routing_feedback"
{
    "id": "request_id",
    "content": "query text",
    "metadata": {
        "router_name": "code-router",
        "selected_model": "anthropic/claude-sonnet",
        "query_type": "code",  # inferred
        "user_rating": 5,  # if feedback collected
        "response_time_ms": 1200,
        "created_at": "...",
    }
}
```

This creates a self-improving system where routing decisions get better over time based on actual usage patterns.

---

## Smart Augmentor (Web Context)

**Status:** Planned  
**Priority:** High

### Overview
Context augmentation that intelligently fetches web search results and/or scrapes URLs to enhance LLM requests with up-to-date information.

### How It Works
1. User requests model by augmentor name (e.g., "research-assistant")
2. Designator LLM analyzes query and decides augmentation type:
   - `direct` - pass through unchanged
   - `search:query terms` - search via SearXNG/Perplexity, inject results
   - `scrape:url1,url2` - fetch URLs mentioned in query, inject content
   - `search+scrape:query` - search then scrape top results
3. Results cached in ChromaDB for semantic reuse
4. Context is injected into system prompt
5. Request forwarded to configured target model

### Key Features
- Extensible search provider system (SearXNG, Perplexity, future: Tavily, Brave)
- Tiered web scraping (built-in or external services)
- ChromaDB caching with TTL-based expiry
- Semantic deduplication (find similar cached results)
- Accessed by name like aliases
- Admin UI similar to Smart Routers
- Cost tracking via existing tag system

### Web Scraping Providers

Tiered approach - simple built-in for most cases, external services for complex sites:

| Provider | Description | Best For |
|----------|-------------|----------|
| `builtin` | httpx + BeautifulSoup | Documentation, blogs, Wikipedia, most content sites |
| `jina` | Jina Reader API (`r.jina.ai`) | JS-heavy sites, cleaner markdown output, free |
| `firecrawl` | Firecrawl API (self-hostable) | Complex sites, structured extraction |
| `custom` | User-provided endpoint | Custom scraping infrastructure |

```python
# Built-in scraper (default)
# Simple, fast, no external dependencies
async def scrape_builtin(url: str) -> ScrapedContent:
    response = await httpx.get(url, follow_redirects=True)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove scripts, styles, nav, footer
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    return ScrapedContent(url=url, title=soup.title.string, content=text)

# Jina Reader (external, free)
# Returns clean markdown, handles JS rendering
async def scrape_jina(url: str) -> ScrapedContent:
    jina_url = f"https://r.jina.ai/{url}"
    response = await httpx.get(jina_url)
    return ScrapedContent(url=url, content=response.text, format="markdown")

# Firecrawl (external, self-hostable)
# More features: structured data, screenshots, etc.
async def scrape_firecrawl(url: str, api_url: str) -> ScrapedContent:
    response = await httpx.post(f"{api_url}/v0/scrape", json={"url": url})
    data = response.json()
    return ScrapedContent(url=url, content=data["markdown"], metadata=data)

# Custom endpoint (user-provided)
# POST {custom_url} with {"url": "..."} -> {"content": "...", "title": "..."}
async def scrape_custom(url: str, endpoint: str) -> ScrapedContent:
    response = await httpx.post(endpoint, json={"url": url})
    return ScrapedContent(**response.json())
```

**Configuration per SmartAugmentor:**
```python
scrape_provider: str = "builtin"  # "builtin" | "jina" | "firecrawl" | "custom"
scrape_provider_url: str | None   # Required for "firecrawl" and "custom"
```

**Environment variables for global defaults:**
```bash
SCRAPE_PROVIDER=builtin           # Default scrape provider
JINA_API_KEY=                     # Optional, for higher Jina rate limits
FIRECRAWL_API_URL=                # Self-hosted Firecrawl URL
FIRECRAWL_API_KEY=                # Firecrawl API key
```

### ChromaDB Integration
```python
# Collection: "web_cache"
# Each document contains:
{
    "id": "hash_of_url_or_query",
    "content": "scraped text or search result",
    "metadata": {
        "type": "search_result" | "scraped_page",
        "url": "https://...",
        "query": "original search query",  # if search result
        "title": "Page Title",
        "fetched_at": "2024-01-07T12:00:00Z",
        "expires_at": "2024-01-08T12:00:00Z",  # TTL
    }
}

# Before fetching, check for semantically similar cached content
similar = collection.query(query_texts=[user_query], n_results=5)
if similar and not expired:
    use cached content
else:
    fetch fresh, store in ChromaDB
```

### Database Schema
```python
class SmartAugmentor(Base):
    __tablename__ = "smart_augmentors"
    
    id: int
    name: str  # unique, e.g., "research-assistant"
    designator_model: str  # "openai/gpt-4o-mini"
    purpose: str  # Context for augmentation decisions
    target_model: str  # Where to send augmented request
    
    # Search config
    search_provider: str  # "searxng" | "perplexity"
    search_provider_url: str | None  # Override default URL
    max_search_results: int = 5
    
    # Scrape config
    scrape_provider: str = "builtin"  # "builtin" | "jina" | "firecrawl" | "custom"
    scrape_provider_url: str | None  # Required for "firecrawl" and "custom"
    max_scrape_urls: int = 3
    
    # Context config
    max_context_tokens: int = 4000
    
    # Caching config
    cache_ttl_hours: int = 24  # How long to cache results
    use_semantic_cache: bool = True  # Check for similar cached queries
    
    tags_json: str
    enabled: bool
```

### Implementation Phases
1. Database layer (model, CRUD, migration)
2. ChromaDB client wrapper
3. Search providers (SearXNG, Perplexity)
4. Web scraper
5. Augmentation engine & registry integration
6. Admin UI

---

## Smart RAG (Document Context)

**Status:** Planned  
**Priority:** Medium

### Overview
Retrieval-Augmented Generation that extracts relevant information from user-provided document collections and injects it as context for queries.

### How It Works
1. User configures document source (Nextcloud, WebDAV, local folder)
2. Documents are indexed into ChromaDB collection
3. User requests model by RAG name (e.g., "docs-assistant")
4. Designator analyzes query and decides if RAG retrieval is needed
5. Relevant document chunks retrieved via semantic search
6. Context injected into system prompt
7. Request forwarded to target model

### Document Source Connectors

#### WebDAV (Priority 1)
- Standard protocol, works with Nextcloud, ownCloud, many others
- Simple file listing and retrieval
- Wide compatibility

#### Nextcloud API (Priority 2)
- Native integration with Nextcloud
- Access to shares, tags, metadata
- Could leverage Nextcloud's own AI features

#### Local Folder (Priority 3)
- Mount point or local path
- Simplest for self-hosted setups
- Good for testing

### Document Processing Pipeline
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│ Source      │ -> │ Fetch       │ -> │ Extract     │ -> │ Chunk    │
│ (WebDAV/..) │    │ Document    │    │ Text        │    │ Content  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────────┘
                                                               │
                                                               ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐
│ Query at    │ <- │ Store in    │ <- │ Generate Embeddings         │
│ Runtime     │    │ ChromaDB    │    │ (ChromaDB default model or  │
└─────────────┘    └─────────────┘    │  custom embedding provider) │
                                      └─────────────────────────────┘
```

### Text Extraction
- PDF: `pypdf` or `pdfplumber`
- DOCX: `python-docx`
- Plain text, Markdown, HTML: built-in
- Future: OCR for scanned documents

### ChromaDB Integration
```python
# Collection: "docs_{rag_id}" (one per SmartRAG config)
# Each document chunk contains:
{
    "id": "hash_of_source_path_and_chunk_index",
    "content": "chunk text content",
    "metadata": {
        "source_path": "/documents/report.pdf",
        "source_name": "report.pdf",
        "chunk_index": 3,
        "total_chunks": 12,
        "file_type": "pdf",
        "file_size": 102400,
        "modified_at": "2024-01-07T12:00:00Z",
        "indexed_at": "2024-01-07T14:00:00Z",
    }
}

# Retrieval
results = collection.query(
    query_texts=[user_query],
    n_results=max_chunks,
    where={"source_path": {"$in": allowed_paths}}  # optional filtering
)
```

### Database Schema
```python
class SmartRAG(Base):
    __tablename__ = "smart_rags"
    
    id: int
    name: str  # unique, e.g., "docs-assistant"
    designator_model: str
    purpose: str
    target_model: str
    
    # Document source config
    source_type: str  # "webdav" | "nextcloud" | "local"
    source_url: str | None  # WebDAV/Nextcloud URL
    source_credentials_json: str | None  # encrypted username/password
    source_path: str  # folder/path to index
    
    # Indexing config
    file_patterns: str  # "*.pdf,*.docx,*.md" - which files to index
    chunk_size: int = 1000  # tokens per chunk
    chunk_overlap: int = 200  # overlap between chunks
    
    # Retrieval config
    max_chunks: int = 5
    max_context_tokens: int = 4000
    similarity_threshold: float = 0.7
    
    # Index status
    chroma_collection: str | None  # ChromaDB collection name
    last_indexed: datetime | None
    document_count: int = 0
    chunk_count: int = 0
    
    tags_json: str
    enabled: bool
```

### Admin UI Features
- Configure document source connection
- Test connection button
- Trigger manual re-index
- View indexing status and stats
- Browse indexed documents
- Preview chunks

### Implementation Phases
1. Database layer (model, CRUD, migration)
2. ChromaDB collection management
3. WebDAV connector
4. Document processing (text extraction, chunking)
5. Indexing pipeline (background job)
6. RAG engine & registry integration
7. Admin UI

---

## Smart Context (Future: Unified)

**Status:** Concept  
**Priority:** Low (after Augmentor + RAG)

### Overview
A unified context provider that can pull from multiple sources (web search, documents, cached results) based on the query.

### How It Works
1. User configures which sources to include (web search, specific RAG collections)
2. Designator analyzes query and decides which sources are relevant
3. Retrieves from multiple ChromaDB collections in parallel
4. Merges and ranks results
5. Injects combined context

### Example Config
```python
class SmartContext(Base):
    name: str  # "universal-assistant"
    designator_model: str
    target_model: str
    
    # Sources to query
    enable_web_search: bool = True
    web_augmentor_id: int | None  # FK to SmartAugmentor
    rag_ids: list[int]  # FKs to SmartRAGs to include
    
    # Merging config
    max_total_context_tokens: int = 6000
    source_weights: dict  # {"web": 0.3, "docs": 0.7}
```

---

## Shared Components

### ChromaDB Client (`context/chroma_client.py`)
```python
class ChromaClient:
    """Wrapper for ChromaDB operations."""
    
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("CHROMA_URL", "http://localhost:8000")
        self.client = chromadb.HttpClient(host=self.url)
    
    def get_or_create_collection(self, name: str) -> Collection:
        """Get or create a collection with standard settings."""
        pass
    
    def add_documents(self, collection: str, docs: list[dict]):
        """Add documents with embeddings."""
        pass
    
    def query(self, collection: str, query: str, n_results: int, filters: dict = None):
        """Semantic search with optional metadata filters."""
        pass
    
    def delete_expired(self, collection: str):
        """Remove documents past their TTL."""
        pass
```

### Context Builder (`context/builder.py`)
```python
def format_web_results(results: list[dict]) -> str:
    """Format search/scrape results for injection."""
    pass

def format_document_chunks(chunks: list[dict]) -> str:
    """Format RAG chunks for injection."""
    pass

def merge_contexts(web: str, docs: str, max_tokens: int) -> str:
    """Merge multiple context sources within token limit."""
    pass
```

---

## Smart Cache (Response Caching)

**Status:** Planned  
**Priority:** High (Phase 1.5 - quick win after foundation)

### Overview
A smart router type that caches LLM responses and returns cached answers for semantically similar queries. Reduces token usage and costs for repeated/similar questions.

### How It Works
```
User Query: "What's the capital of France?"
    │
    ▼
┌─────────────────────────────────────────┐
│ ChromaDB: "cache_{smart_cache_id}"      │
│                                         │
│ Semantic search for similar queries     │
│ Found: "capital of France" (0.96 sim)   │
│ → Cache HIT                             │
└─────────────────────────────────────────┘
    │
    ▼
Return cached response (no LLM call, $0 cost!)
```

### Why a Router Type?
Users decide when caching makes sense:
- ✅ Factual Q&A assistants - high cache value
- ✅ Documentation lookups - same questions repeated
- ✅ Expensive models (Opus, o3) - cost savings
- ❌ Creative writing - users want variety
- ❌ Personal assistants - context-dependent
- ❌ Code with specific context - too variable

### Key Features
- Semantic similarity matching (not just exact match)
- Configurable similarity threshold (0.90 - 0.99)
- TTL-based expiry
- Optional system prompt matching
- Cache hit statistics
- Manual cache clear per router
- Skip cache via header (`X-Skip-Cache: true`)

### ChromaDB Integration
```python
# Collection: "cache_{smart_cache_id}"
{
    "id": "query_hash",
    "content": "query text for embedding similarity",
    "metadata": {
        "query": "What is the capital of France?",
        "system_hash": "abc123",  # hash of system prompt (if match_system=True)
        "response": "The capital of France is Paris...",
        "model": "anthropic/claude-sonnet",
        "input_tokens": 50,
        "output_tokens": 120,
        "cost": 0.0015,
        "created_at": "2024-01-07T12:00:00Z",
        "expires_at": "2024-01-14T12:00:00Z",
        "hit_count": 0,
    }
}

# On request:
results = collection.query(
    query_texts=[user_query],
    n_results=1,
    where={"system_hash": system_hash} if match_system else None
)

if results and results.distances[0] <= (1 - similarity_threshold):
    # Cache HIT - return cached response
    update hit_count
    return cached response
else:
    # Cache MISS - call LLM, store result
    response = call_target_model()
    store in ChromaDB
    return response
```

### Database Schema
```python
class SmartCache(Base):
    __tablename__ = "smart_caches"
    
    id: int
    name: str  # unique, e.g., "cached-claude"
    target_model: str  # Model to call on cache miss
    
    # Cache behavior
    similarity_threshold: float = 0.95  # 0.90-0.99, higher = stricter
    match_system_prompt: bool = True  # Include system prompt in matching
    cache_ttl_days: int = 7  # How long to keep cached responses
    max_cached_tokens: int = 4000  # Don't cache very long responses
    
    # ChromaDB
    chroma_collection: str | None  # Auto-generated collection name
    
    # Stats (updated periodically)
    total_requests: int = 0
    cache_hits: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    
    tags_json: str
    enabled: bool
```

### Resolution Flow
```
SmartRouter → SmartCache → SmartAugmentor → Alias → Provider → Default
```

SmartCache is checked after SmartRouter (routing decisions) but before SmartAugmentor (context enhancement).

### Admin UI Features
- Configure target model and similarity threshold
- View cache statistics (hit rate, tokens saved, cost saved)
- Clear cache button
- Browse cached entries (optional)
- Test query against cache

### Request Flow
```python
class SmartCacheEngine:
    def process(self, messages, system) -> CacheResult:
        # 1. Build cache key from last user message
        query = get_last_user_message(messages)
        system_hash = hash(system) if self.cache.match_system_prompt else None
        
        # 2. Check ChromaDB for similar cached query
        hit = self.check_cache(query, system_hash)
        
        if hit:
            # 3a. Cache HIT - return cached response
            self.record_hit(hit)
            return CacheResult(
                cached=True,
                response=hit.response,
                original_cost=hit.cost,
            )
        else:
            # 3b. Cache MISS - call target model
            resolved = registry.resolve_model(self.cache.target_model)
            response = call_model(resolved, messages, system)
            
            # 4. Store in cache (if response not too long)
            if response.output_tokens <= self.cache.max_cached_tokens:
                self.store_in_cache(query, system_hash, response)
            
            return CacheResult(
                cached=False,
                response=response,
            )
```

### Tracking
- Cache hits logged with `cache_hit=True` in RequestLog
- Original model's cost tracked as "saved"
- Statistics aggregated for admin UI

### Implementation Phases
1. Database model & CRUD
2. ChromaDB cache collection management
3. SmartCacheEngine with hit/miss logic
4. Registry integration
5. Admin UI with stats

---

## Implementation Roadmap

### Phase 1: Foundation (Detailed)

**Goal:** Establish ChromaDB integration and shared utilities that all smart context features will use.

#### 1.1 ChromaDB Client Wrapper

**New file:** `context/chroma.py`

```python
"""
ChromaDB client wrapper for LLM Relay.

Provides a unified interface for all ChromaDB operations across
Smart Cache, Smart Augmentor, and Smart RAG features.
"""

import os
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Singleton client instance
_client: Optional[chromadb.HttpClient] = None


def get_chroma_client() -> chromadb.HttpClient:
    """Get or create ChromaDB client singleton."""
    global _client
    if _client is None:
        url = os.environ.get("CHROMA_URL", "http://localhost:8000")
        _client = chromadb.HttpClient(
            host=url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(url.split(":")[-1]) if ":" in url.split("/")[-1] else 8000,
        )
        logger.info(f"Connected to ChromaDB at {url}")
    return _client


def get_collection(name: str, create: bool = True):
    """Get a collection by name, optionally creating it."""
    client = get_chroma_client()
    prefix = os.environ.get("CHROMA_COLLECTION_PREFIX", "llmrelay_")
    full_name = f"{prefix}{name}"
    
    if create:
        return client.get_or_create_collection(name=full_name)
    return client.get_collection(name=full_name)


def delete_collection(name: str) -> bool:
    """Delete a collection by name."""
    client = get_chroma_client()
    prefix = os.environ.get("CHROMA_COLLECTION_PREFIX", "llmrelay_")
    full_name = f"{prefix}{name}"
    
    try:
        client.delete_collection(name=full_name)
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection {full_name}: {e}")
        return False


def is_chroma_available() -> bool:
    """Check if ChromaDB is configured and reachable."""
    try:
        client = get_chroma_client()
        client.heartbeat()
        return True
    except Exception:
        return False


class CollectionWrapper:
    """
    Wrapper for a ChromaDB collection with convenience methods.
    
    Usage:
        cache = CollectionWrapper("response_cache")
        cache.add(id="abc", content="query text", metadata={...})
        results = cache.query("similar query", n_results=5)
    """
    
    def __init__(self, name: str):
        self.name = name
        self._collection = None
    
    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_collection(self.name)
        return self._collection
    
    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ):
        """Add documents to the collection."""
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> dict:
        """Query for similar documents."""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )
    
    def get(self, ids: list[str]) -> dict:
        """Get documents by ID."""
        return self.collection.get(ids=ids)
    
    def delete(self, ids: list[str] | None = None, where: dict | None = None):
        """Delete documents by ID or filter."""
        self.collection.delete(ids=ids, where=where)
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()
    
    def clear(self):
        """Delete all documents in the collection."""
        # ChromaDB doesn't have a clear method, so we delete and recreate
        delete_collection(self.name)
        self._collection = None  # Force recreation on next access
```

#### 1.2 Context Builder Utilities

**New file:** `context/builder.py`

```python
"""
Context building utilities for injecting retrieved content into prompts.
"""

def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Truncated due to length limit]"


def format_context_header(source: str) -> str:
    """Format a header for injected context."""
    return f"## Context from {source}\n\n"


def inject_context_to_system(
    system: str | None,
    context: str,
    position: str = "prepend",  # "prepend" | "append"
) -> str:
    """Inject context into system prompt."""
    if not context:
        return system or ""
    
    if position == "prepend":
        if system:
            return f"{context}\n\n---\n\n{system}"
        return context
    else:  # append
        if system:
            return f"{system}\n\n---\n\n{context}"
        return context


def format_as_context_message(context: str) -> dict:
    """Format context as a user message for injection."""
    return {
        "role": "user",
        "content": f"[System Context]\n\n{context}\n\n[End Context]"
    }
```

#### 1.3 Module Structure

**New directory:** `context/`

```
context/
  __init__.py          # Exports main utilities
  chroma.py            # ChromaDB client wrapper
  builder.py           # Context formatting utilities
```

**`context/__init__.py`:**
```python
"""
Context module for LLM Relay.

Provides ChromaDB integration and utilities for Smart Cache,
Smart Augmentor, and Smart RAG features.
"""

from .chroma import (
    get_chroma_client,
    get_collection,
    delete_collection,
    is_chroma_available,
    CollectionWrapper,
)
from .builder import (
    estimate_tokens,
    truncate_to_tokens,
    format_context_header,
    inject_context_to_system,
    format_as_context_message,
)

__all__ = [
    # ChromaDB
    "get_chroma_client",
    "get_collection", 
    "delete_collection",
    "is_chroma_available",
    "CollectionWrapper",
    # Builder
    "estimate_tokens",
    "truncate_to_tokens",
    "format_context_header",
    "inject_context_to_system",
    "format_as_context_message",
]
```

#### 1.4 Admin API Endpoint

**Add to `admin/app.py`:**

```python
@admin.route("/api/chroma/status", methods=["GET"])
@require_auth_api
def chroma_status():
    """Check ChromaDB connection status."""
    from context import is_chroma_available, get_chroma_client
    
    available = is_chroma_available()
    
    result = {
        "available": available,
        "url": os.environ.get("CHROMA_URL", "http://localhost:8000"),
    }
    
    if available:
        try:
            client = get_chroma_client()
            collections = client.list_collections()
            prefix = os.environ.get("CHROMA_COLLECTION_PREFIX", "llmrelay_")
            our_collections = [c.name for c in collections if c.name.startswith(prefix)]
            result["collections"] = our_collections
            result["collection_count"] = len(our_collections)
        except Exception as e:
            result["error"] = str(e)
    
    return jsonify(result)
```

#### 1.5 Requirements Update

**Add to `requirements.txt`:**
```
chromadb>=0.4.0
```

#### 1.6 Environment Variables

**Document in README/env.example:**
```bash
# ChromaDB Configuration (optional - required for Smart Cache/Augmentor/RAG)
CHROMA_URL=http://localhost:8000
CHROMA_COLLECTION_PREFIX=llmrelay_
```

#### Implementation Checklist

- [ ] Create `context/` directory
- [ ] Implement `context/chroma.py` with client wrapper
- [ ] Implement `context/builder.py` with utilities  
- [ ] Create `context/__init__.py` with exports
- [ ] Add `chromadb` to requirements.txt
- [ ] Add `/api/chroma/status` endpoint to admin/app.py
- [ ] Test ChromaDB connection with local instance
- [ ] Document environment variables

#### Testing

```python
# Manual testing script
from context import is_chroma_available, CollectionWrapper

# Test connection
print(f"ChromaDB available: {is_chroma_available()}")

# Test collection operations
test = CollectionWrapper("test_collection")
test.add(
    ids=["doc1"],
    documents=["What is the capital of France?"],
    metadatas=[{"answer": "Paris"}]
)
results = test.query("capital of France", n_results=1)
print(f"Query results: {results}")
test.clear()
```

### Phase 1.5: Smart Cache
- [ ] Database model & CRUD
- [ ] Cache engine with ChromaDB
- [ ] Registry integration
- [ ] Admin UI with statistics

### Phase 2: Smart Augmentor
- [ ] Database model & CRUD
- [ ] Search providers (SearXNG, Perplexity)
- [ ] Web scraper
- [ ] Web cache collection in ChromaDB
- [ ] Augmentation engine
- [ ] Registry integration
- [ ] Admin UI

### Phase 2.5: Model Intelligence (Smart Router Enhancement)
**Dependencies:** Phase 1 (ChromaDB), Phase 2 (Search Providers)

- [ ] Add `use_model_intelligence` field to SmartRouter model
- [ ] Create `model_intelligence` ChromaDB collection
- [ ] Implement intelligence gathering service
  - [ ] Search integration (reuse from Smart Augmentor)
  - [ ] Result aggregation
  - [ ] LLM summarization with balanced prompt
- [ ] Modify SmartRouterEngine to include intelligence in designator prompt
- [ ] Admin UI
  - [ ] "Refresh Model Intelligence" button
  - [ ] View cached intelligence per model
  - [ ] Per-router enable/disable toggle
- [ ] Optional: Background job for periodic auto-refresh

### Phase 3: Smart RAG
- [ ] Database model & CRUD
- [ ] WebDAV connector
- [ ] Document text extraction
- [ ] Chunking pipeline
- [ ] Indexing background job
- [ ] RAG engine
- [ ] Registry integration
- [ ] Admin UI

### Phase 4: Enhancements
- [ ] Nextcloud connector
- [ ] Local folder connector
- [ ] Smart Context (unified)
- [ ] Custom context templates
- [ ] Re-ranking models

---

## Environment Variables

```bash
# ChromaDB (required for Smart Cache/Augmentor/RAG)
CHROMA_URL=http://localhost:8000
CHROMA_COLLECTION_PREFIX=llmrelay_

# Search Providers
SEARXNG_URL=http://localhost:8888

# Web Scraping Providers
SCRAPE_PROVIDER=builtin              # Default: "builtin" | "jina" | "firecrawl" | "custom"
JINA_API_KEY=                        # Optional, for higher Jina rate limits
FIRECRAWL_API_URL=                   # Self-hosted Firecrawl URL
FIRECRAWL_API_KEY=                   # Firecrawl API key (if using hosted version)

# Document Processing (optional)
MAX_DOCUMENT_SIZE_MB=50
```

## Docker Compose Addition (when ready)

```yaml
chroma:
  image: chromadb/chroma:latest
  container_name: llm-relay-chroma
  volumes:
    - ./chroma-data:/chroma/chroma
  ports:
    - "8000:8000"
  environment:
    - ANONYMIZED_TELEMETRY=false
  restart: unless-stopped
```
