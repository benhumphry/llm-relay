# Live Document Lookup for File-Based Sources

## Overview

Add live document lookup capability to file-based document sources (local filesystem, Paperless, Nextcloud, etc.), enabling full document retrieval when users reference specific files by name.

### Current Behavior

When a user asks "what is the content of INVOICE - OCTOBER 2022.docx?":
1. RAG retrieves chunks that semantically match the query
2. The specific file may be found in metadata, but only chunks are returned
3. User cannot access the full document content

### Proposed Behavior

1. RAG retrieves relevant chunks (as now)
2. System detects file reference intent (explicit filename or "that document")
3. Live lookup resolves filename to doc_id and fetches full content
4. Full document content injected into context

---

## Current State Analysis

### Unified Sources Today

Unified sources (Gmail, Notion, etc.) already combine RAG + Live:
- Line 1291-1293 in `smart_enricher.py`: "unified sources work even when traditional live data is disabled"
- They're added to `live_source_info` based on `supports_live=True` in the plugin class
- Designator can select them via `live_params`

### When Smart Source Selection is OFF

- `use_smart_source_selection=False`: No designator call, no `live_params`
- RAG runs with all linked stores (no token budgets)
- Unified sources are NOT queried (no `live_params` to trigger them)
- **Gap**: File-based stores can't do live lookup without Smart Source Selection

### When Smart Source Selection is ON

- Designator sees unified sources in `live_source_info` 
- Can allocate to them via `live_params: {"store_name": {"query": "..."}}`
- Unified sources queried in `_retrieve_live_context()`
- **Works for API-based sources** (Notion, Gmail), but file-based sources have no unified plugin

---

## Implementation Plan

### Phase 1: DocumentSource Live Lookup Interface

**File: `mcp/sources.py`**

Add live lookup methods to the `DocumentSource` base class:

```python
class DocumentSource(ABC):
    # Existing methods
    @abstractmethod
    def list_documents(self) -> Iterator[DocumentInfo]: ...
    
    @abstractmethod
    def read_document(self, doc_id: str) -> DocumentContent: ...
    
    # NEW: Live lookup capability
    @property
    def supports_live_lookup(self) -> bool:
        """Whether this source supports on-demand document fetch."""
        return True  # Most sources can - they implement read_document()
    
    def resolve_document(
        self, 
        query: str, 
        indexed_metadata: list[dict]
    ) -> Optional[str]:
        """
        Resolve a natural language reference to a doc_id.
        
        Args:
            query: User's reference ("INVOICE - OCTOBER 2022.docx", "the tax PDF")
            indexed_metadata: Metadata from RAG chunks for this store
                              (includes source_path, title, etc.)
        
        Returns:
            doc_id if resolved, None if ambiguous/not found
        """
        # Default implementation: fuzzy match against source_path/title
        return self._fuzzy_match_document(query, indexed_metadata)
    
    def _fuzzy_match_document(
        self, 
        query: str, 
        indexed_metadata: list[dict]
    ) -> Optional[str]:
        """Default fuzzy matching implementation."""
        from rapidfuzz import fuzz, process
        
        # Build candidates from metadata
        candidates = {}
        for meta in indexed_metadata:
            doc_id = meta.get("source_path") or meta.get("doc_id")
            if doc_id:
                # Use filename or title as match target
                title = meta.get("title") or Path(doc_id).name
                candidates[title.lower()] = doc_id
        
        if not candidates:
            return None
        
        # Fuzzy match
        query_lower = query.lower()
        match = process.extractOne(
            query_lower, 
            candidates.keys(),
            scorer=fuzz.token_set_ratio,
            score_cutoff=70
        )
        
        return candidates[match[0]] if match else None
```

### Phase 2: Unified Source Plugin for File-Based Stores

**New file: `builtin_plugins/unified_sources/filesystem.py`**

Create a unified source that wraps file-based DocumentSources:

```python
class FilesystemUnifiedSource(PluginUnifiedSource):
    """Unified source for local filesystem document stores."""
    
    source_type = "local"
    supports_live = True
    supports_rag = True  # Uses existing RAG index
    
    data_type = "documents"
    best_for = "Retrieving full document content by filename"
    description = "Local filesystem with live document lookup"
    
    @classmethod
    def get_param_definitions(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="filename",
                param_type=ParamType.STRING,
                description="Filename or document title to retrieve",
                required=False,
            ),
            ParamDefinition(
                name="query",
                param_type=ParamType.STRING, 
                description="Search query for RAG retrieval",
                required=False,
            ),
        ]
    
    @classmethod
    def get_designator_hint(cls) -> str:
        return (
            "Use 'filename' param when user asks for full content of a specific file. "
            "Use 'query' param for semantic search across documents."
        )
    
    def query(
        self,
        params: dict,
        rag_search_fn: Optional[Callable] = None,
        designator_model: Optional[str] = None,
    ) -> UnifiedQueryResult:
        filename = params.get("filename")
        search_query = params.get("query")
        
        # If filename specified, do live lookup
        if filename:
            return self._fetch_full_document(filename, rag_search_fn)
        
        # Otherwise, just do RAG search
        if search_query and rag_search_fn:
            chunks = rag_search_fn(search_query)
            return UnifiedQueryResult(
                success=True,
                formatted=self._format_chunks(chunks),
                rag_count=len(chunks),
            )
        
        return UnifiedQueryResult(success=False, error="No filename or query provided")
    
    def _fetch_full_document(
        self, 
        filename: str,
        rag_search_fn: Optional[Callable]
    ) -> UnifiedQueryResult:
        """Fetch full document content by filename."""
        # Get indexed metadata from RAG to help resolve filename
        indexed_metadata = []
        if rag_search_fn:
            # Search for the filename to get metadata
            chunks = rag_search_fn(filename, limit=20)
            indexed_metadata = [c.metadata for c in chunks if c.metadata]
        
        # Resolve filename to doc_id
        source = self._get_document_source()
        doc_id = source.resolve_document(filename, indexed_metadata)
        
        if not doc_id:
            return UnifiedQueryResult(
                success=False,
                error=f"Could not find document matching '{filename}'"
            )
        
        # Fetch full document
        try:
            content = source.read_document(doc_id)
            
            # Parse document content (reuse Docling for complex formats)
            text = self._extract_text(content)
            
            return UnifiedQueryResult(
                success=True,
                formatted=f"## Full content of {filename}\n\n{text}",
                live_count=1,
            )
        except Exception as e:
            return UnifiedQueryResult(
                success=False,
                error=f"Failed to read {filename}: {e}"
            )
    
    def _extract_text(self, content: DocumentContent) -> str:
        """Extract text from document content."""
        if content.text:
            return content.text
        
        if content.binary:
            # Use indexer's document processing for binary content
            from rag.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            chunks = indexer._process_content(
                content, 
                chunk_size=50000,  # Large chunk to get full doc
                chunk_overlap=0
            )
            return "\n\n".join(c["content"] for c in chunks)
        
        return ""
```

### Phase 3: Additional File-Based Unified Sources

**`builtin_plugins/unified_sources/paperless.py`**

```python
class PaperlessUnifiedSource(PluginUnifiedSource):
    source_type = "paperless"
    supports_live = True
    supports_rag = True
    
    # Similar structure to FilesystemUnifiedSource
    # Uses Paperless API for document fetch
```

**`builtin_plugins/unified_sources/nextcloud.py`**

```python
class NextcloudUnifiedSource(PluginUnifiedSource):
    source_type = "nextcloud"
    supports_live = True
    supports_rag = True
    
    # Uses WebDAV for document fetch
```

### Phase 4: Enricher Integration

**Modify `routing/smart_enricher.py`**

The existing unified source handling should work automatically once plugins are registered. Key integration points:

1. **Source discovery** (lines 1289-1397): Already scans document stores for unified plugins
2. **Designator prompt**: Already includes unified sources in `live_source_info`
3. **Query execution** (lines 3010-3160): Already routes to unified source `query()` method

**One change needed**: When Smart Source Selection is OFF, enable a simpler "on-demand lookup" mode:

```python
# In _retrieve_rag_context(), after retrieving chunks:

# Check if any retrieved chunks reference specific documents
# that the user might want full content of
if self._detect_document_lookup_intent(query, chunks):
    # Get unique source docs from chunk metadata
    doc_refs = self._extract_doc_references(chunks)
    
    # For each document store that supports live lookup,
    # offer to fetch full content
    for store_id, doc_id in doc_refs:
        store = self._get_store_by_id(store_id)
        if self._store_supports_live_lookup(store):
            # Fetch full document
            full_content = self._fetch_document(store, doc_id)
            if full_content:
                # Append to context
                rag_context += f"\n\n## Full Document: {doc_id}\n{full_content}"
```

### Phase 5: Intent Detection

Add document lookup intent detection:

```python
def _detect_document_lookup_intent(self, query: str, chunks: list) -> bool:
    """Detect if user wants full document content."""
    patterns = [
        r"(full |entire |complete |whole )?content of",
        r"what('s| is) in",
        r"show me",
        r"read( me)?",
        r"open",
        r"(can you )?(get|fetch|retrieve|pull)",
    ]
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in patterns)

def _extract_doc_references(self, chunks: list) -> list[tuple[int, str]]:
    """Extract document references from chunk metadata."""
    refs = []
    seen = set()
    for chunk in chunks:
        if chunk.metadata:
            store_id = chunk.metadata.get("store_id")
            doc_id = chunk.metadata.get("source_path") or chunk.metadata.get("doc_id")
            if store_id and doc_id and (store_id, doc_id) not in seen:
                refs.append((store_id, doc_id))
                seen.add((store_id, doc_id))
    return refs
```

---

## Source Support Matrix

| Source Type | Indexed | Live Lookup | Resolution Method |
|-------------|---------|-------------|-------------------|
| `local` | ✓ | ✓ | Fuzzy match on filename |
| `paperless` | ✓ | ✓ | API search + fuzzy match on title |
| `nextcloud` | ✓ | ✓ | WebDAV path match |
| `notion` | ✓ | ✓ (existing) | Page title match |
| `mcp:github` | ✓ | ✓ | Repo + path match |
| `website` | ✓ | ✓ | URL match + re-scrape |
| `mcp:gdrive` | ✓ | ✓ | File name/ID match |
| `mcp:gmail` | ✓ | ✓ (existing) | Message ID |

---

## Questions to Resolve

### 1. Smart Source Selection Dependency

**Current**: Unified sources only queried when `live_params` provided by designator.

**Options**:
- A) **Require Smart Source Selection** for live lookup - simpler, consistent
- B) **Add fallback mode** when SSS is OFF - detect intent, auto-lookup
- C) **Separate toggle** for live document lookup on Smart Alias

**Recommendation**: Option A initially. Live lookup is a "smart" feature that benefits from designator orchestration. Document the requirement.

### 2. Token Budget for Full Documents

Full documents can be large. Options:
- A) **Hard limit** (e.g., 8000 tokens) with truncation
- B) **Use remaining context budget** after RAG/Web
- C) **Designator allocates** specific budget for document lookup

**Recommendation**: Option B - use remaining budget, with configurable max per document.

### 3. Multiple Document References

What if user asks about multiple documents?
- A) **Fetch all** up to budget
- B) **Ask for clarification** if ambiguous
- C) **Fetch most relevant** based on RAG scores

**Recommendation**: Option C - use RAG relevance scores to prioritize.

---

## Implementation Order

1. **Phase 1**: Add `resolve_document()` to DocumentSource base class
2. **Phase 2**: Create `FilesystemUnifiedSource` plugin  
3. **Phase 3**: Test with local filesystem stores
4. **Phase 4**: Add Paperless and Nextcloud unified sources
5. **Phase 5**: Document the feature and requirements

---

## Files to Create/Modify

### New Files
- `builtin_plugins/unified_sources/filesystem.py`
- `builtin_plugins/unified_sources/paperless.py`
- `builtin_plugins/unified_sources/nextcloud.py`

### Modified Files
- `mcp/sources.py` - Add `resolve_document()` method
- `plugin_base/document_source.py` - Add live lookup interface to plugin base
- `routing/smart_enricher.py` - Minor tweaks for document lookup intent (optional)

### Dependencies
- `rapidfuzz` - For fuzzy filename matching (add to requirements.txt)
