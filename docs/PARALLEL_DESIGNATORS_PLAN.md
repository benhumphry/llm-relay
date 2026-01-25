# Parallel Designators Implementation Plan

## Overview

Split the current unified designator into 4 parallel, focused designators that can each use a different model and run concurrently for reduced latency and improved decision quality.

## Current Problem

The unified designator in `routing/smart_enricher.py:_select_sources()` (lines ~1027-1400) makes ONE call that handles ALL decisions:
- Model routing (selecting from candidates)
- RAG store selection and token budgets
- Web search query generation  
- Live data source selection and parameters

**Issues:**
- Single model handles everything - cannot use specialized models
- Serial execution - all decisions wait for one LLM call
- Prompt bloat (~1500+ tokens) makes decisions less focused
- One model may excel at routing but be poor at live data extraction

## Solution Architecture

### 4 Parallel Designators

| Designator | Responsibility | Output |
|------------|----------------|--------|
| **Router** | Select best model from candidates | `selected_model` |
| **RAG** | Select stores and allocate token budgets | `{store_id: tokens, ...}` |
| **Web** | Decide if web needed, generate search query | `{use_web, search_query}` |
| **Live** | Select live sources and extract parameters | `{source: [{params}], ...}` |

### Configuration

Each designator is **optional**. New SmartAlias fields:
- `router_designator_model` - Model for routing decisions
- `rag_designator_model` - Model for RAG store selection
- `web_designator_model` - Model for web search decisions
- `live_designator_model` - Model for live data extraction

**Fallback behavior:** If a domain-specific designator is not configured, that domain is handled by the unified `designator_model` (current behavior preserved).

---

## Implementation Steps

### Phase 1: Database Changes ✅ PARTIALLY DONE

**File: `db/models.py`** (~line 1448, after `designator_model`) - ✅ DONE

Added 4 new columns to SmartAlias:
```python
# Parallel designators (optional - if not set, falls back to unified designator_model)
router_designator_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
rag_designator_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
web_designator_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
live_designator_model: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
```

**File: `db/connection.py`** (~line 1920 area, after action_notes_store_id migration) - TODO

Add migration:
```python
# Migration: Add parallel designator columns to smart_aliases (v2.0.2)
if "smart_aliases" in inspector.get_table_names():
    columns = [c["name"] for c in inspector.get_columns("smart_aliases")]
    
    parallel_designator_cols = [
        ("router_designator_model", "VARCHAR(150)"),
        ("rag_designator_model", "VARCHAR(150)"),
        ("web_designator_model", "VARCHAR(150)"),
        ("live_designator_model", "VARCHAR(150)"),
    ]
    
    cols_to_add = [(name, dtype) for name, dtype in parallel_designator_cols if name not in columns]
    
    if cols_to_add:
        logger.info(f"Adding parallel designator columns to smart_aliases (v2.0.2)")
        with engine.connect() as conn:
            for col_name, col_type in cols_to_add:
                conn.execute(text(f"ALTER TABLE smart_aliases ADD COLUMN {col_name} {col_type}"))
            conn.commit()
```

**File: `db/smart_aliases.py`** - TODO

Update `create_smart_alias()` function signature (~line 50):
```python
def create_smart_alias(
    # ... existing params ...
    designator_model: str | None = None,
    router_designator_model: str | None = None,  # ADD
    rag_designator_model: str | None = None,     # ADD
    web_designator_model: str | None = None,     # ADD
    live_designator_model: str | None = None,    # ADD
    # ... rest of params ...
)
```

Update SmartAlias instantiation in create_smart_alias():
```python
alias = SmartAlias(
    # ... existing fields ...
    router_designator_model=router_designator_model,
    rag_designator_model=rag_designator_model,
    web_designator_model=web_designator_model,
    live_designator_model=live_designator_model,
    # ... rest ...
)
```

Update `update_smart_alias()` function similarly.

Update `_alias_to_detached()` to copy these fields:
```python
detached.router_designator_model = alias.router_designator_model
detached.rag_designator_model = alias.rag_designator_model
detached.web_designator_model = alias.web_designator_model
detached.live_designator_model = alias.live_designator_model
```

---

### Phase 2: Focused Designator Functions

**File: `routing/smart_enricher.py`**

Add 4 new methods to `SmartEnricherEngine` class (after `_select_sources` method, ~line 1400):

#### 2.1 `_designate_routing()`

```python
def _designate_routing(
    self,
    query: str,
    messages: list[dict] | None,
    candidates: list[dict],
    token_count: int,
    has_images: bool,
) -> tuple[str | None, dict | None]:
    """
    Focused routing designator - selects best model from candidates.
    
    Returns:
        (selected_model, usage_dict) or (None, None) on failure
    """
    model = getattr(self.enricher, "router_designator_model", None)
    if not model:
        return None, None  # Caller will include in unified call
    
    # Format candidates
    models_section = []
    for c in candidates:
        line = f"- {c['model']}"
        caps = []
        if c.get("context_window"):
            caps.append(f"context: {c['context_window']}")
        if c.get("capabilities"):
            caps.append(f"caps: {','.join(c['capabilities'])}")
        if caps:
            line += f" ({', '.join(caps)})"
        models_section.append(line)
    
    # Get query preview from last user message
    query_preview = query[:500] if query else ""
    if messages:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    query_preview = content[:500]
                break
    
    prompt = f"""Select the best model for this query.

PURPOSE: {self.enricher.purpose or 'General assistant'}

AVAILABLE MODELS:
{chr(10).join(models_section)}

QUERY INFO:
- Estimated tokens: {token_count}
- Contains images: {has_images}
- Query: {query_preview}

Respond with ONLY the model identifier (e.g., "anthropic/claude-sonnet-4"). No explanation."""

    try:
        resolved = self.registry._resolve_actual_model(model)
        provider = resolved.provider
        
        result = provider.chat_completion(
            model=resolved.model_id,
            messages=[{"role": "user", "content": prompt}],
            system=None,
            options={"max_tokens": 100, "temperature": 0},
        )
        
        selected = result.get("content", "").strip()
        usage = {
            "prompt_tokens": result.get("input_tokens", 0),
            "completion_tokens": result.get("output_tokens", 0),
            "purpose": "routing",
        }
        
        # Validate against candidates
        valid_models = [c["model"] for c in candidates]
        if selected in valid_models:
            logger.info(f"Parallel router designator selected: {selected}")
            return selected, usage
        else:
            logger.warning(f"Router designator returned invalid model: {selected}")
            return None, usage
            
    except Exception as e:
        logger.error(f"Router designator failed: {e}")
        return None, None
```

#### 2.2 `_designate_rag()`

```python
def _designate_rag(
    self,
    query: str,
    store_info: list[dict],
    total_budget: int,
    preview_samples: dict[int, list[str]] | None = None,
) -> tuple[dict[int, int] | None, dict | None]:
    """
    Focused RAG designator - selects stores and allocates token budgets.
    
    Returns:
        (store_budgets dict, usage_dict) or (None, None) on failure
    """
    model = getattr(self.enricher, "rag_designator_model", None)
    if not model:
        return None, None
    
    # Format store descriptions
    stores_section = []
    for store in store_info:
        store_id = store["id"]
        lines = [f"- store_{store_id}: {store['name']}"]
        if store.get("description"):
            lines.append(f"  Description: {store['description']}")
        if store.get("themes"):
            lines.append(f"  Themes: {store['themes']}")
        if preview_samples and store_id in preview_samples:
            lines.append(f"  Sample content: {preview_samples[store_id][:2]}")
        stores_section.append("\n".join(lines))
    
    prompt = f"""Allocate {total_budget} tokens across document stores for this query.

DOCUMENT STORES:
{chr(10).join(stores_section)}

USER QUERY: {query}

Respond with JSON only: {{"store_ID": tokens, ...}}
- Only include stores relevant to the query
- Allocate 0 to irrelevant stores (or omit them)
- Total should use most of {total_budget} for best results"""

    try:
        resolved = self.registry._resolve_actual_model(model)
        provider = resolved.provider
        
        result = provider.chat_completion(
            model=resolved.model_id,
            messages=[{"role": "user", "content": prompt}],
            system=None,
            options={"max_tokens": 300, "temperature": 0},
        )
        
        response_text = result.get("content", "").strip()
        usage = {
            "prompt_tokens": result.get("input_tokens", 0),
            "completion_tokens": result.get("output_tokens", 0),
            "purpose": "rag_selection",
        }
        
        # Parse JSON
        import json
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            allocations = json.loads(json_match.group())
            # Convert store_X keys to int IDs
            store_budgets = {}
            for key, value in allocations.items():
                try:
                    store_id = int(key.replace("store_", "").replace("store", ""))
                    store_budgets[store_id] = int(value)
                except (ValueError, TypeError):
                    pass
            
            if store_budgets:
                logger.info(f"Parallel RAG designator: {store_budgets}")
                return store_budgets, usage
        
        return None, usage
        
    except Exception as e:
        logger.error(f"RAG designator failed: {e}")
        return None, None
```

#### 2.3 `_designate_web()`

```python
def _designate_web(
    self,
    query: str,
) -> tuple[bool | None, str | None, dict | None]:
    """
    Focused web designator - decides if web search needed and generates query.
    
    Returns:
        (use_web, search_query, usage_dict)
    """
    model = getattr(self.enricher, "web_designator_model", None)
    if not model:
        return None, None, None
    
    prompt = f"""Analyze if web search would help answer this query.

USER QUERY: {query}

Respond with JSON only:
{{"use_web": true/false, "search_query": "optimized 3-8 word query"}}

Rules:
- use_web=true for: current events, recent news, real-time data, facts that change
- use_web=false for: general knowledge, coding, math, creative writing, personal data
- search_query: Only needed if use_web=true, optimize for search engines"""

    try:
        resolved = self.registry._resolve_actual_model(model)
        provider = resolved.provider
        
        result = provider.chat_completion(
            model=resolved.model_id,
            messages=[{"role": "user", "content": prompt}],
            system=None,
            options={"max_tokens": 150, "temperature": 0},
        )
        
        response_text = result.get("content", "").strip()
        usage = {
            "prompt_tokens": result.get("input_tokens", 0),
            "completion_tokens": result.get("output_tokens", 0),
            "purpose": "web_decision",
        }
        
        # Parse JSON
        import json
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            use_web = data.get("use_web", False)
            search_query = data.get("search_query") if use_web else None
            logger.info(f"Parallel web designator: use_web={use_web}, query={search_query}")
            return use_web, search_query, usage
        
        return None, None, usage
        
    except Exception as e:
        logger.error(f"Web designator failed: {e}")
        return None, None, None
```

#### 2.4 `_designate_live()`

```python
def _designate_live(
    self,
    query: str,
    live_source_hints: list[str],
) -> tuple[dict[str, list[dict]] | None, dict | None]:
    """
    Focused live data designator - selects sources and extracts parameters.
    
    Returns:
        (live_params dict, usage_dict) or (None, None) on failure
    """
    model = getattr(self.enricher, "live_designator_model", None)
    if not model:
        return None, None
    
    prompt = f"""Extract live data parameters needed for this query.

LIVE DATA SOURCES:
{chr(10).join(live_source_hints)}

USER QUERY: {query}

Respond with JSON only: {{"source_name": [{{"param": "value"}}]}}
- Only include sources actually needed for this query
- Extract specific parameters (symbols, locations, team names, etc.)
- Return empty {{}} if no live data needed"""

    try:
        resolved = self.registry._resolve_actual_model(model)
        provider = resolved.provider
        
        result = provider.chat_completion(
            model=resolved.model_id,
            messages=[{"role": "user", "content": prompt}],
            system=None,
            options={"max_tokens": 500, "temperature": 0},
        )
        
        response_text = result.get("content", "").strip()
        usage = {
            "prompt_tokens": result.get("input_tokens", 0),
            "completion_tokens": result.get("output_tokens", 0),
            "purpose": "live_extraction",
        }
        
        # Parse JSON
        import json
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            live_params = json.loads(json_match.group())
            if live_params:
                logger.info(f"Parallel live designator: {list(live_params.keys())}")
            return live_params, usage
        
        return {}, usage
        
    except Exception as e:
        logger.error(f"Live designator failed: {e}")
        return None, None
```

---

### Phase 3: Parallel Execution Engine

**File: `routing/smart_enricher.py`**

Add `_select_sources_parallel()` method (after the focused designator methods):

```python
def _select_sources_parallel(
    self,
    query: str,
    messages: list[dict] | None = None,
    routing_config: dict | None = None,
) -> dict | None:
    """
    Execute focused designators in parallel, falling back to unified for unconfigured domains.
    
    Returns dict with keys: allocations, search_query, live_params, selected_model
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Check which parallel designators are configured
    has_router = bool(getattr(self.enricher, "router_designator_model", None))
    has_rag = bool(getattr(self.enricher, "rag_designator_model", None))
    has_web = bool(getattr(self.enricher, "web_designator_model", None))
    has_live = bool(getattr(self.enricher, "live_designator_model", None))
    
    # If none configured, use unified call
    if not any([has_router, has_rag, has_web, has_live]):
        return self._select_sources(query, messages, routing_config)
    
    logger.info(f"Using parallel designators: router={has_router}, rag={has_rag}, web={has_web}, live={has_live}")
    
    # Prepare data for designators
    store_info = self._get_store_info_for_designator()
    live_source_hints = self._get_live_source_hints()
    preview_samples = {}
    if getattr(self.enricher, "use_two_pass_retrieval", False) and has_rag:
        preview_samples = self._get_preview_samples_for_stores(query, store_info)
    
    # Results
    results = {
        "allocations": {},
        "search_query": None,
        "live_params": {},
        "selected_model": None,
    }
    designator_calls = []
    unified_domains = []  # Domains needing unified fallback
    
    # Determine what needs parallel vs unified
    needs_routing = routing_config and routing_config.get("candidates")
    needs_rag = getattr(self.enricher, "use_rag", False) and store_info
    needs_web = getattr(self.enricher, "use_web", False)
    needs_live = getattr(self.enricher, "use_live_data", False) and live_source_hints
    
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit parallel designator calls
        if needs_routing:
            if has_router:
                futures["router"] = executor.submit(
                    self._designate_routing,
                    query,
                    messages,
                    routing_config["candidates"],
                    routing_config.get("token_count", 0),
                    routing_config.get("has_images", False),
                )
            else:
                unified_domains.append("routing")
        
        if needs_rag:
            if has_rag:
                futures["rag"] = executor.submit(
                    self._designate_rag,
                    query,
                    store_info,
                    getattr(self.enricher, "max_context_tokens", 4000),
                    preview_samples,
                )
            else:
                unified_domains.append("rag")
        
        if needs_web:
            if has_web:
                futures["web"] = executor.submit(
                    self._designate_web,
                    query,
                )
            else:
                unified_domains.append("web")
        
        if needs_live:
            if has_live:
                futures["live"] = executor.submit(
                    self._designate_live,
                    query,
                    live_source_hints,
                )
            else:
                unified_domains.append("live")
        
        # Collect parallel results
        for key, future in futures.items():
            try:
                if key == "router":
                    model, usage = future.result(timeout=30)
                    if model:
                        results["selected_model"] = model
                    if usage:
                        designator_calls.append(usage)
                        
                elif key == "rag":
                    budgets, usage = future.result(timeout=30)
                    if budgets:
                        # Convert to allocations format
                        for store_id, tokens in budgets.items():
                            results["allocations"][str(store_id)] = tokens
                    if usage:
                        designator_calls.append(usage)
                        
                elif key == "web":
                    use_web, search_query, usage = future.result(timeout=30)
                    if use_web:
                        results["allocations"]["web"] = getattr(
                            self.enricher, "max_context_tokens", 4000
                        ) // 4  # Default 25% to web
                        results["search_query"] = search_query
                    if usage:
                        designator_calls.append(usage)
                        
                elif key == "live":
                    params, usage = future.result(timeout=30)
                    if params:
                        results["live_params"] = params
                    if usage:
                        designator_calls.append(usage)
                        
            except Exception as e:
                logger.error(f"Parallel designator '{key}' failed: {e}")
                # Add to unified domains as fallback
                unified_domains.append(key.replace("router", "routing"))
    
    # Handle unified call for remaining domains
    if unified_domains and self.enricher.designator_model:
        logger.info(f"Unified fallback for domains: {unified_domains}")
        unified_result = self._select_sources_unified(
            query, messages, routing_config, unified_domains
        )
        if unified_result:
            # Merge unified results
            if "routing" in unified_domains and unified_result.get("selected_model"):
                results["selected_model"] = unified_result["selected_model"]
            if "rag" in unified_domains:
                for k, v in unified_result.get("allocations", {}).items():
                    if k != "web":
                        results["allocations"][k] = v
            if "web" in unified_domains:
                if unified_result.get("allocations", {}).get("web"):
                    results["allocations"]["web"] = unified_result["allocations"]["web"]
                    results["search_query"] = unified_result.get("search_query")
            if "live" in unified_domains and unified_result.get("live_params"):
                results["live_params"] = unified_result["live_params"]
            
            if unified_result.get("designator_usage"):
                designator_calls.append(unified_result["designator_usage"])
    
    # Attach designator calls for tracking
    results["designator_calls"] = designator_calls
    
    logger.info(f"Parallel selection complete: stores={list(results['allocations'].keys())}, "
                f"web={'web' in results['allocations']}, live={list(results['live_params'].keys())}, "
                f"model={results['selected_model']}")
    
    return results
```

Also add helper method `_select_sources_unified()` for partial unified calls:

```python
def _select_sources_unified(
    self,
    query: str,
    messages: list[dict] | None,
    routing_config: dict | None,
    domains: list[str],
) -> dict | None:
    """
    Make a unified designator call for specific domains only.
    This is used when some parallel designators are configured but not all.
    """
    # Build a trimmed version of _select_sources that only includes requested domains
    # For now, just call the full _select_sources - it handles all domains
    # The caller will extract only what it needs
    return self._select_sources(query, messages, routing_config if "routing" in domains else None)
```

---

### Phase 4: Integration

**File: `routing/smart_enricher.py`** (~line 730, in `enrich()` method)

Find the existing call to `_select_sources`:
```python
if getattr(self.enricher, "use_smart_source_selection", False):
    selection = self._select_sources(
        query, messages=messages, routing_config=routing_config
    )
```

Replace with:
```python
if getattr(self.enricher, "use_smart_source_selection", False):
    # Check if any parallel designators are configured
    has_parallel = any([
        getattr(self.enricher, "router_designator_model", None),
        getattr(self.enricher, "rag_designator_model", None),
        getattr(self.enricher, "web_designator_model", None),
        getattr(self.enricher, "live_designator_model", None),
    ])
    
    if has_parallel:
        selection = self._select_sources_parallel(
            query, messages=messages, routing_config=routing_config
        )
    else:
        selection = self._select_sources(
            query, messages=messages, routing_config=routing_config
        )
```

---

### Phase 5: Admin UI

**File: `admin/templates/smart_aliases.html`**

1. Add form fields in Alpine.js data initialization (~line 50):
```javascript
form: {
    // ... existing fields ...
    router_designator_model: '',
    rag_designator_model: '',
    web_designator_model: '',
    live_designator_model: '',
}
```

2. Add UI section after the main designator_model field (~line 700):
```html
<!-- Parallel Designators Section -->
<div x-show="form.use_smart_source_selection" x-transition class="mt-4">
    <label class="label">
        <span class="label-text font-medium">Parallel Designators (Optional)</span>
    </label>
    <div class="bg-base-200 rounded-lg p-4 space-y-3">
        <p class="text-sm text-base-content/60 mb-3">
            Configure specialized models for each decision type. Leave empty to use the unified designator.
        </p>
        
        <!-- Router Designator -->
        <div class="form-control" x-show="form.use_routing">
            <label class="label py-1">
                <span class="label-text text-sm">Router Designator</span>
            </label>
            <select x-model="form.router_designator_model" class="select select-bordered select-sm">
                <option value="">Use unified designator</option>
                <template x-for="model in availableModels" :key="model.id">
                    <option :value="model.id" x-text="model.id"></option>
                </template>
            </select>
            <label class="label py-0">
                <span class="label-text-alt">Selects best model from candidates</span>
            </label>
        </div>
        
        <!-- RAG Designator -->
        <div class="form-control" x-show="form.use_rag">
            <label class="label py-1">
                <span class="label-text text-sm">RAG Designator</span>
            </label>
            <select x-model="form.rag_designator_model" class="select select-bordered select-sm">
                <option value="">Use unified designator</option>
                <template x-for="model in availableModels" :key="model.id">
                    <option :value="model.id" x-text="model.id"></option>
                </template>
            </select>
            <label class="label py-0">
                <span class="label-text-alt">Selects document stores and allocates token budgets</span>
            </label>
        </div>
        
        <!-- Web Designator -->
        <div class="form-control" x-show="form.use_web">
            <label class="label py-1">
                <span class="label-text text-sm">Web Designator</span>
            </label>
            <select x-model="form.web_designator_model" class="select select-bordered select-sm">
                <option value="">Use unified designator</option>
                <template x-for="model in availableModels" :key="model.id">
                    <option :value="model.id" x-text="model.id"></option>
                </template>
            </select>
            <label class="label py-0">
                <span class="label-text-alt">Decides if web search needed and generates query</span>
            </label>
        </div>
        
        <!-- Live Designator -->
        <div class="form-control" x-show="form.use_live_data">
            <label class="label py-1">
                <span class="label-text text-sm">Live Data Designator</span>
            </label>
            <select x-model="form.live_designator_model" class="select select-bordered select-sm">
                <option value="">Use unified designator</option>
                <template x-for="model in availableModels" :key="model.id">
                    <option :value="model.id" x-text="model.id"></option>
                </template>
            </select>
            <label class="label py-0">
                <span class="label-text-alt">Extracts parameters for live data sources</span>
            </label>
        </div>
    </div>
</div>
```

3. Update `openEditModal()` to load existing values:
```javascript
this.form.router_designator_model = alias.router_designator_model || '';
this.form.rag_designator_model = alias.rag_designator_model || '';
this.form.web_designator_model = alias.web_designator_model || '';
this.form.live_designator_model = alias.live_designator_model || '';
```

4. Update `saveAlias()` to include new fields in the POST/PUT request.

**File: `admin/app.py`**

Update the smart alias create/update API endpoints to accept and pass the new fields.

---

## Key Design Decisions

### 1. Designators are Independent
- No dependencies between parallel designators
- Router selection does NOT affect RAG/Web/Live decisions
- Each operates independently on the query
- Enables true parallel execution

### 2. Simple Result Merging
Each designator owns its domain - no conflict resolution needed:
- Router → `selected_model`
- RAG → `store_budgets` (in allocations)
- Web → `use_web`, `search_query`, `allocations["web"]`
- Live → `live_params`

### 3. Focused Prompts
| Designator | Current Unified | New Focused |
|------------|-----------------|-------------|
| Router | ~1500 tokens | ~300 tokens |
| RAG | ~1500 tokens | ~500 tokens |
| Web | ~1500 tokens | ~100 tokens |
| Live | ~1500 tokens | ~400 tokens |

### 4. Error Handling
If a parallel designator fails:
1. Log the error
2. Add that domain to unified call (if `designator_model` configured)
3. Or use defaults

### 5. 100% Backward Compatible
- Existing aliases with only `designator_model` work unchanged
- New fields default to `None` (use unified)
- Parallel execution only when fields explicitly set

---

## Verification

### Manual Testing

1. **Existing alias (no parallel designators)**: Verify unified call still works
2. **One parallel designator**: Configure only `rag_designator_model`
3. **All parallel**: Configure all 4
4. **Error handling**: Force one to fail, verify fallback

### Check Logs
```bash
docker logs llm-relay-dev 2>&1 | grep -i "parallel\|designat"
```

### Performance
- Unified: ~2-3 seconds
- All parallel: ~0.5-1 second (max of individual times)
