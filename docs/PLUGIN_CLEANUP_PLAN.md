# Plugin System Cleanup Plan

## Problem Statement

The current plugin system is incomplete. Adding a new source type (like IMAP) required editing 7+ files instead of just creating a plugin. This document outlines the changes needed to make the plugin system truly self-contained.

---

## Current State Analysis

### What's Wrong

| Component | Current State | Ideal State |
|-----------|--------------|-------------|
| **Document Store Schema** | 50+ columns for source-specific fields (imap_host, gmail_label_id, etc.) | `source_type` + `config_json` (arbitrary JSON) |
| **Document Store CRUD** | Explicit parameters for every field | Generic `config: dict` parameter |
| **Admin UI Forms** | Hardcoded HTML for each source type | Dynamic rendering from `get_config_fields()` |
| **Content Categories** | Hardcoded `LEGACY_SOURCE_TYPE_CATEGORIES` dict | Query plugin's `content_category` attribute |
| **Account Discovery** | Hardcoded `if source_type == "x"` branches | Ask plugin via standard interface |
| **Action Handlers** | Provider-specific `_execute_gmail()`, `_execute_imap()` | Delegate to unified source's action methods |
| **Legacy mcp/sources.py** | 3000+ lines of document source classes | Delete entirely, use plugins |
| **Legacy live/sources.py** | Mix of legacy providers and plugin adapters | Remove legacy providers |

### Files That Need Changes for Each New Source (Current)

1. `db/models.py` - Add columns
2. `db/connection.py` - Add migrations
3. `db/document_stores.py` - Add CRUD parameters
4. `admin/app.py` - Add API handling
5. `admin/templates/document_stores.html` - Add form HTML
6. `plugin_base/common.py` - Add to content category mapping
7. `routing/smart_enricher.py` - Add account discovery logic
8. `builtin_plugins/actions/*.py` - Add provider-specific execution

### Files That Should Need Changes (Target)

1. `builtin_plugins/unified_sources/new_source.py` - Create the plugin

---

## Implementation Plan

### Phase 1: Database Schema Consolidation

**Goal:** Replace 50+ source-specific columns with a single `config_json` column.

#### 1.1 Add New Column

```python
# db/models.py
class DocumentStore(Base):
    # Keep existing columns for now (backwards compatibility)
    # Add new unified config column
    config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    @property
    def config(self) -> dict:
        """Get parsed config dict."""
        if self.config_json:
            return json.loads(self.config_json)
        # Fallback: build from legacy columns
        return self._build_config_from_legacy_columns()
    
    @config.setter
    def config(self, value: dict):
        self.config_json = json.dumps(value) if value else None
```

#### 1.2 Migration Strategy

1. Add `config_json` column (nullable)
2. Write migration that populates `config_json` from existing columns for all rows
3. Update CRUD to read/write `config_json` preferentially
4. After verification period, mark legacy columns as deprecated
5. Eventually drop legacy columns

#### 1.3 Update CRUD

```python
# db/document_stores.py

def create_document_store(
    name: str,
    source_type: str,
    config: dict,  # <-- Single config dict replaces 50+ parameters
    # Keep only truly common fields as explicit params
    display_name: Optional[str] = None,
    embedding_provider: str = "local",
    embedding_model: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    index_schedule: Optional[str] = None,
    enabled: bool = True,
    db: Optional[Session] = None,
) -> DocumentStore:
    ...
```

---

### Phase 2: Dynamic Admin UI

**Goal:** Render source configuration forms dynamically from plugin definitions.

#### 2.1 New API Endpoint

```python
# admin/app.py

@admin.route("/api/document-sources/<source_type>/fields")
@require_auth_api
def get_source_fields(source_type: str):
    """Get configuration fields for a source type."""
    from plugin_base.loader import unified_source_registry
    
    plugin_class = unified_source_registry.get(source_type)
    if not plugin_class:
        return jsonify({"error": "Unknown source type"}), 404
    
    fields = plugin_class.get_config_fields()
    return jsonify({
        "source_type": source_type,
        "display_name": plugin_class.display_name,
        "description": plugin_class.description,
        "icon": plugin_class.icon,
        "supports_rag": plugin_class.supports_rag,
        "supports_live": plugin_class.supports_live,
        "supports_actions": plugin_class.supports_actions,
        "fields": [field.to_dict() for field in fields],
    })
```

#### 2.2 Dynamic Form Component

```html
<!-- admin/templates/components/dynamic_form.html -->
<!-- Alpine.js component that renders forms from field definitions -->

<template x-for="field in fields" :key="field.name">
    <div class="form-control" x-show="!field.hidden">
        <label class="label">
            <span class="label-text" x-text="field.label"></span>
            <span x-show="field.required" class="text-error">*</span>
        </label>
        
        <!-- Text input -->
        <template x-if="field.field_type === 'text'">
            <input type="text" 
                   x-model="form.config[field.name]"
                   :placeholder="field.placeholder"
                   :required="field.required"
                   class="input input-bordered">
        </template>
        
        <!-- Password input -->
        <template x-if="field.field_type === 'password'">
            <input type="password"
                   x-model="form.config[field.name]"
                   :placeholder="field.placeholder"
                   class="input input-bordered">
        </template>
        
        <!-- Select dropdown -->
        <template x-if="field.field_type === 'select'">
            <select x-model="form.config[field.name]" class="select select-bordered">
                <template x-for="opt in field.options" :key="opt.value">
                    <option :value="opt.value" x-text="opt.label"></option>
                </template>
            </select>
        </template>
        
        <!-- Integer input -->
        <template x-if="field.field_type === 'integer'">
            <input type="number"
                   x-model.number="form.config[field.name]"
                   :min="field.min"
                   :max="field.max"
                   class="input input-bordered">
        </template>
        
        <!-- Boolean checkbox -->
        <template x-if="field.field_type === 'boolean'">
            <input type="checkbox"
                   x-model="form.config[field.name]"
                   class="checkbox">
        </template>
        
        <!-- OAuth account picker -->
        <template x-if="field.field_type === 'oauth_account'">
            <div class="flex gap-2">
                <select x-model="form.config[field.name]" class="select select-bordered flex-1">
                    <option value="">Select account...</option>
                    <template x-for="acc in getOAuthAccounts(field.picker_options?.provider)">
                        <option :value="acc.id" x-text="acc.email"></option>
                    </template>
                </select>
                <button type="button" @click="connectOAuth(field.picker_options?.provider)"
                        class="btn btn-outline">Connect</button>
            </div>
        </template>
        
        <!-- Help text -->
        <label class="label" x-show="field.help_text">
            <span class="label-text-alt" x-text="field.help_text"></span>
        </label>
    </div>
</template>
```

#### 2.3 Refactor Document Stores Page

```javascript
// In document_stores.html Alpine.js data

// Instead of hardcoded form fields:
form: {
    name: "",
    display_name: "",
    source_type: "local",
    config: {},  // <-- Dynamic config object
    // Common fields only
    embedding_provider: "local",
    chunk_size: 512,
    index_schedule: "",
},

// Load fields when source type changes
async onSourceTypeChange() {
    const response = await fetch(`/admin/api/document-sources/${this.form.source_type}/fields`);
    const data = await response.json();
    this.currentSourceFields = data.fields;
    
    // Initialize config with defaults
    this.form.config = {};
    for (const field of data.fields) {
        this.form.config[field.name] = field.default ?? null;
    }
},
```

#### 2.4 Remove Hardcoded Form Sections

Delete these from `document_stores.html`:
- `<!-- IMAP Email Configuration -->`
- `<!-- Todoist Configuration -->`
- `<!-- Slack Configuration -->`
- `<!-- Paperless Configuration -->`
- `<!-- Nextcloud Configuration -->`
- `<!-- Notion Configuration -->`
- `<!-- GitHub Configuration -->`
- `<!-- Website Configuration -->`
- `<!-- WebSearch Configuration -->`
- All Google OAuth sections
- All Microsoft OAuth sections

Replace with single:
```html
<!-- Dynamic Source Configuration -->
<div class="mt-4 space-y-4" x-show="currentSourceFields.length > 0">
    <template x-for="field in currentSourceFields" :key="field.name">
        <!-- Dynamic field rendering -->
    </template>
</div>
```

---

### Phase 3: Plugin-Driven Content Categories

**Goal:** Remove `LEGACY_SOURCE_TYPE_CATEGORIES` dict, query plugins directly.

#### 3.1 Update get_content_category()

```python
# plugin_base/common.py

def get_content_category(source_type: str) -> ContentCategory:
    """Get content category from plugin, with legacy fallback."""
    from plugin_base.loader import unified_source_registry
    
    # Try to get from plugin
    plugin_class = unified_source_registry.get(source_type)
    if plugin_class and hasattr(plugin_class, 'content_category'):
        if plugin_class.content_category is not None:
            return plugin_class.content_category
    
    # Legacy fallback (to be removed)
    return LEGACY_SOURCE_TYPE_CATEGORIES.get(source_type, ContentCategory.OTHER)
```

#### 3.2 Eventually Remove Legacy Dict

Once all sources are migrated to plugins, delete `LEGACY_SOURCE_TYPE_CATEGORIES`.

---

### Phase 4: Plugin-Driven Account Discovery

**Goal:** Replace hardcoded account discovery in `_build_available_accounts()`.

#### 4.1 Add Plugin Interface

```python
# plugin_base/unified_source.py

class PluginUnifiedSource:
    # Existing attributes...
    
    @classmethod
    def get_account_info(cls, store: "DocumentStore") -> Optional[dict]:
        """
        Extract account info from a document store for action handlers.
        
        Returns dict with:
        - provider: str (e.g., "google", "microsoft", "imap", "todoist")
        - email: str (account email/identifier)
        - oauth_account_id: Optional[int] (for OAuth-based sources)
        - store_id: int (document store ID)
        - Extra source-specific fields (calendar_id, project_id, etc.)
        
        Returns None if this store doesn't provide actionable accounts.
        """
        # Default implementation - subclasses override
        return None
```

#### 4.2 Implement in Unified Sources

```python
# builtin_plugins/unified_sources/gmail.py

class GmailUnifiedSource(PluginUnifiedSource):
    @classmethod
    def get_account_info(cls, store) -> Optional[dict]:
        if not store.google_account_id:
            return None
        
        from db.oauth_tokens import get_oauth_token_info
        token_info = get_oauth_token_info(store.google_account_id)
        
        return {
            "provider": "google",
            "email": token_info.get("account_email", "") if token_info else "",
            "oauth_account_id": store.google_account_id,
            "store_id": store.id,
            "name": store.display_name or store.name,
        }
```

```python
# builtin_plugins/unified_sources/imap.py

class IMAPUnifiedSource(PluginUnifiedSource):
    @classmethod
    def get_account_info(cls, store) -> Optional[dict]:
        return {
            "provider": "imap",
            "email": store.imap_username or "",
            "store_id": store.id,
            "name": store.display_name or store.name,
        }
```

#### 4.3 Refactor Account Discovery

```python
# routing/smart_enricher.py

def _build_available_accounts(self) -> dict[str, list[dict]]:
    """Build available_accounts from linked document stores using plugins."""
    from plugin_base.loader import unified_source_registry
    
    available_accounts = {"email": [], "calendar": [], "tasks": []}
    
    CONTENT_TO_ACTION_CATEGORY = {
        ContentCategory.EMAILS: "email",
        ContentCategory.CALENDARS: "calendar",
        ContentCategory.TASKS: "tasks",
    }
    
    for store in self._get_linked_stores():
        source_type = store.source_type
        plugin_class = unified_source_registry.get(source_type)
        
        if not plugin_class:
            continue
        
        # Get content category from plugin
        content_category = plugin_class.content_category
        category = CONTENT_TO_ACTION_CATEGORY.get(content_category)
        
        if not category:
            continue
        
        # Get account info from plugin
        account_info = plugin_class.get_account_info(store)
        if account_info:
            available_accounts[category].append(account_info)
    
    return available_accounts
```

---

### Phase 5: Action Handler Delegation

**Goal:** Action handlers delegate to unified sources instead of reimplementing.

#### 5.1 Unified Source Action Interface

```python
# plugin_base/unified_source.py

class PluginUnifiedSource:
    supports_actions: bool = False
    
    def execute_action(self, action: str, params: dict) -> dict:
        """
        Execute an action on this source.
        
        Args:
            action: Action name (e.g., "send_email", "create_draft")
            params: Action parameters
            
        Returns:
            dict with: success (bool), message (str), error (str), data (dict)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support actions")
```

#### 5.2 Implement in IMAP Source

```python
# builtin_plugins/unified_sources/imap.py

class IMAPUnifiedSource(PluginUnifiedSource):
    supports_actions = True
    
    def execute_action(self, action: str, params: dict) -> dict:
        """Route action to appropriate method."""
        action_map = {
            "draft_new": self._action_draft_new,
            "draft_reply": self._action_draft_reply,
            "send_new": self._action_send_new,
            "send_reply": self._action_send_reply,
            "mark_read": self._action_mark_read,
            "mark_unread": self._action_mark_unread,
            "archive": self._action_archive,
        }
        
        handler = action_map.get(action)
        if not handler:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        return handler(params)
    
    def _action_draft_new(self, params: dict) -> dict:
        return self.create_draft(
            to=params.get("to", []),
            subject=params.get("subject", ""),
            body=params.get("body", ""),
            cc=params.get("cc"),
        )
    
    # ... etc
```

#### 5.3 Refactor Email Action Handler

```python
# builtin_plugins/actions/email.py

class EmailActionHandler(PluginActionHandler):
    """Unified email handler that delegates to source plugins."""
    
    def execute(self, action: str, params: dict, context: ActionContext) -> ActionResult:
        # Get the unified source for the selected account
        source = self._get_unified_source(params, context)
        if not source:
            return ActionResult(success=False, error="No email source configured")
        
        # Delegate to the source
        result = source.execute_action(action, params)
        
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            error=result.get("error"),
            data=result.get("data"),
        )
    
    def _get_unified_source(self, params: dict, context: ActionContext):
        """Load the appropriate unified source based on account selection."""
        account = self._select_account(params, context)
        if not account:
            return None
        
        store_id = account.get("store_id")
        if not store_id:
            return None
        
        from db.document_stores import get_document_store_by_id
        from plugin_base.loader import unified_source_registry
        
        store = get_document_store_by_id(store_id)
        if not store:
            return None
        
        plugin_class = unified_source_registry.get(store.source_type)
        if not plugin_class:
            return None
        
        # Build config and instantiate
        config = plugin_class.build_config_from_store(store)
        return plugin_class(config)
```

This removes:
- `_execute_gmail()` (~300 lines)
- `_execute_outlook()` (~250 lines)  
- `_execute_imap()` (~400 lines)

The email handler becomes ~100 lines of delegation logic.

---

### Phase 6: Delete Legacy Code

**Goal:** Remove unused legacy implementations.

#### 6.1 Delete Legacy Document Sources

Delete from `mcp/sources.py`:
- `GoogleDriveDocumentSource`
- `GmailDocumentSource`
- `GoogleCalendarDocumentSource`
- `GoogleTasksDocumentSource`
- `GoogleContactsDocumentSource`
- `OneDriveDocumentSource`
- `OutlookMailDocumentSource`
- `OneNoteDocumentSource`
- `TeamsDocumentSource`
- `WebSearchDocumentSource`
- `GoogleKeepDocumentSource`
- `_setup_google_oauth_files()`
- `_setup_gdrive_oauth_files()`

Keep only:
- `DocumentSource` base class (for interface reference)
- `get_document_source()` factory (updated to use plugins only)

#### 6.2 Delete Legacy Live Providers

Delete from `live/sources.py`:
- `StocksProvider` (replaced by `SmartStocksProvider` plugin)
- `WeatherProvider` (replaced by plugin)
- `TransportProvider` (replaced by plugin)
- `OpenMeteoProvider` (duplicate)
- `GoogleMapsProvider` (replaced by `RoutesSmartProvider`)
- `OuraLiveProvider` (replaced by plugin)
- `WithingsLiveProvider` (replaced by plugin)

Keep:
- `RoutesSmartProvider` (until migrated to plugin)
- `MCPProvider`
- `PluginLiveSourceAdapter`

#### 6.3 Delete Legacy Action Handler Code

After Phase 5, delete from `builtin_plugins/actions/email.py`:
- `_execute_gmail()` and all `_gmail_*` methods
- `_execute_outlook()` and all `_outlook_*` methods
- `_execute_imap()` and all `_imap_*` methods

---

## Migration Timeline

### Sprint 1: Foundation (Phase 1)
- Add `config_json` column
- Write migration to populate from existing columns
- Update CRUD to use `config_json`

### Sprint 2: Dynamic UI (Phase 2)
- Implement field definitions API
- Create dynamic form component
- Test with one source type (e.g., IMAP)
- Gradually migrate other source types

### Sprint 3: Plugin Integration (Phases 3-4)
- Update `get_content_category()` to use plugins
- Add `get_account_info()` to unified sources
- Refactor `_build_available_accounts()`

### Sprint 4: Action Delegation (Phase 5)
- Add `execute_action()` to unified sources
- Refactor email handler to delegate
- Refactor calendar handler to delegate
- Refactor tasks handler to delegate

### Sprint 5: Cleanup (Phase 6)
- Delete legacy document sources
- Delete legacy live providers
- Delete legacy action handler code
- Remove deprecated database columns

---

## Risk Mitigation

1. **Backwards Compatibility**: Keep legacy columns readable during migration, only delete after full verification.

2. **Gradual Rollout**: Migrate one source type at a time, verify before moving to next.

3. **Feature Flags**: Add `USE_DYNAMIC_FORMS` flag to toggle between legacy and new UI during testing.

4. **Rollback Plan**: Keep legacy code paths available (commented or behind flag) until migration complete.

---

## Success Metrics

After cleanup is complete:

| Metric | Before | After |
|--------|--------|-------|
| Files to edit for new source | 7+ | 1 |
| Lines of code in document_stores.py | ~800 | ~200 |
| Lines of code in email.py action handler | ~2000 | ~200 |
| Lines of code in mcp/sources.py | ~3500 | ~100 |
| Source-specific columns in DocumentStore | 50+ | 0 |
| Hardcoded form sections in UI | 15+ | 0 |

---

## Appendix: Current Source-Specific Columns

Columns that would be consolidated into `config_json`:

```
# Google
google_account_id, gdrive_folder_id, gdrive_folder_name, gmail_label_id, 
gmail_label_name, gcalendar_calendar_id, gcalendar_calendar_name,
gtasks_tasklist_id, gtasks_tasklist_name, gcontacts_group_id, gcontacts_group_name

# Microsoft  
microsoft_account_id, onedrive_folder_id, onedrive_folder_name,
outlook_folder_id, outlook_folder_name, outlook_days_back,
onenote_notebook_id, onenote_notebook_name, teams_team_id, teams_team_name,
teams_channel_id, teams_channel_name, teams_days_back

# Paperless
paperless_url, paperless_token, paperless_tag_id, paperless_tag_name

# GitHub
github_repo, github_branch, github_path

# Notion
notion_database_id, notion_page_id, notion_is_task_database

# Nextcloud
nextcloud_folder

# Website
website_url, website_crawl_depth, website_max_pages, website_include_pattern,
website_exclude_pattern, website_crawler_override

# Slack
slack_channel_id, slack_channel_types, slack_days_back

# Todoist
todoist_project_id, todoist_project_name, todoist_filter, todoist_include_completed

# IMAP
imap_host, imap_port, imap_username, imap_password, imap_use_ssl,
imap_folders, imap_index_days, smtp_host, smtp_port

# WebSearch
websearch_query, websearch_max_results, websearch_pages_to_scrape,
websearch_time_range, websearch_category
```

All of these become simply:
```python
config_json = '{"host": "imap.example.com", "port": 993, ...}'
```
