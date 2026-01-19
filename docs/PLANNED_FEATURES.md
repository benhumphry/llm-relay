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
- **Context Priority**: Token allocation for hybrid RAG+Web (balanced/prefer_rag/prefer_web)

## v1.8: Memory & Smart Source Selection

- **Persistent Memory**: Remembers explicit user facts across sessions
  - Only captures what users directly state about themselves
  - Does NOT infer from topics asked about
  - Memory injected into future requests as context
  - Viewable and clearable from Admin UI
- **Smart Source Selection**: Intelligent token budget allocation across RAG stores
  - Document Store intelligence (themes, best_for, summary)
  - Designator allocates 50% priority budget to most relevant sources
  - 50% baseline ensures all sources contribute
  - Works with Web as an additional source
- **Document Store Intelligence**: Auto-generated metadata for routing
  - Themes: High-level domain categories
  - Best For: Types of questions the store can answer
  - Summary: Brief description of content type

## Document Sources (v1.6+)

Direct API integrations (no Node.js required):
- Google Drive, Gmail, Calendar (via OAuth)
- Notion (via REST API)
- GitHub (via REST API)
- Paperless-ngx (via REST API)
- Nextcloud (via WebDAV)
- Websites (via crawler)
- Local filesystem

---

# Planned Features

## v1.9: Extended Data Sources

Final v1.x release focused on expanding document source integrations.

### Communication & Collaboration

| Source | API Type | Status |
|--------|----------|--------|
| Slack | REST API | Planned |
| Microsoft Teams | Graph API | Planned |
| Discord | Bot API | Planned |
| Mattermost | REST API | Planned |

### Cloud Storage & Documents

| Source | API Type | Status |
|--------|----------|--------|
| Dropbox | REST API | Planned |
| OneDrive | Graph API | Planned |
| OneNote | Graph API | Planned |
| Box | REST API | Planned |
| S3/MinIO | S3 API | Planned |

### Email

| Source | API Type | Status |
|--------|----------|--------|
| Outlook | Graph API | Planned |
| IMAP (generic) | IMAP | Planned |

### Project Management

| Source | API Type | Status |
|--------|----------|--------|
| Jira | REST API | Planned |
| Linear | GraphQL | Planned |
| Trello | REST API | Planned |
| Asana | REST API | Planned |
| Confluence | REST API | Planned |

### Notes & Tasks

| Source | API Type | Status |
|--------|----------|--------|
| Todoist | REST API | Planned |
| Evernote | REST API | Planned (deprecated API) |

### Knowledge & Research

| Source | API Type | Status |
|--------|----------|--------|
| Readwise/Reader | REST API | Planned |
| Raindrop.io | REST API | Planned |
| Zotero | REST API | Planned |
| Airtable | REST API | Planned |

### Search Providers

| Provider | Description | Status |
|----------|-------------|--------|
| Exa | Neural/semantic search | Planned |

### Not Feasible (No Public API)

These were considered but lack viable API access:
- WhatsApp (business API only, no personal)
- Signal (no API)
- Telegram (bot-only access)
- Apple iCloud/Notes (no public API)
- Things (Mac-only, no API)

---

## v2.0: Major Release

### Webhooks for Document Stores

Real-time indexing triggered by external system events:

| Feature | Description |
|---------|-------------|
| Generic refresh endpoint | `POST /api/webhooks/stores/{id}/refresh` - trigger re-index |
| Source-specific handlers | Smart incremental updates (e.g., Paperless, Notion, GitHub) |
| Webhook secrets | HMAC signature verification for security |
| Selective re-indexing | Update only changed documents, not full re-index |

**Use cases:**
- Paperless-ngx: Index new documents immediately after OCR
- Notion: Re-index page when updated
- GitHub: Re-index on push to tracked branches
- Google Drive: Push notifications for file changes

### Smart Query Studio

Built-in chat interface for testing:
- OpenWebUI-style conversation UI
- Model selector with all aliases
- Blind A/B testing
- Response comparison

### Model Sync Service

Subscription for high-quality model metadata:
- Regularly refreshed model intelligence
- Accurate pricing data
- Deprecation notices

### Media Endpoints

| Endpoint | Description | Providers |
|----------|-------------|-----------|
| `POST /v1/images/generations` | Image generation | OpenAI, Stability |
| `POST /v1/audio/transcriptions` | Speech-to-text | OpenAI, Groq |
| `POST /v1/audio/speech` | Text-to-speech | OpenAI, ElevenLabs |

---

## v2.1: Smart Actions (Partially Implemented)

Bidirectional integrations - let LLMs take actions. First phase (email drafts) is implemented.

### Concept

LLM outputs structured action blocks in responses:

```xml
<smart_action type="email" action="draft_new">
{"account": "user@gmail.com", "to": ["recipient@example.com"], "subject": "Meeting follow-up", "body": "..."}
</smart_action>
```

The proxy:
1. Detects action blocks during response streaming
2. Strips action blocks from client-visible response
3. Validates and executes allowed actions
4. Logs results (success/failure)

### Implemented Actions

#### Email Actions (Gmail - Implemented)
| Action | Description | Status |
|--------|-------------|--------|
| `email:draft_new` | Create new email draft | ✅ Implemented |
| `email:draft_reply` | Create reply draft | ✅ Implemented |
| `email:draft_forward` | Create forward draft | ✅ Implemented |
| `email:send_new` | Send new email immediately | ✅ Implemented |
| `email:send_reply` | Send reply immediately | ✅ Implemented |
| `email:send_forward` | Send forward immediately | ✅ Implemented |
| `email:mark_read` | Mark email as read | ✅ Implemented |
| `email:mark_unread` | Mark email as unread | ✅ Implemented |
| `email:archive` | Archive email | ✅ Implemented |
| `email:label` | Add/remove labels | ✅ Implemented |

#### Notification Actions (Implemented)
| Action | Description | Status |
|--------|-------------|--------|
| `notification:push` | Send push notification via configured URLs | ✅ Implemented |

#### Calendar Actions (Implemented)
| Action | Description | Status |
|--------|-------------|--------|
| `calendar:create` | Create calendar event | ✅ Implemented |
| `calendar:update` | Modify existing event | ✅ Implemented |
| `calendar:delete` | Cancel/delete event | ✅ Implemented |
| `calendar:rsvp` | Respond to invitation | 🔜 Planned |

#### Scheduled Prompts (Implemented)
| Action | Description | Status |
|--------|-------------|--------|
| `schedule:prompt` | Schedule recurring or one-time prompts | ✅ Implemented |
| `schedule:cancel` | Cancel scheduled prompt | ✅ Implemented |

**Action Schemas:**

```xml
<!-- Create new draft -->
<smart_action type="email" action="draft_new">
{
  "account": "user@gmail.com",
  "to": ["recipient@example.com"],
  "cc": [],
  "bcc": [],
  "subject": "Subject line",
  "body": "Email body text",
  "body_type": "text"
}
</smart_action>

<!-- Reply to email (requires message_id from email chunk metadata) -->
<smart_action type="email" action="draft_reply">
{
  "account": "user@gmail.com",
  "message_id": "abc123xyz",
  "to": ["sender@example.com"],
  "reply_all": false,
  "body": "Reply text"
}
</smart_action>

<!-- Forward email -->
<smart_action type="email" action="draft_forward">
{
  "account": "user@gmail.com",
  "message_id": "abc123xyz",
  "to": ["forward-to@example.com"],
  "body": "FYI - see below"
}
</smart_action>
```

### Planned Actions

#### Document Actions
| Action | Description |
|--------|-------------|
| `drive:create` | Create document |
| `drive:update` | Edit document |
| `drive:share` | Share with users |
| `notion:create` | Create Notion page |
| `notion:update` | Update Notion page |

#### Communication Actions
| Action | Description |
|--------|-------------|
| `slack:message` | Send Slack message |
| `slack:react` | Add reaction |
| `teams:message` | Send Teams message |

#### Task Actions
| Action | Description |
|--------|-------------|
| `task:create` | Create task (Todoist, Asana, etc.) |
| `task:complete` | Mark task done |
| `task:update` | Update task details |

### Scheduled Prompts Architecture

Scheduled prompts turn LLM Relay into a proactive agent:

1. **Scheduler service** runs in background (APScheduler)
2. **Prompt execution** uses configured Smart Alias
3. **Output routing** - results can trigger further actions:
   - Send as email
   - Post to Slack
   - Store in document
   - Just log (for monitoring)
4. **Context injection** - scheduled prompts get full RAG/Web enrichment
5. **Admin UI** - view, pause, edit, delete scheduled prompts

### Security

- **Approval modes**: always-ask, pre-approved-list, never (disabled)
- **Per-alias allow-lists**: which actions each alias can perform (pattern matching: `email:draft_*`)
- **OAuth write scopes**: ✅ Implemented - Google and Microsoft OAuth now request write permissions
  - Google: `gmail.compose`, `calendar.events`, `tasks`, `contacts`, `drive.file`
  - Microsoft: `Mail.ReadWrite`, `Mail.Send`, `Calendars.ReadWrite`, `Tasks.ReadWrite`, etc.
- **Action audit log**: full history of all actions taken
- **Rate limiting**: prevent runaway scheduled prompts
- **Confirmation for destructive actions**: delete, send always require approval
- **Scheduled prompt limits**: max concurrent, max per day

### Extensibility

Smart Actions are designed as a plugin architecture:

```python
class ActionHandler(ABC):
    """Base class for action handlers."""
    
    @property
    @abstractmethod
    def action_type(self) -> str:
        """e.g., 'email', 'calendar', 'slack'"""
    
    @property
    @abstractmethod
    def supported_actions(self) -> list[str]:
        """e.g., ['send', 'draft', 'archive']"""
    
    @abstractmethod
    def validate(self, action: str, params: dict) -> tuple[bool, str]:
        """Validate action params before execution."""
    
    @abstractmethod
    def execute(self, action: str, params: dict, context: ActionContext) -> ActionResult:
        """Execute the action."""
    
    @abstractmethod
    def get_approval_summary(self, action: str, params: dict) -> str:
        """Human-readable summary for approval UI."""
```

**Adding new actions:**
1. Create handler class extending `ActionHandler`
2. Register in `actions/registry.py`
3. Add UI components if needed (optional)
4. Actions automatically available to LLMs via system prompt injection

**Future action ideas:**
- `homeassistant:*` - Smart home control
- `webhook:*` - Call arbitrary webhooks
- `database:*` - Query/update databases
- `code:*` - Execute sandboxed code
- `notification:*` - Push notifications (mobile, desktop)
- `payment:*` - Invoice, payment requests (with extra security)
- `social:*` - Post to Twitter/LinkedIn/etc.

---

# Implementation Priority

## Completed
- [x] Cross-encoder reranking
- [x] Jina integration (search, scraper, reranker)
- [x] Smart Aliases unification
- [x] Smart Tags
- [x] @relay query commands
- [x] Document sources (Notion, GitHub, Google, Paperless, Nextcloud, Websites, Todoist)
- [x] Persistent Memory (explicit user facts only)
- [x] Smart Source Selection (designator-controlled budget allocation)
- [x] Document Store Intelligence (themes, best_for, summary)
- [x] Smart Actions framework (detection, parsing, execution, action block stripping)
- [x] Smart Actions - Email (Gmail: draft, send, mark read/unread, archive, label)
- [x] Smart Actions - Calendar (Google: create, update, delete events)
- [x] Smart Actions - Notifications (push via configured URLs)
- [x] Smart Actions - Scheduled Prompts (calendar-triggered prompt execution)
- [x] Live Data Sources (real-time API queries for Gmail, Calendar, Stocks, Weather, Transport, etc.)
- [x] OAuth write scopes for Google (gmail.modify, calendar full access)

## Next Up (v1.9)
- [x] Slack integration (Document Store)
- [ ] Microsoft Graph integrations (Teams, OneDrive, OneNote, Outlook)
- [ ] Dropbox integration
- [ ] Jira / Linear integration
- [ ] IMAP generic email
- [ ] Exa search provider
- [ ] Smart Actions - Outlook email drafts

## Future (v2.0+)
- [ ] Smart Query Studio
- [ ] Model Sync service
- [ ] Media endpoints (images, audio)
- [ ] Smart Actions - Calendar, scheduled prompts, send actions
