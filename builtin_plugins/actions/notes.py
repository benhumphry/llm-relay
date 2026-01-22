"""
Notes action plugin.

Handles note creation and management across multiple providers:
- Notion (via API token)
- OneNote (via Microsoft OAuth)

Unlike tasks, notes use a SINGLE default store configured on the Smart Alias,
rather than allowing the LLM to choose between multiple accounts.

Configuration comes from Smart Alias context at runtime:
- default_accounts["notes"]["store_id"] = Document store ID
- default_accounts["notes"]["provider"] = "notion" or "onenote"
- default_accounts["notes"]["oauth_account_id"] = OAuth account ID (for onenote)
"""

import logging
import os
import re
from typing import Optional

import httpx

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
    ResourceRequirement,
    ResourceType,
)
from plugin_base.common import (
    FieldDefinition,
    FieldType,
    ValidationError,
    ValidationResult,
)
from plugin_base.oauth import OAuthMixin

logger = logging.getLogger(__name__)


class NotesActionHandler(OAuthMixin, PluginActionHandler):
    """
    Note creation and management across Notion and OneNote.

    Provider is determined from the default notes store configured on the Smart Alias.
    """

    action_type = "notes"
    display_name = "Notes"
    description = "Create and manage notes (Notion or OneNote)"
    icon = "📝"
    category = "productivity"
    supported_sources = ["Notion", "OneNote"]
    supported_source_types = ["notion", "onenote"]

    _abstract = False

    # API endpoints
    NOTION_API_BASE = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"
    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """No config fields - everything comes from Smart Alias context."""
        return []

    @classmethod
    def get_resource_requirements(cls) -> list[ResourceRequirement]:
        """Define resources needed from Smart Alias."""
        return [
            ResourceRequirement(
                key="notes",
                label="Notes Store",
                resource_type=ResourceType.DOCUMENT_STORE,
                help_text="Document store for note creation (Notion database or OneNote notebook)",
                required=False,
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="create",
                description="Create a new note",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="title",
                        label="Title",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Note title",
                    ),
                    FieldDefinition(
                        name="content",
                        label="Content",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                        help_text="Note content (markdown supported)",
                    ),
                    FieldDefinition(
                        name="folder",
                        label="Folder/Section",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Target folder or section (uses default if not specified)",
                    ),
                ],
            ),
            ActionDefinition(
                name="update",
                description="Update an existing note",
                risk=ActionRisk.MEDIUM,
                params=[
                    FieldDefinition(
                        name="note_id",
                        label="Note ID",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Note/page ID",
                    ),
                    FieldDefinition(
                        name="title",
                        label="Title",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Search by title if note_id not provided",
                    ),
                    FieldDefinition(
                        name="content",
                        label="New Content",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                        help_text="New content to append or replace",
                    ),
                    FieldDefinition(
                        name="append",
                        label="Append Mode",
                        field_type=FieldType.BOOLEAN,
                        required=False,
                        default=True,
                        help_text="Append to existing content (default) or replace",
                    ),
                ],
            ),
            ActionDefinition(
                name="delete",
                description="Delete a note",
                risk=ActionRisk.HIGH,
                params=[
                    FieldDefinition(
                        name="note_id",
                        label="Note ID",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Note/page ID",
                    ),
                    FieldDefinition(
                        name="title",
                        label="Title",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Search by title if note_id not provided",
                    ),
                ],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Static LLM instructions for notes actions."""
        return cls._build_instructions(None)

    def get_llm_instructions_with_context(
        self, available_accounts: Optional[dict[str, list[dict]]] = None
    ) -> str:
        """Dynamic LLM instructions - notes use default store, no account selection needed."""
        return self._build_instructions(available_accounts)

    @staticmethod
    def _build_instructions(
        available_accounts: Optional[dict[str, list[dict]]] = None,
    ) -> str:
        """Build LLM instructions for notes actions."""
        store_info = ""
        if available_accounts:
            notes_accounts = available_accounts.get("notes", [])
            if notes_accounts and len(notes_accounts) > 0:
                acc = notes_accounts[0]
                store_name = acc.get("name", "Notes")
                provider = acc.get("provider", "")
                store_info = (
                    f"\n**Notes will be saved to:** {store_name} ({provider})\n"
                )

        return f"""## Notes
{store_info}
Create and manage notes.

### notes:create
Create a new note.

**Parameters:**
- title (required): Note title
- content (required): Note content (markdown formatting supported)
- folder (optional): Target folder or section name

**Example:**
```xml
<smart_action type="notes" action="create">
{{"title": "Meeting Notes - Project X", "content": "## Key Points\\n\\n- Budget approved\\n- Timeline: Q2 2026\\n- Next steps: schedule kickoff"}}
</smart_action>
```

### notes:update
Update an existing note (append content by default).

**Parameters:**
- note_id OR title (required): Identify the note
- content: New content to add
- append: true (default) to append, false to replace

**Example:**
```xml
<smart_action type="notes" action="update">
{{"title": "Meeting Notes - Project X", "content": "\\n## Follow-up (Jan 22)\\n\\n- Confirmed budget allocation"}}
</smart_action>
```

### notes:delete
Delete a note.

**Parameters:**
- note_id OR title (required): Identify the note to delete

**Example:**
```xml
<smart_action type="notes" action="delete">
{{"title": "Old Draft Notes"}}
</smart_action>
```
"""

    def __init__(self, config: dict):
        """Initialize the handler."""
        super().__init__(config)
        self.provider: Optional[str] = None
        self.store_id: Optional[int] = None

        # Notion
        self.notion_api_token: Optional[str] = None
        self.notion_database_id: Optional[str] = None
        self._notion_client: Optional[httpx.Client] = None

        # OneNote (Microsoft Graph)
        self.onenote_oauth_account_id: Optional[int] = None
        self.onenote_notebook_id: Optional[str] = None
        self.onenote_section_id: Optional[str] = None
        self._graph_client: Optional[httpx.Client] = None

    def _configure_from_context(self, context: ActionContext) -> tuple[bool, str]:
        """Configure provider from Smart Alias context."""
        default_accounts = context.default_accounts or {}
        notes_config = default_accounts.get("notes", {})

        if not notes_config:
            return False, "No notes store configured on this Smart Alias"

        provider = notes_config.get("provider", "")
        self.store_id = notes_config.get("store_id")

        if provider == "notion":
            return self._configure_notion(notes_config)
        elif provider == "onenote":
            return self._configure_onenote(notes_config, context)
        else:
            return False, f"Unknown notes provider: {provider}"

    def _configure_notion(self, config: dict) -> tuple[bool, str]:
        """Configure Notion provider."""
        api_token = os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
        if not api_token:
            return False, "Notion API token not configured (NOTION_TOKEN)"

        database_id = config.get("database_id")
        if not database_id:
            return False, "Notion database ID not configured"

        self.provider = "notion"
        self.notion_api_token = api_token
        self.notion_database_id = database_id
        self._notion_client = httpx.Client(
            base_url=self.NOTION_API_BASE,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Notion-Version": self.NOTION_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        return True, ""

    def _configure_onenote(
        self, config: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Configure OneNote provider via Microsoft Graph."""
        oauth_account_id = config.get("oauth_account_id")
        if not oauth_account_id:
            return False, "OneNote OAuth account not configured"

        # Get access token
        access_token = self._get_oauth_token("microsoft", oauth_account_id)
        if not access_token:
            return False, "Failed to get Microsoft OAuth token"

        self.provider = "onenote"
        self.onenote_oauth_account_id = oauth_account_id
        self.onenote_notebook_id = config.get("notebook_id")
        self.onenote_section_id = config.get("section_id")

        self._graph_client = httpx.Client(
            base_url=self.GRAPH_API_BASE,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        return True, ""

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action == "create":
            if not params.get("title"):
                errors.append(
                    ValidationError(field="title", message="Title is required")
                )
            if not params.get("content"):
                errors.append(
                    ValidationError(field="content", message="Content is required")
                )

        elif action in ("update", "delete"):
            if not params.get("note_id") and not params.get("title"):
                errors.append(
                    ValidationError(
                        field="note_id", message="Either note_id or title is required"
                    )
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute a notes action."""
        # Configure from context
        success, error = self._configure_from_context(context)
        if not success:
            return ActionResult(success=False, message="", error=error)

        try:
            if self.provider == "notion":
                return self._execute_notion(action, params)
            elif self.provider == "onenote":
                return self._execute_onenote(action, params)
            else:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Provider not configured: {self.provider}",
                )
        except Exception as e:
            logger.exception(f"Notes action failed: {e}")
            return ActionResult(
                success=False, message="", error=f"Action failed: {str(e)}"
            )
        finally:
            # Cleanup clients
            if self._notion_client:
                self._notion_client.close()
                self._notion_client = None
            if self._graph_client:
                self._graph_client.close()
                self._graph_client = None

    # =========================================================================
    # Notion Implementation
    # =========================================================================

    def _get_notion_title_property_name(self) -> Optional[str]:
        """Get the name of the title property in the Notion database."""
        try:
            db_response = self._notion_client.get(
                f"/databases/{self.notion_database_id}"
            )
            db_response.raise_for_status()
            db_props = db_response.json().get("properties", {})

            for prop_name, prop_info in db_props.items():
                if prop_info.get("type") == "title":
                    return prop_name
            return None
        except Exception as e:
            logger.error(f"Error getting Notion database schema: {e}")
            return None

    def _execute_notion(self, action: str, params: dict) -> ActionResult:
        """Execute action via Notion API."""
        if action == "create":
            return self._notion_create(params)
        elif action == "update":
            return self._notion_update(params)
        elif action == "delete":
            return self._notion_delete(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _notion_create(self, params: dict) -> ActionResult:
        """Create a new page in Notion database."""
        title = params.get("title", "")
        content = params.get("content", "")

        # Get the actual title property name from database schema
        title_prop_name = self._get_notion_title_property_name()
        if not title_prop_name:
            return ActionResult(
                success=False,
                message="",
                error="Could not determine title property name in Notion database",
            )

        # Build page properties using the actual title property name
        properties = {title_prop_name: {"title": [{"text": {"content": title}}]}}

        # Convert markdown content to Notion blocks
        children = self._markdown_to_notion_blocks(content)

        payload = {
            "parent": {"database_id": self.notion_database_id},
            "properties": properties,
            "children": children,
        }

        try:
            response = self._notion_client.post("/pages", json=payload)
            response.raise_for_status()
            data = response.json()
            page_id = data.get("id", "")

            return ActionResult(
                success=True,
                message=f"Created note: {title}",
                data={"page_id": page_id, "title": title},
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Notion create failed: {e} - {error_body}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create note: {error_body}",
            )

    def _notion_update(self, params: dict) -> ActionResult:
        """Update an existing Notion page."""
        note_id = params.get("note_id")
        title = params.get("title")
        content = params.get("content", "")
        append = params.get("append", True)

        # Find page by title if no ID
        if not note_id and title:
            note_id = self._find_notion_page_by_title(title)
            if not note_id:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Note not found: {title}",
                )

        if not note_id:
            return ActionResult(
                success=False,
                message="",
                error="No note_id or title provided",
            )

        try:
            if content:
                # Convert markdown to blocks
                new_blocks = self._markdown_to_notion_blocks(content)

                if append:
                    # Append blocks to existing page
                    response = self._notion_client.patch(
                        f"/blocks/{note_id}/children", json={"children": new_blocks}
                    )
                else:
                    # To replace, we need to delete existing blocks first
                    # Get existing blocks
                    blocks_response = self._notion_client.get(
                        f"/blocks/{note_id}/children"
                    )
                    blocks_response.raise_for_status()
                    existing_blocks = blocks_response.json().get("results", [])

                    # Delete each block
                    for block in existing_blocks:
                        self._notion_client.delete(f"/blocks/{block['id']}")

                    # Add new blocks
                    response = self._notion_client.patch(
                        f"/blocks/{note_id}/children", json={"children": new_blocks}
                    )

                response.raise_for_status()

            return ActionResult(
                success=True,
                message=f"Updated note: {title or note_id}",
                data={"page_id": note_id},
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Notion update failed: {e} - {error_body}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to update note: {error_body}",
            )

    def _notion_delete(self, params: dict) -> ActionResult:
        """Delete (archive) a Notion page."""
        note_id = params.get("note_id")
        title = params.get("title")

        # Find page by title if no ID
        if not note_id and title:
            note_id = self._find_notion_page_by_title(title)
            if not note_id:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Note not found: {title}",
                )

        if not note_id:
            return ActionResult(
                success=False,
                message="",
                error="No note_id or title provided",
            )

        try:
            # Archive the page (Notion doesn't have true delete via API)
            response = self._notion_client.patch(
                f"/pages/{note_id}", json={"archived": True}
            )
            response.raise_for_status()

            return ActionResult(
                success=True,
                message=f"Deleted note: {title or note_id}",
                data={"page_id": note_id},
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Notion delete failed: {e} - {error_body}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to delete note: {error_body}",
            )

    def _find_notion_page_by_title(self, title: str) -> Optional[str]:
        """Find a Notion page by title (case-insensitive)."""
        try:
            # Get the actual title property name from database schema
            title_prop_name = self._get_notion_title_property_name()
            if not title_prop_name:
                logger.error("No title property found in Notion database")
                return None

            # Query the database for pages with matching title
            response = self._notion_client.post(
                f"/databases/{self.notion_database_id}/query",
                json={
                    "filter": {
                        "property": title_prop_name,
                        "title": {"contains": title},
                    },
                    "page_size": 10,
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            # Find exact or close match
            title_lower = title.lower()
            for page in results:
                page_title = ""
                title_prop = page.get("properties", {}).get(title_prop_name, {})
                if title_prop.get("title"):
                    page_title = "".join(
                        t.get("plain_text", "") for t in title_prop["title"]
                    )

                if (
                    page_title.lower() == title_lower
                    or title_lower in page_title.lower()
                ):
                    return page["id"]

            return None
        except Exception as e:
            logger.error(f"Error finding Notion page: {e}")
            return None

    def _markdown_to_notion_blocks(self, markdown: str) -> list[dict]:
        """Convert markdown text to Notion blocks."""
        blocks = []
        lines = markdown.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Headings
            if line.startswith("### "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[4:]}}
                            ]
                        },
                    }
                )
            elif line.startswith("## "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[3:]}}
                            ]
                        },
                    }
                )
            elif line.startswith("# "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[2:]}}
                            ]
                        },
                    }
                )
            # Bullet list
            elif line.startswith("- ") or line.startswith("* "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[2:]}}
                            ]
                        },
                    }
                )
            # Numbered list
            elif re.match(r"^\d+\.\s", line):
                content = re.sub(r"^\d+\.\s", "", line)
                blocks.append(
                    {
                        "object": "block",
                        "type": "numbered_list_item",
                        "numbered_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": content}}
                            ]
                        },
                    }
                )
            # Code block
            elif line.startswith("```"):
                code_lines = []
                language = line[3:].strip() or "plain text"
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                blocks.append(
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": "\n".join(code_lines)},
                                }
                            ],
                            "language": language,
                        },
                    }
                )
            # Quote
            elif line.startswith("> "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "quote",
                        "quote": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[2:]}}
                            ]
                        },
                    }
                )
            # Regular paragraph
            else:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": line}}]
                        },
                    }
                )

            i += 1

        return blocks

    # =========================================================================
    # OneNote Implementation
    # =========================================================================

    def _execute_onenote(self, action: str, params: dict) -> ActionResult:
        """Execute action via Microsoft Graph API (OneNote)."""
        if action == "create":
            return self._onenote_create(params)
        elif action == "update":
            return self._onenote_update(params)
        elif action == "delete":
            return self._onenote_delete(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _onenote_create(self, params: dict) -> ActionResult:
        """Create a new OneNote page."""
        title = params.get("title", "")
        content = params.get("content", "")
        folder = params.get("folder")  # Section name

        # Determine target section
        section_id = self.onenote_section_id
        if folder:
            # Try to find section by name
            found_section = self._find_onenote_section(folder)
            if found_section:
                section_id = found_section

        if not section_id:
            # Use default section from first notebook
            section_id = self._get_default_onenote_section()
            if not section_id:
                return ActionResult(
                    success=False,
                    message="",
                    error="No OneNote section found. Please create a notebook first.",
                )

        # Convert markdown to HTML
        html_content = self._markdown_to_html(content)

        # OneNote page content is HTML
        page_html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
</head>
<body>
{html_content}
</body>
</html>"""

        try:
            response = self._graph_client.post(
                f"/me/onenote/sections/{section_id}/pages",
                content=page_html.encode("utf-8"),
                headers={"Content-Type": "application/xhtml+xml"},
            )
            response.raise_for_status()
            data = response.json()
            page_id = data.get("id", "")

            return ActionResult(
                success=True,
                message=f"Created note: {title}",
                data={"page_id": page_id, "title": title},
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"OneNote create failed: {e} - {error_body}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to create note: {error_body}",
            )

    def _onenote_update(self, params: dict) -> ActionResult:
        """Update a OneNote page (append content)."""
        note_id = params.get("note_id")
        title = params.get("title")
        content = params.get("content", "")

        # Find page by title if no ID
        if not note_id and title:
            note_id = self._find_onenote_page_by_title(title)
            if not note_id:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Note not found: {title}",
                )

        if not note_id:
            return ActionResult(
                success=False,
                message="",
                error="No note_id or title provided",
            )

        if not content:
            return ActionResult(
                success=True,
                message="No content to update",
            )

        # Convert markdown to HTML
        html_content = self._markdown_to_html(content)

        # OneNote PATCH uses specific format
        # Note: OneNote API only supports appending, not replacing
        patch_content = [
            {
                "target": "body",
                "action": "append",
                "content": f"<div>{html_content}</div>",
            }
        ]

        try:
            response = self._graph_client.patch(
                f"/me/onenote/pages/{note_id}/content", json=patch_content
            )
            response.raise_for_status()

            return ActionResult(
                success=True,
                message=f"Updated note: {title or note_id}",
                data={"page_id": note_id},
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"OneNote update failed: {e} - {error_body}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to update note: {error_body}",
            )

    def _onenote_delete(self, params: dict) -> ActionResult:
        """Delete a OneNote page."""
        note_id = params.get("note_id")
        title = params.get("title")

        # Find page by title if no ID
        if not note_id and title:
            note_id = self._find_onenote_page_by_title(title)
            if not note_id:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Note not found: {title}",
                )

        if not note_id:
            return ActionResult(
                success=False,
                message="",
                error="No note_id or title provided",
            )

        try:
            response = self._graph_client.delete(f"/me/onenote/pages/{note_id}")
            response.raise_for_status()

            return ActionResult(
                success=True,
                message=f"Deleted note: {title or note_id}",
                data={"page_id": note_id},
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"OneNote delete failed: {e} - {error_body}")
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to delete note: {error_body}",
            )

    def _find_onenote_section(self, name: str) -> Optional[str]:
        """Find a OneNote section by name."""
        try:
            response = self._graph_client.get("/me/onenote/sections")
            response.raise_for_status()
            sections = response.json().get("value", [])

            name_lower = name.lower()
            for section in sections:
                if section.get("displayName", "").lower() == name_lower:
                    return section["id"]

            return None
        except Exception as e:
            logger.error(f"Error finding OneNote section: {e}")
            return None

    def _get_default_onenote_section(self) -> Optional[str]:
        """Get the default (first) OneNote section."""
        try:
            # If notebook ID is configured, get sections from that notebook
            if self.onenote_notebook_id:
                response = self._graph_client.get(
                    f"/me/onenote/notebooks/{self.onenote_notebook_id}/sections"
                )
            else:
                # Get all sections
                response = self._graph_client.get("/me/onenote/sections")

            response.raise_for_status()
            sections = response.json().get("value", [])

            if sections:
                return sections[0]["id"]

            return None
        except Exception as e:
            logger.error(f"Error getting default OneNote section: {e}")
            return None

    def _find_onenote_page_by_title(self, title: str) -> Optional[str]:
        """Find a OneNote page by title."""
        try:
            # Search pages by title
            response = self._graph_client.get(
                "/me/onenote/pages", params={"$filter": f"contains(title, '{title}')"}
            )
            response.raise_for_status()
            pages = response.json().get("value", [])

            title_lower = title.lower()
            for page in pages:
                page_title = page.get("title", "")
                if (
                    page_title.lower() == title_lower
                    or title_lower in page_title.lower()
                ):
                    return page["id"]

            return None
        except Exception as e:
            logger.error(f"Error finding OneNote page: {e}")
            return None

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to simple HTML for OneNote."""
        html_lines = []
        lines = markdown.split("\n")
        in_code_block = False
        code_lines = []

        for line in lines:
            # Code block handling
            if line.startswith("```"):
                if in_code_block:
                    html_lines.append(
                        f"<pre><code>{'<br/>'.join(code_lines)}</code></pre>"
                    )
                    code_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(self._escape_html(line))
                continue

            # Empty line
            if not line.strip():
                html_lines.append("<br/>")
                continue

            # Headings
            if line.startswith("### "):
                html_lines.append(f"<h3>{self._escape_html(line[4:])}</h3>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{self._escape_html(line[3:])}</h2>")
            elif line.startswith("# "):
                html_lines.append(f"<h1>{self._escape_html(line[2:])}</h1>")
            # Bullet list
            elif line.startswith("- ") or line.startswith("* "):
                html_lines.append(f"<ul><li>{self._escape_html(line[2:])}</li></ul>")
            # Numbered list
            elif re.match(r"^\d+\.\s", line):
                content = re.sub(r"^\d+\.\s", "", line)
                html_lines.append(f"<ol><li>{self._escape_html(content)}</li></ol>")
            # Quote
            elif line.startswith("> "):
                html_lines.append(
                    f"<blockquote>{self._escape_html(line[2:])}</blockquote>"
                )
            # Regular paragraph
            else:
                # Handle inline formatting
                text = self._escape_html(line)
                # Bold
                text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                # Italic
                text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
                # Inline code
                text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
                html_lines.append(f"<p>{text}</p>")

        return "\n".join(html_lines)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
