"""
Unified Tasks action plugin.

Handles task actions across multiple providers:
- Google Tasks (via OAuth)
- Todoist (via API token)

The LLM uses generic actions (create, complete, update, delete, list)
and the system routes to the configured provider based on Smart Alias settings.

NO CONFIG FIELDS - all configuration comes from Smart Alias context at runtime:
- default_accounts["tasks"]["provider"] = "gtasks" or "todoist"
- default_accounts["tasks"]["id"] = OAuth account ID (for gtasks)
- default_accounts["tasks"]["api_token"] = API token (for todoist)
- default_accounts["tasks"]["list_id"] = default task list/project
"""

import logging
import os
from datetime import datetime, timezone
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


class TasksActionHandler(OAuthMixin, PluginActionHandler):
    """
    Unified task management across Google Tasks and Todoist.

    Provider is determined at runtime from Smart Alias context - no plugin config needed.
    """

    action_type = "tasks"
    display_name = "Tasks"
    description = "Create, complete, and manage tasks (Google Tasks or Todoist)"
    icon = "✅"
    category = "productivity"
    supported_sources = ["Google Tasks", "Todoist"]

    _abstract = False

    # API endpoints
    GTASKS_API_BASE = "https://tasks.googleapis.com/tasks/v1"
    TODOIST_API_BASE = "https://api.todoist.com/rest/v2"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """No config fields - everything comes from Smart Alias context."""
        return []

    @classmethod
    def get_resource_requirements(cls) -> list[ResourceRequirement]:
        """Define resources needed from Smart Alias."""
        return [
            ResourceRequirement(
                key="tasks",
                label="Tasks Account",
                resource_type=ResourceType.OAUTH_ACCOUNT,
                providers=["google"],  # GTasks via OAuth
                help_text="Google account for Google Tasks, or select Todoist below",
                required=False,
            ),
            ResourceRequirement(
                key="tasks",
                sub_key="provider",
                label="Task Provider",
                resource_type=ResourceType.SELECT,
                options=[
                    {"value": "gtasks", "label": "Google Tasks"},
                    {"value": "todoist", "label": "Todoist"},
                ],
                help_text="Which task service to use",
            ),
            ResourceRequirement(
                key="tasks",
                sub_key="list_id",
                label="Default Task List",
                resource_type=ResourceType.TASKLIST_PICKER,
                depends_on="tasks",
                help_text="Default task list for new tasks",
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="create",
                description="Create a new task",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="title",
                        label="Task Title",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Title of the task",
                    ),
                    FieldDefinition(
                        name="notes",
                        label="Notes/Description",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                        help_text="Additional details",
                    ),
                    FieldDefinition(
                        name="due",
                        label="Due Date",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Due date (YYYY-MM-DD or natural language for Todoist)",
                    ),
                    FieldDefinition(
                        name="priority",
                        label="Priority",
                        field_type=FieldType.INTEGER,
                        required=False,
                        help_text="Priority 1-4 (Todoist only, 4=urgent)",
                    ),
                    FieldDefinition(
                        name="list",
                        label="Task List/Project",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Task list (GTasks) or project name (Todoist)",
                    ),
                ],
                examples=[
                    {"title": "Review quarterly report", "due": "2026-01-25"},
                    {"title": "Call dentist", "due": "tomorrow", "list": "Personal"},
                ],
            ),
            ActionDefinition(
                name="complete",
                description="Mark a task as complete",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="task_id",
                        label="Task ID",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="The ID of the task to complete",
                    ),
                ],
                examples=[{"task_id": "abc123"}],
            ),
            ActionDefinition(
                name="update",
                description="Update an existing task",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="task_id",
                        label="Task ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="title",
                        label="New Title",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="notes",
                        label="New Notes",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                    FieldDefinition(
                        name="due",
                        label="New Due Date",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                ],
                examples=[{"task_id": "abc123", "due": "2026-02-01"}],
            ),
            ActionDefinition(
                name="delete",
                description="Delete a task permanently",
                risk=ActionRisk.DESTRUCTIVE,
                params=[
                    FieldDefinition(
                        name="task_id",
                        label="Task ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                ],
                examples=[{"task_id": "abc123"}],
            ),
            ActionDefinition(
                name="list",
                description="List tasks",
                risk=ActionRisk.READ_ONLY,
                params=[
                    FieldDefinition(
                        name="list",
                        label="Task List/Project",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Filter by list/project name",
                    ),
                    FieldDefinition(
                        name="limit",
                        label="Max Results",
                        field_type=FieldType.INTEGER,
                        required=False,
                        default=20,
                    ),
                ],
                examples=[{"list": "Work", "limit": 10}],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Static LLM instructions for tasks actions (no account info)."""
        return cls._build_instructions(None)

    def get_llm_instructions_with_context(
        self, available_accounts: Optional[dict[str, list[dict]]] = None
    ) -> str:
        """Dynamic LLM instructions with available accounts listed."""
        return self._build_instructions(available_accounts)

    @staticmethod
    def _build_instructions(
        available_accounts: Optional[dict[str, list[dict]]] = None,
    ) -> str:
        """Build LLM instructions, optionally with account info."""
        # Build account selection text if accounts available
        account_text = ""
        if available_accounts:
            tasks_accounts = available_accounts.get("tasks", [])
            if tasks_accounts:
                account_list = []
                for acc in tasks_accounts:
                    provider = acc.get("provider", "")
                    email = acc.get("email", "")
                    name = acc.get("name", "")  # Display name (friendly name)
                    slug = acc.get("slug", "")  # Store slug (identifier)
                    source_type = acc.get("source_type", "")

                    # Show both slug and display name so model can use either
                    # The slug is what appears in RAG context headers
                    if slug and name and slug != name:
                        # Different slug and display name - show both
                        if provider == "todoist" or source_type == "todoist":
                            account_list.append(f'  - "{slug}" or "{name}" (Todoist)')
                        elif email:
                            account_list.append(
                                f'  - "{slug}" or "{name}" ({email}, Google Tasks)'
                            )
                        else:
                            account_list.append(
                                f'  - "{slug}" or "{name}" (Google Tasks)'
                            )
                    elif slug or name:
                        # Same or only one available
                        identifier = slug or name
                        if provider == "todoist" or source_type == "todoist":
                            account_list.append(f'  - "{identifier}" (Todoist)')
                        elif email:
                            account_list.append(
                                f'  - "{identifier}" ({email}, Google Tasks)'
                            )
                        else:
                            account_list.append(f'  - "{identifier}" (Google Tasks)')
                    elif provider == "todoist":
                        account_list.append('  - "Todoist"')
                    elif email:
                        account_list.append(f'  - "{email}" (Google Tasks)')

                if account_list:
                    account_text = f"""
**Available Task Accounts:**
{chr(10).join(account_list)}

To use a specific account, add `"account": "Account Name"` to your action params.
If only one account is available, it will be used automatically.
"""

        return f"""## Tasks
{account_text}
Manage tasks in the user's task list (Google Tasks or Todoist).

### tasks:create
Create a new task.

**Parameters:**
- title (required): Task title
- notes (optional): Additional details
- due (optional): Due date - use YYYY-MM-DD format
- list (optional): Task list or project name
- account (optional): Which account to use

**Example:**
```xml
<smart_action type="tasks" action="create">
{{"title": "Review quarterly report", "due": "2026-01-25", "notes": "Check Q4 figures"}}
</smart_action>
```

### tasks:complete
Mark a task as complete.

**Parameters:**
- task_id (required): The task ID

**Example:**
```xml
<smart_action type="tasks" action="complete">
{{"task_id": "abc123"}}
</smart_action>
```

### tasks:update
Update an existing task.

**Parameters:**
- task_id (required): The task ID
- title, notes, due (optional): Fields to update

### tasks:delete
Delete a task permanently.

**Parameters:**
- task_id (required): The task ID

### tasks:list
List tasks from a list/project.

**Parameters:**
- list (optional): Filter by list/project name
- limit (optional): Max results (default 20)
"""

    def __init__(self, config: dict):
        """Initialize the tasks handler - config is ignored, uses context."""
        # These will be set from context at execution time
        self.provider: Optional[str] = None
        self.oauth_account_id: Optional[int] = None
        self.oauth_provider = "google"
        self.default_tasklist_id = "@default"
        self.todoist_api_token: Optional[str] = None
        self.default_project: Optional[str] = None

        # Caches
        self._tasklist_cache: dict[str, str] = {}
        self._project_cache: dict[str, str] = {}
        self._todoist_client: Optional[httpx.Client] = None

    def _find_account(
        self, account_identifier: str, context: ActionContext
    ) -> Optional[dict]:
        """
        Find an account by name, email, or provider.

        Args:
            account_identifier: Email address, account name, or provider (e.g., "todoist")
            context: Action context with available_accounts

        Returns:
            Account dict if found, None otherwise
        """
        available = context.available_accounts.get("tasks", [])
        identifier_lower = account_identifier.lower()

        logger.info(
            f"_find_account: Looking for '{identifier_lower}' in {len(available)} accounts"
        )
        for acc in available:
            logger.info(
                f"  - name={acc.get('name')}, provider={acc.get('provider')}, "
                f"email={acc.get('email')}, project_id={acc.get('project_id')}, "
                f"list_id={acc.get('list_id')}, tasklist_id={acc.get('tasklist_id')}"
            )

        for account in available:
            # Match by provider name (e.g., "todoist")
            provider = account.get("provider", "").lower()
            if provider == identifier_lower:
                return account

            # Match by email
            email = account.get("email", "").lower()
            if email == identifier_lower:
                return account

            # Match by name (display name)
            name = account.get("name", "").lower()
            if name == identifier_lower:
                return account

            # Match by slug (store identifier)
            slug = account.get("slug", "").lower()
            if slug == identifier_lower:
                return account

            # Match partial email (before @)
            if email and identifier_lower == email.split("@")[0]:
                return account

        return None

    def _get_available_accounts_str(self, context: ActionContext) -> str:
        """Get comma-separated list of available task accounts."""
        available = context.available_accounts.get("tasks", [])
        accounts = []
        for a in available:
            # Show slug or name, whichever is more useful
            slug = a.get("slug", "")
            name = a.get("name", "")
            if slug and slug != name:
                accounts.append(f"{slug} ({name})")
            elif name:
                accounts.append(name)
            elif a.get("email"):
                accounts.append(a.get("email"))
            elif a.get("provider"):
                accounts.append(a.get("provider"))
        return ", ".join(accounts) if accounts else "none"

    def _configure_from_params_or_context(
        self, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """
        Configure provider from params (account field) or fall back to default.

        The account can be specified by:
        - Store name (friendly name from document store)
        - Email address
        - Provider name (e.g., "todoist")

        Returns:
            (success, error_message)
        """
        # Check if LLM specified an account
        account_param = params.get("account")

        if account_param:
            # Look up the specified account
            account = self._find_account(account_param, context)
            if not account:
                available = self._get_available_accounts_str(context)
                return (
                    False,
                    f"Account '{account_param}' not found. Available: {available}",
                )

            # Configure based on account type
            source_type = account.get("source_type", "")
            logger.info(
                f"Tasks: Found account for '{account_param}': provider={account.get('provider')}, "
                f"source_type={source_type}, project_id={account.get('project_id')}, "
                f"list_id={account.get('list_id')}"
            )
            if account.get("provider") == "todoist" or source_type == "todoist":
                success, error = self._configure_todoist()
                if success:
                    # Set default project from account config (project_id field)
                    self.default_project = account.get("project_id") or account.get(
                        "list_id"
                    )
                    logger.info(
                        f"Tasks: Todoist default project from account: {self.default_project}"
                    )
                return success, error
            else:
                self.provider = "gtasks"
                # Store-based accounts use oauth_account_id, legacy uses id
                self.oauth_account_id = account.get(
                    "oauth_account_id", account.get("id")
                )
                self.oauth_provider = "google"
                self._init_oauth_client()
                # Use tasklist_id from account if available, else default
                self.default_tasklist_id = (
                    account.get("tasklist_id") or account.get("list_id") or "@default"
                )
                logger.info(
                    f"Tasks: GTasks default list from account: {self.default_tasklist_id}"
                )
                return True, ""
        else:
            # Check if there's exactly one available account - use it as default
            available_accounts = context.available_accounts.get("tasks", [])
            if len(available_accounts) == 1:
                account = available_accounts[0]
                source_type = account.get("source_type", "")
                if account.get("provider") == "todoist" or source_type == "todoist":
                    return self._configure_todoist()
                else:
                    self.provider = "gtasks"
                    self.oauth_account_id = account.get(
                        "oauth_account_id", account.get("id")
                    )
                    self.oauth_provider = "google"
                    self._init_oauth_client()
                    self.default_tasklist_id = account.get("tasklist_id", "@default")
                    logger.info(
                        f"Tasks: Using only available account: {account.get('name', account.get('email'))}"
                    )
                    return True, ""

            # Fall back to default account from context
            return self._configure_from_context(context)

    def _configure_todoist(self) -> tuple[bool, str]:
        """Configure Todoist provider."""
        api_token = os.environ.get("TODOIST_API_KEY") or os.environ.get(
            "TODOIST_API_TOKEN"
        )
        if api_token:
            self.provider = "todoist"
            self.todoist_api_token = api_token
            self._todoist_client = httpx.Client(
                base_url=self.TODOIST_API_BASE,
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=15,
            )
            return True, ""
        else:
            return False, "Todoist API token not configured"

    def _configure_from_context(self, context: ActionContext) -> tuple[bool, str]:
        """Configure provider and credentials from Smart Alias context defaults."""
        default_accounts = getattr(context, "default_accounts", {})
        tasks_config = default_accounts.get("tasks", {})

        if not tasks_config:
            # No default - check if there are any available accounts
            available = self._get_available_accounts_str(context)
            if available == "none":
                return False, "No task accounts available"
            return (
                False,
                f"No default task account configured. Specify 'account' parameter. Available: {available}",
            )

        # Get provider
        self.provider = tasks_config.get("provider", "gtasks")
        logger.info(f"Tasks: Using provider '{self.provider}' from context")

        if self.provider == "todoist":
            success, error = self._configure_todoist()
            if success:
                self.default_project = tasks_config.get("list_id")
            return success, error
        else:
            # Google Tasks uses OAuth
            account_id = tasks_config.get("id")
            if account_id:
                self.oauth_account_id = account_id
                self.oauth_provider = "google"
                self._init_oauth_client()
                self.default_tasklist_id = tasks_config.get("list_id", "@default")
                return True, ""
            else:
                return False, "Google Tasks OAuth account ID not found in context"

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the task action."""
        # Configure from params (account field) or fall back to context defaults
        success, error = self._configure_from_params_or_context(params, context)
        if not success:
            return ActionResult(
                success=False,
                message="",
                error=error,
            )

        try:
            if self.provider == "todoist":
                return self._execute_todoist(action, params)
            else:
                return self._execute_gtasks(action, params)
        except httpx.HTTPStatusError as e:
            logger.error(f"Tasks API error: {e.response.status_code}")
            error_text = e.response.text[:200] if e.response.text else str(e)
            return ActionResult(
                success=False,
                message="",
                error=f"API error ({e.response.status_code}): {error_text}",
            )
        except Exception as e:
            logger.exception(f"Tasks action failed: {action}")
            return ActionResult(
                success=False,
                message="",
                error=f"Action failed: {str(e)}",
            )

    # -------------------------------------------------------------------------
    # Google Tasks Implementation
    # -------------------------------------------------------------------------

    def _execute_gtasks(self, action: str, params: dict) -> ActionResult:
        """Execute action via Google Tasks API."""
        if action == "create":
            return self._gtasks_create(params)
        elif action == "complete":
            return self._gtasks_complete(params)
        elif action == "update":
            return self._gtasks_update(params)
        elif action == "delete":
            return self._gtasks_delete(params)
        elif action == "list":
            return self._gtasks_list(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _resolve_gtasks_list_id(self, list_name: Optional[str]) -> str:
        """Resolve task list name to ID."""
        if not list_name:
            return self.default_tasklist_id or "@default"

        if list_name.startswith("@") or " " not in list_name:
            if list_name.lower() in self._tasklist_cache:
                return self._tasklist_cache[list_name.lower()]
            return list_name

        name_lower = list_name.lower()
        if name_lower in self._tasklist_cache:
            return self._tasklist_cache[name_lower]

        try:
            response = self.oauth_get(f"{self.GTASKS_API_BASE}/users/@me/lists")
            response.raise_for_status()
            for tl in response.json().get("items", []):
                tl_name = tl.get("title", "").lower()
                self._tasklist_cache[tl_name] = tl["id"]
                if tl_name == name_lower:
                    return tl["id"]
        except Exception as e:
            logger.warning(f"Failed to resolve task list '{list_name}': {e}")

        return list_name

    def _gtasks_create(self, params: dict) -> ActionResult:
        """Create task via Google Tasks."""
        title = params.get("title")
        if not title:
            return ActionResult(
                success=False, message="", error="Task title is required"
            )

        tasklist_id = self._resolve_gtasks_list_id(params.get("list"))

        task_body = {"title": title}
        if params.get("notes"):
            task_body["notes"] = params["notes"]
        if params.get("due"):
            due = params["due"]
            if len(due) == 10:
                task_body["due"] = f"{due}T00:00:00.000Z"
            else:
                task_body["due"] = due

        response = self.oauth_post(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks",
            json=task_body,
        )
        response.raise_for_status()

        task = response.json()
        due_info = f" (due: {task['due'][:10]})" if task.get("due") else ""

        return ActionResult(
            success=True,
            message=f"Created task: {title}{due_info}",
            data={"task_id": task["id"], "provider": "gtasks"},
        )

    def _gtasks_complete(self, params: dict) -> ActionResult:
        """Complete task via Google Tasks."""
        task_id = params.get("task_id")
        if not task_id:
            return ActionResult(success=False, message="", error="Task ID is required")

        tasklist_id = self._resolve_gtasks_list_id(params.get("list"))

        get_response = self.oauth_get(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks/{task_id}"
        )
        if get_response.status_code != 200:
            return ActionResult(
                success=False, message="", error=f"Task not found: {task_id}"
            )

        task = get_response.json()
        task["status"] = "completed"
        task["completed"] = datetime.now(timezone.utc).isoformat()

        response = self.oauth_put(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks/{task_id}",
            json=task,
        )
        response.raise_for_status()

        return ActionResult(
            success=True,
            message=f"Completed task: {task.get('title', task_id)}",
            data={"task_id": task_id, "provider": "gtasks"},
        )

    def _gtasks_update(self, params: dict) -> ActionResult:
        """Update task via Google Tasks."""
        task_id = params.get("task_id")
        if not task_id:
            return ActionResult(success=False, message="", error="Task ID is required")

        tasklist_id = self._resolve_gtasks_list_id(params.get("list"))

        get_response = self.oauth_get(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks/{task_id}"
        )
        if get_response.status_code != 200:
            return ActionResult(
                success=False, message="", error=f"Task not found: {task_id}"
            )

        task = get_response.json()

        if params.get("title"):
            task["title"] = params["title"]
        if params.get("notes") is not None:
            task["notes"] = params["notes"]
        if params.get("due"):
            due = params["due"]
            task["due"] = f"{due}T00:00:00.000Z" if len(due) == 10 else due

        response = self.oauth_put(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks/{task_id}",
            json=task,
        )
        response.raise_for_status()

        return ActionResult(
            success=True,
            message=f"Updated task: {task.get('title', task_id)}",
            data={"task_id": task_id, "provider": "gtasks"},
        )

    def _gtasks_delete(self, params: dict) -> ActionResult:
        """Delete task via Google Tasks."""
        task_id = params.get("task_id")
        if not task_id:
            return ActionResult(success=False, message="", error="Task ID is required")

        tasklist_id = self._resolve_gtasks_list_id(params.get("list"))

        response = self.oauth_delete(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks/{task_id}"
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to delete: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message=f"Deleted task {task_id}",
            data={"task_id": task_id, "provider": "gtasks"},
        )

    def _gtasks_list(self, params: dict) -> ActionResult:
        """List tasks via Google Tasks."""
        tasklist_id = self._resolve_gtasks_list_id(params.get("list"))
        max_results = min(params.get("limit", 20), 100)

        response = self.oauth_get(
            f"{self.GTASKS_API_BASE}/lists/{tasklist_id}/tasks",
            params={"maxResults": max_results, "showCompleted": "false"},
        )
        response.raise_for_status()

        tasks = response.json().get("items", [])

        task_list = []
        for task in tasks:
            if not task.get("title", "").strip():
                continue
            due_str = f" (due: {task['due'][:10]})" if task.get("due") else ""
            task_list.append(f"- [{task['id'][:12]}...] {task['title']}{due_str}")

        message = (
            f"Found {len(task_list)} task(s):\n" + "\n".join(task_list)
            if task_list
            else "No tasks found"
        )

        return ActionResult(
            success=True,
            message=message,
            data={"tasks": tasks, "count": len(tasks), "provider": "gtasks"},
        )

    # -------------------------------------------------------------------------
    # Todoist Implementation
    # -------------------------------------------------------------------------

    def _execute_todoist(self, action: str, params: dict) -> ActionResult:
        """Execute action via Todoist API."""
        if action == "create":
            return self._todoist_create(params)
        elif action == "complete":
            return self._todoist_complete(params)
        elif action == "update":
            return self._todoist_update(params)
        elif action == "delete":
            return self._todoist_delete(params)
        elif action == "list":
            return self._todoist_list(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _resolve_todoist_project_id(self, project_name: Optional[str]) -> Optional[str]:
        """Resolve project name to ID."""
        logger.info(
            f"_resolve_todoist_project_id: project_name={project_name}, default_project={self.default_project}"
        )
        if not project_name:
            project_name = self.default_project
        if not project_name:
            logger.info(
                "_resolve_todoist_project_id: No project specified, returning None"
            )
            return None

        # If it's already a numeric ID, return it directly
        if str(project_name).isdigit():
            logger.info(
                f"_resolve_todoist_project_id: Using numeric ID directly: {project_name}"
            )
            return str(project_name)

        name_lower = project_name.lower()
        if name_lower in self._project_cache:
            return self._project_cache[name_lower]

        try:
            response = self._todoist_client.get("/projects")
            response.raise_for_status()
            for project in response.json():
                self._project_cache[project["name"].lower()] = project["id"]
                if project["name"].lower() == name_lower:
                    return project["id"]
        except Exception as e:
            logger.warning(f"Failed to resolve project '{project_name}': {e}")

        return None

    def _todoist_create(self, params: dict) -> ActionResult:
        """Create task via Todoist."""
        title = params.get("title")
        if not title:
            return ActionResult(
                success=False, message="", error="Task title is required"
            )

        data = {"content": title}

        if params.get("notes"):
            data["description"] = params["notes"]
        if params.get("due"):
            data["due_string"] = params["due"]
        if params.get("priority"):
            data["priority"] = int(params["priority"])

        project_id = self._resolve_todoist_project_id(params.get("list"))
        logger.info(
            f"_todoist_create: params.list={params.get('list')}, resolved project_id={project_id}"
        )
        if project_id:
            data["project_id"] = project_id

        logger.info(f"_todoist_create: Sending to Todoist API: {data}")
        response = self._todoist_client.post("/tasks", json=data)
        response.raise_for_status()

        task = response.json()
        due_info = f" (due: {task['due']['string']})" if task.get("due") else ""

        return ActionResult(
            success=True,
            message=f"Created task: {task['content']}{due_info}",
            data={"task_id": task["id"], "provider": "todoist"},
        )

    def _todoist_complete(self, params: dict) -> ActionResult:
        """Complete task via Todoist."""
        task_id = params.get("task_id")
        if not task_id:
            return ActionResult(success=False, message="", error="Task ID is required")

        response = self._todoist_client.post(f"/tasks/{task_id}/close")
        response.raise_for_status()

        return ActionResult(
            success=True,
            message=f"Completed task {task_id}",
            data={"task_id": task_id, "provider": "todoist"},
        )

    def _todoist_update(self, params: dict) -> ActionResult:
        """Update task via Todoist."""
        task_id = params.get("task_id")
        if not task_id:
            return ActionResult(success=False, message="", error="Task ID is required")

        data = {}
        if params.get("title"):
            data["content"] = params["title"]
        if params.get("notes") is not None:
            data["description"] = params["notes"]
        if params.get("due"):
            data["due_string"] = params["due"]
        if params.get("priority"):
            data["priority"] = int(params["priority"])

        if not data:
            return ActionResult(success=False, message="", error="No fields to update")

        response = self._todoist_client.post(f"/tasks/{task_id}", json=data)
        response.raise_for_status()

        task = response.json()
        return ActionResult(
            success=True,
            message=f"Updated task: {task['content']}",
            data={"task_id": task_id, "provider": "todoist"},
        )

    def _todoist_delete(self, params: dict) -> ActionResult:
        """Delete task via Todoist."""
        task_id = params.get("task_id")
        if not task_id:
            return ActionResult(success=False, message="", error="Task ID is required")

        response = self._todoist_client.delete(f"/tasks/{task_id}")
        response.raise_for_status()

        return ActionResult(
            success=True,
            message=f"Deleted task {task_id}",
            data={"task_id": task_id, "provider": "todoist"},
        )

    def _todoist_list(self, params: dict) -> ActionResult:
        """List tasks via Todoist."""
        limit = params.get("limit", 20)
        project_name = params.get("list")

        request_params = {}
        if project_name:
            request_params["filter"] = f"##{project_name}"

        response = self._todoist_client.get("/tasks", params=request_params)
        response.raise_for_status()

        tasks = response.json()[:limit]

        task_list = []
        for task in tasks:
            due_str = f" (due: {task['due']['string']})" if task.get("due") else ""
            task_list.append(f"- [{task['id']}] {task['content']}{due_str}")

        message = (
            f"Found {len(tasks)} task(s):\n" + "\n".join(task_list)
            if task_list
            else "No tasks found"
        )

        return ActionResult(
            success=True,
            message=message,
            data={"tasks": tasks, "count": len(tasks), "provider": "todoist"},
        )

    # -------------------------------------------------------------------------
    # Common methods
    # -------------------------------------------------------------------------

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action == "create":
            if not params.get("title"):
                errors.append(ValidationError("title", "Task title is required"))
        elif action in ("complete", "update", "delete"):
            if not params.get("task_id"):
                errors.append(ValidationError("task_id", "Task ID is required"))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary."""
        if action == "create":
            title = params.get("title", "?")
            due = f" (due: {params['due']})" if params.get("due") else ""
            return f'Create task: "{title}"{due}'
        elif action == "complete":
            return f"Mark task {params.get('task_id', '?')[:20]} as complete"
        elif action == "update":
            return f"Update task {params.get('task_id', '?')[:20]}"
        elif action == "delete":
            return f"DELETE task {params.get('task_id', '?')[:20]} (permanent)"
        elif action == "list":
            list_name = params.get("list", "default")
            return f"List tasks from {list_name}"

        return f"Tasks: {action}"

    def is_available(self) -> bool:
        """Check if plugin is available (always true - config comes from context)."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection - cannot test without context."""
        return (
            True,
            "Tasks handler ready (provider configured per Smart Alias)",
        )

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_oauth_client") and self._oauth_client:
            self._oauth_client.close()
        if self._todoist_client:
            self._todoist_client.close()
