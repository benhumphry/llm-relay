"""
Todoist action plugin - create and manage tasks.

This serves as the reference implementation for action plugins.
"""

import logging

import httpx

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
)
from plugin_base.common import FieldDefinition, FieldType

logger = logging.getLogger(__name__)


class TodoistActionHandler(PluginActionHandler):
    """Create and manage Todoist tasks."""

    action_type = "todoist"
    display_name = "Todoist"
    description = "Create, complete, and manage tasks in Todoist"
    icon = "✅"
    category = "productivity"

    # Not abstract - can be registered
    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="api_token",
                label="API Token",
                field_type=FieldType.PASSWORD,
                required=True,
                help_text="From Todoist Settings > Integrations > Developer > API token",
            ),
            FieldDefinition(
                name="default_project",
                label="Default Project",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Project name for new tasks (leave empty for Inbox)",
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="create",
                description="Create a new task in Todoist",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="content",
                        label="Task title",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="description",
                        label="Description",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                    FieldDefinition(
                        name="due_string",
                        label="Due date (natural language)",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="e.g., 'tomorrow', 'next monday', 'Jan 15 2pm'",
                    ),
                    FieldDefinition(
                        name="priority",
                        label="Priority (1=low, 4=urgent)",
                        field_type=FieldType.INTEGER,
                        required=False,
                        default=1,
                        min_value=1,
                        max_value=4,
                    ),
                    FieldDefinition(
                        name="project",
                        label="Project name",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="labels",
                        label="Labels (comma-separated)",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                ],
                examples=[
                    {
                        "content": "Review quarterly report",
                        "due_string": "tomorrow 2pm",
                        "priority": 2,
                    },
                    {
                        "content": "Call dentist",
                        "due_string": "next monday",
                        "labels": "health,calls",
                    },
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
                    ),
                ],
                examples=[{"task_id": "123456789"}],
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
                        name="content",
                        label="New title",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="description",
                        label="New description",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                    FieldDefinition(
                        name="due_string",
                        label="New due date",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="priority",
                        label="New priority",
                        field_type=FieldType.INTEGER,
                        required=False,
                        min_value=1,
                        max_value=4,
                    ),
                ],
                examples=[
                    {"task_id": "123456789", "due_string": "next friday", "priority": 3}
                ],
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
                examples=[{"task_id": "123456789"}],
            ),
            ActionDefinition(
                name="list",
                description="List tasks, optionally filtered by project",
                risk=ActionRisk.READ_ONLY,
                params=[
                    FieldDefinition(
                        name="project",
                        label="Project name (optional filter)",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="limit",
                        label="Maximum tasks to return",
                        field_type=FieldType.INTEGER,
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=50,
                    ),
                ],
                examples=[{"project": "Work", "limit": 5}],
            ),
        ]

    def __init__(self, config: dict):
        self.api_token = config["api_token"]
        self.default_project = config.get("default_project")
        self.client = httpx.Client(
            base_url="https://api.todoist.com/rest/v2",
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=15,
        )
        # Cache for project name -> ID mapping
        self._project_cache: dict[str, str] = {}

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        try:
            if action == "create":
                return self._create_task(params)
            elif action == "complete":
                return self._complete_task(params)
            elif action == "update":
                return self._update_task(params)
            elif action == "delete":
                return self._delete_task(params)
            elif action == "list":
                return self._list_tasks(params)
            else:
                return ActionResult(
                    success=False, message="", error=f"Unknown action: {action}"
                )
        except httpx.HTTPStatusError as e:
            error_msg = f"Todoist API error: {e.response.status_code}"
            try:
                error_body = e.response.json()
                if "error" in error_body:
                    error_msg = f"Todoist: {error_body['error']}"
            except Exception:
                pass
            logger.error(f"Todoist API error: {e}")
            return ActionResult(success=False, message="", error=error_msg)
        except httpx.TimeoutException:
            return ActionResult(success=False, message="", error="Todoist API timeout")
        except Exception as e:
            logger.error(f"Todoist action error: {e}", exc_info=True)
            return ActionResult(success=False, message="", error=str(e))

    def _create_task(self, params: dict) -> ActionResult:
        data = {"content": params["content"]}

        if params.get("description"):
            data["description"] = params["description"]
        if params.get("due_string"):
            data["due_string"] = params["due_string"]
        if params.get("priority"):
            data["priority"] = int(params["priority"])
        if params.get("labels"):
            labels = params["labels"]
            if isinstance(labels, str):
                labels = [l.strip() for l in labels.split(",")]
            data["labels"] = labels

        # Resolve project name to ID if provided
        project_name = params.get("project") or self.default_project
        if project_name:
            project_id = self._resolve_project_id(project_name)
            if project_id:
                data["project_id"] = project_id
            else:
                logger.warning(f"Project not found: {project_name}, using Inbox")

        response = self.client.post("/tasks", json=data)
        response.raise_for_status()
        task = response.json()

        due_info = ""
        if task.get("due"):
            due_info = (
                f" (due: {task['due'].get('string', task['due'].get('date', ''))})"
            )

        return ActionResult(
            success=True,
            message=f"Created task: {task['content']}{due_info}",
            data={"task_id": task["id"], "url": task.get("url")},
        )

    def _complete_task(self, params: dict) -> ActionResult:
        task_id = params["task_id"]
        response = self.client.post(f"/tasks/{task_id}/close")
        response.raise_for_status()

        return ActionResult(success=True, message=f"Completed task {task_id}")

    def _update_task(self, params: dict) -> ActionResult:
        task_id = params["task_id"]

        # Build update data from provided params
        data = {}
        if params.get("content"):
            data["content"] = params["content"]
        if params.get("description") is not None:
            data["description"] = params["description"]
        if params.get("due_string"):
            data["due_string"] = params["due_string"]
        if params.get("priority"):
            data["priority"] = int(params["priority"])

        if not data:
            return ActionResult(success=False, message="", error="No fields to update")

        response = self.client.post(f"/tasks/{task_id}", json=data)
        response.raise_for_status()
        task = response.json()

        return ActionResult(
            success=True,
            message=f"Updated task: {task['content']}",
            data={"task_id": task["id"]},
        )

    def _delete_task(self, params: dict) -> ActionResult:
        task_id = params["task_id"]
        response = self.client.delete(f"/tasks/{task_id}")
        response.raise_for_status()

        return ActionResult(success=True, message=f"Deleted task {task_id}")

    def _list_tasks(self, params: dict) -> ActionResult:
        limit = params.get("limit", 10)
        project_name = params.get("project")

        # Build filter
        filter_query = None
        if project_name:
            filter_query = f"##{project_name}"

        request_params = {}
        if filter_query:
            request_params["filter"] = filter_query

        response = self.client.get("/tasks", params=request_params)
        response.raise_for_status()
        tasks = response.json()

        # Limit results
        tasks = tasks[:limit]

        # Format for display
        task_list = []
        for task in tasks:
            due_str = ""
            if task.get("due"):
                due_str = (
                    f" (due: {task['due'].get('string', task['due'].get('date', ''))})"
                )
            task_list.append(f"- [{task['id']}] {task['content']}{due_str}")

        message = f"Found {len(tasks)} task(s)"
        if project_name:
            message += f" in {project_name}"
        message += ":\n" + "\n".join(task_list) if task_list else ""

        return ActionResult(
            success=True,
            message=message,
            data={"tasks": tasks, "count": len(tasks)},
        )

    def _resolve_project_id(self, project_name: str) -> str | None:
        """Resolve project name to ID, with caching."""
        # Normalize name for comparison
        name_lower = project_name.lower()

        # Check cache first
        if name_lower in self._project_cache:
            return self._project_cache[name_lower]

        try:
            response = self.client.get("/projects")
            response.raise_for_status()
            projects = response.json()

            for project in projects:
                # Cache all projects while we're at it
                self._project_cache[project["name"].lower()] = project["id"]

                if project["name"].lower() == name_lower:
                    return project["id"]

            return None
        except Exception as e:
            logger.error(f"Failed to resolve project '{project_name}': {e}")
            return None

    def get_approval_summary(self, action: str, params: dict) -> str:
        if action == "create":
            due = f" (due: {params['due_string']})" if params.get("due_string") else ""
            project = f" in {params.get('project', self.default_project or 'Inbox')}"
            return f'Create Todoist task: "{params.get("content", "?")}"{due}{project}'
        elif action == "complete":
            return f"Mark Todoist task {params.get('task_id')} as complete"
        elif action == "update":
            changes = []
            if params.get("content"):
                changes.append(f'title="{params["content"]}"')
            if params.get("due_string"):
                changes.append(f'due="{params["due_string"]}"')
            if params.get("priority"):
                changes.append(f"priority={params['priority']}")
            return f"Update Todoist task {params.get('task_id')}: {', '.join(changes) or 'no changes'}"
        elif action == "delete":
            return f"⚠️ DELETE Todoist task {params.get('task_id')} (permanent)"
        elif action == "list":
            project = f" in {params['project']}" if params.get("project") else ""
            return f"List Todoist tasks{project}"
        return super().get_approval_summary(action, params)

    def test_connection(self) -> tuple[bool, str]:
        try:
            response = self.client.get("/projects")
            response.raise_for_status()
            projects = response.json()
            return True, f"Connected. Found {len(projects)} projects."
        except httpx.HTTPStatusError as e:
            return False, f"API error: {e.response.status_code}"
        except Exception as e:
            return False, str(e)

    def is_available(self) -> bool:
        return bool(self.api_token)
