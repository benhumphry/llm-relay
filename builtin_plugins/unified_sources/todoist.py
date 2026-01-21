"""
Todoist Unified Source Plugin.

Combines Todoist document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index tasks by project/label for semantic search
- Live side: Query pending tasks, due today, overdue, search by content
- Intelligent routing: Analyze queries to choose optimal data source
- Actions: Link to todoist action plugin for create/complete/update

Query routing examples:
- "tasks due today" -> Live only (time-sensitive status)
- "completed tasks from last month" -> RAG only (historical)
- "what did I need to do for the project?" -> Both, prefer RAG
- "my todo list" -> Live only (current state)
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class TodoistUnifiedSource(PluginUnifiedSource):
    """
    Unified Todoist source - RAG for history, Live for current state.

    Single configuration provides:
    - Document indexing: Tasks indexed by project for RAG semantic search
    - Live queries: Current pending tasks, due today, overdue, search
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "todoist"
    display_name = "Todoist"
    description = "Todoist with historical search (RAG) and real-time queries"
    category = "productivity"
    icon = "✅"

    # Document store types this unified source handles
    handles_doc_source_types = ["todoist"]

    supports_rag = True
    supports_live = True
    supports_actions = True  # Links to todoist action plugin
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "api_token": os.environ.get("TODOIST_API_TOKEN", ""),
            "project_ids": store.todoist_project_id or "",
            "filter_expression": store.todoist_filter or "",
            "include_completed": store.todoist_include_completed or False,
        }

    # Todoist API endpoints
    TODOIST_API_BASE = "https://api.todoist.com/rest/v2"
    TODOIST_SYNC_API = "https://api.todoist.com/sync/v9"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="api_token",
                label="API Token",
                field_type=FieldType.PASSWORD,
                required=True,
                help_text="From Todoist Settings > Integrations > Developer > API token",
            ),
            FieldDefinition(
                name="index_project_ids",
                label="Projects to Index",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated project IDs to index (empty = all projects)",
            ),
            FieldDefinition(
                name="include_completed",
                label="Include Completed Tasks",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index completed tasks for historical search",
            ),
            FieldDefinition(
                name="completed_since_days",
                label="Completed Tasks History (days)",
                field_type=FieldType.INTEGER,
                default=90,
                help_text="How many days of completed task history to index (max 365)",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 * * * *", "label": "Hourly"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                ],
                help_text="How often to re-index tasks",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=50,
                help_text="Maximum tasks to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: pending, due_today, overdue, all, search",
                param_type="string",
                required=False,
                default="pending",
                examples=["pending", "due_today", "overdue", "search"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for task content",
                param_type="string",
                required=False,
                examples=["project review", "call", "meeting prep"],
            ),
            ParamDefinition(
                name="project",
                description="Specific project name to query",
                param_type="string",
                required=False,
                examples=["Work", "Personal", "Shopping"],
            ),
            ParamDefinition(
                name="filter",
                description="Todoist filter query (advanced)",
                param_type="string",
                required=False,
                examples=["today | overdue", "p1", "@urgent"],
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum tasks to return",
                param_type="integer",
                required=False,
                default=50,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.api_token = config.get("api_token")
        self.index_project_ids = []
        if config.get("index_project_ids"):
            self.index_project_ids = [
                p.strip() for p in config["index_project_ids"].split(",") if p.strip()
            ]
        self.include_completed = config.get("include_completed", True)
        self.completed_since_days = min(config.get("completed_since_days", 90), 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 50)

        self._client = httpx.Client(
            timeout=30,
            headers={"Authorization": f"Bearer {self.api_token}"},
        )
        self._project_cache: dict[str, dict] = {}  # id -> project data
        self._project_name_cache: dict[str, str] = {}  # name -> id

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate tasks for indexing.

        Lists active tasks and optionally completed tasks.
        """
        if not self.api_token:
            logger.error("Cannot list tasks - no API token")
            return

        # Refresh project cache
        self._refresh_project_cache()

        # List active tasks
        logger.info("Listing active Todoist tasks for indexing")
        active_count = 0

        try:
            response = self._client.get(f"{self.TODOIST_API_BASE}/tasks")
            response.raise_for_status()
            tasks = response.json()

            for task in tasks:
                # Filter by project if configured
                if self.index_project_ids:
                    if task.get("project_id") not in self.index_project_ids:
                        continue

                active_count += 1
                project_name = self._get_project_name(task.get("project_id", ""))

                yield DocumentInfo(
                    uri=f"todoist://{task['id']}",
                    title=f"{project_name}: {task['content'][:80]}",
                    mime_type="text/plain",
                    metadata={
                        "project_id": task.get("project_id"),
                        "project_name": project_name,
                        "is_completed": False,
                    },
                )

            logger.info(f"Found {active_count} active tasks")

        except Exception as e:
            logger.error(f"Failed to list active tasks: {e}")

        # List completed tasks if enabled
        if self.include_completed:
            logger.info(
                f"Listing completed Todoist tasks (last {self.completed_since_days} days)"
            )
            completed_count = 0

            try:
                # Use Sync API for completed tasks
                since = datetime.now(timezone.utc) - timedelta(
                    days=self.completed_since_days
                )
                since_str = since.strftime("%Y-%m-%dT%H:%M:%S")

                # Get completed tasks - paginate through results
                offset = 0
                limit = 100

                while True:
                    response = self._client.get(
                        f"{self.TODOIST_SYNC_API}/completed/get_all",
                        params={
                            "since": since_str,
                            "limit": limit,
                            "offset": offset,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    items = data.get("items", [])
                    if not items:
                        break

                    for task in items:
                        # Filter by project if configured
                        if self.index_project_ids:
                            if task.get("project_id") not in self.index_project_ids:
                                continue

                        completed_count += 1
                        project_name = self._get_project_name(
                            task.get("project_id", "")
                        )

                        yield DocumentInfo(
                            uri=f"todoist://completed/{task['task_id']}",
                            title=f"{project_name}: {task['content'][:80]}",
                            mime_type="text/plain",
                            modified_at=task.get("completed_at"),
                            metadata={
                                "project_id": task.get("project_id"),
                                "project_name": project_name,
                                "is_completed": True,
                            },
                        )

                    offset += limit
                    if len(items) < limit:
                        break

                logger.info(f"Found {completed_count} completed tasks")

            except Exception as e:
                logger.error(f"Failed to list completed tasks: {e}")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read task content for indexing.

        Fetches the task and formats it for embedding.
        """
        if not uri.startswith("todoist://"):
            logger.error(f"Invalid Todoist URI: {uri}")
            return None

        is_completed = "/completed/" in uri

        if is_completed:
            # Completed task - format from what we know
            task_id = uri.replace("todoist://completed/", "")
            # For completed tasks, we need to get from sync API
            try:
                response = self._client.get(
                    f"{self.TODOIST_SYNC_API}/items/get",
                    params={"item_id": task_id},
                )
                if response.status_code != 200:
                    logger.warning(f"Could not fetch completed task {task_id}")
                    return None
                task = response.json().get("item", {})
            except Exception as e:
                logger.error(f"Failed to fetch completed task {task_id}: {e}")
                return None
        else:
            # Active task
            task_id = uri.replace("todoist://", "")
            try:
                response = self._client.get(f"{self.TODOIST_API_BASE}/tasks/{task_id}")
                response.raise_for_status()
                task = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch task {task_id}: {e}")
                return None

        # Extract task details
        content = task.get("content", "(No title)")
        description = task.get("description", "")
        priority = task.get("priority", 1)
        due = task.get("due", {})
        labels = task.get("labels", [])
        project_id = task.get("project_id", "")

        # Get project name
        project_name = self._get_project_name(project_id)

        # Format priority
        priority_names = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}
        priority_text = priority_names.get(priority, "Normal")

        # Format as readable text
        content_parts = [f"Task: {content}", f"Project: {project_name}"]

        if priority > 1:
            content_parts.append(f"Priority: {priority_text}")

        if due:
            due_str = due.get("string") or due.get("date", "")
            content_parts.append(f"Due: {due_str}")

        if labels:
            content_parts.append(f"Labels: {', '.join(labels)}")

        if is_completed:
            content_parts.append("Status: Completed")
            if task.get("completed_at"):
                content_parts.append(f"Completed: {task['completed_at'][:10]}")
        else:
            content_parts.append("Status: Pending")

        if description:
            content_parts.append(f"\nDescription:\n{description}")

        content_text = "\n".join(content_parts)

        # Extract due date for metadata
        due_date = None
        if due:
            due_date = due.get("date")

        return DocumentContent(
            content=content_text,
            mime_type="text/plain",
            metadata={
                "task_id": task_id,
                "project_id": project_id,
                "project_name": project_name,
                "priority": priority,
                "is_completed": is_completed,
                "due_date": due_date,
                "labels": ",".join(labels),  # Join list for ChromaDB compatibility
                "source_type": "task",
            },
        )

    def _refresh_project_cache(self) -> None:
        """Refresh the project cache."""
        try:
            response = self._client.get(f"{self.TODOIST_API_BASE}/projects")
            response.raise_for_status()
            projects = response.json()

            self._project_cache = {}
            self._project_name_cache = {}

            for project in projects:
                self._project_cache[project["id"]] = project
                self._project_name_cache[project["name"].lower()] = project["id"]

        except Exception as e:
            logger.error(f"Failed to refresh project cache: {e}")

    def _get_project_name(self, project_id: str) -> str:
        """Get project name by ID."""
        if not project_id:
            return "Inbox"

        if project_id in self._project_cache:
            return self._project_cache[project_id].get("name", "Unknown")

        # Refresh cache and try again
        self._refresh_project_cache()
        if project_id in self._project_cache:
            return self._project_cache[project_id].get("name", "Unknown")

        return "Unknown"

    def _resolve_project_id(self, name: str) -> Optional[str]:
        """Resolve project name to ID."""
        name_lower = name.lower()

        if name_lower in self._project_name_cache:
            return self._project_name_cache[name_lower]

        # Refresh cache
        self._refresh_project_cache()

        return self._project_name_cache.get(name_lower)

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live task data.

        Supports actions: pending, due_today, overdue, all, search
        """
        start_time = time.time()

        if not self.api_token:
            return LiveDataResult(
                success=False,
                error="No Todoist API token configured",
            )

        action = params.get("action", "pending")
        search_query = params.get("query", "")
        project_name = params.get("project", "")
        filter_query = params.get("filter", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            # Refresh project cache for name resolution
            self._refresh_project_cache()

            # Fetch tasks based on action/filter
            tasks = self._fetch_tasks_live(
                action, search_query, project_name, filter_query, max_results
            )

            # Format for LLM context
            formatted = self._format_tasks(tasks, action, project_name)

            latency_ms = int((time.time() - start_time) * 1000)

            return LiveDataResult(
                success=True,
                data=tasks,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Todoist live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_tasks_live(
        self,
        action: str,
        search_query: str,
        project_name: str,
        filter_query: str,
        max_results: int,
    ) -> list[dict]:
        """Fetch tasks from Todoist API."""
        today = datetime.now(timezone.utc).date()

        # Build filter based on action
        if filter_query:
            # User provided explicit filter
            todoist_filter = filter_query
        elif action == "due_today":
            todoist_filter = "today"
        elif action == "overdue":
            todoist_filter = "overdue"
        elif project_name:
            todoist_filter = f"#{project_name}"
        else:
            todoist_filter = None  # Get all tasks

        # Fetch tasks
        params = {}
        if todoist_filter:
            params["filter"] = todoist_filter

        try:
            response = self._client.get(f"{self.TODOIST_API_BASE}/tasks", params=params)
            response.raise_for_status()
            tasks = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch tasks: {e}")
            return []

        # Add project names to tasks
        for task in tasks:
            task["project_name"] = self._get_project_name(task.get("project_id", ""))

        # Filter by search query if provided
        if search_query:
            query_lower = search_query.lower()
            tasks = [
                t
                for t in tasks
                if query_lower in t.get("content", "").lower()
                or query_lower in t.get("description", "").lower()
            ]

        # Sort by due date, then priority
        def sort_key(t):
            due = t.get("due", {})
            due_date = due.get("date", "9999-99-99") if due else "9999-99-99"
            priority = 5 - t.get("priority", 1)  # Higher priority first
            return (due_date, priority)

        tasks.sort(key=sort_key)

        return tasks[:max_results]

    def _format_tasks(self, tasks: list[dict], action: str, project_name: str) -> str:
        """Format tasks for LLM context."""
        if not tasks:
            action_msgs = {
                "pending": "No pending tasks.",
                "due_today": "No tasks due today.",
                "overdue": "No overdue tasks.",
                "search": "No tasks found matching your search.",
                "all": "No tasks found.",
            }
            return (
                f"### Todoist ({action})\n{action_msgs.get(action, 'No tasks found.')}"
            )

        action_titles = {
            "pending": "Pending Tasks",
            "due_today": "Tasks Due Today",
            "overdue": "Overdue Tasks",
            "search": "Task Search Results",
            "all": "All Tasks",
        }

        lines = [f"### {action_titles.get(action, 'Todoist Tasks')}"]
        if project_name:
            lines.append(f"Project: {project_name}")
        lines.append(f"Found {len(tasks)} task(s):\n")

        today = datetime.now(timezone.utc).date()

        for task in tasks:
            task_id = task.get("id", "")
            content = task.get("content", "(No title)")
            description = task.get("description", "")
            priority = task.get("priority", 1)
            due = task.get("due", {})
            labels = task.get("labels", [])
            proj_name = task.get("project_name", "Inbox")

            # Priority marker
            priority_markers = {1: "", 2: "!", 3: "!!", 4: "!!!"}
            priority_mark = priority_markers.get(priority, "")
            if priority_mark:
                priority_mark = f" {priority_mark}"

            lines.append(f"**{content}**{priority_mark}")
            lines.append(f"   ID: {task_id}")
            lines.append(f"   Project: {proj_name}")

            if due:
                due_date_str = due.get("date", "")
                due_display = due.get("string") or due_date_str

                # Check if overdue
                if due_date_str:
                    try:
                        due_date = datetime.strptime(
                            due_date_str[:10], "%Y-%m-%d"
                        ).date()
                        if due_date < today:
                            lines.append(f"   Due: {due_display} ⚠️ OVERDUE")
                        elif due_date == today:
                            lines.append(f"   Due: {due_display} (TODAY)")
                        else:
                            lines.append(f"   Due: {due_display}")
                    except Exception:
                        lines.append(f"   Due: {due_display}")
                else:
                    lines.append(f"   Due: {due_display}")

            if labels:
                lines.append(f"   Labels: {', '.join(labels)}")

            if description:
                desc_preview = description.replace("\n", " ").strip()
                if len(desc_preview) > 100:
                    desc_preview = desc_preview[:100] + "..."
                lines.append(f"   Notes: {desc_preview}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Query Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze query to determine optimal routing.

        Routing logic:
        - Current state queries (pending, overdue, due today) -> Live only
        - Completed/historical queries -> RAG only
        - Search queries -> Both, merge
        - Default -> Both with deduplication
        """
        query_lower = query.lower()
        action = params.get("action", "")

        # Current state queries -> Live only (need real-time status)
        current_state_patterns = [
            "pending",
            "to do",
            "todo",
            "to-do",
            "due today",
            "overdue",
            "what.*need to do",
            "my tasks",
            "my todo",
            "what's on my",
            "what do i have",
            "inbox",
        ]
        if action in ["pending", "due_today", "overdue"] or any(
            re.search(p, query_lower) for p in current_state_patterns
        ):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": action or "pending"},
                reason="Current task status requires live API",
                max_live_results=self.live_max_results,
            )

        # Completed/historical queries -> RAG only
        historical_patterns = [
            "completed",
            "finished",
            "done",
            "last month",
            "last year",
            r"20\d{2}",
            "what did i",
            "what have i",
        ]
        if any(re.search(p, query_lower) for p in historical_patterns):
            return QueryAnalysis(
                routing=QueryRouting.RAG_ONLY,
                rag_query=query,
                reason="Historical task query - using RAG index",
                max_rag_results=20,
            )

        # Search queries -> Both, merge
        if params.get("query") or action == "search":
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=params.get("query", query),
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.DEDUPE,
                reason="Search query - checking both sources",
                max_rag_results=15,
                max_live_results=self.live_max_results,
            )

        # Project-specific queries -> Both, prefer live
        if params.get("project"):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params=params,
                merge_strategy=MergeStrategy.LIVE_FIRST,
                reason="Project query - live for current, RAG for context",
                max_rag_results=10,
                max_live_results=self.live_max_results,
            )

        # Default -> Both with deduplication
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.LIVE_FIRST,
            reason="General query - using both sources, prefer live for current state",
            max_rag_results=10,
            max_live_results=self.live_max_results,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Todoist is accessible."""
        return bool(self.api_token)

    def test_connection(self) -> tuple[bool, str]:
        """Test Todoist API connection."""
        results = []
        overall_success = True

        try:
            # Test API access - get projects
            response = self._client.get(f"{self.TODOIST_API_BASE}/projects")
            response.raise_for_status()
            projects = response.json()
            results.append(f"Connected. Found {len(projects)} project(s)")

            # List project names
            if projects:
                names = [p.get("name", "Untitled") for p in projects[:5]]
                results.append(f"Projects: {', '.join(names)}")

            # Test document listing (RAG side)
            if self.supports_rag:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found tasks to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "pending", "max_results": 5})
                    if live_result.success:
                        task_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {task_count} pending tasks")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except httpx.HTTPStatusError as e:
            return False, f"API error: {e.response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
