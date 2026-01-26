"""
Google Tasks Unified Source Plugin.

Combines Google Tasks document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index tasks by list/status for semantic search
- Live side: Query pending tasks, due today, overdue, search by content
- Smart routing: Analyze queries to choose optimal data source
- Actions: Link to task action plugin for create/complete/update

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

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.oauth import OAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class GTasksUnifiedSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Google Tasks source - RAG for history, Live for current state.

    Single configuration provides:
    - Document indexing: Tasks indexed by list for RAG semantic search
    - Live queries: Current pending tasks, due today, overdue, search
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "gtasks"
    display_name = "Google Tasks"
    description = "Google Tasks with historical search (RAG) and real-time queries"
    category = "google"
    icon = "✅"
    content_category = ContentCategory.TASKS

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:gtasks"]

    # Live data source types this unified source handles (for legacy live sources)
    handles_live_source_types = ["google_tasks_live"]

    supports_rag = True
    supports_live = True
    supports_actions = True  # Can link to task action plugin
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results
    default_index_days = 365  # Index a full year of tasks

    _abstract = False

    @classmethod
    def get_account_info(cls, store) -> dict | None:
        """Extract account info for action handlers."""
        if not store.google_account_id:
            return None

        # Get email from OAuth token
        try:
            from db.oauth_tokens import get_oauth_token_info

            token_info = get_oauth_token_info(store.google_account_id)
            email = token_info.get("account_email", "") if token_info else ""
        except Exception:
            email = ""

        return {
            "provider": "google",
            "email": email,
            "name": store.display_name or store.name,
            "store_id": store.id,
            "oauth_account_id": store.google_account_id,
            # Tasks-specific fields
            "tasklist_id": store.gtasks_tasklist_id,
        }

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME Google Tasks access. Actions: "
            "action='pending' for incomplete tasks, "
            "action='due_today' for tasks due today, "
            "action='overdue' for overdue tasks, "
            "action='completed' for completed tasks, "
            "action='search' with query='...' to search task content, "
            "action='all' for all tasks. "
            "Optional: tasklist='ListName' to filter by task list."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.google_account_id,
            "tasklist_ids": store.gtasks_tasklist_id or "",
            "include_completed": False,
            "index_schedule": store.index_schedule or "",
        }

    # Tasks API endpoints
    TASKS_API_BASE = "https://tasks.googleapis.com/tasks/v1"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "google", "scopes": ["tasks"]},
                help_text="Select a connected Google account with Tasks access",
            ),
            FieldDefinition(
                name="index_tasklist_ids",
                label="Task Lists to Index",
                field_type=FieldType.MULTISELECT,
                required=False,
                help_text="Which task lists to index for RAG (empty = all lists)",
                picker_options={"provider": "google", "type": "tasklists"},
            ),
            FieldDefinition(
                name="include_completed",
                label="Include Completed Tasks",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index completed tasks for historical search",
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
                description="Query type: pending, due_today, overdue, completed, all, search",
                param_type="string",
                required=False,
                default="pending",
                examples=["pending", "due_today", "overdue", "completed", "search"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for task content",
                param_type="string",
                required=False,
                examples=["project review", "call", "meeting prep"],
            ),
            ParamDefinition(
                name="tasklist",
                description="Specific task list name to query",
                param_type="string",
                required=False,
                examples=["Work", "Personal", "Shopping"],
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
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"

        self.index_tasklist_ids = config.get("index_tasklist_ids", [])
        self.include_completed = config.get("include_completed", True)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 50)

        self._client = httpx.Client(timeout=30)
        self._init_oauth_client()
        self._tasklist_cache: dict[str, str] = {}  # name -> id mapping

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate tasks for indexing.

        Lists tasks from configured task lists.
        """
        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot list tasks - no valid access token")
            return

        # Get task lists
        tasklists = self._get_task_lists()
        if self.index_tasklist_ids:
            tasklists = [tl for tl in tasklists if tl["id"] in self.index_tasklist_ids]

        logger.info(f"Listing tasks from {len(tasklists)} task list(s)")

        total_tasks = 0

        for tasklist in tasklists:
            tasklist_id = tasklist.get("id")
            tasklist_title = tasklist.get("title", "Untitled")

            page_token = None
            while True:
                params = {
                    "maxResults": 100,
                    "showCompleted": str(self.include_completed).lower(),
                    "showHidden": "true",
                }
                if page_token:
                    params["pageToken"] = page_token

                try:
                    response = self._oauth_client.get(
                        f"{self.TASKS_API_BASE}/lists/{tasklist_id}/tasks",
                        headers=self._get_auth_headers(),
                        params=params,
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.error(f"Tasks API error: {e}")
                    break

                tasks = data.get("items", [])

                for task in tasks:
                    task_id = task.get("id")
                    title = task.get("title", "")
                    updated = task.get("updated")

                    if not title.strip():
                        continue  # Skip empty tasks

                    total_tasks += 1
                    yield DocumentInfo(
                        uri=f"gtasks://{tasklist_id}/{task_id}",
                        title=f"{tasklist_title}: {title[:80]}",
                        mime_type="text/plain",
                        modified_at=updated,
                        metadata={
                            "tasklist_id": tasklist_id,
                            "tasklist_title": tasklist_title,
                        },
                    )

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        logger.info(f"Found {total_tasks} tasks to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read task content for indexing.

        Fetches the task and formats it for embedding.
        """
        if not uri.startswith("gtasks://"):
            logger.error(f"Invalid Tasks URI: {uri}")
            return None

        parts = uri.replace("gtasks://", "").split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid Tasks URI format: {uri}")
            return None

        tasklist_id, task_id = parts

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot read task - no valid access token")
            return None

        try:
            response = self._oauth_client.get(
                f"{self.TASKS_API_BASE}/lists/{tasklist_id}/tasks/{task_id}",
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            task = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch task {task_id}: {e}")
            return None

        # Extract task details
        title = task.get("title", "(No title)")
        notes = task.get("notes", "")
        status = task.get("status", "needsAction")
        due = task.get("due", "")
        completed = task.get("completed", "")
        updated = task.get("updated", "")

        # Format status nicely
        status_text = "Completed" if status == "completed" else "Pending"

        # Format as readable text
        content_parts = [f"Task: {title}", f"Status: {status_text}"]

        if due:
            due_date = due[:10] if len(due) >= 10 else due
            content_parts.append(f"Due: {due_date}")

        if completed:
            completed_date = completed[:10] if len(completed) >= 10 else completed
            content_parts.append(f"Completed: {completed_date}")

        if notes:
            content_parts.append(f"\nNotes:\n{notes}")

        content = "\n".join(content_parts)

        # Get tasklist name for metadata
        tasklist_name = self._get_tasklist_name(tasklist_id)

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "task_id": task_id,
                "tasklist_id": tasklist_id,
                "tasklist_name": tasklist_name,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "status": status,
                "is_completed": status == "completed",
                "due_date": due[:10] if due and len(due) >= 10 else None,
                "source_type": "task",
            },
        )

    def _get_task_lists(self) -> list[dict]:
        """Get all task lists for the user."""
        tasklists = []
        page_token = None

        while True:
            params = {"maxResults": 100}
            if page_token:
                params["pageToken"] = page_token

            try:
                response = self._oauth_client.get(
                    f"{self.TASKS_API_BASE}/users/@me/lists",
                    headers=self._get_auth_headers(),
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to get task lists: {e}")
                break

            items = data.get("items", [])
            tasklists.extend(items)

            # Cache name -> id mapping
            for tl in items:
                self._tasklist_cache[tl.get("title", "").lower()] = tl.get("id")

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return tasklists

    def _get_tasklist_name(self, tasklist_id: str) -> str:
        """Get task list name by ID."""
        try:
            response = self._oauth_client.get(
                f"{self.TASKS_API_BASE}/users/@me/lists/{tasklist_id}",
                headers=self._get_auth_headers(),
            )
            if response.status_code == 200:
                return response.json().get("title", "Unknown")
        except Exception:
            pass
        return "Unknown"

    def _resolve_tasklist_id(self, name: str) -> Optional[str]:
        """Resolve task list name to ID."""
        name_lower = name.lower()

        # Check cache first
        if name_lower in self._tasklist_cache:
            return self._tasklist_cache[name_lower]

        # Refresh cache
        self._get_task_lists()

        return self._tasklist_cache.get(name_lower)

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live task data.

        Supports actions: pending, due_today, overdue, completed, all, search
        """
        start_time = time.time()

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            return LiveDataResult(
                success=False,
                error="No valid Google Tasks access token",
            )

        action = params.get("action", "pending")
        search_query = params.get("query", "")
        tasklist_name = params.get("tasklist", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            # Resolve task list if specified
            tasklist_id = None
            if tasklist_name:
                tasklist_id = self._resolve_tasklist_id(tasklist_name)
                if not tasklist_id:
                    return LiveDataResult(
                        success=False,
                        error=f"Task list not found: {tasklist_name}",
                    )

            # Fetch tasks
            tasks = self._fetch_tasks_live(action, tasklist_id, max_results)

            # Filter by search query if provided
            if search_query and action == "search":
                query_lower = search_query.lower()
                tasks = [
                    t
                    for t in tasks
                    if query_lower in t.get("title", "").lower()
                    or query_lower in t.get("notes", "").lower()
                ]

            # Format for LLM context
            formatted = self._format_tasks(tasks, action, tasklist_name)

            latency_ms = int((time.time() - start_time) * 1000)

            return LiveDataResult(
                success=True,
                data=tasks,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Tasks live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_tasks_live(
        self, action: str, tasklist_id: Optional[str], max_results: int
    ) -> list[dict]:
        """Fetch tasks from Google Tasks API."""
        # Get task lists to query
        if tasklist_id:
            tasklists = [{"id": tasklist_id}]
        else:
            tasklists = self._get_task_lists()

        today = datetime.now(timezone.utc).date()
        all_tasks = []

        for tasklist in tasklists:
            tl_id = tasklist.get("id")

            # Set parameters based on action
            params = {"maxResults": 100}

            if action == "completed":
                params["showCompleted"] = "true"
                params["showHidden"] = "true"
            elif action in ["pending", "due_today", "overdue", "search"]:
                params["showCompleted"] = "false"
            else:  # all
                params["showCompleted"] = "true"
                params["showHidden"] = "true"

            page_token = None
            while True:
                if page_token:
                    params["pageToken"] = page_token

                try:
                    response = self._oauth_client.get(
                        f"{self.TASKS_API_BASE}/lists/{tl_id}/tasks",
                        headers=self._get_auth_headers(),
                        params=params,
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.warning(f"Failed to fetch tasks from list {tl_id}: {e}")
                    break

                tasks = data.get("items", [])

                for task in tasks:
                    if not task.get("title", "").strip():
                        continue

                    # Add tasklist info
                    task["tasklist_id"] = tl_id
                    task["tasklist_title"] = tasklist.get(
                        "title", self._get_tasklist_name(tl_id)
                    )

                    # Filter by action
                    if action == "completed":
                        if task.get("status") == "completed":
                            all_tasks.append(task)
                    elif action == "due_today":
                        due = task.get("due", "")
                        if due:
                            due_date = datetime.fromisoformat(
                                due.replace("Z", "+00:00")
                            ).date()
                            if due_date == today:
                                all_tasks.append(task)
                    elif action == "overdue":
                        due = task.get("due", "")
                        if due and task.get("status") != "completed":
                            due_date = datetime.fromisoformat(
                                due.replace("Z", "+00:00")
                            ).date()
                            if due_date < today:
                                all_tasks.append(task)
                    else:
                        all_tasks.append(task)

                    if len(all_tasks) >= max_results:
                        break

                page_token = data.get("nextPageToken")
                if not page_token or len(all_tasks) >= max_results:
                    break

            if len(all_tasks) >= max_results:
                break

        return all_tasks[:max_results]

    def _format_tasks(self, tasks: list[dict], action: str, tasklist_name: str) -> str:
        """Format tasks for LLM context."""
        account_email = self.get_account_email()

        if not tasks:
            action_msgs = {
                "pending": "No pending tasks.",
                "due_today": "No tasks due today.",
                "overdue": "No overdue tasks.",
                "completed": "No completed tasks found.",
                "search": "No tasks found matching your search.",
                "all": "No tasks found.",
            }
            return f"### Google Tasks ({action})\n{action_msgs.get(action, 'No tasks found.')}"

        action_titles = {
            "pending": "Pending Tasks",
            "due_today": "Tasks Due Today",
            "overdue": "Overdue Tasks",
            "completed": "Completed Tasks",
            "search": "Task Search Results",
            "all": "All Tasks",
        }

        lines = [f"### {action_titles.get(action, 'Tasks')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        if tasklist_name:
            lines.append(f"List: {tasklist_name}")
        lines.append(f"Found {len(tasks)} task(s):\n")

        for task in tasks:
            task_id = task.get("id", "")
            title = task.get("title", "(No title)")
            notes = task.get("notes", "")
            status = task.get("status", "needsAction")
            due = task.get("due", "")
            tasklist_title = task.get("tasklist_title", "")

            status_marker = "✅" if status == "completed" else "⬜"

            lines.append(f"{status_marker} **{title}**")
            lines.append(f"   ID: {task_id}")
            if tasklist_title:
                lines.append(f"   List: {tasklist_title}")

            if due:
                due_date = due[:10] if len(due) >= 10 else due
                today = datetime.now(timezone.utc).date()
                try:
                    task_due = datetime.fromisoformat(due.replace("Z", "+00:00")).date()
                    if task_due < today and status != "completed":
                        lines.append(f"   Due: {due_date} ⚠️ OVERDUE")
                    elif task_due == today:
                        lines.append(f"   Due: {due_date} (TODAY)")
                    else:
                        lines.append(f"   Due: {due_date}")
                except Exception:
                    lines.append(f"   Due: {due_date}")

            if notes:
                notes_preview = notes.replace("\n", " ").strip()
                if len(notes_preview) > 100:
                    notes_preview = notes_preview[:100] + "..."
                lines.append(f"   Notes: {notes_preview}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
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
        if action == "completed" or any(
            re.search(p, query_lower) for p in historical_patterns
        ):
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
        """Check if Google Tasks is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Google Tasks API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return (
                    False,
                    "Failed to get access token - check OAuth configuration",
                )

            # Test API access - get task lists
            tasklists = self._get_task_lists()
            results.append(f"Connected. Found {len(tasklists)} task list(s)")

            # List task list names
            if tasklists:
                names = [tl.get("title", "Untitled") for tl in tasklists[:5]]
                results.append(f"Lists: {', '.join(names)}")

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

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
