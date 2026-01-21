"""
Integration tests for the Todoist action plugin.

Uses mocked HTTP responses to test the plugin without a real Todoist account.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

# Import the plugin
from builtin_plugins.actions.todoist import TodoistActionHandler
from plugin_base.action import ActionContext, ActionRisk


class TestTodoistPluginMetadata:
    """Tests for plugin metadata."""

    def test_action_type(self):
        assert TodoistActionHandler.action_type == "todoist"

    def test_display_name(self):
        assert TodoistActionHandler.display_name == "Todoist"

    def test_category(self):
        assert TodoistActionHandler.category == "productivity"

    def test_not_abstract(self):
        assert TodoistActionHandler._abstract is False

    def test_config_fields(self):
        fields = TodoistActionHandler.get_config_fields()
        field_names = [f.name for f in fields]
        assert "api_token" in field_names
        assert "default_project" in field_names

        # api_token should be required
        api_token_field = next(f for f in fields if f.name == "api_token")
        assert api_token_field.required is True

    def test_actions(self):
        actions = TodoistActionHandler.get_actions()
        action_names = [a.name for a in actions]
        assert "create" in action_names
        assert "complete" in action_names
        assert "update" in action_names
        assert "delete" in action_names
        assert "list" in action_names

    def test_action_risks(self):
        actions = {a.name: a for a in TodoistActionHandler.get_actions()}

        assert actions["create"].risk == ActionRisk.LOW
        assert actions["complete"].risk == ActionRisk.LOW
        assert actions["update"].risk == ActionRisk.LOW
        assert actions["delete"].risk == ActionRisk.DESTRUCTIVE
        assert actions["list"].risk == ActionRisk.READ_ONLY


class TestTodoistPluginValidation:
    """Tests for config validation."""

    def test_valid_config(self):
        result = TodoistActionHandler.validate_config({"api_token": "test-token"})
        assert result.valid is True

    def test_missing_api_token(self):
        result = TodoistActionHandler.validate_config({})
        assert result.valid is False
        assert any(e.field == "api_token" for e in result.errors)

    def test_with_optional_default_project(self):
        result = TodoistActionHandler.validate_config(
            {"api_token": "test-token", "default_project": "Work"}
        )
        assert result.valid is True


class TestTodoistCreateTask:
    """Tests for create task action."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_create_simple_task(self, handler):
        """Test creating a basic task."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "123",
            "content": "Test task",
            "url": "https://todoist.com/task/123",
        }
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute("create", {"content": "Test task"}, ActionContext())

        assert result.success is True
        assert "Test task" in result.message
        assert result.data["task_id"] == "123"

        # Verify API call
        handler.client.post.assert_called_once()
        call_args = handler.client.post.call_args
        assert call_args[0][0] == "/tasks"
        assert call_args[1]["json"]["content"] == "Test task"

    def test_create_task_with_due_date(self, handler):
        """Test creating a task with due date."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "124",
            "content": "Task with due",
            "due": {"string": "tomorrow", "date": "2026-01-20"},
        }
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {"content": "Task with due", "due_string": "tomorrow"},
            ActionContext(),
        )

        assert result.success is True
        assert "tomorrow" in result.message

        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["due_string"] == "tomorrow"

    def test_create_task_with_priority(self, handler):
        """Test creating a task with priority."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "125", "content": "Urgent task"}
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {"content": "Urgent task", "priority": 4},
            ActionContext(),
        )

        assert result.success is True

        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["priority"] == 4

    def test_create_task_with_labels(self, handler):
        """Test creating a task with labels."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "126", "content": "Labeled task"}
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "create",
            {"content": "Labeled task", "labels": "work,urgent"},
            ActionContext(),
        )

        assert result.success is True

        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["labels"] == ["work", "urgent"]

    def test_create_task_with_project(self, handler):
        """Test creating a task in a specific project."""
        # Mock project list response
        mock_projects_response = MagicMock()
        mock_projects_response.json.return_value = [
            {"id": "proj1", "name": "Work"},
            {"id": "proj2", "name": "Personal"},
        ]
        mock_projects_response.raise_for_status = MagicMock()

        # Mock task creation response
        mock_task_response = MagicMock()
        mock_task_response.json.return_value = {"id": "127", "content": "Work task"}
        mock_task_response.raise_for_status = MagicMock()

        handler.client.get.return_value = mock_projects_response
        handler.client.post.return_value = mock_task_response

        result = handler.execute(
            "create",
            {"content": "Work task", "project": "Work"},
            ActionContext(),
        )

        assert result.success is True

        # Verify project ID was included
        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["project_id"] == "proj1"


class TestTodoistCompleteTask:
    """Tests for complete task action."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_complete_task(self, handler):
        """Test completing a task."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute("complete", {"task_id": "123456"}, ActionContext())

        assert result.success is True
        assert "123456" in result.message

        handler.client.post.assert_called_once_with("/tasks/123456/close")


class TestTodoistUpdateTask:
    """Tests for update task action."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_update_task_content(self, handler):
        """Test updating task content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "123", "content": "Updated title"}
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "update",
            {"task_id": "123", "content": "Updated title"},
            ActionContext(),
        )

        assert result.success is True
        assert "Updated title" in result.message

    def test_update_task_due_date(self, handler):
        """Test updating task due date."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "123", "content": "Task"}
        mock_response.raise_for_status = MagicMock()
        handler.client.post.return_value = mock_response

        result = handler.execute(
            "update",
            {"task_id": "123", "due_string": "next week"},
            ActionContext(),
        )

        assert result.success is True

        call_args = handler.client.post.call_args
        assert call_args[1]["json"]["due_string"] == "next week"

    def test_update_no_fields(self, handler):
        """Test update with no fields returns error."""
        result = handler.execute("update", {"task_id": "123"}, ActionContext())

        assert result.success is False
        assert "No fields to update" in result.error


class TestTodoistDeleteTask:
    """Tests for delete task action."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_delete_task(self, handler):
        """Test deleting a task."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        handler.client.delete.return_value = mock_response

        result = handler.execute("delete", {"task_id": "123456"}, ActionContext())

        assert result.success is True
        assert "123456" in result.message

        handler.client.delete.assert_called_once_with("/tasks/123456")


class TestTodoistListTasks:
    """Tests for list tasks action."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_list_all_tasks(self, handler):
        """Test listing all tasks."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "1", "content": "Task 1"},
            {"id": "2", "content": "Task 2", "due": {"string": "tomorrow"}},
        ]
        mock_response.raise_for_status = MagicMock()
        handler.client.get.return_value = mock_response

        result = handler.execute("list", {}, ActionContext())

        assert result.success is True
        assert "2 task(s)" in result.message
        assert result.data["count"] == 2

    def test_list_tasks_with_limit(self, handler):
        """Test listing tasks with limit."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": str(i), "content": f"Task {i}"} for i in range(10)
        ]
        mock_response.raise_for_status = MagicMock()
        handler.client.get.return_value = mock_response

        result = handler.execute("list", {"limit": 3}, ActionContext())

        assert result.success is True
        assert result.data["count"] == 3

    def test_list_tasks_by_project(self, handler):
        """Test listing tasks filtered by project."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "1", "content": "Work task"}]
        mock_response.raise_for_status = MagicMock()
        handler.client.get.return_value = mock_response

        result = handler.execute("list", {"project": "Work"}, ActionContext())

        assert result.success is True
        assert "in Work" in result.message

        # Check filter was passed
        call_args = handler.client.get.call_args
        assert call_args[1]["params"]["filter"] == "##Work"


class TestTodoistErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_http_error(self, handler):
        """Test handling HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid token"}
        error = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        handler.client.post.return_value.raise_for_status.side_effect = error
        handler.client.post.return_value.json.return_value = {"error": "Invalid token"}

        result = handler.execute("create", {"content": "Test"}, ActionContext())

        assert result.success is False
        assert "invalid token" in result.error.lower() or "401" in result.error

    def test_timeout_error(self, handler):
        """Test handling timeout errors."""
        handler.client.post.side_effect = httpx.TimeoutException("Timeout")

        result = handler.execute("create", {"content": "Test"}, ActionContext())

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_unknown_action(self, handler):
        """Test handling unknown action."""
        result = handler.execute("unknown", {}, ActionContext())

        assert result.success is False
        assert "Unknown action" in result.error


class TestTodoistApprovalSummary:
    """Tests for approval summary generation."""

    @pytest.fixture
    def handler(self):
        return TodoistActionHandler({"api_token": "test-token"})

    def test_create_summary(self, handler):
        summary = handler.get_approval_summary(
            "create", {"content": "Test task", "due_string": "tomorrow"}
        )
        assert "Create Todoist task" in summary
        assert "Test task" in summary
        assert "tomorrow" in summary

    def test_complete_summary(self, handler):
        summary = handler.get_approval_summary("complete", {"task_id": "123"})
        assert "complete" in summary.lower()
        assert "123" in summary

    def test_delete_summary(self, handler):
        summary = handler.get_approval_summary("delete", {"task_id": "456"})
        assert "DELETE" in summary
        assert "456" in summary
        assert "⚠️" in summary

    def test_list_summary(self, handler):
        summary = handler.get_approval_summary("list", {"project": "Work"})
        assert "List" in summary
        assert "Work" in summary


class TestTodoistTestConnection:
    """Tests for test_connection method."""

    @pytest.fixture
    def handler(self):
        handler = TodoistActionHandler({"api_token": "test-token"})
        handler.client = MagicMock()
        return handler

    def test_connection_success(self, handler):
        """Test successful connection test."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "1", "name": "Inbox"},
            {"id": "2", "name": "Work"},
        ]
        mock_response.raise_for_status = MagicMock()
        handler.client.get.return_value = mock_response

        success, message = handler.test_connection()

        assert success is True
        assert "2 projects" in message

    def test_connection_failure(self, handler):
        """Test failed connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        error = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        handler.client.get.side_effect = error

        success, message = handler.test_connection()

        assert success is False
        assert "401" in message or "error" in message.lower()


class TestTodoistLLMInstructions:
    """Tests for LLM instruction generation."""

    def test_instructions_contain_actions(self):
        instructions = TodoistActionHandler.get_llm_instructions()

        assert "todoist:create" in instructions
        assert "todoist:complete" in instructions
        assert "todoist:delete" in instructions
        assert "todoist:list" in instructions

    def test_instructions_contain_parameters(self):
        instructions = TodoistActionHandler.get_llm_instructions()

        assert "content (required)" in instructions
        assert "due_string (optional)" in instructions
        assert "task_id (required)" in instructions

    def test_instructions_contain_examples(self):
        instructions = TodoistActionHandler.get_llm_instructions()

        assert "<smart_action" in instructions
        assert 'type="todoist"' in instructions
        assert "Review quarterly report" in instructions  # From example
