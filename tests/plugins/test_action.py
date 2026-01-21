"""
Unit tests for plugin_base/action.py

Tests the action handler base class, validation, and LLM instruction generation.
"""

import pytest

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
)
from plugin_base.common import FieldDefinition, FieldType, ValidationResult
from tests.plugins.fixtures.mock_action import MockActionHandler


class TestActionRisk:
    """Tests for ActionRisk enum."""

    def test_risk_values(self):
        """Test that all risk levels have expected values."""
        assert ActionRisk.READ_ONLY.value == "read_only"
        assert ActionRisk.LOW.value == "low"
        assert ActionRisk.MEDIUM.value == "medium"
        assert ActionRisk.HIGH.value == "high"
        assert ActionRisk.DESTRUCTIVE.value == "destructive"

    def test_risk_ordering(self):
        """Test that risks can be compared conceptually."""
        # This is more of a documentation test - risks should escalate
        risks = [
            ActionRisk.READ_ONLY,
            ActionRisk.LOW,
            ActionRisk.MEDIUM,
            ActionRisk.HIGH,
            ActionRisk.DESTRUCTIVE,
        ]
        assert len(risks) == 5


class TestActionDefinition:
    """Tests for ActionDefinition dataclass."""

    def test_minimal_definition(self):
        action = ActionDefinition(
            name="test",
            description="Test action",
            risk=ActionRisk.LOW,
            params=[],
        )
        assert action.name == "test"
        assert action.risk == ActionRisk.LOW
        assert action.examples == []

    def test_to_dict(self):
        action = ActionDefinition(
            name="create",
            description="Create something",
            risk=ActionRisk.MEDIUM,
            params=[
                FieldDefinition(
                    name="title",
                    label="Title",
                    field_type=FieldType.TEXT,
                    required=True,
                ),
            ],
            examples=[{"title": "Example"}],
        )
        result = action.to_dict()

        assert result["name"] == "create"
        assert result["description"] == "Create something"
        assert result["risk"] == "medium"
        assert len(result["params"]) == 1
        assert result["params"][0]["name"] == "title"
        assert result["examples"] == [{"title": "Example"}]


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_success_result(self):
        result = ActionResult(
            success=True,
            message="Task created",
            data={"id": "123"},
        )
        assert result.success is True
        assert result.message == "Task created"
        assert result.data["id"] == "123"
        assert result.error is None

    def test_error_result(self):
        result = ActionResult(
            success=False,
            message="",
            error="API rate limit exceeded",
        )
        assert result.success is False
        assert result.error == "API rate limit exceeded"


class TestActionContext:
    """Tests for ActionContext dataclass."""

    def test_default_context(self):
        context = ActionContext()
        assert context.session_key is None
        assert context.user_tags == []
        assert context.smart_alias_name == ""
        assert context.conversation_id is None

    def test_full_context(self):
        context = ActionContext(
            session_key="abc123",
            user_tags=["user1", "project-x"],
            smart_alias_name="assistant",
            conversation_id="conv-456",
        )
        assert context.session_key == "abc123"
        assert "user1" in context.user_tags
        assert context.smart_alias_name == "assistant"


class TestPluginActionHandlerValidation:
    """Tests for PluginActionHandler config validation."""

    def test_validate_config_valid(self):
        """Test validation with valid config."""
        result = MockActionHandler.validate_config(
            {
                "api_key": "test-key-123",
                "verbose": True,
            }
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_config_missing_required(self):
        """Test validation with missing required field."""
        result = MockActionHandler.validate_config(
            {
                "verbose": True,
            }
        )
        assert len(result.errors) == 1
        assert result.errors[0].field == "api_key"

    def test_validate_config_empty(self):
        """Test validation with empty config."""
        result = MockActionHandler.validate_config({})
        assert result.valid is False


class TestPluginActionHandlerActionValidation:
    """Tests for action parameter validation."""

    def test_validate_action_params_valid(self):
        """Test validation of valid action params."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.validate_action_params("test", {"message": "Hello"})
        assert result.valid is True

    def test_validate_action_params_missing_required(self):
        """Test validation with missing required param."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.validate_action_params("test", {})
        assert result.valid is False
        assert any(e.field == "message" for e in result.errors)

    def test_validate_action_params_unknown_action(self):
        """Test validation with unknown action."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.validate_action_params("nonexistent", {})
        assert result.valid is False
        assert any(e.field == "action" for e in result.errors)

    def test_validate_action_params_optional_missing(self):
        """Test validation with optional param missing is OK."""
        handler = MockActionHandler({"api_key": "test"})
        # "create" action has optional "description" param
        result = handler.validate_action_params("create", {"name": "Test"})
        assert result.valid is True


class TestPluginActionHandlerGetAction:
    """Tests for get_action method."""

    def test_get_action_exists(self):
        """Test getting an existing action."""
        action = MockActionHandler.get_action("test")
        assert action is not None
        assert action.name == "test"
        assert action.risk == ActionRisk.READ_ONLY

    def test_get_action_not_found(self):
        """Test getting non-existent action."""
        action = MockActionHandler.get_action("nonexistent")
        assert action is None

    def test_get_action_risk(self):
        """Test getting action risk level."""
        handler = MockActionHandler({"api_key": "test"})

        assert handler.get_action_risk("test") == ActionRisk.READ_ONLY
        assert handler.get_action_risk("create") == ActionRisk.LOW
        assert handler.get_action_risk("delete") == ActionRisk.DESTRUCTIVE
        # Unknown action defaults to HIGH
        assert handler.get_action_risk("unknown") == ActionRisk.HIGH


class TestPluginActionHandlerApprovalSummary:
    """Tests for approval summary generation."""

    def test_approval_summary_normal(self):
        """Test approval summary for normal action."""
        handler = MockActionHandler({"api_key": "test"})
        summary = handler.get_approval_summary("test", {"message": "Hello"})
        assert "mock_action:test" in summary
        assert "1 parameters" in summary

    def test_approval_summary_destructive(self):
        """Test approval summary for destructive action."""
        handler = MockActionHandler({"api_key": "test"})
        summary = handler.get_approval_summary("delete", {"id": "123"})
        assert "DESTRUCTIVE" in summary
        assert "mock_action:delete" in summary


class TestPluginActionHandlerLLMInstructions:
    """Tests for LLM instruction generation."""

    def test_llm_instructions_contains_header(self):
        """Test that instructions contain display name header."""
        instructions = MockActionHandler.get_llm_instructions()
        assert "## Mock Actions" in instructions

    def test_llm_instructions_contains_actions(self):
        """Test that instructions list all actions."""
        instructions = MockActionHandler.get_llm_instructions()
        assert "mock_action:test" in instructions
        assert "mock_action:create" in instructions
        assert "mock_action:delete" in instructions

    def test_llm_instructions_contains_descriptions(self):
        """Test that instructions include action descriptions."""
        instructions = MockActionHandler.get_llm_instructions()
        assert "Run a test action" in instructions
        assert "Create something" in instructions
        assert "Delete something permanently" in instructions

    def test_llm_instructions_contains_parameters(self):
        """Test that instructions list parameters."""
        instructions = MockActionHandler.get_llm_instructions()
        assert "message (required)" in instructions
        assert "name (required)" in instructions
        assert "description (optional)" in instructions

    def test_llm_instructions_contains_examples(self):
        """Test that instructions include examples."""
        instructions = MockActionHandler.get_llm_instructions()
        assert "<smart_action" in instructions
        assert 'type="mock_action"' in instructions
        assert "Hello, world!" in instructions

    def test_llm_instructions_xml_format(self):
        """Test that examples use correct XML format."""
        instructions = MockActionHandler.get_llm_instructions()
        # Should have opening and closing tags
        assert '<smart_action type="mock_action" action="test">' in instructions
        assert "</smart_action>" in instructions


class TestPluginActionHandlerExecution:
    """Tests for action execution."""

    def test_execute_test_action(self):
        """Test executing the test action."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.execute("test", {"message": "Hello"}, ActionContext())

        assert result.success is True
        assert "Hello" in result.message
        assert result.data["echoed"] == "Hello"

    def test_execute_create_action(self):
        """Test executing the create action."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.execute(
            "create",
            {"name": "My Item", "description": "A test item"},
            ActionContext(),
        )

        assert result.success is True
        assert "My Item" in result.message
        assert result.data["name"] == "My Item"
        assert "id" in result.data

    def test_execute_delete_action(self):
        """Test executing the delete action."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.execute("delete", {"id": "item-123"}, ActionContext())

        assert result.success is True
        assert "item-123" in result.message

    def test_execute_unknown_action(self):
        """Test executing unknown action returns error."""
        handler = MockActionHandler({"api_key": "test"})
        result = handler.execute("unknown", {}, ActionContext())

        assert result.success is False
        assert "Unknown action" in result.error


class TestPluginActionHandlerUtilities:
    """Tests for utility methods."""

    def test_is_available_default(self):
        """Test default is_available returns True."""
        handler = MockActionHandler({"api_key": "test"})
        assert handler.is_available() is True

    def test_test_connection_default(self):
        """Test default test_connection returns OK."""
        handler = MockActionHandler({"api_key": "test"})
        success, message = handler.test_connection()
        assert success is True
        assert message == "OK"


class TestPluginActionHandlerMetadata:
    """Tests for class-level metadata."""

    def test_action_type(self):
        """Test action_type attribute."""
        assert MockActionHandler.action_type == "mock_action"

    def test_display_name(self):
        """Test display_name attribute."""
        assert MockActionHandler.display_name == "Mock Actions"

    def test_description(self):
        """Test description attribute."""
        assert "Test action handler" in MockActionHandler.description

    def test_icon(self):
        """Test icon attribute."""
        assert MockActionHandler.icon == "🎭"

    def test_category(self):
        """Test category attribute."""
        assert MockActionHandler.category == "testing"

    def test_get_config_fields(self):
        """Test get_config_fields returns expected fields."""
        fields = MockActionHandler.get_config_fields()
        assert len(fields) == 2

        field_names = [f.name for f in fields]
        assert "api_key" in field_names
        assert "verbose" in field_names

    def test_get_actions(self):
        """Test get_actions returns expected actions."""
        actions = MockActionHandler.get_actions()
        assert len(actions) == 3

        action_names = [a.name for a in actions]
        assert "test" in action_names
        assert "create" in action_names
        assert "delete" in action_names
