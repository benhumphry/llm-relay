"""
Mock action handler for testing the plugin loader.
"""

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
)
from plugin_base.common import FieldDefinition, FieldType


class MockActionHandler(PluginActionHandler):
    """Mock action handler for testing."""

    action_type = "mock_action"
    display_name = "Mock Actions"
    description = "Test action handler for unit tests"
    icon = "🎭"
    category = "testing"

    # Override _abstract to allow registration
    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="api_key",
                label="API Key",
                field_type=FieldType.PASSWORD,
                required=True,
            ),
            FieldDefinition(
                name="verbose",
                label="Verbose Mode",
                field_type=FieldType.BOOLEAN,
                default=False,
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="test",
                description="Run a test action",
                risk=ActionRisk.READ_ONLY,
                params=[
                    FieldDefinition(
                        name="message",
                        label="Message",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                ],
                examples=[{"message": "Hello, world!"}],
            ),
            ActionDefinition(
                name="create",
                description="Create something",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="name",
                        label="Name",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="description",
                        label="Description",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                ],
                examples=[{"name": "Test Item", "description": "A test item"}],
            ),
            ActionDefinition(
                name="delete",
                description="Delete something permanently",
                risk=ActionRisk.DESTRUCTIVE,
                params=[
                    FieldDefinition(
                        name="id",
                        label="ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                ],
                examples=[{"id": "123"}],
            ),
        ]

    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.verbose = config.get("verbose", False)

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        if action == "test":
            message = params.get("message", "")
            return ActionResult(
                success=True,
                message=f"Test successful: {message}",
                data={"echoed": message},
            )
        elif action == "create":
            name = params.get("name", "")
            return ActionResult(
                success=True,
                message=f"Created: {name}",
                data={"id": "new-123", "name": name},
            )
        elif action == "delete":
            item_id = params.get("id", "")
            return ActionResult(
                success=True,
                message=f"Deleted: {item_id}",
            )
        else:
            return ActionResult(
                success=False,
                message="",
                error=f"Unknown action: {action}",
            )
