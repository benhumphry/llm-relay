"""
Action Handler Loader.

Initializes and registers all available action handlers.
Supports both legacy handlers (actions/handlers/) and plugin handlers (builtin_plugins/actions/).
"""

import logging
from typing import Optional

from .base import ActionContext, ActionHandler, ActionResult, ActionStatus
from .registry import register_handler

logger = logging.getLogger(__name__)


class PluginActionAdapter(ActionHandler):
    """
    Adapter that wraps a PluginActionHandler to work with the existing action registry.

    This allows plugin-based action handlers to be used alongside the legacy handlers
    without modifying the existing executor infrastructure.
    """

    def __init__(self, plugin_class, config: dict):
        """
        Initialize the adapter.

        Args:
            plugin_class: The PluginActionHandler class
            config: Configuration dict for the plugin
        """
        self._plugin_class = plugin_class
        self._config = config
        self._instance = plugin_class(config)

    @property
    def action_type(self) -> str:
        return self._plugin_class.action_type

    @property
    def supported_actions(self) -> list[str]:
        return [a.name for a in self._plugin_class.get_actions()]

    @property
    def requires_oauth(self) -> bool:
        # Check if any config field is of type 'oauth_account'
        from plugin_base.common import FieldType

        for field in self._plugin_class.get_config_fields():
            if field.field_type == FieldType.OAUTH_ACCOUNT:
                return True

        # Also check resource requirements for OAUTH_ACCOUNT type
        from plugin_base.action import ResourceType

        if hasattr(self._plugin_class, "get_resource_requirements"):
            for req in self._plugin_class.get_resource_requirements():
                if req.resource_type == ResourceType.OAUTH_ACCOUNT:
                    return True

        return False

    @property
    def oauth_provider(self) -> Optional[str]:
        # Extract from oauth config fields if present
        from plugin_base.common import FieldType

        for field in self._plugin_class.get_config_fields():
            if field.field_type == FieldType.OAUTH_ACCOUNT:
                # Provider may be in picker_options
                return (
                    field.picker_options.get("provider")
                    if field.picker_options
                    else None
                )
        return None

    def validate(
        self, action: str, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Validate action parameters."""
        # Convert legacy context to plugin context
        from plugin_base.action import ActionContext as PluginContext

        plugin_context = PluginContext(
            session_key=context.session_key,
            user_tags=context.tags,
            smart_alias_name=context.alias_name or "",
            conversation_id=context.request_id,
        )

        result = self._instance.validate_action_params(action, params)
        if result.valid:
            return True, ""
        else:
            errors = "; ".join(f"{e.field}: {e.message}" for e in result.errors)
            return False, errors

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the action."""
        from plugin_base.action import ActionContext as PluginContext

        # Use available_accounts from context if provided (built from document stores)
        # This is the new path - accounts derived from linked document stores
        available_accounts = getattr(context, "available_accounts", None)

        if not available_accounts:
            # Legacy fallback: build from OAuth accounts
            import os

            available_accounts = {
                "email": [],
                "calendar": [],
                "tasks": [],
            }

            oauth_accounts = getattr(context, "oauth_accounts", {})

            # Google accounts can do email, calendar, tasks
            for account in oauth_accounts.get("google", []):
                account_info = {
                    "id": account.get("id"),
                    "type": "oauth",
                    "provider": "google",
                    "email": account.get("email", ""),
                    "name": account.get("name", ""),
                }
                available_accounts["email"].append(account_info)
                available_accounts["calendar"].append(account_info)
                available_accounts["tasks"].append(account_info)

            # Microsoft accounts can do email, calendar
            for account in oauth_accounts.get("microsoft", []):
                account_info = {
                    "id": account.get("id"),
                    "type": "oauth",
                    "provider": "microsoft",
                    "email": account.get("email", ""),
                    "name": account.get("name", ""),
                }
                available_accounts["email"].append(account_info)
                available_accounts["calendar"].append(account_info)

            # Todoist (API-based) for tasks
            if os.environ.get("TODOIST_API_KEY"):
                available_accounts["tasks"].append(
                    {
                        "id": "todoist",
                        "type": "api",
                        "provider": "todoist",
                        "name": "Todoist",
                    }
                )

        plugin_context = PluginContext(
            session_key=context.session_key,
            user_tags=context.tags,
            smart_alias_name=context.alias_name or "",
            conversation_id=context.request_id,
            available_accounts=available_accounts,
            default_accounts=context.default_accounts,
        )

        try:
            plugin_result = self._instance.execute(action, params, plugin_context)

            # Convert plugin result to legacy result
            if plugin_result.success:
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    action_type=self.action_type,
                    action=action,
                    message=plugin_result.message,
                    details={"data": plugin_result.data} if plugin_result.data else {},
                )
            else:
                return ActionResult(
                    status=ActionStatus.FAILED,
                    action_type=self.action_type,
                    action=action,
                    message=plugin_result.error or plugin_result.message,
                )
        except Exception as e:
            logger.exception(
                f"Plugin action execution error: {self.action_type}:{action}"
            )
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message=f"Execution error: {str(e)}",
            )

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate approval summary."""
        return self._instance.get_approval_summary(action, params)

    def get_system_prompt_instructions(
        self, available_accounts: Optional[dict[str, list[dict]]] = None
    ) -> str:
        """Get LLM instructions from the plugin."""
        # Check if the plugin instance has a dynamic instructions method
        if hasattr(self._instance, "get_llm_instructions_with_context"):
            return self._instance.get_llm_instructions_with_context(available_accounts)
        # Fall back to static class method
        return self._plugin_class.get_llm_instructions()


def load_action_handlers() -> None:
    """
    Load and register all action handlers from plugins.

    Plugin handlers get OAuth configuration from Smart Alias context at runtime,
    so they don't require saved config to be registered.

    Call this during application startup.
    """
    logger.info("Loading action handlers...")

    try:
        from plugin_base.loader import action_registry
    except ImportError as e:
        logger.warning(f"Could not import plugin system: {e}")
        return

    count = 0
    all_plugins = action_registry.get_all()

    for action_type, plugin_class in all_plugins.items():
        try:
            # Check if there's a saved config for this plugin
            from db.plugin_configs import get_plugin_configs_by_type

            configs = get_plugin_configs_by_type("action", action_type)

            if configs:
                # Use the first enabled config
                for config in configs:
                    # config is a PluginConfig object, not a dict
                    if getattr(config, "enabled", True):
                        config_data = getattr(config, "config", {}) or {}
                        adapter = PluginActionAdapter(plugin_class, config_data)
                        register_handler(adapter)
                        logger.info(
                            f"Registered action handler: {action_type} "
                            f"(config: {getattr(config, 'name', 'unnamed')})"
                        )
                        count += 1
                        break
            else:
                # No saved config - register with empty config
                # OAuth will be configured from Smart Alias context at runtime
                adapter = PluginActionAdapter(plugin_class, {})
                register_handler(adapter)
                logger.info(
                    f"Registered action handler: {action_type} (context-based config)"
                )
                count += 1

        except Exception as e:
            logger.error(
                f"Failed to load action handler {action_type}: {e}",
                exc_info=True,
            )

    logger.info(f"Loaded {count} action handlers")
