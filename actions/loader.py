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
        return False

    @property
    def oauth_provider(self) -> Optional[str]:
        # Extract from oauth config fields if present
        from plugin_base.common import FieldType

        for field in self._plugin_class.get_config_fields():
            if field.field_type == FieldType.OAUTH_ACCOUNT:
                # Provider may be in field metadata
                return field.metadata.get("provider") if field.metadata else None
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

        plugin_context = PluginContext(
            session_key=context.session_key,
            user_tags=context.tags,
            smart_alias_name=context.alias_name or "",
            conversation_id=context.request_id,
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

    def get_system_prompt_instructions(self) -> str:
        """Get LLM instructions from the plugin."""
        return self._plugin_class.get_llm_instructions()


def load_action_handlers() -> None:
    """
    Load and register all action handlers.

    Loads both legacy handlers and plugin-based handlers.
    Call this during application startup.
    """
    logger.info("Loading action handlers...")

    # Import and register legacy handlers
    from .handlers.calendar import CalendarActionHandler
    from .handlers.email import EmailActionHandler
    from .handlers.notification import NotificationActionHandler
    from .handlers.schedule import ScheduleActionHandler

    register_handler(EmailActionHandler())
    register_handler(CalendarActionHandler())
    register_handler(NotificationActionHandler())
    register_handler(ScheduleActionHandler())

    logger.info("Legacy action handlers loaded")

    # Load plugin-based action handlers
    _load_plugin_action_handlers()


def _load_plugin_action_handlers() -> int:
    """
    Load action handlers from the plugin registry.

    Plugin handlers are wrapped in PluginActionAdapter to match the
    existing ActionHandler interface.

    Returns:
        Number of plugin handlers loaded.
    """
    try:
        from plugin_base.loader import action_registry
        from db.plugin_configs import get_plugin_configs_by_type
    except ImportError as e:
        logger.warning(f"Could not import plugin system: {e}")
        return 0

    count = 0
    all_plugins = action_registry.get_all()

    for action_type, plugin_class in all_plugins.items():
        try:
            # Check if there's a saved config for this plugin
            configs = get_plugin_configs_by_type("action", action_type)

            if configs:
                # Use the first enabled config (plugins may have multiple instances later)
                for config in configs:
                    if config.get("enabled", True):
                        config_data = config.get("config", {})
                        adapter = PluginActionAdapter(plugin_class, config_data)
                        register_handler(adapter)
                        logger.info(
                            f"Registered plugin action handler: {action_type} "
                            f"(config: {config.get('name', 'unnamed')})"
                        )
                        count += 1
                        break
            else:
                # No config saved - check if plugin has required config fields
                required_fields = [
                    f for f in plugin_class.get_config_fields() if f.required
                ]
                if not required_fields:
                    # No required config - register with empty config
                    adapter = PluginActionAdapter(plugin_class, {})
                    register_handler(adapter)
                    logger.info(
                        f"Registered plugin action handler (no config required): {action_type}"
                    )
                    count += 1
                else:
                    logger.debug(
                        f"Skipping plugin action handler {action_type}: "
                        f"requires configuration ({[f.name for f in required_fields]})"
                    )

        except Exception as e:
            logger.error(
                f"Failed to load plugin action handler {action_type}: {e}",
                exc_info=True,
            )

    if count > 0:
        logger.info(f"Loaded {count} plugin action handlers")

    return count
