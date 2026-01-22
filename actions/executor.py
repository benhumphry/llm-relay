"""
Action Executor for Smart Actions.

Processes actions from LLM responses and executes allowed actions.
"""

import logging
from typing import Optional

from .base import ActionContext, ActionResult, ActionStatus, ParsedAction
from .parser import has_actions, parse_actions, strip_actions
from .registry import get_handler, is_action_allowed

logger = logging.getLogger(__name__)


def execute_actions(
    response_text: str,
    alias_name: str,
    allowed_actions: list[str],
    request_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    session_key: Optional[str] = None,
    # New: Store-based accounts from enrichment
    available_accounts: Optional[dict[str, list[dict]]] = None,
    default_accounts: Optional[dict[str, dict]] = None,
    # Legacy: Action-category defaults (for backwards compatibility)
    default_email_account_id: Optional[int] = None,
    default_calendar_account_id: Optional[int] = None,
    default_calendar_id: Optional[str] = None,
    default_tasks_account_id: Optional[int] = None,
    default_tasks_provider: Optional[str] = None,
    default_tasks_list_id: Optional[str] = None,
    default_notification_urls: Optional[list[str]] = None,
    # Scheduled prompts config
    scheduled_prompts_account_id: Optional[int] = None,
    scheduled_prompts_calendar_id: Optional[str] = None,
) -> tuple[list[ActionResult], str]:
    """
    Parse and execute actions from LLM response text.

    Args:
        response_text: The full LLM response text
        alias_name: Name of the Smart Alias (for context/logging)
        allowed_actions: List of allowed action patterns (e.g., ["email:draft_*"])
        request_id: Optional request ID for tracking
        tags: Optional tags for context
        session_key: Optional session key for session-scoped caching
        available_accounts: Store-based accounts from linked document stores
        default_accounts: Default accounts configured on Smart Alias
        default_email_account_id: (Legacy) Default OAuth account ID for email actions
        default_calendar_account_id: (Legacy) Default OAuth account ID for calendar actions
        default_calendar_id: (Legacy) Default calendar ID for calendar actions
        default_tasks_account_id: (Legacy) Default OAuth account ID for task actions
        default_tasks_provider: (Legacy) Non-OAuth task provider ("todoist")
        default_tasks_list_id: (Legacy) Default task list ID for task actions

    Returns:
        Tuple of (list of ActionResults, cleaned response text with actions stripped)
    """
    if not response_text or not allowed_actions:
        return [], response_text

    # Quick check - avoid parsing if no actions
    if not has_actions(response_text):
        return [], response_text

    # Parse actions from response
    parsed_actions = parse_actions(response_text)
    if not parsed_actions:
        return [], response_text

    # Build action context
    context = _build_action_context(
        alias_name,
        allowed_actions,
        request_id,
        tags,
        session_key,
        available_accounts,
        default_accounts,
        default_email_account_id,
        default_calendar_account_id,
        default_calendar_id,
        default_tasks_account_id,
        default_tasks_provider,
        default_tasks_list_id,
        default_notification_urls,
        scheduled_prompts_account_id,
        scheduled_prompts_calendar_id,
    )

    # Execute each action
    results = []
    for parsed in parsed_actions:
        result = _execute_single_action(parsed, context, allowed_actions)
        results.append(result)

    # Strip action blocks from response text
    cleaned_text = strip_actions(response_text)

    return results, cleaned_text


def _build_action_context(
    alias_name: str,
    allowed_actions: list[str],
    request_id: Optional[str],
    tags: Optional[list[str]],
    session_key: Optional[str] = None,
    # New: Pre-built accounts from enricher
    available_accounts: Optional[dict[str, list[dict]]] = None,
    default_accounts_param: Optional[dict[str, dict]] = None,
    # Legacy parameters (used when new params not provided)
    default_email_account_id: Optional[int] = None,
    default_calendar_account_id: Optional[int] = None,
    default_calendar_id: Optional[str] = None,
    default_tasks_account_id: Optional[int] = None,
    default_tasks_provider: Optional[str] = None,
    default_tasks_list_id: Optional[str] = None,
    default_notification_urls: Optional[list[str]] = None,
    scheduled_prompts_account_id: Optional[int] = None,
    scheduled_prompts_calendar_id: Optional[str] = None,
) -> ActionContext:
    """Build ActionContext with available accounts and defaults.

    If available_accounts and default_accounts_param are provided (from enricher),
    uses those directly. Otherwise falls back to legacy behavior of building
    from OAuth tokens and individual account ID parameters.
    """
    from db.oauth_tokens import get_oauth_token_info, list_oauth_tokens

    # Use pre-built accounts if provided, otherwise build from legacy params
    if available_accounts is not None and default_accounts_param is not None:
        # New path: use pre-built accounts from enricher
        oauth_accounts: dict[str, list[dict]] = {}  # Legacy, kept for compatibility
        default_accounts = default_accounts_param

        # Add notification and schedule to defaults if provided via legacy params
        # (these may not be in default_accounts_param yet)
        if default_notification_urls and "notification" not in default_accounts:
            default_accounts["notification"] = {"urls": default_notification_urls}

        if scheduled_prompts_account_id and "schedule" not in default_accounts:
            default_accounts["schedule"] = {
                "account_id": scheduled_prompts_account_id,
                "calendar_id": scheduled_prompts_calendar_id or "primary",
            }

        logger.info(
            f"Using store-based accounts: {sum(len(v) for v in available_accounts.values())} available"
        )
        # Debug: log task accounts specifically
        tasks_accounts = available_accounts.get("tasks", [])
        logger.info(f"  Tasks accounts: {[a.get('name', '?') for a in tasks_accounts]}")

    else:
        # Legacy path: build from OAuth tokens and individual params
        oauth_accounts = {}
        default_accounts = {}
        available_accounts = {"email": [], "calendar": [], "tasks": []}

        try:
            tokens = list_oauth_tokens()
            for token in tokens:
                provider = token.get("provider", "unknown")
                if provider not in oauth_accounts:
                    oauth_accounts[provider] = []

                oauth_accounts[provider].append(
                    {
                        "id": token.get("id"),
                        "email": token.get("account_email", ""),
                        "name": token.get("account_name", ""),
                        "is_valid": token.get("is_valid", True),
                    }
                )

            # Set default accounts by action category
            if default_email_account_id:
                email_token = get_oauth_token_info(default_email_account_id)
                if email_token:
                    default_accounts["email"] = {
                        "id": email_token.get("id"),
                        "email": email_token.get("account_email", ""),
                        "provider": email_token.get("provider", ""),
                    }
                    logger.info(
                        f"Using default email account: {default_accounts['email']['email']}"
                    )

            if default_calendar_account_id:
                calendar_token = get_oauth_token_info(default_calendar_account_id)
                if calendar_token:
                    default_accounts["calendar"] = {
                        "id": calendar_token.get("id"),
                        "email": calendar_token.get("account_email", ""),
                        "provider": calendar_token.get("provider", ""),
                        "calendar_id": default_calendar_id or "primary",
                    }
                    logger.info(
                        f"Using default calendar account: {default_accounts['calendar']['email']}"
                    )

            if default_tasks_account_id:
                tasks_token = get_oauth_token_info(default_tasks_account_id)
                if tasks_token:
                    default_accounts["tasks"] = {
                        "id": tasks_token.get("id"),
                        "email": tasks_token.get("account_email", ""),
                        "provider": tasks_token.get("provider", ""),
                        "list_id": default_tasks_list_id,
                    }
                    logger.info(
                        f"Using default tasks account: {default_accounts['tasks']['email']}"
                    )
            elif default_tasks_provider == "todoist":
                import os

                from db.plugin_configs import get_plugin_configs_by_type

                api_token = None
                configs = get_plugin_configs_by_type("action", "todoist")
                if configs:
                    for config in configs:
                        if getattr(config, "enabled", True):
                            config_data = getattr(config, "config", {}) or {}
                            api_token = config_data.get("api_token")
                            break

                if not api_token:
                    api_token = os.environ.get("TODOIST_API_TOKEN")

                if api_token:
                    default_accounts["tasks"] = {
                        "provider": "todoist",
                        "api_token": api_token,
                        "list_id": default_tasks_list_id,
                    }
                    logger.info("Using Todoist for task actions")

        except Exception as e:
            logger.warning(f"Failed to load OAuth tokens for action context: {e}")

        # Set notification defaults
        if default_notification_urls:
            default_accounts["notification"] = {"urls": default_notification_urls}
            logger.info(
                f"Using {len(default_notification_urls)} default notification URL(s)"
            )

        # Set scheduled prompts config
        if scheduled_prompts_account_id:
            default_accounts["schedule"] = {
                "account_id": scheduled_prompts_account_id,
                "calendar_id": scheduled_prompts_calendar_id or "primary",
            }
            logger.info(
                f"Scheduled prompts enabled with account {scheduled_prompts_account_id}"
            )

    return ActionContext(
        request_id=request_id,
        session_key=session_key,
        alias_name=alias_name,
        tags=tags or [],
        oauth_accounts=oauth_accounts,
        allowed_actions=allowed_actions,
        require_approval=False,
        default_accounts=default_accounts,
        available_accounts=available_accounts,
    )


def _execute_single_action(
    parsed: ParsedAction,
    context: ActionContext,
    allowed_actions: list[str],
) -> ActionResult:
    """Execute a single parsed action."""
    full_action = parsed.full_action

    # Check if action is allowed
    if not is_action_allowed(full_action, allowed_actions):
        logger.warning(
            f"Action not allowed: {full_action} (allowed: {allowed_actions})"
        )
        return ActionResult(
            status=ActionStatus.REJECTED,
            action_type=parsed.action_type,
            action=parsed.action,
            message=f"Action '{full_action}' is not in the allowed actions list",
        )

    # Get handler
    handler = get_handler(parsed.action_type)
    if not handler:
        logger.warning(f"No handler registered for action type: {parsed.action_type}")
        return ActionResult(
            status=ActionStatus.INVALID,
            action_type=parsed.action_type,
            action=parsed.action,
            message=f"No handler registered for action type: {parsed.action_type}",
        )

    # Validate
    is_valid, error_msg = handler.validate(parsed.action, parsed.params, context)
    if not is_valid:
        logger.warning(f"Action validation failed: {full_action} - {error_msg}")
        return ActionResult(
            status=ActionStatus.INVALID,
            action_type=parsed.action_type,
            action=parsed.action,
            message=f"Validation failed: {error_msg}",
        )

    # Execute
    logger.info(f"Executing action: {full_action}")
    try:
        result = handler.execute(parsed.action, parsed.params, context)
        if result.status == ActionStatus.SUCCESS:
            logger.info(f"Action succeeded: {full_action} - {result.message}")
        else:
            logger.warning(f"Action failed: {full_action} - {result.message}")
        return result
    except Exception as e:
        logger.exception(f"Action execution error: {full_action}")
        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=parsed.action_type,
            action=parsed.action,
            message=f"Execution error: {str(e)}",
        )


def get_action_instructions_for_alias(
    alias_id: int,
) -> Optional[str]:
    """
    Get action instructions to inject into system prompt for a Smart Alias.

    Args:
        alias_id: The Smart Alias ID

    Returns:
        System prompt instructions string, or None if actions not enabled.
    """
    from db import get_smart_alias_by_id

    alias = get_smart_alias_by_id(alias_id)
    if not alias:
        return None

    if not getattr(alias, "use_actions", False):
        return None

    allowed = getattr(alias, "allowed_actions", [])
    if not allowed:
        return None

    from .registry import get_system_prompt_for_actions

    return get_system_prompt_for_actions(allowed)
