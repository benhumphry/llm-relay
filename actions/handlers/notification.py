"""
Notification Action Handler.

Handles notification actions via Apprise API (external service).
Apprise API supports 80+ notification services including:
- Slack, Discord, Telegram, Microsoft Teams
- Email (SMTP), Pushover, ntfy, Gotify
- And many more: https://github.com/caronc/apprise-api

Configuration:
- Set APPRISE_API_URL environment variable (e.g., http://apprise-api:8000)
- Configure notification URLs in Smart Alias (e.g., "gotify://gotify/token")
"""

import logging
import os

import requests

from actions.base import ActionContext, ActionHandler, ActionResult, ActionStatus

logger = logging.getLogger(__name__)


class NotificationActionHandler(ActionHandler):
    """
    Handler for notification actions via Apprise API.

    Sends notifications to Apprise API which dispatches to configured services.
    Uses stateless mode - URLs passed directly per request.
    """

    @property
    def action_type(self) -> str:
        return "notification"

    @property
    def supported_actions(self) -> list[str]:
        return ["send"]

    def _get_api_url(self) -> str:
        """Get Apprise API base URL from environment."""
        return os.environ.get("APPRISE_API_URL", "")

    def validate(
        self, action: str, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Validate notification action parameters."""
        if action not in self.supported_actions:
            return False, f"Unknown notification action: {action}"

        if action == "send":
            # Check API URL is configured
            if not self._get_api_url():
                return False, "APPRISE_API_URL environment variable not set"

            # Need at least a body/message
            body = params.get("body") or params.get("message")
            if not body:
                return False, "Missing required parameter: body (notification message)"

            # Need URLs - from params or default config
            urls = params.get("urls") or params.get("url")
            default_urls = context.default_accounts.get("notification", {}).get("urls")

            if not urls and not default_urls:
                return (
                    False,
                    "Missing required parameter: urls (Apprise notification URLs)",
                )

        return True, ""

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the notification action."""
        if action == "send":
            return self._send_notification(params, context)

        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=self.action_type,
            action=action,
            message=f"Unknown action: {action}",
        )

    def _send_notification(self, params: dict, context: ActionContext) -> ActionResult:
        """Send a notification via Apprise API stateless endpoint."""
        # Get notification content
        title = params.get("title", "")
        body = params.get("body") or params.get("message", "")

        # Get notification type (info, success, warning, failure)
        notify_type = params.get("type", "info").lower()
        if notify_type not in ("info", "success", "warning", "failure"):
            notify_type = "info"

        # Get URLs - from params or default config
        urls = params.get("urls") or params.get("url")
        if not urls:
            default_config = context.default_accounts.get("notification", {})
            urls = default_config.get("urls")

        if not urls:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send",
                message="No notification URLs configured",
            )

        # Normalize to list
        if isinstance(urls, str):
            urls = [urls]

        # Build API URL - stateless endpoint
        api_url = self._get_api_url()
        notify_url = f"{api_url.rstrip('/')}/notify/"

        # Send the notification
        try:
            response = requests.post(
                notify_url,
                json={
                    "urls": urls,
                    "title": title,
                    "body": body,
                    "type": notify_type,
                },
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(f"Notification sent via Apprise API: {title or body[:50]}")
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    action_type=self.action_type,
                    action="send",
                    message=f"Notification sent to {len(urls)} service(s)",
                    details={
                        "title": title,
                        "body_preview": body[:100] if len(body) > 100 else body,
                        "services_count": len(urls),
                    },
                )
            else:
                error_detail = response.text[:200] if response.text else "Unknown error"
                return ActionResult(
                    status=ActionStatus.FAILED,
                    action_type=self.action_type,
                    action="send",
                    message=f"Apprise API error ({response.status_code}): {error_detail}",
                )

        except requests.exceptions.ConnectionError:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send",
                message=f"Cannot connect to Apprise API at {api_url}",
            )
        except Exception as e:
            logger.exception("Notification send failed")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="send",
                message=f"Notification failed: {str(e)}",
            )

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        if action == "send":
            title = params.get("title", "")
            body = params.get("body") or params.get("message", "")
            preview = title or (body[:50] + "..." if len(body) > 50 else body)
            return f'Send notification: "{preview}"'

        return f"Notification action: {action}"

    def get_system_prompt_instructions(self) -> str:
        """Get notification-specific instructions for system prompt."""
        return """- notification: send

  **Send Notification:**
  ```xml
  <smart_action type="notification" action="send">
  {"title": "Task Complete", "body": "The report has been generated successfully.", "type": "success"}
  </smart_action>
  ```

  **Parameters:**
  - body/message: Notification content (required)
  - title: Notification title (optional)
  - type: info, success, warning, failure (optional, defaults to "info")
  - urls: List of Apprise URLs (optional if default configured on alias)
"""
