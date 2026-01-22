"""
Notification action plugin.

Sends notifications via Apprise API (external service).
Apprise API supports 80+ notification services including:
- Slack, Discord, Telegram, Microsoft Teams
- Email (SMTP), Pushover, ntfy, Gotify
- And many more: https://github.com/caronc/apprise-api

Configuration:
- Set apprise_api_url in plugin config (e.g., http://apprise-api:8000)
- Optionally set default_urls for default notification destinations
"""

import logging
import os
from typing import Optional

import httpx

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
    ResourceRequirement,
    ResourceType,
)
from plugin_base.common import (
    FieldDefinition,
    FieldType,
    ValidationError,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class NotificationActionHandler(PluginActionHandler):
    """
    Send notifications via Apprise API.

    Apprise is a universal notification library that supports 80+ services.
    This plugin uses the Apprise API (stateless mode) to send notifications.
    """

    action_type = "notification"
    display_name = "Notifications"
    description = (
        "Send push notifications via Apprise API (Slack, Discord, Telegram, ntfy, etc.)"
    )
    icon = "🔔"
    category = "communication"
    supported_sources = []  # Uses Apprise API, not document stores
    supported_source_types = []

    # Mark as non-abstract so it can be registered
    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="apprise_api_url",
                label="Apprise API URL",
                field_type=FieldType.TEXT,
                required=True,
                placeholder="http://apprise-api:8000",
                help_text="URL of your Apprise API instance",
                env_var="APPRISE_API_URL",
            ),
            FieldDefinition(
                name="default_urls",
                label="Default Notification URLs",
                field_type=FieldType.TEXTAREA,
                required=False,
                placeholder="gotify://gotify.example.com/token\nntfy://ntfy.sh/mytopic",
                help_text="Global default Apprise URLs (Smart Aliases can override)",
            ),
        ]

    @classmethod
    def get_resource_requirements(cls) -> list[ResourceRequirement]:
        """Define resources needed from Smart Alias."""
        return [
            ResourceRequirement(
                key="notification",
                sub_key="urls",
                label="Notification URLs (Apprise)",
                resource_type=ResourceType.TEXTAREA,
                help_text="Apprise URLs for this alias (one per line). See Apprise docs for URL formats.",
                required=False,
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="send",
                description="Send a push notification",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="body",
                        label="Message",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                        help_text="Notification message content",
                    ),
                    FieldDefinition(
                        name="title",
                        label="Title",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Notification title (optional)",
                    ),
                    FieldDefinition(
                        name="type",
                        label="Type",
                        field_type=FieldType.SELECT,
                        required=False,
                        default="info",
                        options=[
                            {"value": "info", "label": "Info"},
                            {"value": "success", "label": "Success"},
                            {"value": "warning", "label": "Warning"},
                            {"value": "failure", "label": "Failure"},
                        ],
                        help_text="Notification type/severity",
                    ),
                    FieldDefinition(
                        name="urls",
                        label="Notification URLs",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                        help_text="Apprise URLs to send to (uses default if not specified)",
                    ),
                ],
                examples=[
                    {
                        "body": "The report has been generated successfully.",
                        "title": "Task Complete",
                        "type": "success",
                    },
                    {
                        "body": "Server CPU usage exceeded 90%",
                        "title": "Alert",
                        "type": "warning",
                    },
                ],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Custom LLM instructions for notifications."""
        return """## Notifications

### notification:send
Send a push notification to the user's device (phone/desktop via Gotify, ntfy, Slack, etc.)

**TRIGGER WORDS - USE THIS ACTION WHEN YOU SEE:**
- "send a notification"
- "send me a notification"
- "notify me"
- "push notification"
- "send an alert"
- "alert me"

**CRITICAL**: When the user uses ANY of the trigger words above, you MUST include a <smart_action type="notification" action="send"> block. Do NOT just write the information as text - the user wants it delivered to their phone/device as a push notification.

**Parameters:**
- body (required): The notification message
- title (optional): Notification title
- type (optional): info, success, warning, failure

**Example - "Send me a notification with the weather":**
First write a brief response, then include the action:
```
I'll send that to your device now.
<smart_action type="notification" action="send">
{"title": "Weather", "body": "London: 12°C, partly cloudy"}
</smart_action>
```

**Example - "Send a notification saying nearly time":**
```
Sending that notification now.
<smart_action type="notification" action="send">
{"body": "nearly time!"}
</smart_action>
```

REMEMBER: "Send a notification" = use <smart_action>, NOT just text in chat.
"""

    def __init__(self, config: dict):
        """Initialize the notification handler."""
        self.api_url = config.get("apprise_api_url") or os.environ.get(
            "APPRISE_API_URL", ""
        )

        # Parse default URLs (newline-separated)
        default_urls_str = config.get("default_urls", "")
        if default_urls_str:
            self.default_urls = [
                url.strip()
                for url in default_urls_str.strip().split("\n")
                if url.strip()
            ]
        else:
            self.default_urls = []

        self.client = httpx.Client(timeout=30)

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the notification action."""
        if action == "send":
            return self._send_notification(params, context)

        return ActionResult(
            success=False,
            message="",
            error=f"Unknown action: {action}",
        )

    def _send_notification(self, params: dict, context: ActionContext) -> ActionResult:
        """Send a notification via Apprise API."""
        # Validate API URL
        if not self.api_url:
            return ActionResult(
                success=False,
                message="",
                error="Apprise API URL not configured",
            )

        # Get notification content
        title = params.get("title", "")
        body = params.get("body") or params.get("message", "")

        if not body:
            return ActionResult(
                success=False,
                message="",
                error="Missing required parameter: body",
            )

        # Get notification type
        notify_type = params.get("type", "info").lower()
        if notify_type not in ("info", "success", "warning", "failure"):
            notify_type = "info"

        # Get URLs - priority: params > context (Smart Alias) > plugin config
        urls = params.get("urls")
        if urls:
            # Parse if string (newline or comma-separated)
            if isinstance(urls, str):
                urls = [
                    u.strip() for u in urls.replace(",", "\n").split("\n") if u.strip()
                ]
        else:
            # Check context.default_accounts for Smart Alias configured URLs
            default_accounts = getattr(context, "default_accounts", {}) or {}
            notification_config = default_accounts.get("notification", {})
            if notification_config and notification_config.get("urls"):
                urls = notification_config["urls"]
                if isinstance(urls, str):
                    urls = [
                        u.strip()
                        for u in urls.replace(",", "\n").split("\n")
                        if u.strip()
                    ]
                logger.debug(f"Using notification URLs from Smart Alias: {urls}")
            else:
                # Fall back to plugin config default
                urls = self.default_urls

        if not urls:
            return ActionResult(
                success=False,
                message="",
                error="No notification URLs configured",
            )

        # Log which URLs are being used (mask sensitive tokens)
        masked_urls = [u.split("/")[0] + "/..." if "/" in u else u for u in urls]
        logger.info(f"Sending notification to {len(urls)} URL(s): {masked_urls}")

        # Send via Apprise API stateless endpoint
        notify_url = f"{self.api_url.rstrip('/')}/notify/"

        try:
            response = self.client.post(
                notify_url,
                json={
                    "urls": urls,
                    "title": title,
                    "body": body,
                    "type": notify_type,
                },
            )

            if response.status_code == 200:
                logger.info(f"Notification sent via Apprise: {title or body[:50]}")
                return ActionResult(
                    success=True,
                    message=f"Notification sent to {len(urls)} service(s)",
                    data={
                        "title": title,
                        "body_preview": body[:100] if len(body) > 100 else body,
                        "services_count": len(urls),
                    },
                )
            else:
                error_detail = response.text[:200] if response.text else "Unknown error"
                logger.error(
                    f"Apprise API error: {response.status_code} - {error_detail}"
                )
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Apprise API error ({response.status_code}): {error_detail}",
                )

        except httpx.ConnectError:
            return ActionResult(
                success=False,
                message="",
                error=f"Cannot connect to Apprise API at {self.api_url}",
            )
        except httpx.TimeoutException:
            return ActionResult(
                success=False,
                message="",
                error="Apprise API request timed out",
            )
        except Exception as e:
            logger.exception("Notification send failed")
            return ActionResult(
                success=False,
                message="",
                error=f"Notification failed: {str(e)}",
            )

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        if action == "send":
            title = params.get("title", "")
            body = params.get("body") or params.get("message", "")
            preview = title or (body[:50] + "..." if len(body) > 50 else body)
            return f'Send notification: "{preview}"'

        return f"Notification action: {action}"

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action == "send":
            body = params.get("body") or params.get("message")
            if not body:
                errors.append(
                    ValidationError("body", "Notification message is required")
                )

            # URLs are optional if defaults are configured
            urls = params.get("urls")
            if not urls and not self.default_urls:
                errors.append(
                    ValidationError(
                        "urls",
                        "No notification URLs provided and no defaults configured",
                    )
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def is_available(self) -> bool:
        """Check if the notification service is available."""
        return bool(self.api_url)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Apprise API."""
        if not self.api_url:
            return False, "Apprise API URL not configured"

        try:
            # Try to reach the Apprise API status endpoint
            status_url = f"{self.api_url.rstrip('/')}/status"
            response = self.client.get(status_url)

            if response.status_code == 200:
                urls_count = len(self.default_urls)
                return (
                    True,
                    f"Connected to Apprise API. {urls_count} default URL(s) configured.",
                )
            response = self.client.get(status_url)

            if response.status_code == 200:
                urls_count = len(self.default_urls)
                return (
                    True,
                    f"Connected to Apprise API. {urls_count} default URL(s) configured.",
                )
            else:
                return False, f"Apprise API returned status {response.status_code}"

        except httpx.ConnectError:
            return False, f"Cannot connect to Apprise API at {self.api_url}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def close(self):
        """Clean up resources."""
        if self.client:
            self.client.close()
