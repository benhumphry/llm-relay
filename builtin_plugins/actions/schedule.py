"""
Unified Schedule action plugin.

Handles scheduling prompts for future execution across calendar providers:
- Google Calendar (via OAuth)
- Outlook Calendar (via OAuth)

Scheduled prompts are stored as calendar events where:
- Event title: Brief description for logging (prefixed with [Prompt])
- Event description: The actual prompt to execute
- Event time: When to run the prompt

NO CONFIG FIELDS - all configuration comes from Smart Alias context at runtime:
- default_accounts["schedule"]["id"] = OAuth account ID
- default_accounts["schedule"]["provider"] = "google" or "microsoft"
- default_accounts["schedule"]["calendar_id"] = specific calendar ID
"""

import logging
import re
from datetime import datetime, timedelta
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
from plugin_base.oauth import OAuthMixin

logger = logging.getLogger(__name__)


class ScheduleActionHandler(OAuthMixin, PluginActionHandler):
    """
    Schedule prompts for future or recurring execution.

    Creates calendar events that trigger prompt execution at the scheduled time.
    Supports both Google Calendar and Outlook.
    Provider is determined at runtime from Smart Alias context - no plugin config needed.
    """

    action_type = "schedule"
    display_name = "Scheduled Prompts"
    description = "Schedule prompts for future or recurring execution via calendar"
    icon = "⏰"
    category = "automation"
    supported_sources = ["Google Calendar", "Outlook Calendar"]
    # Uses calendar accounts for scheduling
    supported_source_types = ["mcp:gcalendar", "outlook_calendar"]

    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """No config fields - everything comes from Smart Alias context."""
        return []

    @classmethod
    def get_resource_requirements(cls) -> list[ResourceRequirement]:
        """Define resources needed from Smart Alias."""
        return [
            ResourceRequirement(
                key="schedule",
                label="Scheduled Prompts Account",
                resource_type=ResourceType.OAUTH_ACCOUNT,
                providers=["google", "microsoft"],
                help_text="Calendar account for scheduled prompts (Google or Outlook)",
                required=True,
            ),
            ResourceRequirement(
                key="schedule",
                sub_key="calendar_id",
                label="Prompts Calendar",
                resource_type=ResourceType.CALENDAR_PICKER,
                depends_on="schedule",
                help_text="Calendar to store scheduled prompt events",
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="prompt",
                description="Schedule a prompt for future execution",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="prompt",
                        label="Prompt",
                        field_type=FieldType.TEXTAREA,
                        required=True,
                        help_text="The prompt to execute at the scheduled time",
                    ),
                    FieldDefinition(
                        name="time",
                        label="Time",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="When to run: HH:MM, 'tomorrow at 9am', ISO format, or 'in X minutes'",
                    ),
                    FieldDefinition(
                        name="title",
                        label="Title",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Short description (auto-generated if not provided)",
                    ),
                    FieldDefinition(
                        name="recurrence",
                        label="Recurrence",
                        field_type=FieldType.SELECT,
                        required=False,
                        options=[
                            {"value": "", "label": "One-time"},
                            {"value": "daily", "label": "Daily"},
                            {"value": "weekdays", "label": "Weekdays"},
                            {"value": "weekly", "label": "Weekly"},
                            {"value": "monthly", "label": "Monthly"},
                        ],
                    ),
                    FieldDefinition(
                        name="timezone",
                        label="Timezone",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Timezone override (e.g., 'America/New_York')",
                    ),
                ],
                examples=[
                    {
                        "prompt": "Send me the weather forecast for today",
                        "time": "06:30",
                        "title": "Morning weather",
                        "recurrence": "daily",
                    },
                    {
                        "prompt": "Remind me about the team meeting",
                        "time": "tomorrow at 9am",
                    },
                ],
            ),
            ActionDefinition(
                name="cancel",
                description="Cancel a scheduled prompt",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="event_id",
                        label="Event ID",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="The ID of the scheduled prompt to cancel",
                    ),
                ],
                examples=[
                    {"event_id": "abc123xyz"},
                ],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """LLM instructions for schedule actions."""
        return """## Scheduled Prompts

Schedule prompts to run at specific times. The scheduled prompt runs through this assistant with full capabilities.

### schedule:prompt
Schedule a prompt for one-time or recurring execution.

**Parameters:**
- prompt (required): The prompt to execute
- time (required): When to run - "HH:MM", "tomorrow at 9am", ISO format, or "in X minutes"
- title (optional): Short description for the schedule
- recurrence (optional): "daily", "weekdays", "weekly", "monthly"

**Examples:**
```xml
<smart_action type="schedule" action="prompt">
{"prompt": "Send me a notification with the weather", "time": "06:30", "recurrence": "daily", "title": "Morning weather"}
</smart_action>
```

```xml
<smart_action type="schedule" action="prompt">
{"prompt": "Remind me about the meeting", "time": "tomorrow at 9am"}
</smart_action>
```

```xml
<smart_action type="schedule" action="prompt">
{"prompt": "Check my inbox and summarize new emails", "time": "in 30 minutes"}
</smart_action>
```

### schedule:cancel
Cancel a scheduled prompt.

**Example:**
```xml
<smart_action type="schedule" action="cancel">
{"event_id": "abc123xyz"}
</smart_action>
```
"""

    def __init__(self, config: dict):
        """Initialize the schedule handler - config is ignored, uses context."""
        # These will be set from context at execution time
        self.oauth_account_id: Optional[int] = None
        self.oauth_provider: Optional[str] = None
        self.calendar_id = "primary"
        self.default_timezone = "Europe/London"

    def _configure_from_context(self, context: ActionContext) -> bool:
        """Configure provider and credentials from Smart Alias context."""
        default_accounts = getattr(context, "default_accounts", {})
        schedule_config = default_accounts.get("schedule", {})

        if not schedule_config:
            logger.warning("No schedule configuration in context.default_accounts")
            return False

        account_id = schedule_config.get("account_id") or schedule_config.get("id")
        if not account_id:
            logger.error("Schedule OAuth account ID not found in context")
            return False

        self.oauth_account_id = account_id
        self.oauth_provider = schedule_config.get("provider", "google")
        self.calendar_id = schedule_config.get("calendar_id", "primary")

        logger.info(
            f"Schedule: Using provider '{self.oauth_provider}' account {account_id}"
        )

        self._init_oauth_client()
        return True

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the schedule action."""
        # Always configure from context
        if not self._configure_from_context(context):
            return ActionResult(
                success=False,
                message="",
                error="No schedule account configured - set Scheduled Prompts Account in Smart Alias",
            )

        try:
            if action == "prompt":
                if self.oauth_provider == "microsoft":
                    return self._schedule_prompt_outlook(params, context)
                else:
                    return self._schedule_prompt_google(params, context)
            elif action == "cancel":
                if self.oauth_provider == "microsoft":
                    return self._cancel_prompt_outlook(params)
                else:
                    return self._cancel_prompt_google(params)
            else:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Unknown action: {action}",
                )
        except httpx.HTTPStatusError as e:
            logger.error(f"Schedule API error: {e.response.status_code}")
            error_text = e.response.text[:200] if e.response.text else str(e)
            return ActionResult(
                success=False,
                message="",
                error=f"Calendar API error ({e.response.status_code}): {error_text}",
            )
        except Exception as e:
            logger.exception(f"Schedule action failed: {action}")
            return ActionResult(
                success=False,
                message="",
                error=f"Action failed: {str(e)}",
            )

    # -------------------------------------------------------------------------
    # Google Calendar Implementation
    # -------------------------------------------------------------------------

    def _schedule_prompt_google(
        self, params: dict, context: "ActionContext"
    ) -> ActionResult:
        """Schedule a prompt via Google Calendar."""
        prompt = params.get("prompt")
        title = params.get("title") or self._generate_title(prompt)
        time_str = params.get("time")
        recurrence = params.get("recurrence")
        timezone = params.get("timezone", self.default_timezone)

        # Parse time
        start_time, end_time = self._parse_time(time_str, timezone)
        if not start_time:
            return ActionResult(
                success=False,
                message="",
                error=f"Could not parse time: {time_str}",
            )

        # Build event body
        # Store the alias name in location so scheduler knows which alias to use
        alias_name = context.alias_name or "unknown"
        event_body = {
            "summary": f"[Prompt] {title}",
            "description": prompt,
            "location": f"alias:{alias_name}",
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": timezone,
            },
        }

        # Add recurrence if specified
        if recurrence:
            rrule = self._parse_recurrence_google(recurrence)
            if rrule:
                event_body["recurrence"] = [rrule]

        # Create event
        response = self.oauth_post(
            f"https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}/events",
            json=event_body,
        )
        response.raise_for_status()

        event = response.json()
        event_id = event.get("id")

        # Format response message
        if recurrence:
            schedule_desc = f"recurring ({recurrence})"
        else:
            schedule_desc = start_time.strftime("%Y-%m-%d at %H:%M")

        logger.info(f"Created scheduled prompt (Google): {event_id}")
        return ActionResult(
            success=True,
            message=f"Scheduled prompt '{title}' for {schedule_desc}",
            data={
                "event_id": event_id,
                "title": title,
                "prompt": prompt,
                "scheduled_time": start_time.isoformat(),
                "recurrence": recurrence,
                "provider": "google",
            },
        )

    def _cancel_prompt_google(self, params: dict) -> ActionResult:
        """Cancel a scheduled prompt via Google Calendar."""
        event_id = params.get("event_id")

        response = self.oauth_delete(
            f"https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}/events/{event_id}"
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to cancel scheduled prompt: {response.status_code}",
            )

        logger.info(f"Cancelled scheduled prompt (Google): {event_id}")
        return ActionResult(
            success=True,
            message=f"Cancelled scheduled prompt {event_id}",
            data={"event_id": event_id, "provider": "google"},
        )

    # -------------------------------------------------------------------------
    # Outlook Calendar Implementation
    # -------------------------------------------------------------------------

    def _schedule_prompt_outlook(
        self, params: dict, context: "ActionContext"
    ) -> ActionResult:
        """Schedule a prompt via Outlook Calendar."""
        prompt = params.get("prompt")
        title = params.get("title") or self._generate_title(prompt)
        time_str = params.get("time")
        recurrence = params.get("recurrence")
        timezone = params.get("timezone", self.default_timezone)

        # Parse time
        start_time, end_time = self._parse_time(time_str, timezone)
        if not start_time:
            return ActionResult(
                success=False,
                message="",
                error=f"Could not parse time: {time_str}",
            )

        # Build event body for Outlook
        # Store the alias name in location so scheduler knows which alias to use
        alias_name = context.alias_name or "unknown"
        event_body = {
            "subject": f"[Prompt] {title}",
            "body": {
                "contentType": "text",
                "content": prompt,
            },
            "location": {
                "displayName": f"alias:{alias_name}",
            },
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": timezone,
            },
        }

        # Add recurrence if specified
        if recurrence:
            pattern = self._parse_recurrence_outlook(recurrence)
            if pattern:
                event_body["recurrence"] = pattern

        # Create event
        response = self.oauth_post(
            "https://graph.microsoft.com/v1.0/me/events",
            json=event_body,
        )
        response.raise_for_status()

        event = response.json()
        event_id = event.get("id")

        # Format response message
        if recurrence:
            schedule_desc = f"recurring ({recurrence})"
        else:
            schedule_desc = start_time.strftime("%Y-%m-%d at %H:%M")

        logger.info(f"Created scheduled prompt (Outlook): {event_id}")
        return ActionResult(
            success=True,
            message=f"Scheduled prompt '{title}' for {schedule_desc}",
            data={
                "event_id": event_id,
                "title": title,
                "prompt": prompt,
                "scheduled_time": start_time.isoformat(),
                "recurrence": recurrence,
                "provider": "microsoft",
            },
        )

    def _cancel_prompt_outlook(self, params: dict) -> ActionResult:
        """Cancel a scheduled prompt via Outlook Calendar."""
        event_id = params.get("event_id")

        response = self.oauth_delete(
            f"https://graph.microsoft.com/v1.0/me/events/{event_id}"
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to cancel scheduled prompt: {response.status_code}",
            )

        logger.info(f"Cancelled scheduled prompt (Outlook): {event_id}")
        return ActionResult(
            success=True,
            message=f"Cancelled scheduled prompt {event_id}",
            data={"event_id": event_id, "provider": "microsoft"},
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _generate_title(self, prompt: str) -> str:
        """Generate a short title from the prompt."""
        if len(prompt) <= 50:
            return prompt
        truncated = prompt[:50]
        last_space = truncated.rfind(" ")
        if last_space > 20:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _parse_time(
        self, time_str: str, timezone: str
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse a time string into start and end datetime.

        Supports:
        - ISO format: "2026-01-20T06:30:00"
        - Time only: "06:30" (assumes today or tomorrow)
        - Relative: "in 30 minutes", "tomorrow at 9am"
        """
        if not time_str:
            return None, None

        now = datetime.now()

        # Try ISO format first
        try:
            if "T" in time_str:
                start = datetime.fromisoformat(time_str.replace("Z", ""))
                end = start + timedelta(minutes=5)
                return start, end
        except ValueError:
            pass

        # Try time-only format (HH:MM)
        time_match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if start <= now:
                start += timedelta(days=1)
            end = start + timedelta(minutes=5)
            return start, end

        # Try natural language patterns
        time_str_lower = time_str.lower()

        # "tomorrow at HH:MM"
        tomorrow_match = re.search(
            r"tomorrow\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            time_str_lower,
        )
        if tomorrow_match:
            hour = int(tomorrow_match.group(1))
            minute = int(tomorrow_match.group(2) or 0)
            ampm = tomorrow_match.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            start = (now + timedelta(days=1)).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            end = start + timedelta(minutes=5)
            return start, end

        # "in X minutes/hours" - check this BEFORE generic "at HH:MM"
        in_match = re.search(r"in\s+(\d+)\s+(minute|hour|min|hr)s?", time_str_lower)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)
            if unit.startswith("hour") or unit == "hr":
                delta = timedelta(hours=amount)
            else:
                delta = timedelta(minutes=amount)
            start = now + delta
            end = start + timedelta(minutes=5)
            return start, end

        # "at HH:MM" or "HH:MM am/pm"
        at_match = re.search(
            r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            time_str_lower,
        )
        if at_match:
            hour = int(at_match.group(1))
            minute = int(at_match.group(2) or 0)
            ampm = at_match.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if start <= now:
                start += timedelta(days=1)
            end = start + timedelta(minutes=5)
            return start, end

        return None, None

    def _parse_recurrence_google(self, recurrence: str) -> Optional[str]:
        """Convert recurrence string to Google RRULE format."""
        if not recurrence:
            return None

        recurrence_lower = recurrence.lower().strip()

        # Handle RRULE format directly
        if recurrence_lower.startswith("rrule:"):
            return recurrence.upper()

        # Map common patterns
        patterns = {
            "daily": "RRULE:FREQ=DAILY",
            "every day": "RRULE:FREQ=DAILY",
            "weekly": "RRULE:FREQ=WEEKLY",
            "every week": "RRULE:FREQ=WEEKLY",
            "weekdays": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            "every weekday": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            "monthly": "RRULE:FREQ=MONTHLY",
            "every month": "RRULE:FREQ=MONTHLY",
            "yearly": "RRULE:FREQ=YEARLY",
            "every year": "RRULE:FREQ=YEARLY",
        }

        return patterns.get(recurrence_lower)

    def _parse_recurrence_outlook(self, recurrence: str) -> Optional[dict]:
        """Convert recurrence string to Outlook pattern."""
        if not recurrence:
            return None

        recurrence_lower = recurrence.lower().strip()

        patterns = {
            "daily": {
                "pattern": {"type": "daily", "interval": 1},
                "range": {"type": "noEnd"},
            },
            "weekly": {
                "pattern": {"type": "weekly", "interval": 1, "daysOfWeek": ["monday"]},
                "range": {"type": "noEnd"},
            },
            "weekdays": {
                "pattern": {
                    "type": "weekly",
                    "interval": 1,
                    "daysOfWeek": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                    ],
                },
                "range": {"type": "noEnd"},
            },
            "monthly": {
                "pattern": {"type": "absoluteMonthly", "interval": 1, "dayOfMonth": 1},
                "range": {"type": "noEnd"},
            },
            "yearly": {
                "pattern": {
                    "type": "absoluteYearly",
                    "interval": 1,
                    "dayOfMonth": 1,
                    "month": 1,
                },
                "range": {"type": "noEnd"},
            },
        }

        return patterns.get(recurrence_lower)

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action == "prompt":
            if not params.get("prompt"):
                errors.append(ValidationError("prompt", "Prompt is required"))
            if not params.get("time"):
                errors.append(ValidationError("time", "Time is required"))

        elif action == "cancel":
            if not params.get("event_id"):
                errors.append(ValidationError("event_id", "Event ID is required"))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        if action == "prompt":
            title = params.get("title", "")
            prompt = params.get("prompt", "")
            time = params.get("time", "")
            recurrence = params.get("recurrence", "")

            desc = title or (prompt[:40] + "..." if len(prompt) > 40 else prompt)
            schedule = f"at {time}" if time else ""
            if recurrence:
                schedule += f" ({recurrence})"

            return f'Schedule prompt: "{desc}" {schedule}'

        elif action == "cancel":
            event_id = params.get("event_id", "")[:20]
            return f"Cancel scheduled prompt: {event_id}"

        return f"Schedule action: {action}"

    def is_available(self) -> bool:
        """Check if plugin is available (always true - config comes from context)."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection - cannot test without context."""
        return (
            True,
            "Schedule handler ready (provider configured per Smart Alias)",
        )

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_oauth_client") and self._oauth_client:
            self._oauth_client.close()
