"""
Schedule Action Handler.

Handles scheduling prompts for future execution:
- schedule: Create a scheduled prompt (calendar event with prompt as description)
- cancel: Cancel a scheduled prompt
- list: List scheduled prompts

Scheduled prompts are stored as calendar events where:
- Event title: Brief description for logging
- Event description: The actual prompt to execute
- Event time: When to run the prompt
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from actions.base import ActionContext, ActionHandler, ActionResult, ActionStatus

logger = logging.getLogger(__name__)


class ScheduleActionHandler(ActionHandler):
    """
    Handler for scheduling prompts to run at specific times.

    Creates calendar events on the Smart Alias's configured scheduled prompts
    calendar. The event description contains the prompt to execute.
    """

    @property
    def action_type(self) -> str:
        return "schedule"

    @property
    def supported_actions(self) -> list[str]:
        return ["prompt", "cancel"]

    def validate(
        self, action: str, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Validate schedule action parameters."""
        if action not in self.supported_actions:
            return False, f"Unknown schedule action: {action}"

        if action == "prompt":
            return self._validate_schedule_prompt(params, context)
        elif action == "cancel":
            return self._validate_cancel(params)

        return True, ""

    def _validate_schedule_prompt(
        self, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Validate schedule prompt parameters."""
        prompt = params.get("prompt")
        if not prompt:
            return False, "Missing required parameter: prompt (what to execute)"

        # Need either a time or recurrence
        time = params.get("time")
        recurrence = params.get("recurrence")

        if not time and not recurrence:
            return (
                False,
                "Missing required parameter: time or recurrence (when to execute)",
            )

        # Validate scheduled prompts is configured
        schedule_config = context.default_accounts.get("schedule", {})
        if not schedule_config.get("account_id"):
            return (
                False,
                "Scheduled prompts not configured on this Smart Alias. "
                "Enable 'Scheduled Prompts' in the Smart Alias settings first.",
            )

        return True, ""

    def _validate_cancel(self, params: dict) -> tuple[bool, str]:
        """Validate cancel parameters."""
        event_id = params.get("event_id")
        if not event_id:
            return False, "Missing required parameter: event_id"
        return True, ""

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the schedule action."""
        if action == "prompt":
            return self._schedule_prompt(params, context)
        elif action == "cancel":
            return self._cancel_prompt(params, context)

        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=self.action_type,
            action=action,
            message=f"Unknown action: {action}",
        )

    def _schedule_prompt(self, params: dict, context: ActionContext) -> ActionResult:
        """Schedule a prompt for future execution."""
        import requests as http_requests

        schedule_config = context.default_accounts.get("schedule", {})
        account_id = schedule_config.get("account_id")
        calendar_id = schedule_config.get("calendar_id") or "primary"

        if not account_id:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="prompt",
                message="Scheduled prompts not configured on this Smart Alias",
            )

        # Get OAuth token
        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="prompt",
                message=f"OAuth account {account_id} not found",
            )

        access_token = self._get_valid_access_token(account_id, token_data)
        if not access_token:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="prompt",
                message="Failed to get valid access token",
            )

        # Build event
        prompt = params.get("prompt")
        title = params.get("title") or self._generate_title(prompt)
        time_str = params.get("time")
        recurrence = params.get("recurrence")
        timezone = params.get("timezone", "Europe/London")

        # Parse time
        start_time, end_time = self._parse_time(time_str, timezone)
        if not start_time:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="prompt",
                message=f"Could not parse time: {time_str}",
            )

        # Build event body
        event_body = {
            "summary": f"[Prompt] {title}",
            "description": prompt,
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
            rrule = self._parse_recurrence(recurrence)
            if rrule:
                event_body["recurrence"] = [rrule]

        # Create event
        response = http_requests.post(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=event_body,
            timeout=30,
        )

        if response.status_code not in (200, 201):
            logger.error(f"Failed to create scheduled prompt: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="prompt",
                message=f"Failed to create calendar event: {response.status_code}",
            )

        event = response.json()
        event_id = event.get("id")

        # Format response message
        if recurrence:
            schedule_desc = f"recurring ({recurrence})"
        else:
            schedule_desc = start_time.strftime("%Y-%m-%d at %H:%M")

        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="prompt",
            message=f"Scheduled prompt '{title}' for {schedule_desc}",
            data={
                "event_id": event_id,
                "title": title,
                "prompt": prompt,
                "scheduled_time": start_time.isoformat(),
                "recurrence": recurrence,
            },
        )

    def _cancel_prompt(self, params: dict, context: ActionContext) -> ActionResult:
        """Cancel a scheduled prompt."""
        import requests as http_requests

        schedule_config = context.default_accounts.get("schedule", {})
        account_id = schedule_config.get("account_id")
        calendar_id = schedule_config.get("calendar_id") or "primary"

        if not account_id:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="cancel",
                message="Scheduled prompts not configured",
            )

        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="cancel",
                message=f"OAuth account {account_id} not found",
            )

        access_token = self._get_valid_access_token(account_id, token_data)
        if not access_token:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="cancel",
                message="Failed to get valid access token",
            )

        event_id = params.get("event_id")

        response = http_requests.delete(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="cancel",
                message=f"Failed to cancel scheduled prompt: {response.status_code}",
            )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="cancel",
            message=f"Cancelled scheduled prompt {event_id}",
            data={"event_id": event_id},
        )

    def _generate_title(self, prompt: str) -> str:
        """Generate a short title from the prompt."""
        # Take first 50 chars, cut at word boundary
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
        import re
        from datetime import datetime, timedelta

        if not time_str:
            return None, None

        now = datetime.now()

        # Try ISO format first
        try:
            if "T" in time_str:
                start = datetime.fromisoformat(time_str.replace("Z", ""))
                end = start + timedelta(minutes=5)  # Short duration for prompts
                return start, end
        except ValueError:
            pass

        # Try time-only format (HH:MM)
        time_match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If time has passed today, schedule for tomorrow
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

        # "in X minutes/hours"
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

        return None, None

    def _parse_recurrence(self, recurrence: str) -> Optional[str]:
        """Convert recurrence string to RRULE format."""
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

    def _get_valid_access_token(
        self, account_id: int, token_data: dict
    ) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        import requests as http_requests

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return None

        # Test the token
        test_response = http_requests.get(
            "https://www.googleapis.com/calendar/v3/users/me/calendarList",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"maxResults": 1},
            timeout=10,
        )

        if test_response.status_code == 200:
            return access_token

        # Token expired, try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error("Cannot refresh token - missing credentials")
            return None

        logger.info(f"Refreshing access token for account {account_id}")

        refresh_response = http_requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        if refresh_response.status_code != 200:
            logger.error(f"Token refresh failed: {refresh_response.text}")
            return None

        new_token_data = refresh_response.json()
        new_access_token = new_token_data.get("access_token")

        # Update stored token
        from db.oauth_tokens import update_oauth_token_data

        updated_data = {**token_data, "access_token": new_access_token}
        update_oauth_token_data(account_id, updated_data)

        return new_access_token

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
            event_id = params.get("event_id", "")
            return f"Cancel scheduled prompt: {event_id}"

        return f"Schedule action: {action}"

    def get_system_prompt_instructions(self) -> str:
        """Get schedule-specific instructions for system prompt."""
        return """- schedule: prompt, cancel

  **Schedule a Prompt (one-time):**
  ```xml
  <smart_action type="schedule" action="prompt">
  {"prompt": "Send me a notification with the weather for Harpenden", "time": "06:30", "title": "Morning weather"}
  </smart_action>
  ```

  **Schedule a Recurring Prompt:**
  ```xml
  <smart_action type="schedule" action="prompt">
  {"prompt": "Check my calendar for today and send me a notification summary", "time": "07:00", "recurrence": "daily", "title": "Daily schedule"}
  </smart_action>
  ```

  **Schedule for Tomorrow:**
  ```xml
  <smart_action type="schedule" action="prompt">
  {"prompt": "Remind me about the meeting", "time": "tomorrow at 9am"}
  </smart_action>
  ```

  **Cancel a Scheduled Prompt:**
  ```xml
  <smart_action type="schedule" action="cancel">
  {"event_id": "abc123xyz"}
  </smart_action>
  ```

  **Parameters:**
  - prompt: The prompt to execute at the scheduled time (required)
  - time: When to run - "HH:MM", "tomorrow at HH:MM", ISO format, or "in X minutes" (required)
  - recurrence: "daily", "weekly", "weekdays", "monthly" (optional - makes it recurring)
  - title: Short description for the schedule (optional, auto-generated from prompt)

  Scheduled prompts run through this same assistant with all capabilities (web search, notifications, etc.)
"""
