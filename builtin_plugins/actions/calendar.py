"""
Calendar action plugin.

Handles calendar actions:
- create: Create new events (including recurring)
- update: Modify existing events
- delete: Remove events

Supports Google Calendar via OAuth.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

from plugin_base.action import (
    ActionContext,
    ActionDefinition,
    ActionResult,
    ActionRisk,
    PluginActionHandler,
)
from plugin_base.common import (
    FieldDefinition,
    FieldType,
    ValidationError,
    ValidationResult,
)
from plugin_base.oauth import OAuthMixin

logger = logging.getLogger(__name__)


class CalendarActionHandler(OAuthMixin, PluginActionHandler):
    """
    Create, update, and delete calendar events.

    Supports Google Calendar via OAuth authentication.
    """

    action_type = "calendar"
    display_name = "Google Calendar"
    description = "Create, update, and delete events in Google Calendar"
    icon = "📅"
    category = "productivity"

    # Mark as non-abstract so it can be registered
    _abstract = False

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                help_text="Select a connected Google account with Calendar access",
                picker_options={"provider": "google"},
            ),
            FieldDefinition(
                name="default_calendar_id",
                label="Default Calendar",
                field_type=FieldType.TEXT,
                required=False,
                default="primary",
                help_text="Default calendar ID (leave as 'primary' for main calendar)",
            ),
            FieldDefinition(
                name="default_timezone",
                label="Default Timezone",
                field_type=FieldType.TEXT,
                required=False,
                default="Europe/London",
                help_text="Default timezone for events (e.g., 'America/New_York')",
            ),
        ]

    @classmethod
    def get_actions(cls) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                name="create",
                description="Create a new calendar event",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="title",
                        label="Event Title",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Title/summary of the event",
                    ),
                    FieldDefinition(
                        name="start",
                        label="Start Time",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="Start time in ISO format (e.g., 2026-01-20T14:00:00)",
                    ),
                    FieldDefinition(
                        name="end",
                        label="End Time",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="End time (optional, defaults to 1 hour after start)",
                    ),
                    FieldDefinition(
                        name="description",
                        label="Description",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                    FieldDefinition(
                        name="location",
                        label="Location",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="all_day",
                        label="All-Day Event",
                        field_type=FieldType.BOOLEAN,
                        required=False,
                        default=False,
                    ),
                    FieldDefinition(
                        name="recurrence",
                        label="Recurrence",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Recurrence rule (daily, weekly, monthly, yearly, weekdays, or RRULE)",
                    ),
                    FieldDefinition(
                        name="calendar_id",
                        label="Calendar ID",
                        field_type=FieldType.TEXT,
                        required=False,
                        help_text="Specific calendar ID (optional)",
                    ),
                ],
                examples=[
                    {
                        "title": "Team Meeting",
                        "start": "2026-01-20T14:00:00",
                        "end": "2026-01-20T15:00:00",
                        "description": "Weekly sync",
                        "location": "Conference Room A",
                    },
                    {
                        "title": "Daily Standup",
                        "start": "2026-01-20T09:00:00",
                        "recurrence": "weekdays",
                    },
                ],
            ),
            ActionDefinition(
                name="update",
                description="Update an existing calendar event",
                risk=ActionRisk.MEDIUM,
                params=[
                    FieldDefinition(
                        name="event_id",
                        label="Event ID",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="The ID of the event to update",
                    ),
                    FieldDefinition(
                        name="title",
                        label="New Title",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="start",
                        label="New Start Time",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="end",
                        label="New End Time",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="description",
                        label="New Description",
                        field_type=FieldType.TEXTAREA,
                        required=False,
                    ),
                    FieldDefinition(
                        name="location",
                        label="New Location",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                    FieldDefinition(
                        name="calendar_id",
                        label="Calendar ID",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                ],
                examples=[
                    {
                        "event_id": "abc123xyz",
                        "title": "Updated Meeting Title",
                        "start": "2026-01-20T15:00:00",
                    },
                ],
            ),
            ActionDefinition(
                name="delete",
                description="Delete a calendar event",
                risk=ActionRisk.HIGH,
                params=[
                    FieldDefinition(
                        name="event_id",
                        label="Event ID",
                        field_type=FieldType.TEXT,
                        required=True,
                        help_text="The ID of the event to delete",
                    ),
                    FieldDefinition(
                        name="calendar_id",
                        label="Calendar ID",
                        field_type=FieldType.TEXT,
                        required=False,
                    ),
                ],
                examples=[
                    {"event_id": "abc123xyz"},
                ],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Custom LLM instructions for calendar actions."""
        return """## Calendar

### calendar:create
Create a new calendar event.

**Parameters:**
- title (required): Event title/summary
- start (required): Start time in ISO format (e.g., 2026-01-20T14:00:00)
- end (optional): End time (defaults to 1 hour after start)
- description (optional): Event description
- location (optional): Event location
- all_day (optional): Set to true for all-day events
- recurrence (optional): daily, weekly, monthly, yearly, weekdays, or RRULE format
- calendar_id (optional): Specific calendar ID

**Examples:**
```xml
<smart_action type="calendar" action="create">
{"title": "Team Meeting", "start": "2026-01-20T14:00:00", "end": "2026-01-20T15:00:00", "description": "Weekly sync", "location": "Conference Room A"}
</smart_action>
```

```xml
<smart_action type="calendar" action="create">
{"title": "Daily Standup", "start": "2026-01-20T09:00:00", "recurrence": "weekdays"}
</smart_action>
```

### calendar:update
Update an existing calendar event.

**Parameters:**
- event_id (required): The ID of the event to update
- title, start, end, description, location (optional): Fields to update

**Example:**
```xml
<smart_action type="calendar" action="update">
{"event_id": "abc123", "title": "Updated Title", "start": "2026-01-20T15:00:00"}
</smart_action>
```

### calendar:delete
Delete a calendar event.

**Parameters:**
- event_id (required): The ID of the event to delete

**Example:**
```xml
<smart_action type="calendar" action="delete">
{"event_id": "abc123"}
</smart_action>
```
"""

    def __init__(self, config: dict):
        """Initialize the calendar handler."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"
        self.default_calendar_id = config.get("default_calendar_id", "primary")
        self.default_timezone = config.get("default_timezone", "Europe/London")

        # Initialize OAuth client if account configured
        if self.oauth_account_id:
            self._init_oauth_client()

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the calendar action."""
        if not self.oauth_account_id:
            return ActionResult(
                success=False,
                message="",
                error="OAuth account not configured",
            )

        try:
            if action == "create":
                return self._create_event(params)
            elif action == "update":
                return self._update_event(params)
            elif action == "delete":
                return self._delete_event(params)
            else:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Unknown action: {action}",
                )
        except httpx.HTTPStatusError as e:
            logger.error(f"Calendar API error: {e.response.status_code}")
            error_text = e.response.text[:200] if e.response.text else str(e)
            return ActionResult(
                success=False,
                message="",
                error=f"Calendar API error ({e.response.status_code}): {error_text}",
            )
        except Exception as e:
            logger.exception(f"Calendar action failed: {action}")
            return ActionResult(
                success=False,
                message="",
                error=f"Action failed: {str(e)}",
            )

    def _create_event(self, params: dict) -> ActionResult:
        """Create an event in Google Calendar."""
        calendar_id = params.get("calendar_id", self.default_calendar_id)
        title = params.get("title") or params.get("summary")
        description = params.get("description", "")
        location = params.get("location", "")
        all_day = params.get("all_day", False)
        timezone = params.get("timezone", self.default_timezone)

        if not title:
            return ActionResult(
                success=False,
                message="",
                error="Missing required parameter: title",
            )

        # Build event body
        event_body = {
            "summary": title,
            "description": description,
            "location": location,
        }

        # Handle timing
        if all_day:
            date_str = params.get("date") or params.get("start")
            if not date_str:
                return ActionResult(
                    success=False,
                    message="",
                    error="All-day events require 'date' or 'start' parameter",
                )

            # Parse and format as YYYY-MM-DD
            if "T" in str(date_str):
                date_str = date_str.split("T")[0]
            event_body["start"] = {"date": date_str}

            # End date for all-day is exclusive (next day)
            end_date = params.get("end_date") or params.get("end")
            if end_date:
                if "T" in str(end_date):
                    end_date = end_date.split("T")[0]
                event_body["end"] = {"date": end_date}
            else:
                try:
                    start_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    end_dt = start_dt + timedelta(days=1)
                    event_body["end"] = {"date": end_dt.strftime("%Y-%m-%d")}
                except ValueError:
                    event_body["end"] = {"date": date_str}
        else:
            # Timed event
            start = params.get("start")
            if not start:
                return ActionResult(
                    success=False,
                    message="",
                    error="Missing required parameter: start",
                )

            start_dt = self._parse_datetime(start)
            if not start_dt:
                return ActionResult(
                    success=False,
                    message="",
                    error=f"Invalid start time format: {start}",
                )

            event_body["start"] = {
                "dateTime": start_dt.isoformat(),
                "timeZone": timezone,
            }

            # Parse end time (default to 1 hour after start)
            end = params.get("end")
            if end:
                end_dt = self._parse_datetime(end)
                if not end_dt:
                    end_dt = start_dt + timedelta(hours=1)
            else:
                end_dt = start_dt + timedelta(hours=1)

            event_body["end"] = {
                "dateTime": end_dt.isoformat(),
                "timeZone": timezone,
            }

        # Handle recurrence
        recurrence = params.get("recurrence")
        if recurrence:
            if isinstance(recurrence, str):
                if recurrence.startswith("RRULE:"):
                    event_body["recurrence"] = [recurrence]
                else:
                    rrule = self._simple_to_rrule(recurrence)
                    if rrule:
                        event_body["recurrence"] = [rrule]

        # Handle attendees
        attendees = params.get("attendees", [])
        if attendees:
            if isinstance(attendees, str):
                attendees = [a.strip() for a in attendees.split(",")]
            event_body["attendees"] = [{"email": a} for a in attendees]

        # Create the event
        response = self.oauth_post(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            json=event_body,
        )
        response.raise_for_status()

        event_data = response.json()
        event_id = event_data.get("id")
        html_link = event_data.get("htmlLink")

        logger.info(f"Created Google Calendar event: {event_id}")
        return ActionResult(
            success=True,
            message=f"Event '{title}' created",
            data={
                "event_id": event_id,
                "calendar_id": calendar_id,
                "title": title,
                "link": html_link,
            },
        )

    def _update_event(self, params: dict) -> ActionResult:
        """Update an event in Google Calendar."""
        calendar_id = params.get("calendar_id", self.default_calendar_id)
        event_id = params.get("event_id")

        if not event_id:
            return ActionResult(
                success=False,
                message="",
                error="Missing required parameter: event_id",
            )

        # First, get the existing event
        get_response = self.oauth_get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        )

        if get_response.status_code != 200:
            return ActionResult(
                success=False,
                message="",
                error=f"Event not found: {event_id}",
            )

        event_body = get_response.json()
        timezone = params.get("timezone", self.default_timezone)

        # Update fields
        if params.get("title") or params.get("summary"):
            event_body["summary"] = params.get("title") or params.get("summary")
        if params.get("description") is not None:
            event_body["description"] = params.get("description")
        if params.get("location") is not None:
            event_body["location"] = params.get("location")

        # Update timing if provided
        if params.get("start"):
            start_dt = self._parse_datetime(params["start"])
            if start_dt:
                event_body["start"] = {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": timezone,
                }
        if params.get("end"):
            end_dt = self._parse_datetime(params["end"])
            if end_dt:
                event_body["end"] = {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": timezone,
                }

        # Update the event
        response = self.oauth_put(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            json=event_body,
        )
        response.raise_for_status()

        logger.info(f"Updated Google Calendar event: {event_id}")
        return ActionResult(
            success=True,
            message="Event updated",
            data={"event_id": event_id, "calendar_id": calendar_id},
        )

    def _delete_event(self, params: dict) -> ActionResult:
        """Delete an event from Google Calendar."""
        calendar_id = params.get("calendar_id", self.default_calendar_id)
        event_id = params.get("event_id")

        if not event_id:
            return ActionResult(
                success=False,
                message="",
                error="Missing required parameter: event_id",
            )

        response = self.oauth_delete(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to delete event: {response.status_code}",
            )

        logger.info(f"Deleted Google Calendar event: {event_id}")
        return ActionResult(
            success=True,
            message="Event deleted",
            data={"event_id": event_id, "calendar_id": calendar_id},
        )

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string in various formats."""
        if not dt_str:
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]

        # Handle ISO format with timezone
        if "+" in dt_str or dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(dt_str)
            except ValueError:
                pass

        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        return None

    def _simple_to_rrule(self, simple: str) -> Optional[str]:
        """Convert simple recurrence string to RRULE format."""
        simple = simple.lower().strip()

        mappings = {
            "daily": "RRULE:FREQ=DAILY",
            "weekly": "RRULE:FREQ=WEEKLY",
            "monthly": "RRULE:FREQ=MONTHLY",
            "yearly": "RRULE:FREQ=YEARLY",
            "every day": "RRULE:FREQ=DAILY",
            "every week": "RRULE:FREQ=WEEKLY",
            "every month": "RRULE:FREQ=MONTHLY",
            "every year": "RRULE:FREQ=YEARLY",
            "weekdays": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            "every weekday": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        }

        return mappings.get(simple)

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action == "create":
            title = params.get("title") or params.get("summary")
            if not title:
                errors.append(ValidationError("title", "Event title is required"))

            all_day = params.get("all_day", False)
            start = params.get("start")
            date = params.get("date")

            if all_day:
                if not date and not start:
                    errors.append(
                        ValidationError("date", "All-day events require date or start")
                    )
            else:
                if not start:
                    errors.append(ValidationError("start", "Start time is required"))

        elif action == "update":
            if not params.get("event_id"):
                errors.append(ValidationError("event_id", "Event ID is required"))

            # Check at least one field to update
            updatable = ["title", "summary", "start", "end", "description", "location"]
            if not any(params.get(f) for f in updatable):
                errors.append(
                    ValidationError("_general", "No fields to update provided")
                )

        elif action == "delete":
            if not params.get("event_id"):
                errors.append(ValidationError("event_id", "Event ID is required"))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        if action == "create":
            title = params.get("title") or params.get("summary", "Untitled")
            start = params.get("start") or params.get("date", "")
            return f'Create event "{title}" at {start}'

        elif action == "update":
            event_id = params.get("event_id", "?")[:20]
            return f"Update event {event_id}"

        elif action == "delete":
            event_id = params.get("event_id", "?")[:20]
            return f"⚠️ Delete event {event_id}"

        return f"Calendar action: {action}"

    def is_available(self) -> bool:
        """Check if plugin is configured."""
        return bool(self.oauth_account_id)

    def test_connection(self) -> tuple[bool, str]:
        """Test connection by listing calendars."""
        if not self.oauth_account_id:
            return False, "OAuth account not configured"

        try:
            response = self.oauth_get(
                "https://www.googleapis.com/calendar/v3/users/me/calendarList"
            )
            response.raise_for_status()

            calendars = response.json().get("items", [])
            return True, f"Connected. Found {len(calendars)} calendars."
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_oauth_client") and self._oauth_client:
            self._oauth_client.close()
