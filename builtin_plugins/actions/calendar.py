"""
Unified Calendar action plugin.

Handles calendar actions across multiple providers:
- Google Calendar (via OAuth)
- Outlook Calendar (via OAuth)

The LLM uses generic actions (create, update, delete, rsvp)
and the system routes to the configured provider based on Smart Alias settings.

NO CONFIG FIELDS - all configuration comes from Smart Alias context at runtime:
- default_accounts["calendar"]["id"] = OAuth account ID
- default_accounts["calendar"]["provider"] = "google" or "microsoft"
- default_accounts["calendar"]["calendar_id"] = specific calendar ID
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


class CalendarActionHandler(OAuthMixin, PluginActionHandler):
    """
    Unified calendar management across Google Calendar and Outlook.

    Provider is determined at runtime from Smart Alias context - no plugin config needed.
    """

    action_type = "calendar"
    display_name = "Calendar"
    description = "Create, update, and delete events (Google Calendar or Outlook)"
    icon = "📅"
    category = "productivity"
    supported_sources = ["Google Calendar", "Outlook Calendar"]
    # Document store source_type values that provide calendar accounts
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
                key="calendar",
                label="Calendar Account",
                resource_type=ResourceType.OAUTH_ACCOUNT,
                providers=["google", "microsoft"],
                help_text="Account for calendar actions (Google or Outlook)",
                required=True,
            ),
            ResourceRequirement(
                key="calendar",
                sub_key="calendar_id",
                label="Default Calendar",
                resource_type=ResourceType.CALENDAR_PICKER,
                depends_on="calendar",
                help_text="Calendar for creating events",
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
                        help_text="Recurrence rule (daily, weekly, monthly, yearly, weekdays)",
                    ),
                ],
                examples=[
                    {
                        "title": "Team Meeting",
                        "start": "2026-01-20T14:00:00",
                        "end": "2026-01-20T15:00:00",
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
                ],
                examples=[{"event_id": "abc123", "title": "Updated Title"}],
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
                    ),
                ],
                examples=[{"event_id": "abc123"}],
            ),
            ActionDefinition(
                name="rsvp",
                description="Respond to a calendar invitation",
                risk=ActionRisk.LOW,
                params=[
                    FieldDefinition(
                        name="event_id",
                        label="Event ID",
                        field_type=FieldType.TEXT,
                        required=True,
                    ),
                    FieldDefinition(
                        name="response",
                        label="Response",
                        field_type=FieldType.SELECT,
                        required=True,
                        options=[
                            {"value": "accepted", "label": "Accept"},
                            {"value": "declined", "label": "Decline"},
                            {"value": "tentative", "label": "Maybe"},
                        ],
                    ),
                ],
                examples=[{"event_id": "abc123", "response": "accepted"}],
            ),
        ]

    @classmethod
    def get_llm_instructions(cls) -> str:
        """Static LLM instructions for calendar actions (no account info)."""
        return cls._build_instructions(None)

    def get_llm_instructions_with_context(
        self, available_accounts: Optional[dict[str, list[dict]]] = None
    ) -> str:
        """Dynamic LLM instructions with available accounts listed."""
        return self._build_instructions(available_accounts)

    @staticmethod
    def _build_instructions(
        available_accounts: Optional[dict[str, list[dict]]] = None,
    ) -> str:
        """Build LLM instructions, optionally with account info."""
        # Build account selection text if accounts available
        account_text = ""
        if available_accounts:
            calendar_accounts = available_accounts.get("calendar", [])
            if calendar_accounts:
                account_list = []
                for acc in calendar_accounts:
                    email = acc.get("email", "")
                    name = acc.get("name", "")  # Store name (friendly name)
                    provider = acc.get("provider", "")
                    # Show store name as primary identifier
                    if name:
                        provider_type = (
                            "Google Calendar" if provider == "google" else "Outlook"
                        )
                        if email:
                            account_list.append(
                                f'  - "{name}" ({email}, {provider_type})'
                            )
                        else:
                            account_list.append(f'  - "{name}" ({provider_type})')
                    elif email:
                        provider_type = (
                            "Google Calendar" if provider == "google" else "Outlook"
                        )
                        account_list.append(f'  - "{email}" ({provider_type})')
                if account_list:
                    account_text = f"""
**Available Calendar Accounts:**
{chr(10).join(account_list)}

To use a specific account, add `"account": "Account Name"` or `"account": "email@example.com"`.
If only one account is available, it will be used automatically.
"""

        return f"""## Calendar
{account_text}
Manage calendar events (Google Calendar or Outlook).

### calendar:create
Create a new calendar event.

**Parameters:**
- title (required): Event title
- start (required): Start time in ISO format (e.g., 2026-01-20T14:00:00)
- end (optional): End time (defaults to 1 hour after start)
- description (optional): Event description
- location (optional): Event location
- all_day (optional): Set to true for all-day events
- recurrence (optional): daily, weekly, monthly, yearly, weekdays
- account (optional): Which account to use

**Example:**
```xml
<smart_action type="calendar" action="create">
{{"title": "Team Meeting", "start": "2026-01-20T14:00:00", "end": "2026-01-20T15:00:00", "location": "Conference Room A"}}
</smart_action>
```

### calendar:update
Update an existing event.

**Parameters:**
- event_id (required): The event ID
- title, start, end, description, location (optional): Fields to update

### calendar:delete
Delete an event.

**Parameters:**
- event_id (required): The event ID

### calendar:rsvp
Respond to a calendar invitation.

**Parameters:**
- event_id (required): The event ID
- response (required): "accepted", "declined", or "tentative"
"""

    def __init__(self, config: dict):
        """Initialize the calendar handler - config is ignored, uses context."""
        # These will be set from context at execution time
        self.oauth_account_id: Optional[int] = None
        self.oauth_provider: Optional[str] = None
        self.calendar_id = "primary"
        self.default_timezone = "Europe/London"

    def _find_account(
        self, account_identifier: str, context: ActionContext
    ) -> Optional[dict]:
        """
        Find an account by email address or name.

        Args:
            account_identifier: Email address or account name to find
            context: Action context with available_accounts

        Returns:
            Account dict if found, None otherwise
        """
        available = context.available_accounts.get("calendar", [])
        identifier_lower = account_identifier.lower()

        for account in available:
            email = account.get("email", "").lower()
            name = account.get("name", "").lower()
            if email == identifier_lower or name == identifier_lower:
                return account
            # Also match partial email (before @)
            if email and identifier_lower == email.split("@")[0]:
                return account

        return None

    def _get_available_accounts_str(self, context: ActionContext) -> str:
        """Get comma-separated list of available calendar accounts."""
        available = context.available_accounts.get("calendar", [])
        emails = [a.get("email", a.get("name", "unknown")) for a in available]
        return ", ".join(emails) if emails else "none"

    def _configure_from_params_or_context(
        self, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """
        Configure provider from params (account field) or fall back to default.

        The account can be specified by:
        - Store name (friendly name from document store)
        - Email address
        - Partial email (before @)

        Returns:
            (success, error_message)
        """
        # Check if LLM specified an account
        account_param = params.get("account")

        if account_param:
            # Look up the specified account
            account = self._find_account(account_param, context)
            if not account:
                available = self._get_available_accounts_str(context)
                return (
                    False,
                    f"Account '{account_param}' not found. Available: {available}",
                )

            # Store-based accounts use oauth_account_id, legacy uses id
            self.oauth_account_id = account.get("oauth_account_id", account.get("id"))
            self.oauth_provider = account.get("provider", "google")
            # Use calendar_id from account if available, else default
            self.calendar_id = account.get("calendar_id", "primary")
        else:
            # Check if there's exactly one available account - use it as default
            available_accounts = context.available_accounts.get("calendar", [])
            if len(available_accounts) == 1:
                account = available_accounts[0]
                self.oauth_account_id = account.get(
                    "oauth_account_id", account.get("id")
                )
                self.oauth_provider = account.get("provider", "google")
                self.calendar_id = account.get("calendar_id", "primary")
                logger.info(
                    f"Calendar: Using only available account: {account.get('name', account.get('email'))}"
                )
            elif not available_accounts:
                # Fall back to default account
                default_accounts = getattr(context, "default_accounts", {})
                calendar_config = default_accounts.get("calendar", {})

                if not calendar_config:
                    return False, "No calendar accounts available"

                # Default account uses oauth_account_id or id
                self.oauth_account_id = calendar_config.get(
                    "oauth_account_id", calendar_config.get("id")
                )
                self.oauth_provider = calendar_config.get("provider", "google")
                self.calendar_id = calendar_config.get("calendar_id", "primary")
            else:
                available = self._get_available_accounts_str(context)
                return (
                    False,
                    f"Multiple calendar accounts available. Specify 'account' parameter: {available}",
                )

        if not self.oauth_account_id:
            return False, "Could not determine calendar account ID"

        logger.info(
            f"Calendar: Using provider '{self.oauth_provider}' account {self.oauth_account_id}"
        )

        self._init_oauth_client()
        return True, ""

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the calendar action."""
        # Configure from params (account field) or fall back to context defaults
        success, error = self._configure_from_params_or_context(params, context)
        if not success:
            return ActionResult(
                success=False,
                message="",
                error=error,
            )

        try:
            if self.oauth_provider == "microsoft":
                return self._execute_outlook(action, params)
            else:
                return self._execute_google(action, params)
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

    # -------------------------------------------------------------------------
    # Google Calendar Implementation
    # -------------------------------------------------------------------------

    def _execute_google(self, action: str, params: dict) -> ActionResult:
        """Execute action via Google Calendar API."""
        if action == "create":
            return self._google_create(params)
        elif action == "update":
            return self._google_update(params)
        elif action == "delete":
            return self._google_delete(params)
        elif action == "rsvp":
            return self._google_rsvp(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _google_create(self, params: dict) -> ActionResult:
        """Create event via Google Calendar."""
        calendar_id = params.get("calendar_id", self.calendar_id)
        title = params.get("title") or params.get("summary")

        if not title:
            return ActionResult(
                success=False, message="", error="Event title is required"
            )

        event_body = {
            "summary": title,
            "description": params.get("description", ""),
            "location": params.get("location", ""),
        }

        all_day = params.get("all_day", False)
        timezone = params.get("timezone", self.default_timezone)

        if all_day:
            date_str = params.get("date") or params.get("start")
            if not date_str:
                return ActionResult(
                    success=False, message="", error="All-day events require date"
                )
            if "T" in str(date_str):
                date_str = date_str.split("T")[0]
            event_body["start"] = {"date": date_str}

            end_date = params.get("end")
            if end_date and "T" in str(end_date):
                end_date = end_date.split("T")[0]
            if not end_date:
                start_dt = datetime.strptime(date_str, "%Y-%m-%d")
                end_date = (start_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            event_body["end"] = {"date": end_date}
        else:
            start = params.get("start")
            if not start:
                return ActionResult(
                    success=False, message="", error="Start time is required"
                )

            start_dt = self._parse_datetime(start)
            if not start_dt:
                return ActionResult(
                    success=False, message="", error=f"Invalid start time: {start}"
                )

            event_body["start"] = {
                "dateTime": start_dt.isoformat(),
                "timeZone": timezone,
            }

            end = params.get("end")
            end_dt = self._parse_datetime(end) if end else start_dt + timedelta(hours=1)
            event_body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": timezone}

        # Handle recurrence
        recurrence = params.get("recurrence")
        if recurrence:
            rrule = self._simple_to_rrule(recurrence)
            if rrule:
                event_body["recurrence"] = [rrule]

        response = self.oauth_post(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            json=event_body,
        )
        response.raise_for_status()

        event = response.json()
        return ActionResult(
            success=True,
            message=f"Event '{title}' created",
            data={
                "event_id": event["id"],
                "link": event.get("htmlLink"),
                "provider": "google",
            },
        )

    def _google_update(self, params: dict) -> ActionResult:
        """Update event via Google Calendar."""
        calendar_id = params.get("calendar_id", self.calendar_id)
        event_id = params.get("event_id")

        if not event_id:
            return ActionResult(success=False, message="", error="Event ID is required")

        get_response = self.oauth_get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        )
        if get_response.status_code != 200:
            return ActionResult(
                success=False, message="", error=f"Event not found: {event_id}"
            )

        event = get_response.json()
        timezone = params.get("timezone", self.default_timezone)

        if params.get("title"):
            event["summary"] = params["title"]
        if params.get("description") is not None:
            event["description"] = params["description"]
        if params.get("location") is not None:
            event["location"] = params["location"]
        if params.get("start"):
            start_dt = self._parse_datetime(params["start"])
            if start_dt:
                event["start"] = {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": timezone,
                }
        if params.get("end"):
            end_dt = self._parse_datetime(params["end"])
            if end_dt:
                event["end"] = {"dateTime": end_dt.isoformat(), "timeZone": timezone}

        response = self.oauth_put(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            json=event,
        )
        response.raise_for_status()

        return ActionResult(
            success=True,
            message="Event updated",
            data={"event_id": event_id, "provider": "google"},
        )

    def _google_delete(self, params: dict) -> ActionResult:
        """Delete event via Google Calendar."""
        calendar_id = params.get("calendar_id", self.calendar_id)
        event_id = params.get("event_id")

        if not event_id:
            return ActionResult(success=False, message="", error="Event ID is required")

        response = self.oauth_delete(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to delete: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Event deleted",
            data={"event_id": event_id, "provider": "google"},
        )

    def _google_rsvp(self, params: dict) -> ActionResult:
        """RSVP to event via Google Calendar."""
        calendar_id = params.get("calendar_id", self.calendar_id)
        event_id = params.get("event_id")
        response_status = params.get("response")

        if not event_id:
            return ActionResult(success=False, message="", error="Event ID is required")
        if response_status not in ["accepted", "declined", "tentative"]:
            return ActionResult(success=False, message="", error="Invalid response")

        get_response = self.oauth_get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}"
        )
        if get_response.status_code != 200:
            return ActionResult(
                success=False, message="", error=f"Event not found: {event_id}"
            )

        event = get_response.json()
        user_email = self.get_account_email()

        attendees = event.get("attendees", [])
        found = False
        for attendee in attendees:
            if attendee.get("email", "").lower() == user_email.lower():
                attendee["responseStatus"] = response_status
                found = True
                break

        if not found:
            attendees.append({"email": user_email, "responseStatus": response_status})

        event["attendees"] = attendees

        response = self.oauth_put(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            json=event,
        )
        response.raise_for_status()

        labels = {"accepted": "Accepted", "declined": "Declined", "tentative": "Maybe"}
        return ActionResult(
            success=True,
            message=f"{labels[response_status]} invitation to: {event.get('summary', event_id)}",
            data={
                "event_id": event_id,
                "response": response_status,
                "provider": "google",
            },
        )

    # -------------------------------------------------------------------------
    # Outlook Calendar Implementation
    # -------------------------------------------------------------------------

    def _execute_outlook(self, action: str, params: dict) -> ActionResult:
        """Execute action via Microsoft Graph API."""
        if action == "create":
            return self._outlook_create(params)
        elif action == "update":
            return self._outlook_update(params)
        elif action == "delete":
            return self._outlook_delete(params)
        elif action == "rsvp":
            return self._outlook_rsvp(params)
        else:
            return ActionResult(
                success=False, message="", error=f"Unknown action: {action}"
            )

    def _outlook_create(self, params: dict) -> ActionResult:
        """Create event via Outlook Calendar."""
        title = params.get("title") or params.get("summary")

        if not title:
            return ActionResult(
                success=False, message="", error="Event title is required"
            )

        event_body = {
            "subject": title,
            "body": {"contentType": "text", "content": params.get("description", "")},
        }

        if params.get("location"):
            event_body["location"] = {"displayName": params["location"]}

        all_day = params.get("all_day", False)
        timezone = params.get("timezone", self.default_timezone)

        if all_day:
            date_str = params.get("date") or params.get("start")
            if not date_str:
                return ActionResult(
                    success=False, message="", error="All-day events require date"
                )
            if "T" in str(date_str):
                date_str = date_str.split("T")[0]

            event_body["isAllDay"] = True
            event_body["start"] = {
                "dateTime": f"{date_str}T00:00:00",
                "timeZone": timezone,
            }

            end_date = params.get("end")
            if end_date and "T" in str(end_date):
                end_date = end_date.split("T")[0]
            if not end_date:
                start_dt = datetime.strptime(date_str, "%Y-%m-%d")
                end_date = (start_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            event_body["end"] = {
                "dateTime": f"{end_date}T00:00:00",
                "timeZone": timezone,
            }
        else:
            start = params.get("start")
            if not start:
                return ActionResult(
                    success=False, message="", error="Start time is required"
                )

            start_dt = self._parse_datetime(start)
            if not start_dt:
                return ActionResult(
                    success=False, message="", error=f"Invalid start time: {start}"
                )

            event_body["start"] = {
                "dateTime": start_dt.isoformat(),
                "timeZone": timezone,
            }

            end = params.get("end")
            end_dt = self._parse_datetime(end) if end else start_dt + timedelta(hours=1)
            event_body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": timezone}

        # Handle recurrence
        recurrence = params.get("recurrence")
        if recurrence:
            pattern = self._simple_to_outlook_recurrence(recurrence)
            if pattern:
                event_body["recurrence"] = pattern

        response = self.oauth_post(
            "https://graph.microsoft.com/v1.0/me/events",
            json=event_body,
        )
        response.raise_for_status()

        event = response.json()
        return ActionResult(
            success=True,
            message=f"Event '{title}' created",
            data={
                "event_id": event["id"],
                "link": event.get("webLink"),
                "provider": "microsoft",
            },
        )

    def _outlook_update(self, params: dict) -> ActionResult:
        """Update event via Outlook Calendar."""
        event_id = params.get("event_id")

        if not event_id:
            return ActionResult(success=False, message="", error="Event ID is required")

        event_body = {}
        timezone = params.get("timezone", self.default_timezone)

        if params.get("title"):
            event_body["subject"] = params["title"]
        if params.get("description") is not None:
            event_body["body"] = {
                "contentType": "text",
                "content": params["description"],
            }
        if params.get("location") is not None:
            event_body["location"] = {"displayName": params["location"]}
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

        if not event_body:
            return ActionResult(success=False, message="", error="No fields to update")

        response = self.oauth_patch(
            f"https://graph.microsoft.com/v1.0/me/events/{event_id}",
            json=event_body,
        )
        response.raise_for_status()

        return ActionResult(
            success=True,
            message="Event updated",
            data={"event_id": event_id, "provider": "microsoft"},
        )

    def _outlook_delete(self, params: dict) -> ActionResult:
        """Delete event via Outlook Calendar."""
        event_id = params.get("event_id")

        if not event_id:
            return ActionResult(success=False, message="", error="Event ID is required")

        response = self.oauth_delete(
            f"https://graph.microsoft.com/v1.0/me/events/{event_id}"
        )

        if response.status_code not in (200, 204):
            return ActionResult(
                success=False,
                message="",
                error=f"Failed to delete: {response.status_code}",
            )

        return ActionResult(
            success=True,
            message="Event deleted",
            data={"event_id": event_id, "provider": "microsoft"},
        )

    def _outlook_rsvp(self, params: dict) -> ActionResult:
        """RSVP to event via Outlook Calendar."""
        event_id = params.get("event_id")
        response_status = params.get("response")

        if not event_id:
            return ActionResult(success=False, message="", error="Event ID is required")

        # Map to Outlook actions
        action_map = {
            "accepted": "accept",
            "declined": "decline",
            "tentative": "tentativelyAccept",
        }

        if response_status not in action_map:
            return ActionResult(success=False, message="", error="Invalid response")

        action = action_map[response_status]

        response = self.oauth_post(
            f"https://graph.microsoft.com/v1.0/me/events/{event_id}/{action}",
            json={"sendResponse": True},
        )
        response.raise_for_status()

        labels = {"accepted": "Accepted", "declined": "Declined", "tentative": "Maybe"}
        return ActionResult(
            success=True,
            message=f"{labels[response_status]} invitation",
            data={
                "event_id": event_id,
                "response": response_status,
                "provider": "microsoft",
            },
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string."""
        if not dt_str:
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]

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
        """Convert simple recurrence to Google RRULE."""
        mappings = {
            "daily": "RRULE:FREQ=DAILY",
            "weekly": "RRULE:FREQ=WEEKLY",
            "monthly": "RRULE:FREQ=MONTHLY",
            "yearly": "RRULE:FREQ=YEARLY",
            "weekdays": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        }
        return mappings.get(simple.lower().strip())

    def _simple_to_outlook_recurrence(self, simple: str) -> Optional[dict]:
        """Convert simple recurrence to Outlook pattern."""
        simple = simple.lower().strip()

        patterns = {
            "daily": {
                "pattern": {"type": "daily", "interval": 1},
                "range": {"type": "noEnd"},
            },
            "weekly": {
                "pattern": {"type": "weekly", "interval": 1, "daysOfWeek": ["monday"]},
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
        }
        return patterns.get(simple)

    def validate_action_params(self, action: str, params: dict) -> ValidationResult:
        """Validate action parameters."""
        errors = []

        if action == "create":
            if not (params.get("title") or params.get("summary")):
                errors.append(ValidationError("title", "Event title is required"))
            if not params.get("all_day") and not params.get("start"):
                errors.append(ValidationError("start", "Start time is required"))
        elif action in ("update", "delete", "rsvp"):
            if not params.get("event_id"):
                errors.append(ValidationError("event_id", "Event ID is required"))

        if action == "rsvp":
            if params.get("response") not in ["accepted", "declined", "tentative"]:
                errors.append(ValidationError("response", "Invalid response"))

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary."""
        if action == "create":
            title = params.get("title") or params.get("summary", "Untitled")
            return f'Create event "{title}"'
        elif action == "update":
            return f"Update event {params.get('event_id', '?')[:20]}"
        elif action == "delete":
            return f"Delete event {params.get('event_id', '?')[:20]}"
        elif action == "rsvp":
            return f"RSVP '{params.get('response')}' to event"
        return f"Calendar: {action}"

    def is_available(self) -> bool:
        """Check if plugin is available (always true - config comes from context)."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection - cannot test without context."""
        return (
            True,
            "Calendar handler ready (provider configured per Smart Alias)",
        )

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_oauth_client") and self._oauth_client:
            self._oauth_client.close()
