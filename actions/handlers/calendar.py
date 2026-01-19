"""
Calendar Action Handler.

Handles calendar actions:
- create_event (create new events, including recurring)
- update_event (modify existing events)
- delete_event (remove events)

Supports Google Calendar via OAuth.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from actions.base import ActionContext, ActionHandler, ActionResult, ActionStatus

logger = logging.getLogger(__name__)


class CalendarActionHandler(ActionHandler):
    """
    Handler for calendar actions.

    Supports creating, updating, and deleting events in Google Calendar.
    """

    @property
    def action_type(self) -> str:
        return "calendar"

    @property
    def supported_actions(self) -> list[str]:
        return [
            "create_event",
            "update_event",
            "delete_event",
        ]

    def validate(
        self, action: str, params: dict, context: ActionContext
    ) -> tuple[bool, str]:
        """Validate calendar action parameters."""
        if action not in self.supported_actions:
            return False, f"Unknown calendar action: {action}"

        # Validate account
        account = params.get("account")
        if not account:
            # Try to use default calendar account
            default = context.default_accounts.get("calendar", {})
            if not default:
                return False, "Missing required parameter: account"

        if action == "create_event":
            return self._validate_create_event(params)
        elif action == "update_event":
            return self._validate_update_event(params)
        elif action == "delete_event":
            return self._validate_delete_event(params)

        return True, ""

    def _validate_create_event(self, params: dict) -> tuple[bool, str]:
        """Validate create_event parameters."""
        title = params.get("title") or params.get("summary")
        if not title:
            return False, "Missing required parameter: title (event title/summary)"

        # Need either start/end times OR all_day date
        start = params.get("start")
        end = params.get("end")
        all_day = params.get("all_day", False)
        date = params.get("date")

        if all_day:
            if not date and not start:
                return False, "All-day events require 'date' or 'start' parameter"
        else:
            if not start:
                return False, "Missing required parameter: start (event start time)"
            # End is optional - will default to 1 hour after start

        return True, ""

    def _validate_update_event(self, params: dict) -> tuple[bool, str]:
        """Validate update_event parameters."""
        event_id = params.get("event_id")
        if not event_id:
            return False, "Missing required parameter: event_id"

        # At least one field to update
        updatable = ["title", "summary", "start", "end", "description", "location"]
        if not any(params.get(f) for f in updatable):
            return False, "No fields to update provided"

        return True, ""

    def _validate_delete_event(self, params: dict) -> tuple[bool, str]:
        """Validate delete_event parameters."""
        event_id = params.get("event_id")
        if not event_id:
            return False, "Missing required parameter: event_id"

        return True, ""

    def execute(
        self, action: str, params: dict, context: ActionContext
    ) -> ActionResult:
        """Execute the calendar action."""
        account = params.get("account")

        # Use default calendar account if not specified
        if not account:
            default = context.default_accounts.get("calendar", {})
            if default:
                account = default.get("email")
                params["account"] = account
                # Also get default calendar_id if not specified
                if not params.get("calendar_id"):
                    params["calendar_id"] = default.get("calendar_id", "primary")

        oauth_account = self._find_oauth_account(account, context)

        if not oauth_account:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message=f"OAuth account not found: {account}",
            )

        provider = oauth_account.get("provider", "google")
        account_id = oauth_account.get("id")

        try:
            if provider == "google":
                return self._execute_google(action, params, account_id, context)
            elif provider == "microsoft":
                return self._execute_outlook(action, params, account_id, context)
            else:
                return ActionResult(
                    status=ActionStatus.FAILED,
                    action_type=self.action_type,
                    action=action,
                    message=f"Unsupported calendar provider: {provider}",
                )
        except Exception as e:
            logger.exception(f"Calendar action failed: {action}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message=f"Action failed: {str(e)}",
            )

    def _find_oauth_account(
        self, account_email: str, context: ActionContext
    ) -> Optional[dict]:
        """Find OAuth account by email address."""
        for provider, accounts in context.oauth_accounts.items():
            for acc in accounts:
                if acc.get("email", "").lower() == account_email.lower():
                    return {"provider": provider, **acc}
        return None

    def _execute_google(
        self, action: str, params: dict, account_id: int, context: ActionContext
    ) -> ActionResult:
        """Execute action via Google Calendar API."""
        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message="OAuth token not found",
            )

        access_token = self._get_google_access_token(token_data)
        if not access_token:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action=action,
                message="Failed to get valid access token",
            )

        if action == "create_event":
            return self._google_create_event(access_token, params)
        elif action == "update_event":
            return self._google_update_event(access_token, params)
        elif action == "delete_event":
            return self._google_delete_event(access_token, params)

        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=self.action_type,
            action=action,
            message=f"Unknown action: {action}",
        )

    def _get_google_access_token(self, token_data: dict) -> Optional[str]:
        """Get valid Google access token, refreshing if needed."""
        import time

        import requests as http_requests

        from db.oauth_tokens import update_oauth_token
        from db.settings import get_setting

        token_info = token_data.get("token_data", {})
        access_token = token_info.get("access_token")
        expires_at = token_info.get("expires_at", 0)
        refresh_token = token_info.get("refresh_token")

        # Check if token is expired or about to expire (5 min buffer)
        if time.time() > expires_at - 300:
            if not refresh_token:
                logger.error("No refresh token available")
                return None

            # Refresh the token
            client_id = get_setting("google_client_id")
            client_secret = get_setting("google_client_secret")

            if not client_id or not client_secret:
                logger.error("Google OAuth not configured")
                return None

            try:
                response = http_requests.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    new_token = response.json()
                    new_token["refresh_token"] = refresh_token
                    new_token["expires_at"] = time.time() + new_token.get(
                        "expires_in", 3600
                    )

                    # Update stored token
                    update_oauth_token(token_data["id"], new_token)

                    return new_token["access_token"]
                else:
                    logger.error(f"Token refresh failed: {response.text}")
                    return None

            except Exception as e:
                logger.error(f"Token refresh error: {e}")
                return None

        return access_token

    def _google_create_event(self, access_token: str, params: dict) -> ActionResult:
        """Create an event in Google Calendar."""
        import requests as http_requests

        calendar_id = params.get("calendar_id", "primary")
        title = params.get("title") or params.get("summary")
        description = params.get("description", "")
        location = params.get("location", "")
        all_day = params.get("all_day", False)

        # Build event body
        event_body = {
            "summary": title,
            "description": description,
            "location": location,
        }

        # Handle timing
        if all_day:
            # All-day event uses date (not dateTime)
            date_str = params.get("date") or params.get("start")
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
                # Default to same day (end is exclusive, so use next day)
                try:
                    start_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    end_dt = start_dt + timedelta(days=1)
                    event_body["end"] = {"date": end_dt.strftime("%Y-%m-%d")}
                except ValueError:
                    event_body["end"] = {"date": date_str}
        else:
            # Timed event
            start = params.get("start")
            end = params.get("end")
            timezone = params.get("timezone", "Europe/London")

            # Parse start time
            start_dt = self._parse_datetime(start)
            if not start_dt:
                return ActionResult(
                    status=ActionStatus.FAILED,
                    action_type=self.action_type,
                    action="create_event",
                    message=f"Invalid start time format: {start}",
                )

            event_body["start"] = {
                "dateTime": start_dt.isoformat(),
                "timeZone": timezone,
            }

            # Parse end time (default to 1 hour after start)
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
            # Accept RRULE format or simple strings
            if isinstance(recurrence, str):
                if recurrence.startswith("RRULE:"):
                    event_body["recurrence"] = [recurrence]
                else:
                    # Convert simple format to RRULE
                    rrule = self._simple_to_rrule(recurrence)
                    if rrule:
                        event_body["recurrence"] = [rrule]

        # Handle attendees
        attendees = params.get("attendees", [])
        if attendees:
            event_body["attendees"] = [{"email": a} for a in attendees]

        # Handle reminders
        reminders = params.get("reminders")
        if reminders:
            event_body["reminders"] = {
                "useDefault": False,
                "overrides": reminders,
            }

        # Create the event
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
            logger.error(f"Google Calendar create failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="create_event",
                message=f"Failed to create event: {response.status_code}",
                details={"error": response.text},
            )

        event_data = response.json()
        event_id = event_data.get("id")
        html_link = event_data.get("htmlLink")

        logger.info(f"Created Google Calendar event: {event_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="create_event",
            message=f"Event '{title}' created",
            details={
                "event_id": event_id,
                "calendar_id": calendar_id,
                "title": title,
                "link": html_link,
            },
        )

    def _google_update_event(self, access_token: str, params: dict) -> ActionResult:
        """Update an event in Google Calendar."""
        import requests as http_requests

        calendar_id = params.get("calendar_id", "primary")
        event_id = params.get("event_id")

        # First, get the existing event
        get_response = http_requests.get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if get_response.status_code != 200:
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="update_event",
                message=f"Event not found: {event_id}",
            )

        event_body = get_response.json()

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
                timezone = params.get("timezone", "Europe/London")
                event_body["start"] = {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": timezone,
                }
        if params.get("end"):
            end_dt = self._parse_datetime(params["end"])
            if end_dt:
                timezone = params.get("timezone", "Europe/London")
                event_body["end"] = {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": timezone,
                }

        # Update the event
        response = http_requests.put(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=event_body,
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Google Calendar update failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="update_event",
                message=f"Failed to update event: {response.status_code}",
                details={"error": response.text},
            )

        logger.info(f"Updated Google Calendar event: {event_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="update_event",
            message=f"Event updated",
            details={"event_id": event_id, "calendar_id": calendar_id},
        )

    def _google_delete_event(self, access_token: str, params: dict) -> ActionResult:
        """Delete an event from Google Calendar."""
        import requests as http_requests

        calendar_id = params.get("calendar_id", "primary")
        event_id = params.get("event_id")

        response = http_requests.delete(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code not in (200, 204):
            logger.error(f"Google Calendar delete failed: {response.text}")
            return ActionResult(
                status=ActionStatus.FAILED,
                action_type=self.action_type,
                action="delete_event",
                message=f"Failed to delete event: {response.status_code}",
            )

        logger.info(f"Deleted Google Calendar event: {event_id}")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            action_type=self.action_type,
            action="delete_event",
            message="Event deleted",
            details={"event_id": event_id, "calendar_id": calendar_id},
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

    def _execute_outlook(
        self, action: str, params: dict, account_id: int, context: ActionContext
    ) -> ActionResult:
        """Execute action via Microsoft Graph API."""
        # TODO: Implement Outlook calendar actions
        return ActionResult(
            status=ActionStatus.FAILED,
            action_type=self.action_type,
            action=action,
            message="Outlook calendar actions not yet implemented",
        )

    def get_approval_summary(self, action: str, params: dict) -> str:
        """Generate human-readable summary for approval."""
        if action == "create_event":
            title = params.get("title") or params.get("summary", "Untitled")
            start = params.get("start") or params.get("date", "")
            return f'Create event "{title}" at {start}'

        elif action == "update_event":
            event_id = params.get("event_id", "?")
            return f"Update event {event_id}"

        elif action == "delete_event":
            event_id = params.get("event_id", "?")
            return f"Delete event {event_id}"

        return f"Calendar action: {action}"

    def get_system_prompt_instructions(self) -> str:
        """Get calendar-specific instructions for system prompt."""
        return """- calendar: create_event, update_event, delete_event

  **Create Event:**
  ```xml
  <smart_action type="calendar" action="create_event">
  {"account": "user@gmail.com", "calendar_id": "primary", "title": "Team Meeting", "start": "2026-01-20T14:00:00", "end": "2026-01-20T15:00:00", "description": "Weekly sync", "location": "Conference Room A"}
  </smart_action>
  ```

  **Create Recurring Event:**
  ```xml
  <smart_action type="calendar" action="create_event">
  {"account": "user@gmail.com", "title": "Daily Standup", "start": "2026-01-20T09:00:00", "end": "2026-01-20T09:15:00", "recurrence": "daily"}
  </smart_action>
  ```
  Supported recurrence values: "daily", "weekly", "monthly", "yearly", "weekdays", or RRULE format

  **Create All-Day Event:**
  ```xml
  <smart_action type="calendar" action="create_event">
  {"account": "user@gmail.com", "title": "Holiday", "date": "2026-01-20", "all_day": true}
  </smart_action>
  ```

  **Update Event:**
  ```xml
  <smart_action type="calendar" action="update_event">
  {"account": "user@gmail.com", "event_id": "abc123", "title": "Updated Title", "start": "2026-01-20T15:00:00"}
  </smart_action>
  ```

  **Delete Event:**
  ```xml
  <smart_action type="calendar" action="delete_event">
  {"account": "user@gmail.com", "event_id": "abc123"}
  </smart_action>
  ```

  **Parameters:**
  - account: Email of the Google account (required)
  - calendar_id: Calendar ID (optional, defaults to "primary")
  - title/summary: Event title (required for create)
  - start: Start time in ISO format (required for timed events)
  - end: End time in ISO format (optional, defaults to 1 hour after start)
  - date: Date for all-day events (YYYY-MM-DD)
  - all_day: Set to true for all-day events
  - description: Event description (optional)
  - location: Event location (optional)
  - recurrence: Recurrence rule (optional)
  - attendees: List of email addresses to invite (optional)
  - timezone: Timezone (optional, defaults to "Europe/London")
"""
