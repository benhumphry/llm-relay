"""
Google Calendar Unified Source Plugin.

Combines Google Calendar document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index events by calendar/date range for semantic search
- Live side: Query upcoming events, today's schedule, free/busy, search
- Intelligent routing: Analyze queries to choose optimal data source
- Actions: Link to calendar action plugin for create/update/delete

Query routing examples:
- "what's on my calendar today" -> Live only (real-time schedule)
- "when did I meet with John last year" -> RAG only (historical)
- "upcoming meetings" -> Live only (current schedule)
- "find team meeting notes" -> Both, prefer RAG
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.oauth import OAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class GCalendarUnifiedSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Google Calendar source - RAG for history, Live for schedule.

    Single configuration provides:
    - Document indexing: Events indexed by calendar for RAG semantic search
    - Live queries: Today's events, upcoming, free/busy, search
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "gcalendar"
    display_name = "Google Calendar"
    description = "Google Calendar with historical search (RAG) and real-time queries"
    category = "google"
    icon = "📅"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:gcalendar"]

    # Live data source types this unified source handles (for legacy live sources)
    handles_live_source_types = ["google_calendar_live"]

    supports_rag = True
    supports_live = True
    supports_actions = True  # Links to calendar action plugin
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results
    default_days_back = 90
    default_days_forward = 30

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.google_account_id,
            "calendar_ids": store.gcalendar_calendar_id or "",
            "days_back": 90,
            "days_forward": 30,
            "index_schedule": store.index_schedule or "",
        }

    # Calendar API endpoint
    CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "google", "scopes": ["calendar"]},
                help_text="Select a connected Google account with Calendar access",
            ),
            FieldDefinition(
                name="calendar_ids",
                label="Calendars to Index",
                field_type=FieldType.MULTISELECT,
                required=False,
                help_text="Which calendars to index (empty = primary only)",
                picker_options={"provider": "google", "type": "calendars"},
            ),
            FieldDefinition(
                name="days_back",
                label="Days Back to Index",
                field_type=FieldType.INTEGER,
                default=90,
                help_text="How many days of past events to index",
            ),
            FieldDefinition(
                name="days_forward",
                label="Days Forward to Index",
                field_type=FieldType.INTEGER,
                default=30,
                help_text="How many days of future events to index",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 * * * *", "label": "Hourly"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                ],
                help_text="How often to re-index events",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=25,
                help_text="Maximum events to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: today, upcoming, week, search, freebusy",
                param_type="string",
                required=False,
                default="upcoming",
                examples=["today", "upcoming", "week", "search", "freebusy"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for event content (title, description, attendees)",
                param_type="string",
                required=False,
                examples=["team meeting", "with John", "project review"],
            ),
            ParamDefinition(
                name="date",
                description="Specific date to query (YYYY-MM-DD or natural language)",
                param_type="string",
                required=False,
                examples=["2026-01-25", "tomorrow", "next Monday"],
            ),
            ParamDefinition(
                name="calendar_id",
                description="Specific calendar ID to query",
                param_type="string",
                required=False,
                examples=["primary", "work@example.com"],
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum events to return",
                param_type="integer",
                required=False,
                default=25,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"

        # Handle calendar_ids - can be list, comma-separated string, or single string
        calendar_ids = config.get("calendar_ids", [])
        if isinstance(calendar_ids, str):
            # Could be comma-separated or single ID
            if "," in calendar_ids:
                self.calendar_ids = [
                    c.strip() for c in calendar_ids.split(",") if c.strip()
                ]
            elif calendar_ids.strip():
                self.calendar_ids = [calendar_ids.strip()]
            else:
                self.calendar_ids = []
        elif isinstance(calendar_ids, list):
            self.calendar_ids = calendar_ids
        else:
            self.calendar_ids = []

        if not self.calendar_ids:
            self.calendar_ids = ["primary"]
        self.days_back = config.get("days_back", self.default_days_back)
        self.days_forward = config.get("days_forward", self.default_days_forward)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 25)

        self._client = httpx.Client(timeout=30)
        self._init_oauth_client()
        self._calendar_cache: dict[str, str] = {}  # id -> name mapping

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate events for indexing.

        Lists events from configured calendars within the date range.
        """
        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot list events - no valid access token")
            return

        # Calculate time range
        now = datetime.now(timezone.utc)
        time_min = (now - timedelta(days=self.days_back)).isoformat()
        time_max = (now + timedelta(days=self.days_forward)).isoformat()

        for calendar_id in self.calendar_ids:
            logger.info(
                f"Listing events from calendar {calendar_id} "
                f"({self.days_back} days back, {self.days_forward} days forward)"
            )

            params = {
                "timeMin": time_min,
                "timeMax": time_max,
                "maxResults": 250,
                "singleEvents": "true",
                "orderBy": "startTime",
            }

            page_token = None
            total_events = 0

            while True:
                if page_token:
                    params["pageToken"] = page_token

                try:
                    response = self._oauth_client.get(
                        f"{self.CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                        headers=self._get_auth_headers(),
                        params=params,
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.error(f"Calendar API error: {e}")
                    break

                events = data.get("items", [])

                for event in events:
                    event_id = event.get("id")
                    summary = event.get("summary", "(No title)")
                    modified_time = event.get("updated")

                    total_events += 1
                    yield DocumentInfo(
                        uri=f"gcal://{calendar_id}/{event_id}",
                        title=summary[:100],
                        mime_type="text/calendar",
                        modified_at=modified_time,
                        metadata={"calendar_id": calendar_id},
                    )

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

            logger.info(f"Found {total_events} events in calendar {calendar_id}")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read event content for indexing.

        Fetches the event and formats it for embedding.
        """
        if not uri.startswith("gcal://"):
            logger.error(f"Invalid Calendar URI: {uri}")
            return None

        parts = uri.replace("gcal://", "").split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid Calendar URI format: {uri}")
            return None

        calendar_id, event_id = parts

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot read event - no valid access token")
            return None

        try:
            response = self._oauth_client.get(
                f"{self.CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            event = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch event {event_id}: {e}")
            return None

        # Extract event details
        summary = event.get("summary", "(No title)")
        description = event.get("description", "")
        location = event.get("location", "")

        # Parse start/end times
        start = event.get("start", {})
        end = event.get("end", {})
        start_time = start.get("dateTime") or start.get("date", "")
        end_time = end.get("dateTime") or end.get("date", "")

        # Determine if all-day event
        is_all_day = "date" in start and "dateTime" not in start

        # Get attendees
        attendees = event.get("attendees", [])
        attendee_names = []
        for a in attendees:
            name = a.get("displayName") or a.get("email", "")
            if name:
                attendee_names.append(name)
        attendee_list = ", ".join(attendee_names)

        # Get organizer
        organizer = event.get("organizer", {})
        organizer_name = organizer.get("displayName") or organizer.get("email", "")

        # Format as readable text
        content_parts = [f"Event: {summary}"]

        if is_all_day:
            content_parts.append(f"Date: {start_time} (All day)")
        else:
            content_parts.append(f"Start: {start_time}")
            content_parts.append(f"End: {end_time}")

        if location:
            content_parts.append(f"Location: {location}")

        if organizer_name:
            content_parts.append(f"Organizer: {organizer_name}")

        if attendee_list:
            content_parts.append(f"Attendees: {attendee_list}")

        if description:
            content_parts.append(f"\nDescription:\n{description}")

        content = "\n".join(content_parts)

        # Extract date for metadata
        event_date = None
        if start_time:
            event_date = start_time[:10]

        # Get calendar name
        calendar_name = self._get_calendar_name(calendar_id)

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "event_id": event_id,
                "calendar_id": calendar_id,
                "calendar_name": calendar_name,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "event_date": event_date,
                "location": location,
                "organizer": organizer_name,
                "attendees": attendee_list,  # Join list for ChromaDB compatibility
                "is_all_day": is_all_day,
                "source_type": "calendar_event",
            },
        )

    def _get_calendar_name(self, calendar_id: str) -> str:
        """Get calendar name by ID."""
        if calendar_id in self._calendar_cache:
            return self._calendar_cache[calendar_id]

        if calendar_id == "primary":
            return "Primary Calendar"

        try:
            response = self._oauth_client.get(
                f"{self.CALENDAR_API_BASE}/calendars/{calendar_id}",
                headers=self._get_auth_headers(),
            )
            if response.status_code == 200:
                name = response.json().get("summary", calendar_id)
                self._calendar_cache[calendar_id] = name
                return name
        except Exception:
            pass

        return calendar_id

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """
        Fetch live calendar data.

        Supports actions: today, upcoming, week, search, freebusy
        """
        start_time = time.time()

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            return LiveDataResult(
                success=False,
                error="No valid Google Calendar access token",
            )

        action = params.get("action", "upcoming")
        search_query = params.get("query", "")
        specific_date = params.get("date", "")
        calendar_id = params.get("calendar_id", "primary")
        max_results = params.get("max_results", self.live_max_results)

        try:
            # Handle specific date parsing
            target_date = None
            if specific_date:
                target_date = self._parse_date(specific_date)

            # Fetch events based on action
            events = self._fetch_events_live(
                action, search_query, target_date, calendar_id, max_results
            )

            # Format for LLM context
            formatted = self._format_events(events, action, specific_date)

            latency_ms = int((time.time() - start_time) * 1000)

            return LiveDataResult(
                success=True,
                data=events,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Calendar live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        date_lower = date_str.lower()
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Handle natural language
        if date_lower == "today":
            return today
        elif date_lower == "tomorrow":
            return today + timedelta(days=1)
        elif date_lower == "yesterday":
            return today - timedelta(days=1)
        elif "next " in date_lower:
            # "next Monday", "next week", etc.
            day_names = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            for i, day in enumerate(day_names):
                if day in date_lower:
                    current_day = today.weekday()
                    days_ahead = i - current_day
                    if days_ahead <= 0:
                        days_ahead += 7
                    return today + timedelta(days=days_ahead)

        # Try ISO format
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            pass

        # Try common formats
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                pass

        return None

    def _fetch_events_live(
        self,
        action: str,
        search_query: str,
        target_date: Optional[datetime],
        calendar_id: str,
        max_results: int,
    ) -> list[dict]:
        """Fetch events from Google Calendar API."""
        now = datetime.now(timezone.utc)

        # Calculate time range based on action
        if action == "today" or (target_date and target_date.date() == now.date()):
            time_min = now.replace(hour=0, minute=0, second=0, microsecond=0)
            time_max = time_min + timedelta(days=1)
        elif action == "week":
            time_min = now
            time_max = now + timedelta(days=7)
        elif target_date:
            time_min = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            time_max = time_min + timedelta(days=1)
        else:  # upcoming
            time_min = now
            time_max = now + timedelta(days=14)  # Next 2 weeks

        params = {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        if search_query:
            params["q"] = search_query

        try:
            response = self._oauth_client.get(
                f"{self.CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                headers=self._get_auth_headers(),
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            events = data.get("items", [])

            # Add calendar name to events
            calendar_name = self._get_calendar_name(calendar_id)
            for event in events:
                event["calendar_name"] = calendar_name

            return events

        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []

    def _format_events(
        self, events: list[dict], action: str, specific_date: str
    ) -> str:
        """Format events for LLM context."""
        account_email = self.get_account_email()

        if not events:
            action_msgs = {
                "today": "No events scheduled for today.",
                "upcoming": "No upcoming events.",
                "week": "No events this week.",
                "search": "No events found matching your search.",
            }
            title = action_msgs.get(action, "No events found.")
            if specific_date:
                title = f"No events on {specific_date}."
            return f"### Google Calendar\n{title}"

        action_titles = {
            "today": "Today's Schedule",
            "upcoming": "Upcoming Events",
            "week": "This Week's Events",
            "search": "Calendar Search Results",
        }

        title = action_titles.get(action, "Calendar Events")
        if specific_date:
            title = f"Events on {specific_date}"

        lines = [f"### {title}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(events)} event(s):\n")

        for event in events:
            event_id = event.get("id", "")
            summary = event.get("summary", "(No title)")
            location = event.get("location", "")
            description = event.get("description", "")

            # Parse times
            start = event.get("start", {})
            end = event.get("end", {})
            start_time = start.get("dateTime") or start.get("date", "")
            end_time = end.get("dateTime") or end.get("date", "")
            is_all_day = "date" in start and "dateTime" not in start

            # Get attendees
            attendees = event.get("attendees", [])
            attendee_names = [
                a.get("displayName") or a.get("email", "") for a in attendees[:5]
            ]
            if len(attendees) > 5:
                attendee_names.append(f"... and {len(attendees) - 5} more")

            lines.append(f"**{summary}**")
            lines.append(f"   ID: {event_id}")

            if is_all_day:
                lines.append(f"   Date: {start_time} (All day)")
            else:
                # Format times nicely
                try:
                    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                    start_fmt = start_dt.strftime("%Y-%m-%d %H:%M")
                    end_fmt = end_dt.strftime("%H:%M")
                    if start_dt.date() != end_dt.date():
                        end_fmt = end_dt.strftime("%Y-%m-%d %H:%M")
                    lines.append(f"   Time: {start_fmt} - {end_fmt}")
                except Exception:
                    lines.append(f"   Start: {start_time}")
                    lines.append(f"   End: {end_time}")

            if location:
                lines.append(f"   Location: {location}")

            if attendee_names:
                lines.append(f"   Attendees: {', '.join(attendee_names)}")

            if description:
                desc_preview = description.replace("\n", " ").strip()
                if len(desc_preview) > 100:
                    desc_preview = desc_preview[:100] + "..."
                lines.append(f"   Description: {desc_preview}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Query Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze query to determine optimal routing.

        Routing logic:
        - Current schedule queries (today, upcoming, this week) -> Live only
        - Historical queries (last month, when did I meet...) -> RAG only
        - Search with specific criteria -> Both, merge
        - Default -> Live (calendar is usually about current/future)
        """
        query_lower = query.lower()
        action = params.get("action", "")

        # Current schedule queries -> Live only
        current_patterns = [
            "today",
            "tomorrow",
            "this week",
            "next week",
            "upcoming",
            "what's on",
            "what do i have",
            "my schedule",
            "my calendar",
            "free time",
            "busy",
            "when am i",
            "do i have anything",
        ]
        if action in ["today", "upcoming", "week", "freebusy"] or any(
            p in query_lower for p in current_patterns
        ):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": action or "upcoming"},
                reason="Current schedule requires live API",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            "last month",
            "last year",
            r"20\d{2}",
            "when did i",
            "when was",
            "past meetings",
            "previous",
            "history",
        ]
        if any(re.search(p, query_lower) for p in historical_patterns):
            return QueryAnalysis(
                routing=QueryRouting.RAG_ONLY,
                rag_query=query,
                reason="Historical calendar query - using RAG index",
                max_rag_results=20,
            )

        # Search queries -> Both, prefer live for recency
        if params.get("query") or action == "search":
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=params.get("query", query),
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.LIVE_FIRST,
                reason="Search query - checking both sources",
                max_rag_results=10,
                max_live_results=self.live_max_results,
            )

        # Default -> Live (calendar is usually forward-looking)
        return QueryAnalysis(
            routing=QueryRouting.LIVE_ONLY,
            live_params={**params, "action": action or "upcoming"},
            reason="Calendar queries default to live schedule",
            max_live_results=self.live_max_results,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Google Calendar is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Google Calendar API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return (
                    False,
                    "Failed to get access token - check OAuth configuration",
                )

            # Test API access - get calendar list
            response = self._oauth_client.get(
                f"{self.CALENDAR_API_BASE}/users/me/calendarList",
                headers=self._get_auth_headers(),
                params={"maxResults": 10},
            )
            response.raise_for_status()
            calendars = response.json().get("items", [])

            results.append(f"Connected. Found {len(calendars)} calendar(s)")

            # List calendar names
            if calendars:
                names = [c.get("summary", "Untitled") for c in calendars[:5]]
                results.append(f"Calendars: {', '.join(names)}")

            # Test document listing (RAG side)
            if self.supports_rag:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found events to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "upcoming", "max_results": 5})
                    if live_result.success:
                        event_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {event_count} upcoming events")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
