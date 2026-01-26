"""
Outlook Calendar Unified Source Plugin.

Combines Outlook Calendar indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index calendar events for semantic search
- Live side: Query upcoming events, today's schedule, free/busy info
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "what's on my calendar today" -> Live only (real-time schedule)
- "meetings about project X last quarter" -> RAG only (historical)
- "upcoming meetings with John" -> Both, prefer live
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
from plugin_base.document_source import DocumentContent, DocumentInfo
from plugin_base.live_source import LiveDataResult, ParamDefinition
from plugin_base.oauth import MicrosoftOAuthMixin
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class OutlookCalendarUnifiedSource(MicrosoftOAuthMixin, PluginUnifiedSource):
    """
    Unified Outlook Calendar source - RAG for history, Live for schedule.

    Single configuration provides:
    - Document indexing: Events indexed for RAG semantic search
    - Live queries: Today's schedule, upcoming events, free/busy
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "outlook_calendar"
    display_name = "Outlook Calendar"
    description = "Outlook/Microsoft 365 calendar with historical search (RAG) and real-time queries"
    category = "microsoft"
    icon = "📅"
    content_category = ContentCategory.CALENDARS

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:outlook_calendar"]

    supports_rag = True
    supports_live = True
    supports_actions = True  # Can link to calendar action plugin
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results
    default_index_days = 180  # 6 months of past events

    _abstract = False

    @classmethod
    def get_account_info(cls, store) -> dict | None:
        """Extract account info for action handlers."""
        if not store.microsoft_account_id:
            return None

        # Get email from OAuth token
        try:
            from db.oauth_tokens import get_oauth_token_info

            token_info = get_oauth_token_info(store.microsoft_account_id)
            email = token_info.get("account_email", "") if token_info else ""
        except Exception:
            email = ""

        return {
            "provider": "microsoft",
            "email": email,
            "name": store.display_name or store.name,
            "store_id": store.id,
            "oauth_account_id": store.microsoft_account_id,
            # Calendar-specific - no specific calendar_id field for Outlook yet
        }

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME Outlook Calendar access. Actions: "
            "action='today' for today's events, "
            "action='upcoming' for upcoming events, "
            "action='week' for this week's events, "
            "action='search' with query='...' to search events. "
            "Optional: date='YYYY-MM-DD' for a specific day."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        # Note: Outlook Calendar doesn't have specific doc store fields yet
        # but can use the Microsoft account
        return {
            "oauth_account_id": store.microsoft_account_id,
            "calendar_ids": "",
            "days_back": 90,
            "days_forward": 30,
            "index_schedule": store.index_schedule or "",
        }

    # Microsoft Graph API endpoint
    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Microsoft Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "microsoft", "scopes": ["calendars"]},
                help_text="Select a connected Microsoft account with Calendar access",
            ),
            FieldDefinition(
                name="calendar_id",
                label="Calendar ID",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Specific calendar ID (empty = primary calendar)",
            ),
            FieldDefinition(
                name="index_days_past",
                label="Days to Index (Past)",
                field_type=FieldType.INTEGER,
                default=180,
                help_text="How many days of past events to index",
            ),
            FieldDefinition(
                name="index_days_future",
                label="Days to Index (Future)",
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
                name="live_lookahead_days",
                label="Live Lookahead (Days)",
                field_type=FieldType.INTEGER,
                default=14,
                help_text="How far ahead to look for live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: today, upcoming, week, search",
                param_type="string",
                required=False,
                default="today",
                examples=["today", "upcoming", "week", "search"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for event titles/descriptions",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="date",
                description="Specific date (YYYY-MM-DD) to query",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum events to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.calendar_id = config.get("calendar_id", "")
        self.index_days_past = config.get("index_days_past", 180)
        self.index_days_future = config.get("index_days_future", 30)
        self.index_schedule = config.get("index_schedule", "")
        self.live_lookahead_days = config.get("live_lookahead_days", 14)

        self._init_oauth_client()

    def _get_calendar_endpoint(self) -> str:
        """Get the calendar endpoint URL."""
        if self.calendar_id:
            return f"{self.GRAPH_API_BASE}/me/calendars/{self.calendar_id}"
        return f"{self.GRAPH_API_BASE}/me/calendar"

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate calendar events for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot list events - no valid access token")
            return

        # Calculate date range
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=self.index_days_past)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        end_date = (now + timedelta(days=self.index_days_future)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Build endpoint
        calendar_endpoint = self._get_calendar_endpoint()
        endpoint = f"{calendar_endpoint}/calendarView"

        params = {
            "startDateTime": start_date,
            "endDateTime": end_date,
            "$select": "id,subject,start,end,lastModifiedDateTime",
            "$orderby": "start/dateTime desc",
            "$top": 100,
        }

        logger.info(f"Listing Outlook Calendar events from {start_date} to {end_date}")
        total_events = 0

        while endpoint:
            try:
                response = self._oauth_client.get(
                    endpoint,
                    headers=self._get_auth_headers(),
                    params=params if "?" not in endpoint else None,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Outlook Calendar API error: {e}")
                break

            events = data.get("value", [])

            for event in events:
                total_events += 1
                subject = event.get("subject", "(No Title)")
                start_data = event.get("start", {})
                modified = event.get("lastModifiedDateTime", "")

                yield DocumentInfo(
                    uri=f"outlook-calendar://{event['id']}",
                    title=subject[:100] if subject else event["id"],
                    mime_type="text/calendar",
                    modified_at=modified,
                    metadata={
                        "start": start_data.get("dateTime", ""),
                        "calendar_id": self.calendar_id or "primary",
                    },
                )

            # Handle pagination
            endpoint = data.get("@odata.nextLink")
            params = None

        logger.info(f"Found {total_events} calendar events to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read event content for indexing."""
        if not uri.startswith("outlook-calendar://"):
            logger.error(f"Invalid Outlook Calendar URI: {uri}")
            return None

        event_id = uri.replace("outlook-calendar://", "")

        if not self._refresh_token_if_needed():
            logger.error("Cannot read event - no valid access token")
            return None

        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/events/{event_id}",
                headers=self._get_auth_headers(),
                params={
                    "$select": "id,subject,body,start,end,location,attendees,organizer,isAllDay,recurrence"
                },
            )
            response.raise_for_status()
            event_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch event {event_id}: {e}")
            return None

        # Extract fields
        subject = event_data.get("subject", "(No Title)")
        body_data = event_data.get("body", {})
        body_content = body_data.get("content", "")
        body_type = body_data.get("contentType", "text")

        start_data = event_data.get("start", {})
        end_data = event_data.get("end", {})
        location_data = event_data.get("location", {})
        is_all_day = event_data.get("isAllDay", False)

        # Parse dates
        start_str = start_data.get("dateTime", "")
        end_str = end_data.get("dateTime", "")
        timezone_str = start_data.get("timeZone", "UTC")

        # Format location
        location = location_data.get("displayName", "")

        # Format attendees
        attendees = event_data.get("attendees", [])
        attendee_list = []
        for att in attendees:
            email_data = att.get("emailAddress", {})
            name = email_data.get("name", "")
            email = email_data.get("address", "")
            status = att.get("status", {}).get("response", "")
            attendee_list.append(f"{name} <{email}> ({status})")

        # Format organizer
        organizer_data = event_data.get("organizer", {}).get("emailAddress", {})
        organizer = (
            f"{organizer_data.get('name', '')} <{organizer_data.get('address', '')}>"
        )

        # Clean body content
        if body_type.lower() == "html":
            body_text = self._strip_html(body_content)
        else:
            body_text = body_content

        # Format for indexing
        content = f"""Title: {subject}
Date: {start_str} to {end_str} ({timezone_str})
All Day: {"Yes" if is_all_day else "No"}
Location: {location}
Organizer: {organizer}
Attendees: {", ".join(attendee_list) if attendee_list else "None"}

{body_text}
"""

        # Parse date for metadata
        event_date = None
        if start_str:
            try:
                parsed = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                event_date = parsed.strftime("%Y-%m-%d")
            except Exception:
                pass

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "event_id": event_id,
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "event_date": event_date,
                "start": start_str,
                "end": end_str,
                "location": location,
                "is_all_day": is_all_day,
                "attendee_count": len(attendees),
                "subject": subject,
                "source_type": "calendar",
            },
        )

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and decode entities."""
        import html as html_module

        # Remove script and style
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Replace block elements with newlines
        html = re.sub(r"<(br|p|div|tr|li)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Remove remaining tags
        html = re.sub(r"<[^>]+>", "", html)
        # Decode entities
        html = html_module.unescape(html)
        # Clean whitespace
        html = re.sub(r"\n\s*\n", "\n\n", html)
        return html.strip()

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live calendar data."""
        if not self._refresh_token_if_needed():
            return LiveDataResult(
                success=False,
                error="No valid Outlook access token",
            )

        action = params.get("action", "today")
        search_query = params.get("query", "")
        specific_date = params.get("date", "")
        max_results = params.get("max_results", 20)

        try:
            events = self._fetch_events_live(
                action, search_query, specific_date, max_results
            )
            formatted = self._format_events(events, action)

            return LiveDataResult(
                success=True,
                data=events,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Outlook Calendar live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_events_live(
        self, action: str, search_query: str, specific_date: str, max_results: int
    ) -> list[dict]:
        """Fetch events from Microsoft Graph API."""
        now = datetime.now(timezone.utc)
        calendar_endpoint = self._get_calendar_endpoint()

        # Determine time range based on action
        if action == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
        elif action == "week":
            start_time = now
            end_time = now + timedelta(days=7)
        elif action == "upcoming":
            start_time = now
            end_time = now + timedelta(days=self.live_lookahead_days)
        elif specific_date:
            try:
                start_time = datetime.strptime(specific_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                end_time = start_time + timedelta(days=1)
            except ValueError:
                start_time = now
                end_time = now + timedelta(days=self.live_lookahead_days)
        else:
            start_time = now
            end_time = now + timedelta(days=self.live_lookahead_days)

        params = {
            "startDateTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDateTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "$select": "id,subject,start,end,location,isAllDay,organizer,attendees",
            "$orderby": "start/dateTime asc",
            "$top": min(max_results, 50),
        }

        # Add search filter if query provided
        if search_query:
            params["$filter"] = f"contains(subject, '{search_query}')"

        logger.info(f"Outlook Calendar live query: action={action}, params={params}")

        response = self._oauth_client.get(
            f"{calendar_endpoint}/calendarView",
            headers=self._get_auth_headers(),
            params=params,
        )
        response.raise_for_status()

        events_data = response.json().get("value", [])
        logger.info(f"Outlook Calendar returned {len(events_data)} events")

        # Parse events
        events = []
        for event in events_data[:max_results]:
            start_data = event.get("start", {})
            end_data = event.get("end", {})
            location_data = event.get("location", {})
            organizer_data = event.get("organizer", {}).get("emailAddress", {})

            events.append(
                {
                    "id": event.get("id"),
                    "subject": event.get("subject", "(No Title)"),
                    "start": start_data.get("dateTime", ""),
                    "end": end_data.get("dateTime", ""),
                    "timezone": start_data.get("timeZone", "UTC"),
                    "location": location_data.get("displayName", ""),
                    "is_all_day": event.get("isAllDay", False),
                    "organizer": f"{organizer_data.get('name', '')} <{organizer_data.get('address', '')}>",
                    "attendee_count": len(event.get("attendees", [])),
                    "account_email": self.get_account_email(),
                }
            )

        return events

    def _format_events(self, events: list[dict], action: str) -> str:
        """Format events for LLM context."""
        account_email = self.get_account_email()

        if not events:
            action_msgs = {
                "today": "No events scheduled for today.",
                "week": "No events scheduled for this week.",
                "upcoming": "No upcoming events.",
                "search": "No events found matching your search.",
            }
            return f"### Outlook Calendar ({action})\n{action_msgs.get(action, 'No events found.')}"

        action_titles = {
            "today": "Today's Schedule",
            "week": "This Week's Events",
            "upcoming": "Upcoming Events",
            "search": "Calendar Search Results",
        }

        lines = [f"### {action_titles.get(action, 'Events')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(events)} event(s):\n")

        for event in events:
            subject = event.get("subject", "(No Title)")
            start = event.get("start", "")
            end = event.get("end", "")
            location = event.get("location", "")
            is_all_day = event.get("is_all_day", False)
            attendee_count = event.get("attendee_count", 0)

            lines.append(f"**{subject}**")

            if is_all_day:
                # Parse date only for all-day events
                if start:
                    try:
                        start_date = datetime.fromisoformat(
                            start.replace("Z", "+00:00")
                        )
                        lines.append(
                            f"  Date: {start_date.strftime('%A, %B %d, %Y')} (All Day)"
                        )
                    except Exception:
                        lines.append(f"  Date: {start} (All Day)")
            else:
                if start and end:
                    try:
                        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                        lines.append(
                            f"  Time: {start_dt.strftime('%A, %B %d, %Y %I:%M %p')} - {end_dt.strftime('%I:%M %p')}"
                        )
                    except Exception:
                        lines.append(f"  Time: {start} - {end}")

            if location:
                lines.append(f"  Location: {location}")

            if attendee_count > 0:
                lines.append(f"  Attendees: {attendee_count}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine optimal routing."""
        query_lower = query.lower()
        action = params.get("action", "")

        # Real-time schedule queries -> Live only
        realtime_patterns = [
            "today",
            "right now",
            "next meeting",
            "next event",
            "next hour",
            "this afternoon",
            "this morning",
            "tonight",
            "free now",
            "busy now",
            "am i free",
        ]
        if any(p in query_lower for p in realtime_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "today"},
                reason="Real-time schedule query - using live API only",
                max_live_results=20,
            )

        # Week/upcoming queries -> Live only
        if action in ("week", "upcoming") or any(
            p in query_lower for p in ["this week", "upcoming", "next few days"]
        ):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": action or "upcoming"},
                reason="Near-future schedule query - using live API",
                max_live_results=20,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"last month",
            r"last quarter",
            r"q[1-4] 20\d{2}",
        ]
        for pattern in historical_patterns:
            if re.search(pattern, query_lower):
                return QueryAnalysis(
                    routing=QueryRouting.RAG_ONLY,
                    rag_query=query,
                    reason="Historical reference detected - using RAG index",
                    max_rag_results=20,
                )

        # Search queries -> Both, prefer live for recent
        if params.get("query"):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.LIVE_FIRST,
                reason="Search query - checking both sources",
                max_rag_results=10,
                max_live_results=15,
            )

        # Default -> Live for schedule, RAG for context
        return QueryAnalysis(
            routing=QueryRouting.LIVE_THEN_RAG,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.LIVE_FIRST,
            reason="General query - live first, RAG for context",
            max_rag_results=5,
            max_live_results=15,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Outlook Calendar is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Outlook Calendar API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return False, "Failed to get access token - check OAuth configuration"

            # Test API access
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me",
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            profile = response.json()
            email = profile.get("mail") or profile.get("userPrincipalName", "unknown")
            results.append(f"Connected as: {email}")

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
                    live_result = self.fetch({"action": "today"})
                    if live_result.success:
                        event_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {event_count} events today")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
