"""
Gmail Unified Source Plugin.

Combines Gmail document indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index emails by label/date range for semantic search
- Live side: Query recent emails, search by sender/subject/content
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "emails from last hour" -> Live only (too recent for index)
- "project updates from Q3 2023" -> RAG only (historical)
- "latest email from John" -> Both, prefer live, dedupe
- "find all invoices from Acme Corp" -> RAG with live supplement
"""

import base64
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import httpx

from plugin_base.common import ContentCategory, FieldDefinition, FieldType
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


class GmailUnifiedSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Gmail source - RAG for history, Live for recent/specific.

    Single configuration provides:
    - Document indexing: Emails indexed by label/date for RAG semantic search
    - Live queries: Recent emails, search, specific message lookup
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "gmail"
    display_name = "Gmail"
    description = "Gmail with historical search (RAG) and real-time queries"
    category = "google"
    icon = "📧"
    content_category = ContentCategory.EMAILS

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:gmail"]

    # Live data source types this unified source handles (for legacy live sources)
    handles_live_source_types = ["google_gmail_live"]

    supports_rag = True
    supports_live = True
    supports_actions = True  # Can link to email action plugin
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results
    default_index_days = 90

    _abstract = False

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME Gmail access. Actions: "
            "action='recent' for latest emails, "
            "action='unread' for unread emails, "
            "action='today' for today's emails, "
            "action='search' with query='...' using Gmail search syntax "
            "(e.g. query='from:john@example.com', query='subject:invoice after:2024/01/01'). "
            "Optional: max_results=N to limit results."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.google_account_id,
            "index_label_ids": [store.gmail_label_id] if store.gmail_label_id else [],
            "index_days": 90,
            "index_schedule": store.index_schedule or "",
            "live_max_results": 20,
        }

    @classmethod
    def get_account_info(cls, store) -> dict | None:
        """Extract account info for action handlers."""
        if not store.google_account_id:
            return None

        # Get email from OAuth token
        try:
            from db.oauth_tokens import get_oauth_token_info

            token_info = get_oauth_token_info(store.google_account_id)
            email = token_info.get("account_email", "") if token_info else ""
        except Exception:
            email = ""

        return {
            "provider": "google",
            "email": email,
            "name": store.display_name or store.name,
            "store_id": store.id,
            "oauth_account_id": store.google_account_id,
        }

    # Gmail API endpoints
    GMAIL_API_BASE = "https://www.googleapis.com/gmail/v1/users/me"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Google Account",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "google", "scopes": ["gmail"]},
                help_text="Select a connected Google account with Gmail access",
            ),
            FieldDefinition(
                name="index_label_ids",
                label="Labels to Index",
                field_type=FieldType.MULTISELECT,
                required=False,
                help_text="Which labels to index for RAG (empty = INBOX only)",
                picker_options={"provider": "google", "type": "gmail_labels"},
            ),
            FieldDefinition(
                name="index_days",
                label="Days to Index",
                field_type=FieldType.INTEGER,
                default=90,
                help_text="How many days of email history to index (max 365)",
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
                help_text="How often to re-index emails",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=20,
                help_text="Maximum emails to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="query",
                description="Gmail search query (supports Gmail syntax like from:, subject:, after:)",
                param_type="string",
                required=False,
                examples=[
                    "from:john@example.com",
                    "subject:invoice after:2024/01/01",
                    "is:unread",
                ],
            ),
            ParamDefinition(
                name="action",
                description="Query type: recent, unread, today, search",
                param_type="string",
                required=False,
                default="recent",
                examples=["recent", "unread", "today", "search"],
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum emails to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.oauth_provider = "google"

        self.index_label_ids = config.get("index_label_ids", [])
        self.index_days = min(config.get("index_days", 90), 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        self._client = httpx.Client(timeout=30)
        self._init_oauth_client()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        Enumerate emails for indexing.

        Lists emails from configured labels within the index_days window.
        """
        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot list emails - no valid access token")
            return

        # Calculate date range
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.index_days)
        after_date = cutoff_date.strftime("%Y/%m/%d")

        # Build query
        query_parts = [f"after:{after_date}"]

        # Determine labels to query
        label_ids = self.index_label_ids or ["INBOX"]

        for label_id in label_ids:
            logger.info(f"Listing emails from label {label_id} after {after_date}")

            page_token = None
            total_emails = 0

            while True:
                params = {
                    "q": " ".join(query_parts),
                    "labelIds": label_id,
                    "maxResults": 100,
                }
                if page_token:
                    params["pageToken"] = page_token

                try:
                    response = self._oauth_client.get(
                        f"{self.GMAIL_API_BASE}/messages",
                        headers=self._get_auth_headers(),
                        params=params,
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.error(f"Gmail API error listing messages: {e}")
                    break

                messages = data.get("messages", [])

                for msg in messages:
                    total_emails += 1
                    yield DocumentInfo(
                        uri=f"gmail://{msg['id']}",
                        title=msg["id"],  # Will be replaced with subject when reading
                        mime_type="message/rfc822",
                        metadata={"label_id": label_id},
                    )

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

            logger.info(f"Found {total_emails} emails in label {label_id}")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read email content for indexing.

        Fetches the full email and formats it for embedding.
        """
        if not uri.startswith("gmail://"):
            logger.error(f"Invalid Gmail URI: {uri}")
            return None

        message_id = uri.replace("gmail://", "")

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            logger.error("Cannot read email - no valid access token")
            return None

        try:
            response = self._oauth_client.get(
                f"{self.GMAIL_API_BASE}/messages/{message_id}",
                headers=self._get_auth_headers(),
                params={"format": "full"},
            )
            response.raise_for_status()
            msg_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch email {message_id}: {e}")
            return None

        # Extract headers
        headers = {
            h["name"].lower(): h["value"]
            for h in msg_data.get("payload", {}).get("headers", [])
        }
        subject = headers.get("subject", "(No Subject)")
        from_addr = headers.get("from", "")
        to_addr = headers.get("to", "")
        date_str = headers.get("date", "")

        # Extract body
        body_text = self._extract_body(msg_data.get("payload", {}))

        # Format for indexing
        content = f"""Subject: {subject}
From: {from_addr}
To: {to_addr}
Date: {date_str}

{body_text}
"""

        # Parse date for metadata
        email_date = None
        if date_str:
            try:
                from email.utils import parsedate_to_datetime

                parsed = parsedate_to_datetime(date_str)
                email_date = parsed.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Extract sender name
        from_name = from_addr
        if "<" in from_addr:
            from_name = from_addr.split("<")[0].strip().strip('"')
        elif "@" in from_addr:
            from_name = from_addr.split("@")[0]

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "message_id": message_id,
                "thread_id": msg_data.get("threadId", ""),
                "account_id": self.oauth_account_id,
                "account_email": self.get_account_email(),
                "email_date": email_date,
                "from": from_addr,
                "from_name": from_name,
                "to": to_addr,
                "subject": subject,
                "source_type": "email",
            },
        )

    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from email payload."""
        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})
        parts = payload.get("parts", [])

        # Direct body data
        if body.get("data"):
            try:
                return base64.urlsafe_b64decode(body["data"]).decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                pass

        # Multipart - look for text/plain first
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    part_body = part.get("body", {})
                    if part_body.get("data"):
                        try:
                            return base64.urlsafe_b64decode(part_body["data"]).decode(
                                "utf-8", errors="ignore"
                            )
                        except Exception:
                            pass

            # Fall back to text/html
            for part in parts:
                if part.get("mimeType") == "text/html":
                    part_body = part.get("body", {})
                    if part_body.get("data"):
                        try:
                            html = base64.urlsafe_b64decode(part_body["data"]).decode(
                                "utf-8", errors="ignore"
                            )
                            return self._strip_html(html)
                        except Exception:
                            pass

            # Recurse into nested parts
            for part in parts:
                if part.get("parts"):
                    result = self._extract_body(part)
                    if result:
                        return result

        return ""

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and decode entities."""
        import html as html_module

        # Remove script and style elements
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
        """
        Fetch live email data.

        Supports actions: recent, unread, today, search
        """
        start_time = time.time()

        access_token = self._access_token
        if not access_token:
            self._refresh_token_if_needed()
            access_token = self._access_token

        if not access_token:
            return LiveDataResult(
                success=False,
                error="No valid Gmail access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            # Build Gmail search query
            gmail_query = self._build_gmail_query(action, search_query)

            # Fetch emails
            emails = self._fetch_emails_live(gmail_query, max_results)

            # Format for LLM context
            formatted = self._format_emails(emails, action)

            latency_ms = int((time.time() - start_time) * 1000)

            return LiveDataResult(
                success=True,
                data=emails,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Gmail live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _build_gmail_query(self, action: str, search_query: str) -> str:
        """Build Gmail search query based on action and search terms."""
        today = datetime.now(timezone.utc)

        if action == "unread":
            return "is:unread"
        elif action == "today":
            return f"after:{today.strftime('%Y/%m/%d')}"
        elif action == "search" and search_query:
            # Fix temporal syntax
            return self._fix_gmail_temporal_syntax(search_query)
        else:
            # Recent - last 7 days
            week_ago = (today - timedelta(days=7)).strftime("%Y/%m/%d")
            return f"after:{week_ago}"

    def _fix_gmail_temporal_syntax(self, query: str) -> str:
        """Fix invalid Gmail date syntax like 'after:today' -> 'after:2024/01/20'."""
        today = datetime.now().date()

        temporal_to_date = {
            "today": today,
            "yesterday": today - timedelta(days=1),
            "tomorrow": today + timedelta(days=1),
        }

        result = query
        for word, date in temporal_to_date.items():
            date_str = date.strftime("%Y/%m/%d")
            result = re.sub(
                rf"(after:|before:){word}\b",
                rf"\g<1>{date_str}",
                result,
                flags=re.IGNORECASE,
            )

        return result

    def _fetch_emails_live(self, query: str, max_results: int) -> list[dict]:
        """Fetch emails from Gmail API."""
        logger.info(f"Gmail live search: q='{query}', max_results={max_results}")

        # Get message list
        response = self._oauth_client.get(
            f"{self.GMAIL_API_BASE}/messages",
            headers=self._get_auth_headers(),
            params={"q": query, "maxResults": max_results},
        )
        response.raise_for_status()

        messages = response.json().get("messages", [])
        logger.info(f"Gmail search returned {len(messages)} messages")

        # Fetch metadata for each message
        emails = []
        for msg in messages[:max_results]:
            try:
                msg_response = self._oauth_client.get(
                    f"{self.GMAIL_API_BASE}/messages/{msg['id']}",
                    headers=self._get_auth_headers(),
                    params={
                        "format": "metadata",
                        "metadataHeaders": ["From", "Subject", "Date"],
                    },
                )
                if msg_response.status_code == 200:
                    msg_data = msg_response.json()
                    emails.append(self._parse_email_metadata(msg_data))
            except Exception as e:
                logger.warning(f"Failed to fetch email {msg['id']}: {e}")

        return emails

    def _parse_email_metadata(self, msg_data: dict) -> dict:
        """Parse email metadata into a dict."""
        headers = msg_data.get("payload", {}).get("headers", [])
        header_dict = {h["name"].lower(): h["value"] for h in headers}

        return {
            "id": msg_data.get("id"),
            "thread_id": msg_data.get("threadId"),
            "from": header_dict.get("from", ""),
            "subject": header_dict.get("subject", "(No subject)"),
            "date": header_dict.get("date", ""),
            "snippet": msg_data.get("snippet", ""),
            "labels": msg_data.get("labelIds", []),
            "account_email": self.get_account_email(),
        }

    def _format_emails(self, emails: list[dict], action: str) -> str:
        """Format emails for LLM context."""
        account_email = self.get_account_email()

        if not emails:
            action_msgs = {
                "unread": "No unread emails.",
                "today": "No emails received today.",
                "recent": "No recent emails.",
                "search": "No emails found matching your search.",
            }
            return (
                f"### Gmail ({action})\n{action_msgs.get(action, 'No emails found.')}"
            )

        action_titles = {
            "unread": "Unread Emails",
            "today": "Today's Emails",
            "recent": "Recent Emails",
            "search": "Email Search Results",
        }

        lines = [f"### {action_titles.get(action, 'Emails')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(emails)} email(s):\n")

        for email in emails:
            msg_id = email.get("id", "")
            subject = email.get("subject", "(No subject)")
            sender = email.get("from", "Unknown")
            date = email.get("date", "")
            snippet = email.get("snippet", "")
            labels = email.get("labels", [])

            is_unread = "UNREAD" in labels
            unread_marker = " (UNREAD)" if is_unread else ""

            lines.append(f"**{subject}**{unread_marker}")
            lines.append(f"  ID: {msg_id}")
            lines.append(f"  From: {sender}")
            lines.append(f"  Date: {date}")

            if snippet:
                snippet = snippet.replace("\n", " ").strip()
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                lines.append(f"  Preview: {snippet}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """
        Analyze query to determine optimal routing.

        Routing logic:
        - Very recent (last hour, just now) -> Live only
        - Historical (last year, specific dates in past) -> RAG only
        - Latest/new/recent -> Both, prefer live
        - Search with specific criteria -> Both, merge
        - Default -> Both with deduplication
        """
        query_lower = query.lower()
        action = params.get("action", "")

        # Very recent queries -> Live only
        very_recent_patterns = [
            "last hour",
            "past hour",
            "last 30 minutes",
            "just now",
            "right now",
            "just received",
            "just got",
        ]
        if any(p in query_lower for p in very_recent_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params=params,
                reason="Very recent time reference - using live API only",
                max_live_results=self.live_max_results,
            )

        # Unread action -> Live only (real-time status)
        if action == "unread" or "unread" in query_lower:
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "unread"},
                reason="Unread status requires live API",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"last month",
            r"q[1-4] 20\d{2}",
            r"january|february|march|april|may|june",
            r"july|august|september|october|november|december",
        ]
        for pattern in historical_patterns:
            if re.search(pattern, query_lower):
                return QueryAnalysis(
                    routing=QueryRouting.RAG_ONLY,
                    rag_query=query,
                    reason=f"Historical reference detected - using RAG index",
                    max_rag_results=20,
                )

        # Latest/recent/new -> Both, prefer live
        freshness_patterns = ["latest", "newest", "most recent", "new email"]
        if any(p in query_lower for p in freshness_patterns):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params={**params, "action": "recent"},
                merge_strategy=MergeStrategy.LIVE_FIRST,
                freshness_priority=True,
                reason="Freshness keyword - querying both, preferring live",
                max_rag_results=10,
                max_live_results=self.live_max_results,
            )

        # Today action -> Live with RAG supplement
        if action == "today" or "today" in query_lower:
            return QueryAnalysis(
                routing=QueryRouting.LIVE_THEN_RAG,
                rag_query=query,
                live_params={**params, "action": "today"},
                merge_strategy=MergeStrategy.LIVE_FIRST,
                reason="Today's emails - live first, RAG supplement",
                max_rag_results=5,
                max_live_results=self.live_max_results,
            )

        # Search with specific query -> Both, dedupe
        if params.get("query") or action == "search":
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=params.get("query", query),
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.DEDUPE,
                reason="Search query - checking both sources",
                max_rag_results=15,
                max_live_results=self.live_max_results,
            )

        # Default -> Both with deduplication
        return QueryAnalysis(
            routing=QueryRouting.BOTH_MERGE,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.DEDUPE,
            reason="General query - using both sources",
            max_rag_results=10,
            max_live_results=self.live_max_results,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Gmail is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Gmail API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return False, "Failed to get access token - check OAuth configuration"

            # Test API access
            response = self._oauth_client.get(
                f"{self.GMAIL_API_BASE}/profile",
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            profile = response.json()
            email = profile.get("emailAddress", "unknown")
            results.append(f"Connected as: {email}")

            # Test document listing (RAG side)
            if self.supports_rag:
                try:
                    docs = list(self.list_documents())
                    # Only get first 10 to test
                    doc_count = 0
                    for _ in docs:
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found emails to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "recent", "max_results": 5})
                    if live_result.success:
                        email_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {email_count} recent emails")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
