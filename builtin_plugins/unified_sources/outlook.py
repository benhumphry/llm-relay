"""
Outlook Mail Unified Source Plugin.

Combines Outlook email indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index emails by folder/date range for semantic search
- Live side: Query recent emails, search by sender/subject/content
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "emails from last hour" -> Live only (too recent for index)
- "project updates from Q3 2023" -> RAG only (historical)
- "latest email from John" -> Both, prefer live, dedupe
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
from plugin_base.unified_source import (
    MergeStrategy,
    PluginUnifiedSource,
    QueryAnalysis,
    QueryRouting,
)

logger = logging.getLogger(__name__)


class MicrosoftOAuthMixin:
    """
    OAuth mixin for Microsoft Graph API.

    Similar to OAuthMixin but handles Microsoft-specific token refresh.
    """

    oauth_account_id: int
    _oauth_client: Optional[httpx.Client] = None
    _access_token: Optional[str] = None
    _token_expires_at: float = 0

    def _init_oauth_client(self, timeout: int = 30) -> None:
        """Initialize the OAuth HTTP client."""
        self._oauth_client = httpx.Client(timeout=timeout)
        self._refresh_token_if_needed()

    def _get_token_data(self) -> Optional[dict]:
        """Get token data from the database."""
        try:
            from db.oauth_tokens import get_oauth_token_by_id

            token_data = get_oauth_token_by_id(self.oauth_account_id)
            return token_data
        except Exception as e:
            logger.error(f"Failed to get OAuth token: {e}")
            return None

    def _update_token_data(self, token_data: dict) -> bool:
        """Update token data in the database."""
        try:
            from db.oauth_tokens import update_oauth_token_data

            update_oauth_token_data(self.oauth_account_id, token_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update OAuth token: {e}")
            return False

    def _refresh_token_if_needed(self) -> bool:
        """Check token expiry and refresh if needed."""
        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"OAuth token not found: {self.oauth_account_id}")
            return False

        expires_at = token_data.get("expires_at", 0)
        current_time = time.time()

        # Refresh if: no expiry set (unknown state) OR expired/expiring soon
        if not expires_at or current_time > expires_at - 300:
            logger.info(
                f"Microsoft OAuth token expired or expiring soon for account {self.oauth_account_id}, refreshing..."
            )
            refreshed = self._do_token_refresh(token_data)
            if refreshed:
                self._update_token_data(refreshed)
                token_data = refreshed
            else:
                logger.error("Token refresh failed")
                return False

        self._access_token = token_data.get("access_token")
        self._token_expires_at = token_data.get("expires_at", 0)
        return bool(self._access_token)

    def _do_token_refresh(self, token_data: dict) -> Optional[dict]:
        """Perform Microsoft OAuth token refresh."""
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            logger.error("No refresh token available")
            return None

        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not client_id or not client_secret:
            logger.error("Missing client_id or client_secret in token data")
            return None

        token_url = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"

        try:
            response = httpx.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=30,
            )
            response.raise_for_status()
            new_token = response.json()

            result = token_data.copy()
            result["access_token"] = new_token["access_token"]

            if "refresh_token" in new_token:
                result["refresh_token"] = new_token["refresh_token"]

            if "expires_in" in new_token:
                result["expires_at"] = time.time() + new_token["expires_in"]

            logger.info(
                f"Successfully refreshed Microsoft OAuth token for account {self.oauth_account_id}"
            )
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Token refresh HTTP error: {e.response.status_code}")
            try:
                error_body = e.response.json()
                logger.error(f"Error response: {error_body}")
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

    def _get_auth_headers(self) -> dict:
        """Get authorization headers for requests."""
        current_time = time.time()
        if not self._access_token or current_time > self._token_expires_at - 60:
            self._refresh_token_if_needed()

        if not self._access_token:
            logger.error("No access token available after refresh attempt")

        return {"Authorization": f"Bearer {self._access_token}"}

    def get_account_email(self) -> Optional[str]:
        """Get the email address associated with the OAuth account."""
        try:
            from db.oauth_tokens import get_oauth_token_info

            token_info = get_oauth_token_info(self.oauth_account_id)
            if token_info:
                return token_info.get("account_email")
            return None
        except Exception:
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        if self._oauth_client:
            self._oauth_client.close()
            self._oauth_client = None


class OutlookUnifiedSource(MicrosoftOAuthMixin, PluginUnifiedSource):
    """
    Unified Outlook Mail source - RAG for history, Live for recent/specific.

    Single configuration provides:
    - Document indexing: Emails indexed by folder/date for RAG semantic search
    - Live queries: Recent emails, search, specific message lookup
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "outlook"
    display_name = "Outlook"
    description = (
        "Outlook/Microsoft 365 email with historical search (RAG) and real-time queries"
    )
    category = "microsoft"
    icon = "📬"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:outlook"]

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
            "REAL-TIME Outlook email access. Actions: "
            "action='recent' for latest emails, "
            "action='unread' for unread emails, "
            "action='today' for today's emails, "
            "action='search' with query='...' using Outlook search syntax "
            "(e.g. query='from:john@example.com', query='subject:invoice'). "
            "Optional: max_results=N to limit results."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.microsoft_account_id,
            "folder_ids": store.outlook_folder_id or "",
            "index_days": store.outlook_days_back or 90,
            "index_schedule": store.index_schedule or "",
            "live_max_results": 20,
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
                picker_options={"provider": "microsoft", "scopes": ["mail"]},
                help_text="Select a connected Microsoft account with Outlook access",
            ),
            FieldDefinition(
                name="index_folder_id",
                label="Folder to Index",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Folder ID to index (empty = Inbox). Use 'sentitems' for Sent folder.",
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
                description="Search query (supports Outlook search syntax)",
                param_type="string",
                required=False,
                examples=[
                    "from:john@example.com",
                    "subject:invoice",
                    "hasAttachments:true",
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
        self.index_folder_id = config.get("index_folder_id", "")
        self.index_days = min(config.get("index_days", 90), 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        self._init_oauth_client()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate emails for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot list emails - no valid access token")
            return

        # Calculate date range
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.index_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build endpoint
        if self.index_folder_id:
            endpoint = (
                f"{self.GRAPH_API_BASE}/me/mailFolders/{self.index_folder_id}/messages"
            )
        else:
            endpoint = f"{self.GRAPH_API_BASE}/me/messages"

        # Query parameters
        params = {
            "$filter": f"receivedDateTime ge {cutoff_str}",
            "$select": "id,subject,from,receivedDateTime",
            "$orderby": "receivedDateTime desc",
            "$top": 100,
        }

        logger.info(f"Listing Outlook emails after {cutoff_str}")
        total_emails = 0

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
                logger.error(f"Outlook API error listing messages: {e}")
                break

            messages = data.get("value", [])

            for msg in messages:
                total_emails += 1
                received = msg.get("receivedDateTime", "")
                subject = msg.get("subject", "(No Subject)")

                yield DocumentInfo(
                    uri=f"outlook://{msg['id']}",
                    title=subject[:100] if subject else msg["id"],
                    mime_type="message/rfc822",
                    modified_at=received,
                    metadata={"folder_id": self.index_folder_id or "inbox"},
                )

            # Handle pagination
            endpoint = data.get("@odata.nextLink")
            params = None  # nextLink includes params

        logger.info(f"Found {total_emails} Outlook emails to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read email content for indexing."""
        if not uri.startswith("outlook://"):
            logger.error(f"Invalid Outlook URI: {uri}")
            return None

        message_id = uri.replace("outlook://", "")

        if not self._refresh_token_if_needed():
            logger.error("Cannot read email - no valid access token")
            return None

        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/messages/{message_id}",
                headers=self._get_auth_headers(),
                params={
                    "$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,conversationId"
                },
            )
            response.raise_for_status()
            msg_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch email {message_id}: {e}")
            return None

        # Extract fields
        subject = msg_data.get("subject", "(No Subject)")
        from_data = msg_data.get("from", {}).get("emailAddress", {})
        from_addr = (
            f"{from_data.get('name', '')} <{from_data.get('address', '')}>".strip()
        )

        to_list = msg_data.get("toRecipients", [])
        to_addr = ", ".join(
            f"{r.get('emailAddress', {}).get('name', '')} <{r.get('emailAddress', {}).get('address', '')}>"
            for r in to_list
        )

        date_str = msg_data.get("receivedDateTime", "")
        body_data = msg_data.get("body", {})
        body_content = body_data.get("content", "")
        body_type = body_data.get("contentType", "text")

        # Convert HTML to plain text if needed
        if body_type.lower() == "html":
            body_text = self._strip_html(body_content)
        else:
            body_text = body_content

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
                parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                email_date = parsed.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Extract sender name
        from_name = (
            from_data.get("name", "") or from_data.get("address", "").split("@")[0]
        )

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "message_id": message_id,
                "conversation_id": msg_data.get("conversationId", ""),
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
        """Fetch live email data."""
        start_time = time.time()

        if not self._refresh_token_if_needed():
            return LiveDataResult(
                success=False,
                error="No valid Outlook access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            # Fetch emails
            emails = self._fetch_emails_live(action, search_query, max_results)

            # Format for LLM context
            formatted = self._format_emails(emails, action)

            return LiveDataResult(
                success=True,
                data=emails,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Outlook live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_emails_live(
        self, action: str, search_query: str, max_results: int
    ) -> list[dict]:
        """Fetch emails from Microsoft Graph API."""
        today = datetime.now(timezone.utc)

        # Build filter based on action
        filter_parts = []
        if action == "unread":
            filter_parts.append("isRead eq false")
        elif action == "today":
            today_str = today.strftime("%Y-%m-%dT00:00:00Z")
            filter_parts.append(f"receivedDateTime ge {today_str}")
        else:
            # Recent - last 7 days
            week_ago = (today - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
            filter_parts.append(f"receivedDateTime ge {week_ago}")

        params = {
            "$select": "id,subject,from,receivedDateTime,isRead,bodyPreview",
            "$orderby": "receivedDateTime desc",
            "$top": min(max_results, 50),
        }

        if filter_parts:
            params["$filter"] = " and ".join(filter_parts)

        # Use search if query provided
        if search_query:
            params["$search"] = f'"{search_query}"'
            # Can't use $filter with $search in Graph API
            if "$filter" in params:
                del params["$filter"]

        logger.info(f"Outlook live search: params={params}")

        response = self._oauth_client.get(
            f"{self.GRAPH_API_BASE}/me/messages",
            headers=self._get_auth_headers(),
            params=params,
        )
        response.raise_for_status()

        messages = response.json().get("value", [])
        logger.info(f"Outlook search returned {len(messages)} messages")

        # Parse messages
        emails = []
        for msg in messages[:max_results]:
            from_data = msg.get("from", {}).get("emailAddress", {})
            emails.append(
                {
                    "id": msg.get("id"),
                    "from": f"{from_data.get('name', '')} <{from_data.get('address', '')}>",
                    "subject": msg.get("subject", "(No subject)"),
                    "date": msg.get("receivedDateTime", ""),
                    "snippet": msg.get("bodyPreview", ""),
                    "is_read": msg.get("isRead", True),
                    "account_email": self.get_account_email(),
                }
            )

        return emails

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
                f"### Outlook ({action})\n{action_msgs.get(action, 'No emails found.')}"
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
            is_read = email.get("is_read", True)

            unread_marker = " (UNREAD)" if not is_read else ""

            lines.append(f"**{subject}**{unread_marker}")
            lines.append(f"  ID: {msg_id[:20]}...")
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
        """Analyze query to determine optimal routing."""
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
                    reason="Historical reference detected - using RAG index",
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
        """Check if Outlook is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Outlook API connection."""
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
