"""
Microsoft Teams Unified Source Plugin.

Combines Teams messages indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index channel messages and chat history for semantic search
- Live side: Query recent messages, unread items, search conversations
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "messages from today" -> Live only (real-time)
- "discussion about project X last quarter" -> RAG only (historical)
- "what did John say about the budget" -> Both, merge results

Note: Teams requires delegated permissions and works with organizational accounts.
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

# Import the mixin from outlook module
try:
    from .outlook import MicrosoftOAuthMixin
except ImportError:
    from builtin_plugins.unified_sources.outlook import MicrosoftOAuthMixin

logger = logging.getLogger(__name__)


class TeamsUnifiedSource(MicrosoftOAuthMixin, PluginUnifiedSource):
    """
    Unified Microsoft Teams source - RAG for history, Live for recent messages.

    Single configuration provides:
    - Document indexing: Channel/chat messages indexed for RAG semantic search
    - Live queries: Recent messages, unread items, search
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "teams"
    display_name = "Microsoft Teams"
    description = (
        "Microsoft Teams messages with historical search (RAG) and real-time queries"
    )
    category = "microsoft"
    icon = "💬"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:teams"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # No actions for Teams (yet)
    supports_incremental = True

    default_cache_ttl = 180  # 3 minutes for live results (messages change frequently)
    default_index_days = 90

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "oauth_account_id": store.microsoft_account_id,
            "team_ids": store.teams_team_id or "",
            "channel_ids": store.teams_channel_id or "",
            "days_back": store.teams_days_back or 90,
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
                picker_options={"provider": "microsoft", "scopes": ["teams"]},
                help_text="Select a connected Microsoft work/school account with Teams access",
            ),
            FieldDefinition(
                name="team_id",
                label="Team ID",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Specific team ID to index (empty = all joined teams)",
            ),
            FieldDefinition(
                name="channel_id",
                label="Channel ID",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Specific channel ID to index (empty = all channels in team)",
            ),
            FieldDefinition(
                name="include_chats",
                label="Include 1:1 Chats",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Also index direct messages (1:1 and group chats)",
            ),
            FieldDefinition(
                name="index_days",
                label="Days to Index",
                field_type=FieldType.INTEGER,
                default=90,
                help_text="How many days of message history to index",
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
                help_text="How often to re-index messages",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=30,
                help_text="Maximum messages to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: recent, unread, search",
                param_type="string",
                required=False,
                default="recent",
                examples=["recent", "unread", "search"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for message content",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="team",
                description="Team name or ID to filter by",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="channel",
                description="Channel name or ID to filter by",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum messages to return",
                param_type="integer",
                required=False,
                default=30,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.oauth_account_id = config.get("oauth_account_id")
        self.team_id = config.get("team_id", "")
        self.channel_id = config.get("channel_id", "")
        self.include_chats = config.get("include_chats", False)
        self.index_days = min(config.get("index_days", 90), 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 30)

        self._init_oauth_client()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate messages for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot list messages - no valid access token")
            return

        logger.info("Listing Teams messages")

        # Index channel messages
        if self.channel_id and self.team_id:
            # Specific channel
            yield from self._list_channel_messages(self.team_id, self.channel_id)
        elif self.team_id:
            # All channels in specific team
            channels = self._get_team_channels(self.team_id)
            for channel in channels:
                yield from self._list_channel_messages(self.team_id, channel["id"])
        else:
            # All joined teams
            teams = self._get_joined_teams()
            for team in teams:
                channels = self._get_team_channels(team["id"])
                for channel in channels:
                    yield from self._list_channel_messages(team["id"], channel["id"])

        # Optionally index direct chats
        if self.include_chats:
            yield from self._list_chat_messages()

    def _get_joined_teams(self) -> list[dict]:
        """Get teams the user has joined."""
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/joinedTeams",
                headers=self._get_auth_headers(),
                params={"$select": "id,displayName"},
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            logger.error(f"Failed to list joined teams: {e}")
            return []

    def _get_team_channels(self, team_id: str) -> list[dict]:
        """Get channels in a team."""
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/teams/{team_id}/channels",
                headers=self._get_auth_headers(),
                params={"$select": "id,displayName"},
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            logger.error(f"Failed to list channels for team {team_id}: {e}")
            return []

    def _list_channel_messages(
        self, team_id: str, channel_id: str
    ) -> Iterator[DocumentInfo]:
        """List messages in a channel."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.index_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        endpoint = (
            f"{self.GRAPH_API_BASE}/teams/{team_id}/channels/{channel_id}/messages"
        )
        params = {
            "$select": "id,subject,body,from,createdDateTime",
            "$top": 50,
        }

        # Get channel info for metadata
        channel_name = ""
        team_name = ""
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/teams/{team_id}/channels/{channel_id}",
                headers=self._get_auth_headers(),
                params={"$select": "displayName"},
            )
            if response.status_code == 200:
                channel_name = response.json().get("displayName", "")

            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/teams/{team_id}",
                headers=self._get_auth_headers(),
                params={"$select": "displayName"},
            )
            if response.status_code == 200:
                team_name = response.json().get("displayName", "")
        except Exception:
            pass

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
                logger.error(f"Failed to list messages in channel {channel_id}: {e}")
                break

            messages = data.get("value", [])

            for msg in messages:
                created = msg.get("createdDateTime", "")

                # Skip messages older than cutoff
                if created and created < cutoff_str:
                    continue

                subject = msg.get("subject", "")
                from_data = msg.get("from", {})
                user_data = from_data.get("user", {}) or from_data.get(
                    "application", {}
                )
                sender = user_data.get("displayName", "Unknown")

                title = subject if subject else f"Message from {sender}"

                yield DocumentInfo(
                    uri=f"teams-channel://{team_id}/{channel_id}/{msg['id']}",
                    title=title[:100],
                    mime_type="text/plain",
                    modified_at=created,
                    metadata={
                        "team_id": team_id,
                        "team_name": team_name,
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "message_type": "channel",
                    },
                )

            # Handle pagination
            endpoint = data.get("@odata.nextLink")
            params = None

    def _list_chat_messages(self) -> Iterator[DocumentInfo]:
        """List messages from direct chats."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.index_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Get user's chats
        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/chats",
                headers=self._get_auth_headers(),
                params={"$select": "id,topic,chatType"},
            )
            response.raise_for_status()
            chats = response.json().get("value", [])
        except Exception as e:
            logger.error(f"Failed to list chats: {e}")
            return

        for chat in chats:
            chat_id = chat.get("id")
            chat_topic = chat.get("topic", "Direct Chat")
            chat_type = chat.get("chatType", "")

            endpoint = f"{self.GRAPH_API_BASE}/me/chats/{chat_id}/messages"
            params = {
                "$select": "id,body,from,createdDateTime",
                "$top": 50,
            }

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
                    logger.error(f"Failed to list messages in chat {chat_id}: {e}")
                    break

                messages = data.get("value", [])

                for msg in messages:
                    created = msg.get("createdDateTime", "")

                    if created and created < cutoff_str:
                        continue

                    from_data = msg.get("from", {})
                    user_data = from_data.get("user", {}) or from_data.get(
                        "application", {}
                    )
                    sender = user_data.get("displayName", "Unknown")

                    yield DocumentInfo(
                        uri=f"teams-chat://{chat_id}/{msg['id']}",
                        title=f"Chat: {sender}",
                        mime_type="text/plain",
                        modified_at=created,
                        metadata={
                            "chat_id": chat_id,
                            "chat_topic": chat_topic,
                            "chat_type": chat_type,
                            "message_type": "chat",
                        },
                    )

                endpoint = data.get("@odata.nextLink")
                params = None

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read message content for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot read message - no valid access token")
            return None

        if uri.startswith("teams-channel://"):
            return self._read_channel_message(uri)
        elif uri.startswith("teams-chat://"):
            return self._read_chat_message(uri)
        else:
            logger.error(f"Invalid Teams URI: {uri}")
            return None

    def _read_channel_message(self, uri: str) -> Optional[DocumentContent]:
        """Read a channel message."""
        parts = uri.replace("teams-channel://", "").split("/")
        if len(parts) < 3:
            return None

        team_id, channel_id, message_id = parts[0], parts[1], parts[2]

        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/teams/{team_id}/channels/{channel_id}/messages/{message_id}",
                headers=self._get_auth_headers(),
                params={"$select": "id,subject,body,from,createdDateTime,importance"},
            )
            response.raise_for_status()
            msg_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch channel message {message_id}: {e}")
            return None

        return self._format_message(
            msg_data,
            "channel",
            {
                "team_id": team_id,
                "channel_id": channel_id,
            },
        )

    def _read_chat_message(self, uri: str) -> Optional[DocumentContent]:
        """Read a chat message."""
        parts = uri.replace("teams-chat://", "").split("/")
        if len(parts) < 2:
            return None

        chat_id, message_id = parts[0], parts[1]

        try:
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/chats/{chat_id}/messages/{message_id}",
                headers=self._get_auth_headers(),
                params={"$select": "id,body,from,createdDateTime"},
            )
            response.raise_for_status()
            msg_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch chat message {message_id}: {e}")
            return None

        return self._format_message(msg_data, "chat", {"chat_id": chat_id})

    def _format_message(
        self, msg_data: dict, msg_type: str, extra_metadata: dict
    ) -> DocumentContent:
        """Format message data for indexing."""
        body_data = msg_data.get("body", {})
        body_content = body_data.get("content", "")
        content_type = body_data.get("contentType", "text")

        # Convert HTML to text if needed
        if content_type.lower() == "html":
            body_text = self._strip_html(body_content)
        else:
            body_text = body_content

        from_data = msg_data.get("from", {})
        user_data = from_data.get("user", {}) or from_data.get("application", {})
        sender = user_data.get("displayName", "Unknown")
        sender_email = user_data.get("email", "")

        created = msg_data.get("createdDateTime", "")
        subject = msg_data.get("subject", "")
        importance = msg_data.get("importance", "normal")

        # Format for indexing
        content = f"""From: {sender}
Date: {created}
Type: {msg_type}
{f"Subject: {subject}" if subject else ""}
{f"Importance: {importance}" if importance != "normal" else ""}

{body_text}
"""

        # Parse date for metadata
        msg_date = None
        if created:
            try:
                parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
                msg_date = parsed.strftime("%Y-%m-%d")
            except Exception:
                pass

        metadata = {
            "message_id": msg_data.get("id", ""),
            "sender": sender,
            "sender_email": sender_email,
            "message_date": msg_date,
            "message_type": msg_type,
            "account_id": self.oauth_account_id,
            "account_email": self.get_account_email(),
            "source_type": "message",
            **extra_metadata,
        }

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata=metadata,
        )

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and decode entities."""
        import html as html_module

        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(r"<(br|p|div|tr|li)[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<[^>]+>", "", html)
        html = html_module.unescape(html)
        html = re.sub(r"\n\s*\n", "\n\n", html)
        return html.strip()

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live message data."""
        if not self._refresh_token_if_needed():
            return LiveDataResult(
                success=False,
                error="No valid Teams access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        team_filter = params.get("team", "")
        channel_filter = params.get("channel", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            messages = self._fetch_messages_live(
                action, search_query, team_filter, channel_filter, max_results
            )
            formatted = self._format_messages(messages, action)

            return LiveDataResult(
                success=True,
                data=messages,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Teams live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _fetch_messages_live(
        self,
        action: str,
        search_query: str,
        team_filter: str,
        channel_filter: str,
        max_results: int,
    ) -> list[dict]:
        """Fetch recent messages from Teams."""
        messages = []

        # Determine which teams/channels to query
        if team_filter:
            teams = [{"id": team_filter, "displayName": team_filter}]
        elif self.team_id:
            teams = [{"id": self.team_id, "displayName": ""}]
        else:
            teams = self._get_joined_teams()[:5]  # Limit for live queries

        for team in teams:
            team_id = team["id"]
            team_name = team.get("displayName", "")

            if channel_filter:
                channels = [{"id": channel_filter, "displayName": channel_filter}]
            elif self.channel_id:
                channels = [{"id": self.channel_id, "displayName": ""}]
            else:
                channels = self._get_team_channels(team_id)[:3]  # Limit

            for channel in channels:
                channel_id = channel["id"]
                channel_name = channel.get("displayName", "")

                try:
                    response = self._oauth_client.get(
                        f"{self.GRAPH_API_BASE}/teams/{team_id}/channels/{channel_id}/messages",
                        headers=self._get_auth_headers(),
                        params={
                            "$select": "id,subject,body,from,createdDateTime",
                            "$top": 20,
                        },
                    )
                    response.raise_for_status()
                    channel_messages = response.json().get("value", [])

                    for msg in channel_messages:
                        body_data = msg.get("body", {})
                        body_content = body_data.get("content", "")
                        content_type = body_data.get("contentType", "text")

                        if content_type.lower() == "html":
                            body_text = self._strip_html(body_content)
                        else:
                            body_text = body_content

                        # Filter by search query if provided
                        if (
                            search_query
                            and search_query.lower() not in body_text.lower()
                        ):
                            continue

                        from_data = msg.get("from", {})
                        user_data = from_data.get("user", {}) or from_data.get(
                            "application", {}
                        )

                        messages.append(
                            {
                                "id": msg.get("id"),
                                "team": team_name,
                                "channel": channel_name,
                                "sender": user_data.get("displayName", "Unknown"),
                                "date": msg.get("createdDateTime", ""),
                                "subject": msg.get("subject", ""),
                                "preview": body_text[:200] if body_text else "",
                                "type": "channel",
                                "account_email": self.get_account_email(),
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch messages from {team_name}/{channel_name}: {e}"
                    )

        # Sort by date and limit
        messages.sort(key=lambda m: m.get("date", ""), reverse=True)
        return messages[:max_results]

    def _format_messages(self, messages: list[dict], action: str) -> str:
        """Format messages for LLM context."""
        account_email = self.get_account_email()

        if not messages:
            action_msgs = {
                "recent": "No recent messages.",
                "unread": "No unread messages.",
                "search": "No messages found matching your search.",
            }
            return (
                f"### Teams ({action})\n{action_msgs.get(action, 'No messages found.')}"
            )

        action_titles = {
            "recent": "Recent Messages",
            "unread": "Unread Messages",
            "search": "Message Search Results",
        }

        lines = [f"### {action_titles.get(action, 'Messages')}"]
        if account_email:
            lines.append(f"Account: {account_email}")
        lines.append(f"Found {len(messages)} message(s):\n")

        for msg in messages:
            team = msg.get("team", "")
            channel = msg.get("channel", "")
            sender = msg.get("sender", "Unknown")
            date = msg.get("date", "")
            subject = msg.get("subject", "")
            preview = msg.get("preview", "")

            location = (
                f"{team} / {channel}"
                if team and channel
                else (team or channel or "Chat")
            )

            lines.append(f"**{sender}** in {location}")
            if subject:
                lines.append(f"  Subject: {subject}")
            lines.append(f"  Date: {date}")

            if preview:
                preview = preview.replace("\n", " ").strip()
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                lines.append(f"  Preview: {preview}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine optimal routing."""
        query_lower = query.lower()
        action = params.get("action", "")

        # Real-time queries -> Live only
        realtime_patterns = [
            "today",
            "just now",
            "recent",
            "latest",
            "unread",
            "this morning",
            "this afternoon",
            "right now",
        ]
        if any(p in query_lower for p in realtime_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": "recent"},
                reason="Real-time query - using live API only",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"last month",
            r"last quarter",
        ]
        for pattern in historical_patterns:
            if re.search(pattern, query_lower):
                return QueryAnalysis(
                    routing=QueryRouting.RAG_ONLY,
                    rag_query=query,
                    reason="Historical reference - using RAG index",
                    max_rag_results=20,
                )

        # Search queries -> Both
        if action == "search" or params.get("query"):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params={**params, "action": "search"},
                merge_strategy=MergeStrategy.DEDUPE,
                reason="Search query - checking both sources",
                max_rag_results=15,
                max_live_results=self.live_max_results,
            )

        # Person-based queries -> Both
        person_patterns = ["said", "wrote", "mentioned", "from"]
        if any(p in query_lower for p in person_patterns):
            return QueryAnalysis(
                routing=QueryRouting.BOTH_MERGE,
                rag_query=query,
                live_params=params,
                merge_strategy=MergeStrategy.LIVE_FIRST,
                reason="Person-based query - checking both sources",
                max_rag_results=10,
                max_live_results=self.live_max_results,
            )

        # Default -> Live first for freshness
        return QueryAnalysis(
            routing=QueryRouting.LIVE_THEN_RAG,
            rag_query=query,
            live_params=params,
            merge_strategy=MergeStrategy.LIVE_FIRST,
            reason="General query - live first, RAG supplement",
            max_rag_results=10,
            max_live_results=self.live_max_results,
        )

    # =========================================================================
    # Testing & Availability
    # =========================================================================

    def is_available(self) -> bool:
        """Check if Teams is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Teams API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return False, "Failed to get access token - check OAuth configuration"

            # Test API access - get joined teams
            response = self._oauth_client.get(
                f"{self.GRAPH_API_BASE}/me/joinedTeams",
                headers=self._get_auth_headers(),
                params={"$select": "id,displayName", "$top": 5},
            )
            response.raise_for_status()
            teams = response.json().get("value", [])
            results.append(f"Teams: Found {len(teams)} joined team(s)")

            # Test document listing (RAG side)
            if self.supports_rag and teams:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found messages to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "recent", "max_results": 5})
                    if live_result.success:
                        msg_count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {msg_count} recent messages")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)
