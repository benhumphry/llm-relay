"""
Slack Unified Source Plugin.

Combines Slack message indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index channel messages for semantic search
- Live side: Query recent messages, unread items, search conversations
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "messages from today" -> Live only (real-time)
- "discussion about project X last quarter" -> RAG only (historical)
- "what did John say about the budget" -> Both, merge results
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


class SlackUnifiedSource(OAuthMixin, PluginUnifiedSource):
    """
    Unified Slack source - RAG for history, Live for recent messages.

    Single configuration provides:
    - Document indexing: Channel messages indexed for RAG semantic search
    - Live queries: Recent messages, unread items, search
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "slack"
    display_name = "Slack"
    description = (
        "Slack workspace messages with historical search (RAG) and real-time queries"
    )
    category = "communication"
    icon = "💬"

    # Document store types this unified source handles
    handles_doc_source_types = ["slack"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # Could add message sending later
    supports_incremental = True

    default_cache_ttl = 180  # 3 minutes for live results
    default_index_days = 90

    _abstract = False

    @classmethod
    def get_designator_hint(cls) -> str:
        """Generate hint for designator prompt."""
        return (
            "REAL-TIME Slack access. Actions: "
            "action='recent' for recent messages, "
            "action='search' with query='...' to search message content, "
            "action='channel' with channel='#channel-name' to list channel messages. "
            "Optional: user='username' to filter by sender."
        )

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "api_token": os.environ.get("SLACK_BOT_TOKEN", ""),
            "channel_ids": store.slack_channel_id or "",
            "channel_types": store.slack_channel_types or "public_channel",
            "days_back": store.slack_days_back or 90,
        }

    # Slack API base
    SLACK_API_BASE = "https://slack.com/api"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="oauth_account_id",
                label="Slack Workspace",
                field_type=FieldType.OAUTH_ACCOUNT,
                required=True,
                picker_options={"provider": "slack"},
                help_text="Select a connected Slack workspace",
            ),
            FieldDefinition(
                name="channel_ids",
                label="Channel IDs",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Comma-separated channel IDs to index (empty = all public channels bot is in)",
            ),
            FieldDefinition(
                name="include_private",
                label="Include Private Channels",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Also index private channels bot has been invited to",
            ),
            FieldDefinition(
                name="include_dms",
                label="Include Direct Messages",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Also index direct messages",
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
                description="Query type: recent, search, channel",
                param_type="string",
                required=False,
                default="recent",
                examples=["recent", "search", "channel"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for message content",
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
                name="user",
                description="Username to filter messages by",
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
        self.include_private = config.get("include_private", False)
        self.include_dms = config.get("include_dms", False)
        self.index_days = min(config.get("index_days", 90), 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 30)

        # Parse channel IDs
        channel_str = config.get("channel_ids", "")
        self.channel_ids = (
            [c.strip() for c in channel_str.split(",") if c.strip()]
            if channel_str
            else []
        )

        # Cache for user/channel lookups
        self._user_cache: dict[str, str] = {}
        self._channel_cache: dict[str, str] = {}

        self._init_oauth_client()

    def _slack_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a Slack API request."""
        if not self._refresh_token_if_needed():
            raise Exception("No valid Slack access token")

        url = f"{self.SLACK_API_BASE}/{endpoint}"
        response = self._oauth_client.get(
            url,
            headers=self._get_auth_headers(),
            params=params or {},
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("ok"):
            error = data.get("error", "Unknown error")
            raise Exception(f"Slack API error: {error}")

        return data

    def _get_user_name(self, user_id: str) -> str:
        """Get user display name from ID."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            data = self._slack_request("users.info", {"user": user_id})
            user = data.get("user", {})
            name = user.get("real_name") or user.get("name") or user_id
            self._user_cache[user_id] = name
            return name
        except Exception:
            return user_id

    def _get_channel_name(self, channel_id: str) -> str:
        """Get channel name from ID."""
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]

        try:
            data = self._slack_request("conversations.info", {"channel": channel_id})
            channel = data.get("channel", {})
            name = channel.get("name") or channel_id
            self._channel_cache[channel_id] = name
            return name
        except Exception:
            return channel_id

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate messages for indexing."""
        if not self._refresh_token_if_needed():
            logger.error("Cannot list messages - no valid access token")
            return

        logger.info("Listing Slack messages")

        # Get channels to index
        if self.channel_ids:
            channels = [{"id": cid} for cid in self.channel_ids]
        else:
            channels = self._get_channels()

        cutoff_ts = (
            datetime.now(timezone.utc) - timedelta(days=self.index_days)
        ).timestamp()

        for channel in channels:
            channel_id = channel.get("id")
            channel_name = channel.get("name", "")

            if not channel_name:
                channel_name = self._get_channel_name(channel_id)

            yield from self._list_channel_messages(channel_id, channel_name, cutoff_ts)

    def _get_channels(self) -> list[dict]:
        """Get channels the bot is in."""
        channels = []
        cursor = None

        types = "public_channel"
        if self.include_private:
            types += ",private_channel"
        if self.include_dms:
            types += ",im,mpim"

        while True:
            params = {
                "types": types,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = self._slack_request("conversations.list", params)
                channels.extend(data.get("channels", []))

                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
            except Exception as e:
                logger.error(f"Failed to list channels: {e}")
                break

        return channels

    def _list_channel_messages(
        self, channel_id: str, channel_name: str, cutoff_ts: float
    ) -> Iterator[DocumentInfo]:
        """List messages in a channel."""
        cursor = None

        while True:
            params = {
                "channel": channel_id,
                "limit": 100,
                "oldest": str(cutoff_ts),
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = self._slack_request("conversations.history", params)
                messages = data.get("messages", [])

                for msg in messages:
                    ts = msg.get("ts", "")
                    user_id = msg.get("user", "")
                    text = msg.get("text", "")

                    # Skip bot messages and empty messages
                    if msg.get("subtype") == "bot_message" or not text.strip():
                        continue

                    # Convert timestamp to datetime
                    try:
                        msg_time = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                        modified_at = msg_time.isoformat()
                    except Exception:
                        modified_at = ""

                    user_name = self._get_user_name(user_id) if user_id else "Unknown"

                    yield DocumentInfo(
                        uri=f"slack://{channel_id}/{ts}",
                        title=f"{user_name} in #{channel_name}",
                        mime_type="text/plain",
                        modified_at=modified_at,
                        metadata={
                            "channel_id": channel_id,
                            "channel_name": channel_name,
                            "user_id": user_id,
                            "user_name": user_name,
                        },
                    )

                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            except Exception as e:
                logger.error(f"Failed to list messages in {channel_name}: {e}")
                break

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read message content for indexing."""
        if not uri.startswith("slack://"):
            logger.error(f"Invalid Slack URI: {uri}")
            return None

        parts = uri.replace("slack://", "").split("/")
        if len(parts) < 2:
            return None

        channel_id, ts = parts[0], parts[1]

        if not self._refresh_token_if_needed():
            logger.error("Cannot read message - no valid access token")
            return None

        try:
            # Get the specific message
            data = self._slack_request(
                "conversations.history",
                {
                    "channel": channel_id,
                    "oldest": ts,
                    "latest": ts,
                    "inclusive": "true",
                    "limit": 1,
                },
            )
            messages = data.get("messages", [])
            if not messages:
                return None

            msg = messages[0]
        except Exception as e:
            logger.error(f"Failed to fetch message {ts}: {e}")
            return None

        text = msg.get("text", "")
        user_id = msg.get("user", "")
        user_name = self._get_user_name(user_id) if user_id else "Unknown"
        channel_name = self._get_channel_name(channel_id)

        # Convert timestamp to datetime
        try:
            msg_time = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            date_str = msg_time.strftime("%Y-%m-%d %H:%M:%S")
            msg_date = msg_time.strftime("%Y-%m-%d")
        except Exception:
            date_str = ts
            msg_date = None

        # Resolve user mentions
        text = self._resolve_mentions(text)

        # Format for indexing
        content = f"""From: {user_name}
Channel: #{channel_name}
Date: {date_str}

{text}
"""

        # Handle thread replies if present
        if msg.get("thread_ts") and msg.get("reply_count", 0) > 0:
            try:
                replies_data = self._slack_request(
                    "conversations.replies",
                    {
                        "channel": channel_id,
                        "ts": msg["thread_ts"],
                        "limit": 10,
                    },
                )
                replies = replies_data.get("messages", [])[1:]  # Skip the parent
                if replies:
                    content += "\n\nThread replies:\n"
                    for reply in replies:
                        reply_user = self._get_user_name(reply.get("user", ""))
                        reply_text = self._resolve_mentions(reply.get("text", ""))
                        content += f"- {reply_user}: {reply_text}\n"
            except Exception:
                pass

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "message_ts": ts,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "user_id": user_id,
                "user_name": user_name,
                "message_date": msg_date,
                "has_thread": bool(msg.get("thread_ts")),
                "account_id": self.oauth_account_id,
                "source_type": "message",
            },
        )

    def _resolve_mentions(self, text: str) -> str:
        """Resolve user/channel mentions in message text."""

        # Replace user mentions <@U123> with names
        def replace_user(match):
            user_id = match.group(1)
            return f"@{self._get_user_name(user_id)}"

        text = re.sub(r"<@(U[A-Z0-9]+)>", replace_user, text)

        # Replace channel mentions <#C123|name>
        text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)

        # Replace links <url|text>
        text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2 (\1)", text)
        text = re.sub(r"<([^>]+)>", r"\1", text)

        return text

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live message data."""
        if not self._refresh_token_if_needed():
            return LiveDataResult(
                success=False,
                error="No valid Slack access token",
            )

        action = params.get("action", "recent")
        search_query = params.get("query", "")
        channel_filter = params.get("channel", "")
        user_filter = params.get("user", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            if action == "search" and search_query:
                messages = self._search_messages(search_query, max_results)
            else:
                messages = self._get_recent_messages(
                    channel_filter, user_filter, max_results
                )

            formatted = self._format_messages(messages, action)

            return LiveDataResult(
                success=True,
                data=messages,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"Slack live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _search_messages(self, query: str, max_results: int) -> list[dict]:
        """Search for messages using Slack's search API."""
        try:
            data = self._slack_request(
                "search.messages",
                {
                    "query": query,
                    "count": min(max_results, 100),
                    "sort": "timestamp",
                    "sort_dir": "desc",
                },
            )

            matches = data.get("messages", {}).get("matches", [])
            return self._parse_messages(matches, max_results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _get_recent_messages(
        self, channel_filter: str, user_filter: str, max_results: int
    ) -> list[dict]:
        """Get recent messages from channels."""
        messages = []

        # Get channels to query
        if channel_filter:
            # Try to find channel by name or ID
            channels = [{"id": channel_filter, "name": channel_filter}]
        elif self.channel_ids:
            channels = [{"id": cid} for cid in self.channel_ids[:5]]
        else:
            channels = self._get_channels()[:5]

        for channel in channels:
            channel_id = channel.get("id")
            channel_name = channel.get("name") or self._get_channel_name(channel_id)

            try:
                data = self._slack_request(
                    "conversations.history",
                    {"channel": channel_id, "limit": 20},
                )
                channel_msgs = data.get("messages", [])

                for msg in channel_msgs:
                    if msg.get("subtype") == "bot_message":
                        continue

                    user_id = msg.get("user", "")
                    user_name = self._get_user_name(user_id) if user_id else "Unknown"

                    # Apply user filter if specified
                    if user_filter and user_filter.lower() not in user_name.lower():
                        continue

                    text = self._resolve_mentions(msg.get("text", ""))

                    # Convert timestamp
                    ts = msg.get("ts", "")
                    try:
                        msg_time = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                        date_str = msg_time.isoformat()
                    except Exception:
                        date_str = ts

                    messages.append(
                        {
                            "channel": channel_name,
                            "channel_id": channel_id,
                            "user": user_name,
                            "user_id": user_id,
                            "date": date_str,
                            "text": text[:500] if text else "",
                            "ts": ts,
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to get messages from {channel_name}: {e}")

        # Sort by date and limit
        messages.sort(key=lambda m: m.get("date", ""), reverse=True)
        return messages[:max_results]

    def _parse_messages(self, matches: list[dict], max_results: int) -> list[dict]:
        """Parse search results into standardized format."""
        messages = []

        for match in matches[:max_results]:
            channel = match.get("channel", {})
            user_id = match.get("user", "")
            user_name = match.get("username", "") or self._get_user_name(user_id)

            text = self._resolve_mentions(match.get("text", ""))
            ts = match.get("ts", "")

            try:
                msg_time = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                date_str = msg_time.isoformat()
            except Exception:
                date_str = ts

            messages.append(
                {
                    "channel": channel.get("name", ""),
                    "channel_id": channel.get("id", ""),
                    "user": user_name,
                    "user_id": user_id,
                    "date": date_str,
                    "text": text[:500] if text else "",
                    "ts": ts,
                }
            )

        return messages

    def _format_messages(self, messages: list[dict], action: str) -> str:
        """Format messages for LLM context."""
        if not messages:
            action_msgs = {
                "recent": "No recent messages.",
                "search": "No messages found matching your search.",
                "channel": "No messages in this channel.",
            }
            return (
                f"### Slack ({action})\n{action_msgs.get(action, 'No messages found.')}"
            )

        action_titles = {
            "recent": "Recent Messages",
            "search": "Search Results",
            "channel": "Channel Messages",
        }

        lines = [f"### {action_titles.get(action, 'Messages')}"]
        lines.append(f"Found {len(messages)} message(s):\n")

        for msg in messages:
            channel = msg.get("channel", "")
            user = msg.get("user", "Unknown")
            date = msg.get("date", "")
            text = msg.get("text", "")

            lines.append(f"**{user}** in #{channel}")
            lines.append(f"  Date: {date}")

            if text:
                preview = text.replace("\n", " ").strip()
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                lines.append(f"  {preview}")

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
        """Check if Slack is accessible."""
        try:
            self._refresh_token_if_needed()
            return bool(self._access_token)
        except Exception:
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test Slack API connection."""
        results = []
        overall_success = True

        try:
            self._refresh_token_if_needed()
            if not self._access_token:
                return False, "Failed to get access token - check OAuth configuration"

            # Test API access
            data = self._slack_request("auth.test")
            team = data.get("team", "Unknown")
            user = data.get("user", "Unknown")
            results.append(f"Connected as: {user} in {team}")

            # Test channel listing
            channels = self._get_channels()
            results.append(f"Channels: Found {len(channels)} channel(s)")

            # Test document listing (RAG side)
            if self.supports_rag:
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
