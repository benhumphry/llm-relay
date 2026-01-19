"""
Live Data Providers for LLM Relay.

Provides real-time API data fetching for Smart Alias enrichment.
Each provider handles a specific type of live data source.
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Simple in-memory cache for live data responses
_cache: dict[str, tuple[Any, float]] = {}

# Global cache for MCP tool listings (persists across provider instances)
# Key: mcp_api_host, Value: (tools_list, timestamp)
_mcp_tools_cache: dict[str, tuple[list[dict], float]] = {}
MCP_TOOLS_CACHE_TTL = 3600  # 1 hour

# Session-scoped email cache for Gmail lookups
# Key: "{session_key}:{account_email}:{entity_type}:{query}"
# Value: list of (message_data, timestamp) tuples, sorted by most recent first
# This allows finding the most recently viewed email when there are duplicates
_session_email_cache: dict[str, list[tuple[dict, float]]] = {}
SESSION_EMAIL_CACHE_TTL = 3600  # 1 hour - session caches expire after inactivity

# Session-scoped live data context for LLM
# Key: session_key
# Value: list of (source_name, formatted_data, timestamp) tuples
# Accumulates live data results across multiple queries within a session
# This is injected into context so LLM can reference previous results
_session_live_context: dict[str, list[tuple[str, str, float]]] = {}
SESSION_LIVE_CONTEXT_TTL = 1800  # 30 minutes - context expires after inactivity

# Cache statistics for monitoring
_session_cache_stats = {
    "email_cache_hits": 0,
    "email_cache_misses": 0,
    "live_context_hits": 0,
    "live_context_misses": 0,
}


@dataclass
class LiveDataResult:
    """Result from a live data fetch."""

    source_name: str
    success: bool
    data: dict | list | str | None = None
    formatted: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cache_hit: bool = False
    error: str | None = None
    latency_ms: float = 0.0


class LiveDataProvider(ABC):
    """Abstract base class for live data providers."""

    @abstractmethod
    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch live data based on the query.

        Args:
            query: The user's query or extracted parameters
            context: Optional context dict with additional info

        Returns:
            LiveDataResult with the fetched data
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available."""
        pass

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """
        Test the provider connection.

        Returns:
            Tuple of (success, message)
        """
        pass

    def get_cache_key(self, query: str) -> str:
        """Generate a cache key for the query."""
        return f"{self.__class__.__name__}:{query}"

    def get_cached(self, query: str, ttl_seconds: int) -> Optional[Any]:
        """Get cached result if still valid."""
        key = self.get_cache_key(query)
        if key in _cache:
            data, cached_at = _cache[key]
            if time.time() - cached_at < ttl_seconds:
                return data
            del _cache[key]
        return None

    def set_cached(self, query: str, data: Any) -> None:
        """Cache a result."""
        key = self.get_cache_key(query)
        _cache[key] = (data, time.time())


class MCPProvider(LiveDataProvider):
    """
    MCP (Model Context Protocol) provider for RapidAPI and other MCP servers.

    This provider connects to MCP servers that expose API endpoints as MCP tools
    with LLM-friendly descriptions. It also supports a direct RapidAPI mode that
    makes REST API calls without requiring MCP server connectivity.

    Key benefits over manual REST API configuration:
    - Tool discovery: MCP server provides all available tools automatically
    - LLM-friendly: Tool descriptions are designed for LLM consumption
    - Type-safe: Input schemas define required parameters
    - Simpler config: Just need API host and key

    Configuration fields:
    - source_type: "mcp_server"
    - auth_config_json: {"mcp_api_host": "...", "mcp_api_key": "..."}

    Note: RapidAPI's MCP server (mcp.rapidapi.com) may require specific API
    activation. If MCP is unavailable, the provider can still make direct
    REST API calls using the configured endpoint.
    """

    MCP_BASE_URL = "https://mcp.rapidapi.com"

    def __init__(self, source: Any):
        """Initialize with a LiveDataSource configured for MCP."""
        self.source = source
        self.name = source.name
        self.timeout = source.timeout_seconds or 30
        self.last_endpoint = None  # Track last tool/endpoint called for stats
        self.cache_ttl = source.cache_ttl_seconds or 60

        # Parse auth config for MCP settings
        auth_config = {}
        if source.auth_config_json:
            try:
                auth_config = json.loads(source.auth_config_json)
            except json.JSONDecodeError:
                pass

        self.mcp_api_host = auth_config.get("mcp_api_host", "")
        self.mcp_api_key = auth_config.get("mcp_api_key", "")

        # Use RAPIDAPI_KEY env var as fallback
        if not self.mcp_api_key:
            self.mcp_api_key = os.environ.get("RAPIDAPI_KEY", "")

        # Cache for tool list (refreshed periodically)
        self._tools_cache: list[dict] | None = None
        self._tools_cache_time: float = 0
        self._tools_cache_ttl = 3600  # Cache tools for 1 hour

    def _get_mcp_headers(self) -> dict[str, str]:
        """Get headers for MCP server requests."""
        # MCP server uses x-api-host/x-api-key (not x-rapidapi-*)
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-api-host": self.mcp_api_host,
            "x-api-key": self.mcp_api_key,
        }

    def _get_direct_headers(self) -> dict[str, str]:
        """Get headers for direct RapidAPI REST calls."""
        return {
            "Content-Type": "application/json",
            "X-RapidAPI-Host": self.mcp_api_host,
            "X-RapidAPI-Key": self.mcp_api_key,
        }

    def list_tools(self, include_broken: bool = False) -> list[dict]:
        """
        List available tools from the MCP server.

        Returns list of tools with name, description, and inputSchema.
        Results are cached globally (by API host) to avoid repeated calls.
        Broken tools (those returning 404/not-found errors) are filtered out
        unless include_broken=True.

        If MCP server is unavailable, returns an empty list (direct REST
        mode will be used instead via fetch()).
        """
        # Check global cache first (persists across provider instances)
        cache_key = self.mcp_api_host
        if cache_key in _mcp_tools_cache:
            tools, cached_at = _mcp_tools_cache[cache_key]
            if time.time() - cached_at < MCP_TOOLS_CACHE_TTL:
                # Filter broken tools before returning
                if not include_broken:
                    tools = self._filter_broken_tools(tools)
                return tools

        # Try MCP server with JSON-RPC format (Streamable HTTP transport)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # JSON-RPC 2.0 tools/list request
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }
                # RapidAPI MCP uses POST to root endpoint
                response = client.post(
                    f"{self.MCP_BASE_URL}/",
                    headers=self._get_mcp_headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Handle JSON-RPC response format
                if "result" in data:
                    tools = data["result"].get("tools", [])
                else:
                    tools = data.get("tools", [])

                # Cache globally (all tools, filter on retrieval)
                _mcp_tools_cache[cache_key] = (tools, time.time())

                logger.info(
                    f"MCP {self.name}: Listed {len(tools)} tools from {self.mcp_api_host}"
                )

                # Filter broken tools before returning
                if not include_broken:
                    tools = self._filter_broken_tools(tools)

                return tools

        except Exception as e:
            logger.warning(
                f"MCP {self.name}: MCP server unavailable ({e}). "
                "Direct REST mode will be used."
            )
            # Cache empty list globally to avoid repeated failed requests
            _mcp_tools_cache[cache_key] = ([], time.time())
            return []

    def _filter_broken_tools(self, tools: list[dict]) -> list[dict]:
        """Filter out tools that have been marked as broken."""
        if not tools:
            return tools

        # Get source ID from the source object
        source_id = getattr(self.source, "id", None)
        if not source_id:
            return tools

        try:
            from db.live_data_sources import get_broken_tools

            broken_tool_names = set(get_broken_tools(source_id))
            if not broken_tool_names:
                return tools

            filtered = [t for t in tools if t.get("name") not in broken_tool_names]
            if len(filtered) < len(tools):
                logger.info(
                    f"MCP {self.name}: Filtered {len(tools) - len(filtered)} broken tools"
                )
            return filtered
        except Exception as e:
            logger.warning(f"Error filtering broken tools: {e}")
            return tools

    def get_tool_descriptions(self) -> str:
        """
        Get formatted tool descriptions for the designator.

        Returns a string describing all available tools that can be
        included in the designator prompt.
        """
        tools = self.list_tools()
        if not tools:
            return f"No tools available from {self.name}"

        lines = [f"### {self.name} API Tools ({self.mcp_api_host})"]
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            # Truncate long descriptions
            if len(description) > 200:
                description = description[:197] + "..."
            lines.append(f"- **{name}**: {description}")

            # Show required parameters
            schema = tool.get("inputSchema", {})
            required = schema.get("required", [])
            if required:
                lines.append(f"  Required params: {', '.join(required)}")

        return "\n".join(lines)

    def _get_source_type(self) -> str:
        """Get the source type for cache keying."""
        # Use mcp_api_host as a more specific identifier, or fall back to source_type
        if self.mcp_api_host:
            # Extract a clean name from the API host (e.g., "finnhub" from "finnhub-realtime-stock-price.p.rapidapi.com")
            host_parts = self.mcp_api_host.split(".")
            if host_parts:
                return host_parts[0].lower()
        return getattr(self.source, "source_type", "mcp")

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a specific MCP tool with arguments via MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Dict of arguments matching the tool's input schema

        Returns:
            Tool result dict with content
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # JSON-RPC 2.0 tools/call request
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
                # RapidAPI MCP uses POST to root endpoint
                response = client.post(
                    f"{self.MCP_BASE_URL}/",
                    headers=self._get_mcp_headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Handle JSON-RPC response
                if "error" in data:
                    error = data["error"]
                    raise Exception(error.get("message", f"MCP error: {error}"))
                if "result" in data:
                    return data["result"]
                return data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"MCP {self.name}: Tool call failed: HTTP {e.response.status_code}"
            )
            raise
        except Exception as e:
            logger.error(f"MCP {self.name}: Tool call error: {e}")
            raise

    def call_tool_cached(self, tool_name: str, arguments: dict) -> tuple[dict, bool]:
        """
        Call a tool with database-backed caching.

        Checks the live data cache before making API call, and caches
        successful results with appropriate TTL based on data type.

        Args:
            tool_name: Name of the tool to call
            arguments: Dict of arguments matching the tool's input schema

        Returns:
            Tuple of (result dict, cache_hit bool)
        """
        try:
            from db.live_cache import cache_data_response, lookup_data_cache

            source_type = self._get_source_type()

            # Check cache first
            cached_json = lookup_data_cache(source_type, tool_name, arguments)
            if cached_json:
                try:
                    cached_data = json.loads(cached_json)
                    logger.info(f"MCP {self.name}: Cache hit for {tool_name}")
                    return cached_data, True
                except json.JSONDecodeError:
                    # Invalid cached data, proceed with API call
                    pass

            # Cache miss - make the actual API call
            result = self.call_tool(tool_name, arguments)

            # Cache the result
            try:
                result_json = json.dumps(result)
                # Generate a brief summary for the cache entry
                summary = self._generate_result_summary(result, tool_name)
                cache_data_response(
                    source_type=source_type,
                    tool_name=tool_name,
                    args=arguments,
                    response_data=result_json,
                    response_summary=summary,
                    # TTL is auto-determined based on tool name and args
                )
            except Exception as e:
                logger.warning(f"Failed to cache tool result: {e}")

            return result, False

        except ImportError:
            # If caching module not available, fall back to direct call
            logger.warning("Live cache module not available, using direct call")
            return self.call_tool(tool_name, arguments), False

    def _generate_result_summary(self, result: dict, tool_name: str) -> str:
        """Generate a brief human-readable summary of the result."""
        try:
            content = result.get("content", [])
            if content and isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        # Return first 200 chars
                        return text[:200] + "..." if len(text) > 200 else text
            return f"{tool_name} result"
        except Exception:
            return f"{tool_name} result"

    def _extract_and_cache_entities(
        self, tool_name: str, arguments: dict, result: dict
    ) -> None:
        """
        Extract entity resolutions from tool results and cache them.

        Looks for patterns like symbol lookups, location codes, etc.
        """
        try:
            from db.live_cache import cache_entity

            source_type = self._get_source_type()
            tool_lower = tool_name.lower()

            # Extract text content from result
            content = result.get("content", [])
            result_text = ""
            if content and isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        result_text += item.get("text", "")

            # Try to parse as JSON for structured extraction
            result_data = None
            if result_text.strip().startswith("{"):
                try:
                    result_data = json.loads(result_text)
                except json.JSONDecodeError:
                    pass

            # Stock symbol resolution
            if any(
                term in tool_lower
                for term in ["search", "lookup", "find", "symbol", "ticker"]
            ):
                # Look for symbol in arguments and result
                query = (
                    arguments.get("query")
                    or arguments.get("keywords")
                    or arguments.get("name", "")
                )

                if result_data and isinstance(result_data, dict):
                    # Common patterns for symbol results
                    symbol = (
                        result_data.get("symbol")
                        or result_data.get("ticker")
                        or result_data.get("code")
                    )
                    display_name = (
                        result_data.get("name")
                        or result_data.get("description")
                        or result_data.get("company_name")
                    )

                    if symbol and query:
                        cache_entity(
                            source_type=source_type,
                            entity_type="symbol",
                            query_text=query,
                            resolved_value=symbol,
                            display_name=display_name,
                        )
                        logger.debug(
                            f"Cached entity: {query} -> {symbol} ({display_name})"
                        )

                # Also check for results array
                if (
                    result_data
                    and isinstance(result_data, list)
                    and len(result_data) > 0
                ):
                    first_result = result_data[0]
                    if isinstance(first_result, dict):
                        symbol = (
                            first_result.get("symbol")
                            or first_result.get("ticker")
                            or first_result.get("code")
                        )
                        display_name = first_result.get("name") or first_result.get(
                            "description"
                        )
                        if symbol and query:
                            cache_entity(
                                source_type=source_type,
                                entity_type="symbol",
                                query_text=query,
                                resolved_value=symbol,
                                display_name=display_name,
                            )

            # Location/city resolution (weather APIs)
            if any(term in tool_lower for term in ["location", "city", "place", "geo"]):
                query = (
                    arguments.get("query")
                    or arguments.get("city")
                    or arguments.get("location", "")
                )

                if result_data and isinstance(result_data, dict):
                    location_key = (
                        result_data.get("key")
                        or result_data.get("location_key")
                        or result_data.get("id")
                        or result_data.get("location_id")
                    )
                    display_name = (
                        result_data.get("name")
                        or result_data.get("city")
                        or result_data.get("location_name")
                    )

                    if location_key and query:
                        cache_entity(
                            source_type=source_type,
                            entity_type="location",
                            query_text=query,
                            resolved_value=str(location_key),
                            display_name=display_name,
                        )

        except ImportError:
            pass  # Caching not available
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

    def _call_direct_rest(self, endpoint: str, method: str, params: dict) -> dict:
        """
        Make a direct REST API call to RapidAPI (fallback when MCP unavailable).

        Args:
            endpoint: API endpoint path (e.g., "/api/v1/quote")
            method: HTTP method (GET or POST)
            params: Query parameters or body

        Returns:
            API response as dict
        """
        base_url = f"https://{self.mcp_api_host}"
        url = (
            f"{base_url}{endpoint}"
            if endpoint.startswith("/")
            else f"{base_url}/{endpoint}"
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = client.get(
                        url,
                        params=params,
                        headers=self._get_direct_headers(),
                    )
                else:
                    response = client.post(
                        url,
                        json=params,
                        headers=self._get_direct_headers(),
                    )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"MCP {self.name}: Direct REST call failed: {e}")
            raise

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch data using MCP server or direct REST API.

        The context should contain:
        - tool_name: Which MCP tool to call (for MCP mode)
        - tool_args: Arguments for the tool
        OR for agentic mode (multi-step):
        - agentic: True to enable multi-step tool calling
        - goal: The goal to accomplish (defaults to query)
        OR for direct REST mode:
        - endpoint: API endpoint path
        - method: HTTP method (default: GET)
        - params: Query/body parameters
        """
        start_time = time.time()

        if context is None:
            context = {}

        # Check if agentic mode is requested
        if context.get("agentic") or context.get("_agentic"):
            goal = context.get("goal", query)
            designator_model = context.get("designator_model")
            max_iterations = context.get("max_iterations", 5)
            return self.fetch_agentic(
                goal=goal,
                context=context,
                designator_model=designator_model,
                max_iterations=max_iterations,
            )

        # Check if we have MCP tools available
        tools = self.list_tools()
        use_mcp = bool(tools)

        # Get tool/endpoint info from context
        tool_name = context.get("tool_name") or context.get("_mcp_tool")
        endpoint = context.get("endpoint", "")
        method = context.get("method", "GET")

        # Build arguments/params
        tool_args = context.get("tool_args", {})
        params = context.get("params", {})
        if not tool_args and not params:
            # Extract from top-level context keys
            internal_keys = {
                "query",
                "tool_name",
                "_mcp_tool",
                "tool_args",
                "endpoint",
                "method",
                "params",
            }
            extracted = {k: v for k, v in context.items() if k not in internal_keys}
            tool_args = extracted
            params = extracted

        # Determine cache key
        if use_mcp and tool_name:
            cache_key = f"mcp:{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        elif endpoint:
            cache_key = f"rest:{endpoint}:{json.dumps(params, sort_keys=True)}"
        elif use_mcp:
            # No tool_name specified but MCP tools available - use agentic mode
            # This allows the LLM to figure out which tool to call
            logger.info(
                f"MCP {self.name}: No tool_name specified, falling back to agentic mode"
            )
            goal = context.get("goal") or context.get("query") or query
            return self.fetch_agentic(
                goal=goal,
                context=context,
                designator_model=context.get("designator_model"),
                max_iterations=context.get("max_iterations", 5),
            )
        else:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No tool_name or endpoint specified and no MCP tools available",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Check cache
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_result(cached, tool_name or endpoint),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        try:
            # Try MCP first if available and tool specified
            if use_mcp and tool_name:
                self.last_endpoint = tool_name  # Track for stats
                result = self.call_tool(tool_name, tool_args)
            elif endpoint:
                # Fall back to direct REST
                self.last_endpoint = endpoint  # Track for stats
                result = self._call_direct_rest(endpoint, method, params)
            else:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"No tool_name or endpoint. MCP available: {use_mcp}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            # Extract content from MCP result
            content = result.get("content", [])
            if content and isinstance(content, list):
                # MCP returns content as array of {type, text} objects
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                data = "\n".join(text_parts) if text_parts else content
            else:
                data = result

            # Try to parse as JSON if it looks like JSON
            if isinstance(data, str) and data.strip().startswith("{"):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            # Cache result
            if self.cache_ttl > 0:
                self.set_cached(cache_key, data)

            formatted = self._format_result(data, tool_name or endpoint)

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=data,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            error_msg = str(e)
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=error_msg,
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _format_result(self, data: Any, tool_name: str) -> str:
        """Format MCP tool result for LLM context injection."""
        header = f"### {self.name}: {tool_name}"

        if isinstance(data, dict):
            # Format dict nicely
            lines = [header]
            for key, value in data.items():
                if key.startswith("_"):
                    continue
                display_key = key.replace("_", " ").title()
                if isinstance(value, (dict, list)):
                    lines.append(f"{display_key}: {json.dumps(value)[:200]}")
                else:
                    lines.append(f"{display_key}: {value}")
            return "\n".join(lines)
        elif isinstance(data, str):
            # Truncate very long strings
            if len(data) > 2000:
                data = data[:2000] + "..."
            return f"{header}\n{data}"
        else:
            return f"{header}\n{json.dumps(data, indent=2)[:2000]}"

    def fetch_agentic(
        self,
        goal: str,
        context: dict | None = None,
        designator_model: str | None = None,
        max_iterations: int = 5,
    ) -> LiveDataResult:
        """
        Fetch data using an agentic loop that can make multiple tool calls.

        This method is useful for APIs that require multi-step lookups,
        e.g., first searching for a location code, then fetching data for that location.

        Args:
            goal: The user's goal/query to accomplish
            context: Additional context (user info, etc.)
            designator_model: LLM model to use for tool selection (default: groq/llama-3.1-8b-instant)
            max_iterations: Maximum number of tool calls (default: 5)

        Returns:
            LiveDataResult with the final answer
        """
        start_time = time.time()

        tools = self.list_tools()
        if not tools:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No MCP tools available for agentic fetch",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Build tools description for the LLM
        tools_desc = self._build_tools_description(tools)

        # Track conversation history for the agent
        messages = []
        tool_results = []
        iterations = 0

        # System prompt for the agent
        system_prompt = f"""You are a data retrieval agent for the {self.name} API.
Your goal is to fetch the requested data by calling the appropriate API tools.

AVAILABLE TOOLS:
{tools_desc}

INSTRUCTIONS:
1. Analyze the user's goal and determine which tool(s) to call
2. Some queries require multiple steps (e.g., search for an ID first, then fetch details)
3. After each tool call, you'll see the result. Decide if you need more calls or have enough data.
4. When you have the final answer, respond with: FINAL_ANSWER: <your formatted answer>

RESPONSE FORMAT:
- To call a tool: TOOL_CALL: {{"tool": "tool_name", "args": {{"param": "value"}}}}
- When done: FINAL_ANSWER: <formatted data for the user>

Always extract and format the key information from API responses."""

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Goal: {goal}"})

        if context:
            # Add relevant context
            ctx_parts = []
            if context.get("user_intelligence"):
                ctx_parts.append(f"User context: {context['user_intelligence']}")
            if context.get("user_memory"):
                ctx_parts.append(f"Known info: {context['user_memory']}")
            if ctx_parts:
                messages.append({"role": "user", "content": "\n".join(ctx_parts)})

        # Get designator model
        model = designator_model or os.environ.get(
            "DESIGNATOR_MODEL", "groq/llama-3.1-8b-instant"
        )

        try:
            from providers import registry

            # Ensure providers are registered (may not be if called outside main app)
            if not registry._providers:
                from providers import register_all_providers

                register_all_providers()

            # Resolve model once before the loop
            resolved = registry._resolve_actual_model(model)
            if not resolved or not resolved.provider:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"Designator model not found: {model}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            llm_provider = resolved.provider

            while iterations < max_iterations:
                iterations += 1

                # Call the LLM
                response = llm_provider.chat_completion(
                    model=resolved.model_id,
                    messages=messages,
                    system=None,
                    options={"temperature": 0.1, "max_tokens": 1000},
                )

                assistant_msg = response.get("content", "")
                messages.append({"role": "assistant", "content": assistant_msg})

                logger.debug(
                    f"MCP {self.name} agent iteration {iterations}: {assistant_msg[:200]}"
                )

                # Check for final answer
                if "FINAL_ANSWER:" in assistant_msg:
                    final_answer = assistant_msg.split("FINAL_ANSWER:", 1)[1].strip()
                    return LiveDataResult(
                        source_name=self.name,
                        success=True,
                        data={"tool_results": tool_results, "iterations": iterations},
                        formatted=f"### {self.name}\n{final_answer}",
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                # Check for tool call
                if "TOOL_CALL:" in assistant_msg:
                    try:
                        tool_json = assistant_msg.split("TOOL_CALL:", 1)[1].strip()
                        # Handle case where there might be text after the JSON
                        if "\n" in tool_json:
                            tool_json = tool_json.split("\n")[0]
                        tool_call = json.loads(tool_json)
                        tool_name = tool_call.get("tool")
                        tool_args = tool_call.get("args", {})

                        logger.info(
                            f"MCP {self.name} agent calling tool: {tool_name} with {tool_args}"
                        )

                        # Call the tool with caching
                        self.last_endpoint = tool_name
                        result, cache_hit = self.call_tool_cached(tool_name, tool_args)

                        if cache_hit:
                            logger.info(
                                f"MCP {self.name} agent: Cache hit for {tool_name}"
                            )

                        # Extract and cache any entity resolutions from the result
                        self._extract_and_cache_entities(tool_name, tool_args, result)

                        # Extract content from result
                        content = result.get("content", [])
                        if content and isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    text_parts.append(item.get("text", ""))
                            result_text = (
                                "\n".join(text_parts) if text_parts else str(content)
                            )
                        else:
                            result_text = json.dumps(result, indent=2)

                        # Truncate very long results
                        if len(result_text) > 4000:
                            result_text = result_text[:4000] + "\n... (truncated)"

                        tool_results.append(
                            {
                                "tool": tool_name,
                                "args": tool_args,
                                "result": result_text[
                                    :500
                                ],  # Store abbreviated for metadata
                            }
                        )

                        # Add result to conversation
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Tool result for {tool_name}:\n{result_text}",
                            }
                        )

                    except json.JSONDecodeError as e:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Error parsing tool call: {e}. Please use valid JSON format.",
                            }
                        )
                    except Exception as e:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Tool call failed: {e}. Try a different approach or tool.",
                            }
                        )
                else:
                    # No tool call and no final answer - prompt for action
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please either call a tool (TOOL_CALL: {...}) or provide the final answer (FINAL_ANSWER: ...).",
                        }
                    )

            # Max iterations reached
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=f"Max iterations ({max_iterations}) reached without final answer",
                data={"tool_results": tool_results, "iterations": iterations},
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"MCP {self.name} agentic fetch error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_tools_description(self, tools: list[dict]) -> str:
        """Build a formatted description of available tools for the agent."""
        lines = []
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            schema = tool.get("inputSchema", {})
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            lines.append(f"**{name}**: {description}")
            if properties:
                params = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    req_marker = "*" if param_name in required else ""
                    params.append(
                        f"  - {param_name}{req_marker} ({param_type}): {param_desc}"
                    )
                if params:
                    lines.append("  Parameters:")
                    lines.extend(params)
            lines.append("")

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if MCP source is configured."""
        return bool(self.mcp_api_host and self.mcp_api_key)

    def test_connection(self) -> tuple[bool, str]:
        """Test MCP connection by listing tools."""
        if not self.mcp_api_host:
            return False, "MCP API host not configured"
        if not self.mcp_api_key:
            return False, "MCP API key not configured (set RAPIDAPI_KEY or provide key)"

        try:
            tools = self.list_tools()
            if tools:
                return True, f"Connected. {len(tools)} tools available."
            else:
                return (
                    False,
                    f"Connected but no MCP tools found for {self.mcp_api_host}. Check the API host is correct (some APIs use numbered suffixes like -15).",
                )
        except Exception as e:
            return False, str(e)


class StocksProvider(LiveDataProvider):
    """
    Built-in stocks provider using Finnhub API.

    Requires FINNHUB_API_KEY environment variable.

    Can handle:
    - Single stock queries: "What's AAPL trading at?"
    - Portfolio queries: Extracts all holdings from user context
    """

    BASE_URL = "https://finnhub.io/api/v1"

    # Map company names to stock symbols
    COMPANY_TO_SYMBOL = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "netflix": "NFLX",
        "amd": "AMD",
        "intel": "INTC",
        "unity": "U",
        "adobe": "ADBE",
        "salesforce": "CRM",
        "paypal": "PYPL",
        "spotify": "SPOT",
        "uber": "UBER",
        "airbnb": "ABNB",
        "coinbase": "COIN",
        "palantir": "PLTR",
        "snowflake": "SNOW",
        "shopify": "SHOP",
        "zoom": "ZM",
        "docusign": "DOCU",
        "crowdstrike": "CRWD",
        "datadog": "DDOG",
        "twilio": "TWLO",
        "roku": "ROKU",
        "square": "SQ",
        "block": "SQ",
    }

    def __init__(self, source: Any = None):
        self.source = source
        self.api_key = os.environ.get("FINNHUB_API_KEY", "")
        self.name = source.name if source else "stocks"
        self.cache_ttl = source.cache_ttl_seconds if source else 60
        self.last_endpoint = None  # Track last endpoint called for stats

    def _extract_symbol(self, query: str) -> str:
        """Extract stock symbol from query."""
        # Look for common patterns: "AAPL", "price of AAPL", "AAPL stock"
        query_upper = query.upper()
        # Common stock symbols are 1-5 uppercase letters
        symbols = re.findall(r"\b([A-Z]{1,5})\b", query_upper)
        if symbols:
            # Filter out common words
            common_words = {
                "THE",
                "AND",
                "FOR",
                "OF",
                "TO",
                "IN",
                "IS",
                "IT",
                "A",
                "AN",
            }
            symbols = [s for s in symbols if s not in common_words]
            if symbols:
                return symbols[0]
        return query.upper().strip()

    def _extract_holdings_from_context(self, context: dict) -> list[dict]:
        """
        Extract stock holdings from user context (memory/intelligence).

        Looks for patterns like:
        - "X shares in Company"
        - "X shares of SYMBOL"
        - "holds X SYMBOL"
        - "X stocks/units of Fund Name" (for funds)

        Returns list of {"symbol": "AAPL", "shares": 123.45, "name": "Apple", "is_fund": False}
        """
        holdings = []

        # Get user context text
        user_text = ""
        if context:
            user_text += context.get("user_intelligence", "") + "\n"
            user_text += context.get("user_memory", "") + "\n"

        if not user_text.strip():
            return holdings

        seen_symbols = set()
        seen_fund_names = set()

        # Pattern for funds: "X stocks/units of Multi Word Fund Name"
        # Matches: "15056.323 stocks of L&G Global Technology Index I Acc"
        fund_patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:stocks?|units?)\s+(?:in|of)\s+([A-Za-z&\s]+(?:Index|Fund|ETF|Trust|Acc|Inc)[A-Za-z\s]*)",
        ]

        for pattern in fund_patterns:
            matches = re.findall(pattern, user_text, re.IGNORECASE)
            for match in matches:
                shares_str, fund_name = match
                try:
                    shares = float(shares_str)
                except ValueError:
                    continue

                fund_name = fund_name.strip()
                fund_name_lower = fund_name.lower()

                if fund_name_lower not in seen_fund_names:
                    seen_fund_names.add(fund_name_lower)
                    holdings.append(
                        {
                            "symbol": None,  # No symbol - will need web lookup
                            "shares": shares,
                            "name": fund_name,
                            "is_fund": True,
                        }
                    )

        # Pattern: "X shares in/of Company/Symbol"
        # Matches: "149.847 shares in Unity", "100 shares of AAPL"
        stock_patterns = [
            r"(\d+(?:\.\d+)?)\s*shares?\s+(?:in|of)\s+([A-Za-z]+)",
            r"holds?\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+shares?",
            r"(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+shares?",
        ]

        for pattern in stock_patterns:
            matches = re.findall(pattern, user_text, re.IGNORECASE)
            for match in matches:
                shares_str, name = match
                try:
                    shares = float(shares_str)
                except ValueError:
                    continue

                # Convert company name to symbol
                name_lower = name.lower()
                if name_lower in self.COMPANY_TO_SYMBOL:
                    symbol = self.COMPANY_TO_SYMBOL[name_lower]
                elif name.upper() in [s for s in self.COMPANY_TO_SYMBOL.values()]:
                    # Already a symbol
                    symbol = name.upper()
                else:
                    # Unknown - try as symbol if it's short enough
                    if len(name) <= 5:
                        symbol = name.upper()
                    else:
                        continue

                if symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    holdings.append(
                        {
                            "symbol": symbol,
                            "shares": shares,
                            "name": name.title(),
                            "is_fund": False,
                        }
                    )

        return holdings

    def _is_portfolio_query(self, query: str) -> bool:
        """Check if the query is asking about a portfolio/multiple stocks."""
        portfolio_keywords = [
            "portfolio",
            "all my stocks",
            "all my shares",
            "my holdings",
            "my investments",
            "total value",
            "stock holdings",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in portfolio_keywords)

    def _fetch_single_quote(self, symbol: str) -> dict | None:
        """Fetch a single stock quote. Returns quote data or None on error."""
        # Check cache first
        if self.cache_ttl > 0:
            cached = self.get_cached(symbol, self.cache_ttl)
            if cached is not None:
                return cached

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    f"{self.BASE_URL}/quote",
                    params={"symbol": symbol, "token": self.api_key},
                )
                response.raise_for_status()
                quote_data = response.json()

                if quote_data.get("c", 0) == 0:
                    return None

                # Cache result
                if self.cache_ttl > 0:
                    self.set_cached(symbol, quote_data)

                return quote_data
        except Exception:
            return None

    def _fetch_fund_price_via_web(self, fund_name: str) -> dict | None:
        """
        Fetch fund price via web search and scrape when API lookup fails.

        Returns dict with keys: price, currency, change_pct (if available)
        """
        cache_key = f"fund_web:{fund_name}"

        # Check cache first
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                logger.debug(f"Fund price cache hit for: {fund_name}")
                return cached

        try:
            # Import search and scraper
            from augmentation.scraper import WebScraper
            from augmentation.search.searxng import SearXNGProvider

            searcher = SearXNGProvider()
            scraper = WebScraper()

            if not searcher.is_configured():
                logger.warning("SearXNG not configured for fund price lookup")
                return None

            # Search for fund price
            search_query = f"{fund_name} price GBP"
            logger.info(f"Searching for fund price: {search_query}")

            results = searcher.search(search_query, max_results=3)
            if not results:
                logger.warning(f"No search results for fund: {fund_name}")
                return None

            # Try scraping each result until we find a price
            for result in results:
                try:
                    scrape_result = scraper.scrape(result.url)
                    if not scrape_result.success or not scrape_result.content:
                        continue

                    # Try to extract price from scraped content
                    price_data = self._extract_price_from_content(
                        scrape_result.content, fund_name
                    )

                    if price_data and price_data.get("price", 0) > 0:
                        logger.info(
                            f"Found fund price for {fund_name}: "
                            f"{price_data.get('currency', '£')}{price_data['price']:.2f}"
                        )

                        # Cache result
                        if self.cache_ttl > 0:
                            self.set_cached(cache_key, price_data)

                        return price_data

                except Exception as e:
                    logger.debug(f"Error scraping {result.url}: {e}")
                    continue

            logger.warning(f"Could not extract price for fund: {fund_name}")
            return None

        except Exception as e:
            logger.error(f"Error fetching fund price via web: {e}")
            return None

    def _extract_price_from_content(self, content: str, fund_name: str) -> dict | None:
        """
        Extract fund price from scraped web content using regex patterns.

        Falls back to LLM extraction if patterns don't match.
        """
        # Common price patterns for UK funds
        # Matches: "123.45p", "£1.23", "1.2345", "NAV: 123.45"
        patterns = [
            # Pence format: "123.45p" or "12345.67 p"
            r"(?:price|nav|value)[:\s]*(\d+(?:\.\d+)?)\s*p(?:ence)?",
            # Pound format: "£1.23" or "GBP 1.23"
            r"(?:price|nav|value)[:\s]*[£]?\s*(\d+(?:\.\d+)?)",
            # Generic decimal that looks like a price
            r"(?:current|latest|today)[^£\d]*[£]?\s*(\d+(?:\.\d+)?)",
            # NAV specific
            r"nav[:\s]+[£]?(\d+(?:\.\d+)?)",
            # Bid price
            r"bid[:\s]+[£]?(\d+(?:\.\d+)?)\s*p?",
        ]

        content_lower = content.lower()

        for pattern in patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                try:
                    price = float(matches[0])
                    # If price is > 100, it's likely in pence
                    if price > 100:
                        price = price / 100  # Convert pence to pounds

                    # Sanity check - fund prices are typically 0.50 to 500
                    if 0.1 < price < 1000:
                        return {
                            "price": price,
                            "currency": "GBP",
                            "change_pct": 0,  # Not always available
                        }
                except ValueError:
                    continue

        # Try LLM extraction as fallback
        return self._extract_price_via_llm(content, fund_name)

    def _extract_price_via_llm(self, content: str, fund_name: str) -> dict | None:
        """Use a fast LLM to extract price from content."""
        try:
            # Truncate content to avoid token limits
            max_chars = 4000
            if len(content) > max_chars:
                content = content[:max_chars]

            prompt = f"""Extract the current price/NAV for the fund "{fund_name}" from this content.

Content:
{content}

Return ONLY a JSON object with these fields:
- price: the numeric price value (as a float, in pounds not pence)
- currency: "GBP" or "USD"
- change_pct: percentage change today (or 0 if not found)

If you cannot find the price, return: {{"price": 0}}

JSON:"""

            # Use a fast model for extraction
            response = httpx.post(
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": "gemini/gemini-flash-lite-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 100,
                },
                timeout=15,
            )
            response.raise_for_status()

            result = response.json()
            content_text = result["choices"][0]["message"]["content"].strip()

            # Parse JSON response
            # Handle markdown code blocks
            if "```" in content_text:
                content_text = re.search(
                    r"```(?:json)?\s*(.*?)\s*```", content_text, re.DOTALL
                )
                if content_text:
                    content_text = content_text.group(1)
                else:
                    return None

            data = json.loads(content_text)
            if data.get("price", 0) > 0:
                return data

        except Exception as e:
            logger.debug(f"LLM price extraction failed: {e}")

        return None

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """Fetch stock quote(s) from Finnhub."""
        start_time = time.time()
        self.last_endpoint = "quote"  # Default endpoint

        logger.debug(
            f"StocksProvider.fetch called with query='{query[:50]}...', context keys={list(context.keys()) if context else None}"
        )

        if not self.api_key:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="FINNHUB_API_KEY not configured",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Check if context requests a full portfolio fetch (set by enricher for multi-symbol queries)
        if context and context.get("_portfolio_fetch"):
            holdings = self._extract_holdings_from_context(context)
            if holdings:
                return self._fetch_portfolio(holdings, start_time)

        # If context has a symbol, use it directly (from designator)
        # Only fall back to portfolio extraction if no symbol provided
        if context and "symbol" in context:
            symbol = context["symbol"].upper()
            logger.debug(f"StocksProvider: Using symbol from context: {symbol}")
        elif self._is_portfolio_query(query) and context:
            # No symbol provided but it's a portfolio query - extract holdings from context
            holdings = self._extract_holdings_from_context(context)
            if holdings:
                return self._fetch_portfolio(holdings, start_time)
            # If no holdings found, try to extract symbol from query
            symbol = self._extract_symbol(query)
        else:
            symbol = self._extract_symbol(query)

        # Check cache
        if self.cache_ttl > 0:
            cached = self.get_cached(symbol, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_quote(cached, symbol),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        try:
            with httpx.Client(timeout=10) as client:
                # Get quote
                response = client.get(
                    f"{self.BASE_URL}/quote",
                    params={"symbol": symbol, "token": self.api_key},
                )
                response.raise_for_status()
                quote_data = response.json()

                # Check if we got valid data
                if quote_data.get("c", 0) == 0:
                    logger.warning(
                        f"StocksProvider: No data (price=0) for symbol: {symbol}"
                    )
                    return LiveDataResult(
                        source_name=self.name,
                        success=False,
                        error=f"No data found for symbol: {symbol}",
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                # Cache result
                if self.cache_ttl > 0:
                    self.set_cached(symbol, quote_data)

                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=quote_data,
                    formatted=self._format_quote(quote_data, symbol),
                    latency_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _fetch_portfolio(
        self, holdings: list[dict], start_time: float
    ) -> LiveDataResult:
        """Fetch quotes for all holdings and format as portfolio summary."""
        results = []
        errors = []
        total_value_usd = 0.0
        total_value_gbp = 0.0

        for holding in holdings:
            symbol = holding.get("symbol")
            shares = holding["shares"]
            name = holding.get("name", symbol or "Unknown")
            is_fund = holding.get("is_fund", False)

            if is_fund or symbol is None:
                # Try web search fallback for funds
                logger.info(f"Fetching fund price via web for: {name}")
                fund_data = self._fetch_fund_price_via_web(name)
                if fund_data and fund_data.get("price", 0) > 0:
                    price = fund_data["price"]
                    change_pct = fund_data.get("change_pct", 0)
                    currency = fund_data.get("currency", "GBP")
                    value = price * shares

                    if currency == "GBP":
                        total_value_gbp += value
                    else:
                        total_value_usd += value

                    results.append(
                        {
                            "symbol": None,
                            "name": name,
                            "shares": shares,
                            "price": price,
                            "change_pct": change_pct,
                            "value": value,
                            "currency": currency,
                            "is_fund": True,
                        }
                    )
                else:
                    errors.append(name)
            else:
                # Regular stock - use Finnhub API
                quote = self._fetch_single_quote(symbol)
                if quote:
                    price = quote.get("c", 0)
                    change_pct = quote.get("dp", 0)
                    value = price * shares
                    total_value_usd += value

                    results.append(
                        {
                            "symbol": symbol,
                            "name": name,
                            "shares": shares,
                            "price": price,
                            "change_pct": change_pct,
                            "value": value,
                            "currency": "USD",
                            "is_fund": False,
                        }
                    )
                else:
                    errors.append(symbol)

        if not results:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=f"Could not fetch any quotes. Failed: {', '.join(errors)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Format portfolio summary
        formatted = self._format_portfolio(
            results, total_value_usd, total_value_gbp, errors
        )

        return LiveDataResult(
            source_name=self.name,
            success=True,
            data={
                "holdings": results,
                "total_usd": total_value_usd,
                "total_gbp": total_value_gbp,
                "errors": errors,
            },
            formatted=formatted,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _format_portfolio(
        self, results: list[dict], total_usd: float, total_gbp: float, errors: list[str]
    ) -> str:
        """Format portfolio data for context injection."""
        lines = ["### Stock Portfolio Summary"]
        lines.append(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n")

        # Separate stocks and funds for clearer display
        stocks = [r for r in results if not r.get("is_fund")]
        funds = [r for r in results if r.get("is_fund")]

        # Sort each by value descending
        stocks.sort(key=lambda x: x["value"], reverse=True)
        funds.sort(key=lambda x: x["value"], reverse=True)

        if stocks:
            lines.append("**Stocks (USD):**")
            for r in stocks:
                sign = "+" if r["change_pct"] >= 0 else ""
                symbol_display = r["symbol"] or r["name"]
                lines.append(
                    f"- **{symbol_display}** ({r['name']}): {r['shares']:.3f} shares @ ${r['price']:.2f} = ${r['value']:,.2f} ({sign}{r['change_pct']:.2f}%)"
                )

        if funds:
            lines.append("\n**Funds (GBP):**")
            for r in funds:
                sign = "+" if r["change_pct"] >= 0 else ""
                currency_symbol = "£" if r.get("currency") == "GBP" else "$"
                lines.append(
                    f"- **{r['name']}**: {r['shares']:.3f} units @ {currency_symbol}{r['price']:.4f} = {currency_symbol}{r['value']:,.2f} ({sign}{r['change_pct']:.2f}%)"
                )

        # Totals
        lines.append("")
        if total_usd > 0:
            lines.append(f"**Total Stocks (USD): ${total_usd:,.2f}**")
            # Approximate GBP conversion (rough rate)
            gbp_from_usd = total_usd * 0.79
            lines.append(f"_Approximate GBP: £{gbp_from_usd:,.2f}_")

        if total_gbp > 0:
            lines.append(f"**Total Funds (GBP): £{total_gbp:,.2f}**")

        if total_usd > 0 or total_gbp > 0:
            # Combined total in GBP (approximate)
            combined_gbp = (total_usd * 0.79) + total_gbp
            lines.append(f"\n**Combined Portfolio Value: ~£{combined_gbp:,.2f}**")

        if errors:
            lines.append(f"\n_Could not fetch: {', '.join(errors)}_")

        return "\n".join(lines)

    def _format_quote(self, data: dict, symbol: str) -> str:
        """Format stock quote for context injection."""
        current = data.get("c", 0)
        change = data.get("d", 0)
        change_pct = data.get("dp", 0)
        high = data.get("h", 0)
        low = data.get("l", 0)
        open_price = data.get("o", 0)
        prev_close = data.get("pc", 0)

        direction = "up" if change >= 0 else "down"
        sign = "+" if change >= 0 else ""

        return f"""### Stock: {symbol}
Current Price: ${current:.2f}
Change: {sign}${change:.2f} ({sign}{change_pct:.2f}%) - {direction}
Day Range: ${low:.2f} - ${high:.2f}
Open: ${open_price:.2f} | Previous Close: ${prev_close:.2f}
Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC"""

    def is_available(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "FINNHUB_API_KEY not set"
        result = self.fetch("AAPL", {})
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class AlphaVantageProvider(LiveDataProvider):
    """
    Stock provider using Alpha Vantage API.

    Supports:
    - US stocks (AAPL, MSFT, etc.)
    - UK stocks (TSCO.LON, etc.)
    - International stocks (various exchanges)
    - Historical data (daily, weekly, monthly)
    - UK funds via search (ISIN lookup)

    Requires ALPHA_VANTAGE_API_KEY environment variable.
    Free tier: 25 calls/day
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Map company names to symbols (including UK)
    COMPANY_TO_SYMBOL = {
        # US stocks
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "netflix": "NFLX",
        "amd": "AMD",
        "intel": "INTC",
        "unity": "U",
        "adobe": "ADBE",
        # UK stocks
        "tesco": "TSCO.LON",
        "lloyds": "LLOY.LON",
        "barclays": "BARC.LON",
        "bp": "BP.LON",
        "shell": "SHEL.LON",
        "hsbc": "HSBA.LON",
        "vodafone": "VOD.LON",
        "astrazeneca": "AZN.LON",
        "gsk": "GSK.LON",
        "glaxosmithkline": "GSK.LON",
        "unilever": "ULVR.LON",
        "diageo": "DGE.LON",
        "legal & general": "LGEN.LON",
        "legal and general": "LGEN.LON",
        "l&g": "LGEN.LON",
    }

    # Known UK funds that need web fallback (not available via stock APIs)
    # Maps search terms to fund display name for web search
    FUND_WEB_LOOKUP = {
        "l&g global technology": "L&G Global Technology Index Fund",
        "legal & general global technology": "L&G Global Technology Index Fund",
        "l&g global tech": "L&G Global Technology Index Fund",
        "0p000023mw": "L&G Global Technology Index Fund",  # Morningstar ticker
    }

    def __init__(self, source: Any = None):
        self.source = source
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.name = source.name if source else "alpha-vantage"
        self.cache_ttl = source.cache_ttl_seconds if source else 300  # 5 min default
        self.last_endpoint = None  # Track last endpoint called for stats

    def _get_fund_name(self, query: str) -> str | None:
        """Check if query matches a known fund that needs web lookup."""
        query_lower = query.lower()
        for term, fund_name in self.FUND_WEB_LOOKUP.items():
            if term in query_lower:
                return fund_name
        return None

    def _fetch_fund_via_web(self, fund_name: str) -> dict | None:
        """
        Fetch fund price via web search and scrape.

        Returns dict with keys: price, currency, change_pct, fund_name
        """
        cache_key = f"fund_web:{fund_name}"

        # Check cache first
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                logger.debug(f"Fund price cache hit for: {fund_name}")
                return cached

        try:
            from augmentation.scraper import WebScraper
            from augmentation.search.searxng import SearXNGProvider

            searcher = SearXNGProvider()
            scraper = WebScraper()

            if not searcher.is_configured():
                logger.warning("SearXNG not configured for fund price lookup")
                return None

            # Search for fund price
            search_query = f"{fund_name} price GBP"
            logger.info(f"Searching for fund price: {search_query}")

            results = searcher.search(search_query, max_results=3)
            if not results:
                logger.warning(f"No search results for fund: {fund_name}")
                return None

            # Try scraping each result until we find a price
            for result in results:
                try:
                    scrape_result = scraper.scrape(result.url)
                    if not scrape_result.success or not scrape_result.content:
                        continue

                    # Try to extract price from scraped content
                    price_data = self._extract_fund_price(
                        scrape_result.content, fund_name
                    )

                    if price_data and price_data.get("price", 0) > 0:
                        price_data["fund_name"] = fund_name
                        logger.info(
                            f"Found fund price for {fund_name}: "
                            f"{price_data.get('currency', '£')}{price_data['price']:.2f}"
                        )

                        # Cache result
                        if self.cache_ttl > 0:
                            self.set_cached(cache_key, price_data)

                        return price_data

                except Exception as e:
                    logger.debug(f"Error scraping {result.url}: {e}")
                    continue

            logger.warning(f"Could not extract price for fund: {fund_name}")
            return None

        except Exception as e:
            logger.error(f"Error fetching fund price via web: {e}")
            return None

    def _extract_fund_price(self, content: str, fund_name: str) -> dict | None:
        """Extract fund price from scraped web content."""
        # Common price patterns for UK funds
        patterns = [
            # Pence format: "123.45p" or "223.00 p"
            r"(?:price|nav|value|bid)[:\s]*(\d+(?:\.\d+)?)\s*p(?:ence)?",
            # Pound format with £ symbol
            r"[£]\s*(\d+(?:\.\d+)?)",
            # NAV specific
            r"nav[:\s]+[£]?(\d+(?:\.\d+)?)",
            # Bid price
            r"bid[:\s]+[£]?(\d+(?:\.\d+)?)\s*p?",
        ]

        content_lower = content.lower()

        for pattern in patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                try:
                    price = float(matches[0])
                    # If price is > 100, it's likely in pence
                    if price > 100:
                        price = price / 100  # Convert pence to pounds

                    # Sanity check - fund prices are typically 0.50 to 50
                    if 0.1 < price < 100:
                        return {
                            "price": price,
                            "currency": "GBP",
                            "change_pct": 0,
                        }
                except ValueError:
                    continue

        return None

    def _format_fund_quote(self, fund_data: dict) -> str:
        """Format fund data for context injection."""
        fund_name = fund_data.get("fund_name", "Unknown Fund")
        price = fund_data.get("price", 0)
        currency = "£" if fund_data.get("currency") == "GBP" else "$"

        return (
            f"### Fund: {fund_name}\n"
            f"Current Price: {currency}{price:.2f}\n"
            f"Currency: {fund_data.get('currency', 'GBP')}\n"
            f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Source: Web lookup"
        )

    def _extract_symbol(self, query: str) -> str:
        """Extract stock symbol from query."""
        query_lower = query.lower()

        # Check company name mapping first
        for company, symbol in self.COMPANY_TO_SYMBOL.items():
            if company in query_lower:
                return symbol

        # Look for Morningstar-style fund tickers first (e.g., 0P000023MW.L)
        # These start with 0P and have alphanumeric chars plus optional exchange suffix
        morningstar = re.findall(r"\b(0P[A-Z0-9]+(?:\.[A-Z]{1,4})?)\b", query.upper())
        if morningstar:
            return morningstar[0]

        # Look for ticker patterns (1-5 uppercase letters, optionally with .LON etc)
        symbols = re.findall(r"\b([A-Z]{1,5}(?:\.[A-Z]{2,4})?)\b", query.upper())
        if symbols:
            common_words = {
                "THE",
                "AND",
                "FOR",
                "OF",
                "TO",
                "IN",
                "IS",
                "IT",
                "A",
                "AN",
                "UK",
                "USA",
                "GBP",
                "USD",
            }
            symbols = [s for s in symbols if s not in common_words]
            if symbols:
                return symbols[0]

        return query.upper().strip()[:10]

    def _get_quote(self, symbol: str) -> dict | None:
        """Fetch current quote for a symbol."""
        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(
                    self.BASE_URL,
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": symbol,
                        "apikey": self.api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Check for API limit message
                if "Note" in data or "Information" in data:
                    logger.warning(
                        f"Alpha Vantage API limit: {data.get('Note') or data.get('Information')}"
                    )
                    return None

                quote = data.get("Global Quote", {})
                if not quote or "05. price" not in quote:
                    return None

                return {
                    "symbol": quote.get("01. symbol", symbol),
                    "price": float(quote.get("05. price", 0)),
                    "change": float(quote.get("09. change", 0)),
                    "change_pct": float(
                        quote.get("10. change percent", "0%").replace("%", "")
                    ),
                    "volume": int(quote.get("06. volume", 0)),
                    "prev_close": float(quote.get("08. previous close", 0)),
                }
        except Exception as e:
            logger.error(f"Alpha Vantage quote error for {symbol}: {e}")
            return None

    def _get_historical(self, symbol: str, period: str = "1M") -> list[dict] | None:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Stock symbol
            period: "1W" (week), "1M" (month), "3M", "6M", "1Y", or "5Y"

        Returns:
            List of {date, open, high, low, close, volume} dicts
        """
        # Determine function and outputsize based on period
        if period in ("1W", "1M"):
            function = "TIME_SERIES_DAILY"
            outputsize = "compact"  # Last 100 data points
        elif period in ("3M", "6M", "1Y"):
            function = "TIME_SERIES_DAILY"
            outputsize = "compact"
        else:  # 5Y+
            function = "TIME_SERIES_WEEKLY"
            outputsize = "compact"

        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(
                    self.BASE_URL,
                    params={
                        "function": function,
                        "symbol": symbol,
                        "outputsize": outputsize,
                        "apikey": self.api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Check for API limit
                if "Note" in data or "Information" in data:
                    logger.warning(
                        f"Alpha Vantage API limit: {data.get('Note') or data.get('Information')}"
                    )
                    return None

                # Get the time series data
                series_key = (
                    "Time Series (Daily)"
                    if "DAILY" in function
                    else "Weekly Time Series"
                )
                series = data.get(series_key, {})

                if not series:
                    logger.warning(
                        f"Alpha Vantage: No '{series_key}' in response for {symbol}. "
                        f"Keys: {list(data.keys())}"
                    )
                    return None

                # Convert to list and limit based on period
                period_days = {
                    "1W": 7,
                    "1M": 30,
                    "3M": 90,
                    "6M": 180,
                    "1Y": 365,
                    "5Y": 260,
                }
                limit = period_days.get(period, 30)

                results = []
                for date_str, values in sorted(series.items(), reverse=True)[:limit]:
                    results.append(
                        {
                            "date": date_str,
                            "open": float(values.get("1. open", 0)),
                            "high": float(values.get("2. high", 0)),
                            "low": float(values.get("3. low", 0)),
                            "close": float(values.get("4. close", 0)),
                            "volume": int(values.get("5. volume", 0)),
                        }
                    )

                return results

        except Exception as e:
            logger.error(f"Alpha Vantage historical error for {symbol}: {e}")
            return None

    def _search_symbol(self, keywords: str) -> str | None:
        """Search for a symbol by keywords (company name, ISIN, etc.)."""
        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(
                    self.BASE_URL,
                    params={
                        "function": "SYMBOL_SEARCH",
                        "keywords": keywords,
                        "apikey": self.api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

                matches = data.get("bestMatches", [])
                if matches:
                    # Return the best match symbol
                    return matches[0].get("1. symbol")

        except Exception as e:
            logger.error(f"Alpha Vantage search error: {e}")

        return None

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """Fetch stock data from Alpha Vantage."""
        start_time = time.time()
        self.last_endpoint = "GLOBAL_QUOTE"  # Default endpoint

        # Check if this is a known fund that needs web lookup
        fund_name = self._get_fund_name(query)
        if fund_name:
            logger.info(f"Detected fund query, using web lookup: {fund_name}")
            fund_data = self._fetch_fund_via_web(fund_name)
            if fund_data:
                formatted = self._format_fund_quote(fund_data)
                fund_data["_formatted"] = formatted
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=fund_data,
                    formatted=formatted,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            else:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"Could not find price for fund: {fund_name}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        if not self.api_key:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="ALPHA_VANTAGE_API_KEY not configured",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Check for historical query
        period = None
        if context and "period" in context:
            period = context["period"].upper()
        else:
            # Detect period from query
            query_lower = query.lower()
            if "week" in query_lower:
                period = "1W"
            elif "month" in query_lower:
                period = "1M"
            elif "3 month" in query_lower or "quarter" in query_lower:
                period = "3M"
            elif "6 month" in query_lower or "half year" in query_lower:
                period = "6M"
            elif "year" in query_lower:
                period = "1Y"

        # Get symbol from context or extract from query
        if context and "symbol" in context:
            symbol = context["symbol"].upper()
        else:
            symbol = self._extract_symbol(query)

        # Check cache
        cache_key = f"{symbol}:{period or 'quote'}"
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=cached.get("_formatted", ""),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Fetch data
        if period:
            # Historical query
            history = self._get_historical(symbol, period)
            if history:
                formatted = self._format_historical(symbol, history, period)
                result_data = {
                    "symbol": symbol,
                    "period": period,
                    "history": history,
                    "_formatted": formatted,
                }

                if self.cache_ttl > 0:
                    self.set_cached(cache_key, result_data)

                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=result_data,
                    formatted=formatted,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            else:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"No historical data for {symbol}",
                    latency_ms=(time.time() - start_time) * 1000,
                )
        else:
            # Current quote
            quote = self._get_quote(symbol)
            if quote:
                formatted = self._format_quote(quote)
                quote["_formatted"] = formatted

                if self.cache_ttl > 0:
                    self.set_cached(cache_key, quote)

                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=quote,
                    formatted=formatted,
                    latency_ms=(time.time() - start_time) * 1000,
                )
            else:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"No quote data for {symbol}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

    def _format_quote(self, quote: dict) -> str:
        """Format quote data for context injection."""
        symbol = quote.get("symbol", "Unknown")
        price = quote.get("price", 0)
        change = quote.get("change", 0)
        change_pct = quote.get("change_pct", 0)
        prev_close = quote.get("prev_close", 0)

        # Detect currency from symbol
        currency = "£" if ".LON" in symbol or ".L" in symbol else "$"
        direction = "up" if change >= 0 else "down"
        sign = "+" if change >= 0 else ""

        return f"""### Stock: {symbol}
Current Price: {currency}{price:.2f}
Change: {sign}{currency}{change:.2f} ({sign}{change_pct:.2f}%) - {direction}
Previous Close: {currency}{prev_close:.2f}
Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC"""

    def _format_historical(self, symbol: str, history: list[dict], period: str) -> str:
        """Format historical data for context injection."""
        if not history:
            return f"No historical data for {symbol}"

        currency = "£" if ".LON" in symbol or ".L" in symbol else "$"

        # Get start and end prices
        latest = history[0]
        oldest = history[-1]

        price_change = latest["close"] - oldest["close"]
        pct_change = (price_change / oldest["close"]) * 100 if oldest["close"] else 0

        high = max(d["high"] for d in history)
        low = min(d["low"] for d in history)

        sign = "+" if price_change >= 0 else ""
        direction = "increased" if price_change >= 0 else "decreased"

        period_names = {
            "1W": "week",
            "1M": "month",
            "3M": "3 months",
            "6M": "6 months",
            "1Y": "year",
        }
        period_name = period_names.get(period, period)

        return f"""### {symbol} - {period_name.title()} History
Period: {oldest["date"]} to {latest["date"]}
Start Price: {currency}{oldest["close"]:.2f}
Current Price: {currency}{latest["close"]:.2f}
Change: {sign}{currency}{price_change:.2f} ({sign}{pct_change:.2f}%) - {direction}
{period_name.title()} High: {currency}{high:.2f}
{period_name.title()} Low: {currency}{low:.2f}
Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC"""

    def is_available(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ALPHA_VANTAGE_API_KEY not set"
        result = self.fetch("AAPL", {})
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class WeatherProvider(LiveDataProvider):
    """
    Built-in weather provider using AccuWeather API.

    Requires ACCUWEATHER_API_KEY environment variable.
    """

    BASE_URL = "https://dataservice.accuweather.com"

    def __init__(self, source: Any = None):
        self.source = source
        self.api_key = os.environ.get("ACCUWEATHER_API_KEY", "")
        self.name = source.name if source else "weather"
        self.cache_ttl = source.cache_ttl_seconds if source else 300  # 5 min default
        self.last_endpoint = None  # Track last endpoint called for stats
        self._location_cache: dict[str, str] = {}

    def _extract_location(self, query: str) -> str:
        """Extract location from query."""
        # Remove common weather-related words
        query = query.lower()
        for word in [
            "weather",
            "temperature",
            "forecast",
            "in",
            "for",
            "what's",
            "the",
            "today",
            "now",
            "current",
        ]:
            query = query.replace(word, "")
        return query.strip() or "London"

    def _get_location_key(self, location: str) -> Optional[str]:
        """Get AccuWeather location key for a city."""
        if location in self._location_cache:
            return self._location_cache[location]

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    f"{self.BASE_URL}/locations/v1/cities/search",
                    params={"apikey": self.api_key, "q": location},
                )
                response.raise_for_status()
                locations = response.json()
                if locations:
                    key = locations[0]["Key"]
                    self._location_cache[location] = key
                    return key
        except Exception as e:
            logger.warning(f"Location lookup failed: {e}")
        return None

    def _resolve_placeholder_location(self, location: str, context: dict | None) -> str:
        """
        Resolve placeholder location values to actual locations.

        Handles cases where the designator returns placeholders like
        "user's home location" instead of the actual location name.
        """
        # List of common placeholder patterns
        placeholders = [
            "user's home",
            "users home",
            "home location",
            "user location",
            "my location",
            "current location",
            "default location",
        ]

        location_lower = location.lower()
        is_placeholder = any(p in location_lower for p in placeholders)

        if not is_placeholder:
            return location

        # Try to extract actual location from user context (intelligence)
        if context:
            # Check for user_intelligence passed from enricher
            intelligence = context.get("user_intelligence", "")
            if intelligence:
                # Look for location patterns like "lives in X" or "based in X"
                import re

                patterns = [
                    r"(?:live[s]? in|based in|from|home[: ]+|location[: ]+)\s*([A-Za-z\s]+?)(?:[,.\n]|$)",
                    r"([A-Za-z]+)\s+(?:is my home|is home)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, intelligence, re.IGNORECASE)
                    if match:
                        extracted = match.group(1).strip()
                        if extracted and len(extracted) > 2:
                            logger.info(
                                f"Resolved placeholder '{location}' to '{extracted}' from user intelligence"
                            )
                            return extracted

            # Check for purpose context
            purpose = context.get("purpose", "")
            if purpose:
                import re

                patterns = [
                    r"(?:live[s]? in|based in|from|home[: ]+|location[: ]+)\s*([A-Za-z\s]+?)(?:[,.\n]|$)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, purpose, re.IGNORECASE)
                    if match:
                        extracted = match.group(1).strip()
                        if extracted and len(extracted) > 2:
                            logger.info(
                                f"Resolved placeholder '{location}' to '{extracted}' from purpose"
                            )
                            return extracted

        # Default fallback
        logger.warning(
            f"Could not resolve placeholder location '{location}', defaulting to London"
        )
        return "London"

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """Fetch current weather from AccuWeather."""
        start_time = time.time()
        self.last_endpoint = "currentconditions"  # Default endpoint

        if not self.api_key:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="ACCUWEATHER_API_KEY not configured",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Use structured params from context if available, otherwise extract from query
        if context and "location" in context:
            location = context["location"]
            # Resolve placeholder values like "user's home location"
            location = self._resolve_placeholder_location(location, context)
        else:
            location = self._extract_location(query)

        # Check cache
        if self.cache_ttl > 0:
            cached = self.get_cached(location, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_weather(cached, location),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        try:
            # Get location key
            location_key = self._get_location_key(location)
            if not location_key:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"Location not found: {location}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            # Get current conditions
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    f"{self.BASE_URL}/currentconditions/v1/{location_key}",
                    params={"apikey": self.api_key, "details": "true"},
                )
                response.raise_for_status()
                conditions = response.json()

                if not conditions:
                    return LiveDataResult(
                        source_name=self.name,
                        success=False,
                        error="No weather data returned",
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                weather_data = conditions[0]

                # Cache result
                if self.cache_ttl > 0:
                    self.set_cached(location, weather_data)

                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=weather_data,
                    formatted=self._format_weather(weather_data, location),
                    latency_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _format_weather(self, data: dict, location: str) -> str:
        """Format weather data for context injection."""
        text = data.get("WeatherText", "Unknown")
        temp_c = data.get("Temperature", {}).get("Metric", {}).get("Value", 0)
        temp_f = data.get("Temperature", {}).get("Imperial", {}).get("Value", 0)
        humidity = data.get("RelativeHumidity", 0)
        wind_speed = (
            data.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value", 0)
        )
        wind_dir = data.get("Wind", {}).get("Direction", {}).get("English", "")
        feels_like_c = (
            data.get("RealFeelTemperature", {}).get("Metric", {}).get("Value", temp_c)
        )
        uv_index = data.get("UVIndex", 0)

        return f"""### Weather: {location.title()}
Conditions: {text}
Temperature: {temp_c:.1f}C ({temp_f:.1f}F)
Feels Like: {feels_like_c:.1f}C
Humidity: {humidity}%
Wind: {wind_speed} km/h {wind_dir}
UV Index: {uv_index}
Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC"""

    def is_available(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ACCUWEATHER_API_KEY not set"
        result = self.fetch("London", {})
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class TransportProvider(LiveDataProvider):
    """
    Built-in UK transport provider using TransportAPI.

    Requires TRANSPORTAPI_APP_KEY and TRANSPORTAPI_APP_ID environment variables.
    """

    BASE_URL = "https://transportapi.com/v3/uk"

    # Common UK station codes for quick lookup
    STATION_CODES = {
        "farringdon": "ZFD",
        "kings cross": "KGX",
        "king's cross": "KGX",
        "st pancras": "STP",
        "euston": "EUS",
        "paddington": "PAD",
        "victoria": "VIC",
        "waterloo": "WAT",
        "liverpool street": "LST",
        "london bridge": "LBG",
        "charing cross": "CHX",
        "marylebone": "MYB",
        "blackfriars": "BFR",
        "city thameslink": "CTK",
        "harpenden": "HPD",
        "st albans": "SAC",
        "luton": "LUT",
        "luton airport parkway": "LTN",
        "bedford": "BDM",
        "gatwick airport": "GTW",
        "heathrow": "HXX",
        "brighton": "BTN",
        "cambridge": "CBG",
        "oxford": "OXF",
        "reading": "RDG",
        "birmingham new street": "BHM",
        "manchester piccadilly": "MAN",
        "leeds": "LDS",
        "edinburgh": "EDB",
        "glasgow central": "GLC",
    }

    def __init__(self, source: Any = None):
        self.source = source
        self.app_key = os.environ.get("TRANSPORTAPI_APP_KEY", "")
        self.app_id = os.environ.get("TRANSPORTAPI_APP_ID", "")
        self.name = source.name if source else "transport"
        self.cache_ttl = source.cache_ttl_seconds if source else 60
        self.last_endpoint = None  # Track last endpoint called for stats
        self._station_cache: dict[str, str] = {}  # name -> code cache

    def _lookup_station_code(self, station_name: str) -> str | None:
        """Look up 3-letter CRS code for a station name."""
        name_lower = station_name.lower().strip()

        # Check hardcoded lookup first
        if name_lower in self.STATION_CODES:
            return self.STATION_CODES[name_lower]

        # Check ChromaDB semantic cache
        try:
            from live.param_cache import cache_parameter, lookup_parameter

            cached = lookup_parameter("transport", "station", station_name)
            if cached:
                logger.info(f"Station code cache hit: '{station_name}' -> '{cached}'")
                return cached
        except Exception as e:
            logger.debug(f"Param cache lookup failed: {e}")

        # Query TransportAPI places endpoint
        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                response = client.get(
                    f"{self.BASE_URL}/places.json",
                    params={
                        "app_id": self.app_id,
                        "app_key": self.app_key,
                        "query": station_name,
                        "type": "train_station",
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Find first train station match
                members = data.get("member", [])
                for member in members:
                    if member.get("type") == "train_station":
                        code = member.get("station_code")
                        if code:
                            # Cache in ChromaDB for semantic matching
                            try:
                                cache_parameter(
                                    "transport",
                                    "station",
                                    station_name,
                                    code,
                                    {"station_name": member.get("name", station_name)},
                                )
                            except Exception:
                                pass
                            logger.info(
                                f"Resolved station '{station_name}' to code '{code}'"
                            )
                            return code

        except Exception as e:
            logger.warning(f"Station lookup failed for '{station_name}': {e}")

        return None

    def _extract_station(self, query: str) -> str:
        """Extract station code or name from query."""
        # Common station code patterns
        query_upper = query.upper()
        # 3-letter station codes
        codes = re.findall(r"\b([A-Z]{3})\b", query_upper)
        if codes:
            return codes[0]
        # Otherwise use the query as station name search term
        return query.strip()

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """Fetch live train departures from TransportAPI."""
        start_time = time.time()
        self.last_endpoint = "live_departures"  # Default endpoint

        if not self.app_key or not self.app_id:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="TRANSPORTAPI_APP_KEY/APP_ID not configured",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Use structured params from context if available, otherwise extract from query
        if context and "station" in context:
            station_input = context["station"]
        else:
            station_input = self._extract_station(query)

        # Resolve to 3-letter code if needed
        if len(station_input) == 3 and station_input.isupper():
            station_code = station_input
        else:
            station_code = self._lookup_station_code(station_input)
            if not station_code:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"Could not find station code for: {station_input}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Check cache
        if self.cache_ttl > 0:
            cached = self.get_cached(station_code, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_departures(cached, station_code),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                response = client.get(
                    f"{self.BASE_URL}/train/station/{station_code}/live.json",
                    params={
                        "app_id": self.app_id,
                        "app_key": self.app_key,
                        "darwin": "false",
                        "train_status": "passenger",
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Cache result
                if self.cache_ttl > 0:
                    self.set_cached(station_code, data)

                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=data,
                    formatted=self._format_departures(data, station_code),
                    latency_ms=(time.time() - start_time) * 1000,
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"Station not found: {station_code}",
                    latency_ms=(time.time() - start_time) * 1000,
                )
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _format_departures(self, data: dict, station: str) -> str:
        """Format train departures for context injection."""
        station_name = data.get("station_name", station)
        departures = data.get("departures", {}).get("all", [])

        lines = [f"### Train Departures: {station_name}"]

        if not departures:
            lines.append("No departures found")
        else:
            for dep in departures[:5]:  # Max 5 departures
                time_str = dep.get("aimed_departure_time", "??:??")
                dest = dep.get("destination_name", "Unknown")
                platform = dep.get("platform", "?")
                status = dep.get("status", "On time")
                operator = dep.get("operator_name", "")

                lines.append(f"- {time_str} to {dest} (Plat {platform}) - {status}")
                if operator:
                    lines[-1] += f" [{operator}]"

        lines.append(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        return "\n".join(lines)

    def is_available(self) -> bool:
        return bool(self.app_key and self.app_id)

    def test_connection(self) -> tuple[bool, str]:
        if not self.app_key or not self.app_id:
            return False, "TRANSPORTAPI credentials not set"
        result = self.fetch("KGX", {})  # King's Cross
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class OpenMeteoProvider(LiveDataProvider):
    """
    Built-in weather provider using Open-Meteo API.

    Completely free, no API key required.
    https://open-meteo.com/
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

    # WMO Weather interpretation codes
    WMO_CODES = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    def __init__(self, source: Any = None):
        self.source = source
        self.name = source.name if source else "open-meteo"
        self.cache_ttl = source.cache_ttl_seconds if source else 600  # 10 min default
        self.last_endpoint = None
        self._geocode_cache: dict[str, tuple[float, float, str]] = {}

    def _extract_location(self, query: str) -> str:
        """Extract location from query."""
        query = query.lower()
        for word in [
            "weather",
            "temperature",
            "forecast",
            "in",
            "for",
            "what's",
            "the",
            "today",
            "now",
            "current",
            "tomorrow",
        ]:
            query = query.replace(word, "")
        return query.strip() or "London"

    def _geocode(self, location: str) -> tuple[float, float, str] | None:
        """
        Geocode a location name to coordinates.

        Returns (latitude, longitude, display_name) or None if not found.
        """
        if location in self._geocode_cache:
            return self._geocode_cache[location]

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    self.GEOCODING_URL,
                    params={"name": location, "count": 1, "language": "en"},
                )
                response.raise_for_status()
                data = response.json()

                if data.get("results"):
                    result = data["results"][0]
                    coords = (
                        result["latitude"],
                        result["longitude"],
                        f"{result.get('name', location)}, {result.get('country', '')}".strip(
                            ", "
                        ),
                    )
                    self._geocode_cache[location] = coords
                    return coords
        except Exception as e:
            logger.warning(f"Open-Meteo geocoding failed for '{location}': {e}")

        return None

    def _get_weather_description(self, code: int) -> str:
        """Convert WMO weather code to human-readable description."""
        return self.WMO_CODES.get(code, f"Unknown ({code})")

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """Fetch current weather and forecast from Open-Meteo."""
        start_time = time.time()
        self.last_endpoint = "forecast"

        # Get location from context or query
        if context and "location" in context:
            location = context["location"]
        else:
            location = self._extract_location(query)

        # Handle placeholder locations
        if "user" in location.lower() and "home" in location.lower():
            location = "London"  # Default fallback

        # Check cache
        cache_key = f"openmeteo:{location.lower()}"
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached["data"],
                    formatted=cached["formatted"],
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Geocode the location
        coords = self._geocode(location)
        if not coords:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=f"Location not found: {location}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        lat, lon, display_name = coords

        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(
                    self.BASE_URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_direction_10m",
                        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                        "timezone": "auto",
                        "forecast_days": 3,
                    },
                )
                response.raise_for_status()
                data = response.json()

            # Extract current conditions
            current = data.get("current", {})
            daily = data.get("daily", {})

            weather_data = {
                "location": display_name,
                "coordinates": {"lat": lat, "lon": lon},
                "current": {
                    "temperature_c": current.get("temperature_2m"),
                    "feels_like_c": current.get("apparent_temperature"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_speed_kmh": current.get("wind_speed_10m"),
                    "wind_direction": current.get("wind_direction_10m"),
                    "condition": self._get_weather_description(
                        current.get("weather_code", 0)
                    ),
                    "weather_code": current.get("weather_code"),
                },
                "forecast": [],
            }

            # Add forecast days
            if daily.get("time"):
                for i, date in enumerate(daily["time"]):
                    weather_data["forecast"].append(
                        {
                            "date": date,
                            "condition": self._get_weather_description(
                                daily.get("weather_code", [0])[i]
                                if daily.get("weather_code")
                                else 0
                            ),
                            "high_c": daily.get("temperature_2m_max", [None])[i],
                            "low_c": daily.get("temperature_2m_min", [None])[i],
                            "precipitation_chance": daily.get(
                                "precipitation_probability_max", [None]
                            )[i],
                        }
                    )

            # Format output
            formatted = self._format_weather(weather_data)

            # Cache the result
            if self.cache_ttl > 0:
                self.set_cached(
                    cache_key, {"data": weather_data, "formatted": formatted}
                )

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=weather_data,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except httpx.HTTPStatusError as e:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=f"API error: {e.response.status_code}",
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _format_weather(self, data: dict) -> str:
        """Format weather data for LLM context."""
        lines = [f"### Weather in {data['location']}"]

        current = data.get("current", {})
        if current:
            lines.append(f"\n**Current Conditions:**")
            lines.append(
                f"- Temperature: {current.get('temperature_c')}°C (feels like {current.get('feels_like_c')}°C)"
            )
            lines.append(f"- Condition: {current.get('condition')}")
            lines.append(f"- Humidity: {current.get('humidity')}%")
            lines.append(f"- Wind: {current.get('wind_speed_kmh')} km/h")

        forecast = data.get("forecast", [])
        if forecast:
            lines.append(f"\n**3-Day Forecast:**")
            for day in forecast:
                precip = day.get("precipitation_chance")
                precip_str = f", {precip}% chance of rain" if precip else ""
                lines.append(
                    f"- {day['date']}: {day['condition']}, {day.get('low_c')}°C - {day.get('high_c')}°C{precip_str}"
                )

        return "\n".join(lines)

    def is_available(self) -> bool:
        """Open-Meteo is always available (no API key required)."""
        return True

    def test_connection(self) -> tuple[bool, str]:
        """Test connection to Open-Meteo."""
        result = self.fetch("London", {})
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class GoogleOAuthLiveProvider(LiveDataProvider):
    """
    Base class for Google OAuth-based live data providers.

    Handles OAuth token retrieval and refresh for Calendar, Tasks, and Gmail.
    Subclasses implement specific API calls.
    """

    # Subclasses should override this for token validation
    TOKEN_TEST_URL = "https://www.googleapis.com/oauth2/v1/tokeninfo"

    def __init__(self, source: Any = None):
        self.source = source
        self.name = source.name if source else "google_oauth"
        self.cache_ttl = source.cache_ttl_seconds if source else 60
        self.last_endpoint = None
        self._access_token: str | None = None
        self._token_data: dict | None = None

        # Get account_id from auth_config
        auth_config = {}
        if source and source.auth_config_json:
            try:
                auth_config = json.loads(source.auth_config_json)
            except json.JSONDecodeError:
                pass

        self.account_id = auth_config.get("account_id")

    def _get_token_data(self) -> dict | None:
        """Get and cache the decrypted token data."""
        if self._token_data is None and self.account_id:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> str | None:
        """Get a valid access token, refreshing if necessary."""
        import requests as http_requests

        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"OAuth token {self.account_id} not found")
            return None

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            logger.error("No access token in stored credentials")
            return None

        # Try the access token first with token info endpoint
        test_response = http_requests.get(
            self.TOKEN_TEST_URL,
            params={"access_token": access_token},
            timeout=10,
        )

        if test_response.status_code == 200:
            return access_token

        # Token expired, try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error(
                "Cannot refresh token - missing refresh_token or client credentials"
            )
            return None

        logger.info("Access token expired, refreshing...")
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
        update_oauth_token_data(self.account_id, updated_data)

        self._token_data = updated_data
        logger.info("Token refreshed successfully")
        return new_access_token

    def is_available(self) -> bool:
        """Check if we have a valid OAuth account configured."""
        if not self.account_id:
            return False
        return self._get_access_token() is not None

    def test_connection(self) -> tuple[bool, str]:
        """Test OAuth connection."""
        if not self.account_id:
            return False, "No Google account configured (set account_id in auth_config)"
        token = self._get_access_token()
        if token:
            return True, "Connected to Google account"
        return False, "Failed to get valid access token"


class GoogleCalendarLiveProvider(GoogleOAuthLiveProvider):
    """
    Live Google Calendar provider for real-time calendar queries.

    Use cases:
    - "What's my next meeting?"
    - "What's on my calendar today?"
    - "Am I free at 3pm?"
    - "What meetings do I have tomorrow?"
    """

    TOKEN_TEST_URL = (
        "https://www.googleapis.com/calendar/v3/users/me/calendarList?maxResults=1"
    )

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch calendar events.

        Context params:
        - action: "today", "tomorrow", "next", "week", "search"
        - calendar_id: Specific calendar (default: primary)
        - query: Search term for "search" action
        """
        start_time = time.time()

        access_token = self._get_access_token()
        if not access_token:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No valid Google OAuth token",
                latency_ms=(time.time() - start_time) * 1000,
            )

        action = "today"
        calendar_id = "primary"
        search_query = None

        if context:
            action = context.get("action", "today")
            calendar_id = context.get("calendar_id", "primary")
            search_query = context.get("query")

        self.last_endpoint = f"calendar:{action}"

        try:
            events = self._fetch_events(access_token, action, calendar_id, search_query)
            formatted = self._format_events(events, action)

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=events,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Calendar API error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _fetch_events(
        self,
        access_token: str,
        action: str,
        calendar_id: str,
        search_query: str | None,
    ) -> list[dict]:
        """Fetch events based on action type."""
        from datetime import datetime, timedelta

        import requests as http_requests

        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Determine time range based on action
        if action == "next":
            time_min = now
            time_max = now + timedelta(days=7)
            max_results = 1
        elif action == "today":
            time_min = today_start
            time_max = today_start + timedelta(days=1)
            max_results = 20
        elif action == "tomorrow":
            time_min = today_start + timedelta(days=1)
            time_max = today_start + timedelta(days=2)
            max_results = 20
        elif action == "week":
            time_min = today_start
            time_max = today_start + timedelta(days=7)
            max_results = 50
        elif action == "search":
            time_min = now - timedelta(days=30)
            time_max = now + timedelta(days=90)
            max_results = 20
        else:
            # Default to today
            time_min = today_start
            time_max = today_start + timedelta(days=1)
            max_results = 20

        params = {
            "timeMin": time_min.isoformat() + "Z",
            "timeMax": time_max.isoformat() + "Z",
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        if search_query:
            params["q"] = search_query

        response = http_requests.get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=15,
        )
        response.raise_for_status()

        return response.json().get("items", [])

    def _format_events(self, events: list[dict], action: str) -> str:
        """Format calendar events for LLM context."""
        if not events:
            if action == "next":
                return "### Next Meeting\nNo upcoming meetings scheduled."
            elif action == "today":
                return "### Today's Calendar\nNo events scheduled for today."
            elif action == "tomorrow":
                return "### Tomorrow's Calendar\nNo events scheduled for tomorrow."
            return f"### Calendar ({action})\nNo events found."

        action_titles = {
            "next": "Next Meeting",
            "today": "Today's Calendar",
            "tomorrow": "Tomorrow's Calendar",
            "week": "This Week's Calendar",
            "search": "Calendar Search Results",
        }

        lines = [f"### {action_titles.get(action, 'Calendar Events')}"]
        lines.append(f"Found {len(events)} event(s):\n")

        for event in events:
            summary = event.get("summary", "(No title)")
            start = event.get("start", {})
            end = event.get("end", {})
            location = event.get("location", "")
            description = event.get("description", "")

            # Parse start time
            start_dt = start.get("dateTime", start.get("date", ""))
            end_dt = end.get("dateTime", end.get("date", ""))

            # Format time nicely
            if "T" in start_dt:
                # Has time component
                from datetime import datetime

                try:
                    dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
                    time_str = dt.strftime("%a %b %d, %I:%M %p")
                except ValueError:
                    time_str = start_dt
            else:
                # All-day event
                time_str = f"{start_dt} (All day)"

            lines.append(f"**{summary}**")
            lines.append(f"  When: {time_str}")

            if location:
                lines.append(f"  Where: {location}")

            if description and len(description) < 200:
                lines.append(f"  Notes: {description}")
            elif description:
                lines.append(f"  Notes: {description[:200]}...")

            # Check for video conference
            conference = event.get("conferenceData", {})
            if conference:
                entry_points = conference.get("entryPoints", [])
                for ep in entry_points:
                    if ep.get("entryPointType") == "video":
                        lines.append(f"  Video: {ep.get('uri', '')}")
                        break

            lines.append("")

        return "\n".join(lines)

    def test_connection(self) -> tuple[bool, str]:
        """Test calendar connection."""
        if not self.account_id:
            return False, "No Google account configured"

        result = self.fetch("test", {"action": "next"})
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class GoogleTasksLiveProvider(GoogleOAuthLiveProvider):
    """
    Live Google Tasks provider for real-time task queries.

    Use cases:
    - "What tasks do I have?"
    - "What's due today?"
    - "Show my task list"
    """

    TOKEN_TEST_URL = (
        "https://tasks.googleapis.com/tasks/v1/users/@me/lists?maxResults=1"
    )

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch tasks.

        Context params:
        - action: "all", "due_today", "due_week", "overdue"
        - tasklist_id: Specific task list (default: @default)
        - show_completed: Include completed tasks (default: false)
        """
        start_time = time.time()

        access_token = self._get_access_token()
        if not access_token:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No valid Google OAuth token",
                latency_ms=(time.time() - start_time) * 1000,
            )

        action = "all"
        tasklist_id = "@default"
        show_completed = False

        if context:
            action = context.get("action", "all")
            tasklist_id = context.get("tasklist_id", "@default")
            show_completed = context.get("show_completed", False)

        self.last_endpoint = f"tasks:{action}"

        try:
            tasks = self._fetch_tasks(access_token, action, tasklist_id, show_completed)
            formatted = self._format_tasks(tasks, action)

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=tasks,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Tasks API error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _fetch_tasks(
        self,
        access_token: str,
        action: str,
        tasklist_id: str,
        show_completed: bool,
    ) -> list[dict]:
        """Fetch tasks based on action type."""
        from datetime import datetime, timedelta

        import requests as http_requests

        params = {
            "maxResults": 100,
            "showCompleted": str(show_completed).lower(),
            "showHidden": "false",
        }

        # Filter by due date for specific actions
        now = datetime.utcnow()
        today_end = now.replace(hour=23, minute=59, second=59)

        if action == "due_today":
            params["dueMax"] = today_end.isoformat() + "Z"
        elif action == "due_week":
            week_end = today_end + timedelta(days=7)
            params["dueMax"] = week_end.isoformat() + "Z"

        response = http_requests.get(
            f"https://tasks.googleapis.com/tasks/v1/lists/{tasklist_id}/tasks",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=15,
        )
        response.raise_for_status()

        tasks = response.json().get("items", [])

        # Filter for overdue if requested
        if action == "overdue":
            tasks = [
                t
                for t in tasks
                if t.get("due")
                and t.get("due") < now.isoformat() + "Z"
                and t.get("status") != "completed"
            ]

        return tasks

    def _format_tasks(self, tasks: list[dict], action: str) -> str:
        """Format tasks for LLM context."""
        if not tasks:
            return f"### Tasks ({action})\nNo tasks found."

        action_titles = {
            "all": "All Tasks",
            "due_today": "Tasks Due Today",
            "due_week": "Tasks Due This Week",
            "overdue": "Overdue Tasks",
        }

        lines = [f"### {action_titles.get(action, 'Tasks')}"]
        lines.append(f"Found {len(tasks)} task(s):\n")

        for task in tasks:
            title = task.get("title", "(No title)")
            status = task.get("status", "needsAction")
            due = task.get("due", "")
            notes = task.get("notes", "")

            # Status indicator
            status_icon = "[ ]" if status == "needsAction" else "[x]"

            lines.append(f"{status_icon} **{title}**")

            if due:
                # Parse and format due date
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                    due_str = dt.strftime("%a %b %d, %Y")
                    lines.append(f"    Due: {due_str}")
                except ValueError:
                    lines.append(f"    Due: {due}")

            if notes:
                if len(notes) < 100:
                    lines.append(f"    Notes: {notes}")
                else:
                    lines.append(f"    Notes: {notes[:100]}...")

            lines.append("")

        return "\n".join(lines)

    def test_connection(self) -> tuple[bool, str]:
        """Test tasks connection."""
        if not self.account_id:
            return False, "No Google account configured"

        result = self.fetch("test", {"action": "all"})
        if result.success:
            count = len(result.data) if result.data else 0
            return True, f"Connected. {count} task(s) ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class GmailLiveProvider(GoogleOAuthLiveProvider):
    """
    Live Gmail provider for real-time email queries.

    Use cases:
    - "Do I have any new emails?"
    - "What emails came in today?"
    - "Any unread messages?"
    - "Emails from John"
    """

    TOKEN_TEST_URL = "https://www.googleapis.com/gmail/v1/users/me/profile"

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch emails.

        Context params:
        - action: "unread", "today", "recent", "search"
        - query: Gmail search query for "search" action
        - max_results: Max emails to return (default: 10)
        - agentic: True to use multi-step agentic mode
        - goal: Goal for agentic mode (defaults to query)
        - session_key: Session identifier for session-scoped caching
        """
        if context is None:
            context = {}

        # Store session key for session-scoped caching
        self._session_key = context.get("session_key")

        # Check if agentic mode is requested
        if context.get("agentic") or context.get("_agentic"):
            goal = context.get("goal", query)
            designator_model = context.get("designator_model")
            max_iterations = context.get("max_iterations", 5)
            return self.fetch_agentic(
                goal=goal,
                context=context,
                designator_model=designator_model,
                max_iterations=max_iterations,
            )

        start_time = time.time()

        access_token = self._get_access_token()
        if not access_token:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No valid Google OAuth token",
                latency_ms=(time.time() - start_time) * 1000,
            )

        action = "recent"
        search_query = None

        action = context.get("action", "recent")
        search_query = context.get("query")
        # No artificial limit - let the query filters (date, sender, etc.) constrain results
        # Gmail API max is 500, which is effectively "all" for reasonable queries
        max_results = context.get("max_results", 500)

        # Support alternative "search" key from designator
        # Designator may pass {"search": "gmail query"} instead of {"action": "search", "query": "..."}
        if "search" in context:
            action = "search"
            search_query = context.get("search")

        self.last_endpoint = f"gmail:{action}"

        # For search queries, check entity cache first
        # This allows finding emails by sender/subject without re-querying Gmail
        if action == "search" and search_query:
            logger.info(
                f"Gmail {self.name}: Checking cache for query: {search_query[:100]}"
            )
            cached = self._lookup_cached_email(search_query)
            if cached:
                logger.info(
                    f"Gmail {self.name}: Cache hit for '{search_query}' -> {cached.get('message_id')}"
                )
                # Return the cached email info as a single result
                email_data = {
                    "id": cached.get("message_id"),
                    "thread_id": cached.get("thread_id"),
                    "from": cached.get("from", ""),
                    "subject": cached.get("subject", ""),
                    "date": cached.get("date", ""),
                    "snippet": "",
                    "labels": [],
                    "account_email": cached.get("account_email"),
                }
                formatted = self._format_emails([email_data], action)
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=[email_data],
                    formatted=formatted,
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        try:
            emails = self._fetch_emails(access_token, action, search_query, max_results)
            formatted = self._format_emails(emails, action)

            logger.info(
                f"Gmail {self.name}: action={action}, found {len(emails)} emails"
            )
            if emails:
                # Log subjects to debug inconsistent results
                subjects = [e.get("subject", "")[:40] for e in emails]
                logger.info(f"Gmail {self.name} subjects: {subjects}")
                logger.debug(
                    f"Gmail results: {[e.get('subject', '')[:50] for e in emails]}"
                )

                # Store formatted email list in session context for LLM reference
                # This allows the LLM to reference emails on subsequent turns
                session_key = getattr(self, "_session_key", None)
                if session_key:
                    source_name = f"gmail:{self._get_account_email()}"
                    store_session_live_context(session_key, source_name, formatted)

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=emails,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Gmail API error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _fetch_emails(
        self,
        access_token: str,
        action: str,
        search_query: str | None,
        max_results: int,
    ) -> list[dict]:
        """Fetch emails based on action type."""
        from datetime import datetime, timedelta

        import requests as http_requests

        # Build Gmail search query
        if action == "unread":
            q = "is:unread"
        elif action == "today":
            today = datetime.utcnow().strftime("%Y/%m/%d")
            q = f"after:{today}"
        elif action == "search" and search_query:
            # Fix invalid Gmail date syntax like "after:today" -> "after:2026/01/18"
            q = self._fix_gmail_temporal_syntax(search_query)
        else:
            # Recent - last 7 days
            week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y/%m/%d")
            q = f"after:{week_ago}"

        # Get message list
        logger.info(f"Gmail search: q='{q}', max_results={max_results}")
        response = http_requests.get(
            "https://www.googleapis.com/gmail/v1/users/me/messages",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"q": q, "maxResults": max_results},
            timeout=15,
        )
        response.raise_for_status()

        messages = response.json().get("messages", [])
        logger.info(f"Gmail search returned {len(messages)} messages")

        # Fetch details for each message
        emails = []
        for msg in messages[:max_results]:
            msg_response = http_requests.get(
                f"https://www.googleapis.com/gmail/v1/users/me/messages/{msg['id']}",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date"],
                },
                timeout=15,
            )
            if msg_response.status_code == 200:
                msg_data = msg_response.json()
                emails.append(self._parse_email(msg_data))

        return emails

    def _parse_email(self, msg_data: dict) -> dict:
        """Parse email metadata into a clean dict."""
        headers = msg_data.get("payload", {}).get("headers", [])
        header_dict = {h["name"].lower(): h["value"] for h in headers}

        email_data = {
            "id": msg_data.get("id"),
            "thread_id": msg_data.get("threadId"),
            "from": header_dict.get("from", ""),
            "subject": header_dict.get("subject", "(No subject)"),
            "date": header_dict.get("date", ""),
            "snippet": msg_data.get("snippet", ""),
            "labels": msg_data.get("labelIds", []),
        }

        # Cache this email for future lookups by sender/subject
        self._cache_email_entity(email_data)

        return email_data

    def _cache_email_entity(self, email_data: dict) -> None:
        """
        Cache email entity for quick lookup by sender/subject.

        Uses session-scoped in-memory cache when session_key is available.
        This ensures:
        1. Emails from one session don't leak to other sessions
        2. Most recently viewed emails are returned when there are duplicates
        """
        try:
            account_email = self._get_account_email()
            msg_id = email_data.get("id")
            sender = email_data.get("from", "")
            subject = email_data.get("subject", "")
            session_key = getattr(self, "_session_key", None)

            if not msg_id:
                return

            # Extract sender name/email for cache key
            # "John Doe <john@example.com>" -> "john@example.com"
            sender_email = sender
            if "<" in sender and ">" in sender:
                sender_email = sender.split("<")[1].split(">")[0]
            sender_name = sender.split("<")[0].strip().strip('"')

            # Build email metadata
            metadata = {
                "message_id": msg_id,
                "account_email": account_email,
                "thread_id": email_data.get("thread_id"),
                "subject": subject,
                "from": sender,
                "date": email_data.get("date"),
            }

            # Use session-scoped in-memory cache
            self._add_to_session_cache(
                session_key, account_email, "sender", sender_email.lower(), metadata
            )

            # Also cache by sender name (if different from email)
            if sender_name and sender_name.lower() != sender_email.lower():
                self._add_to_session_cache(
                    session_key, account_email, "sender", sender_name.lower(), metadata
                )

            # Cache by subject (normalized)
            if subject:
                clean_subject = subject.lower()
                for prefix in ["re:", "fwd:", "fw:"]:
                    while clean_subject.startswith(prefix):
                        clean_subject = clean_subject[len(prefix) :].strip()
                if clean_subject:
                    self._add_to_session_cache(
                        session_key,
                        account_email,
                        "subject",
                        clean_subject[:50],
                        metadata,
                    )

        except Exception as e:
            logger.warning(f"Failed to cache email entity: {e}")

    def _add_to_session_cache(
        self,
        session_key: str | None,
        account_email: str,
        entity_type: str,
        query_text: str,
        metadata: dict,
    ) -> None:
        """
        Add an email to the session-scoped cache.

        Each cache key stores a list of (metadata, timestamp) tuples,
        allowing us to return the most recently viewed email when there
        are duplicates (e.g., recurring emails with same subject).
        """
        # Build cache key: session:account:type:query
        # If no session_key, use account-only scope (less ideal but functional)
        if session_key:
            cache_key = f"{session_key}:{account_email}:{entity_type}:{query_text}"
        else:
            cache_key = f"nosession:{account_email}:{entity_type}:{query_text}"

        now = time.time()

        # Get or create the list for this key
        if cache_key not in _session_email_cache:
            _session_email_cache[cache_key] = []

        entries = _session_email_cache[cache_key]

        # Check if this message_id is already cached - if so, update timestamp
        msg_id = metadata.get("message_id")
        for i, (existing_meta, _) in enumerate(entries):
            if existing_meta.get("message_id") == msg_id:
                # Update timestamp to mark as most recently viewed
                entries[i] = (metadata, now)
                # Re-sort by timestamp (most recent first)
                entries.sort(key=lambda x: x[1], reverse=True)
                return

        # Add new entry at the front (most recent)
        entries.insert(0, (metadata, now))

        # Limit entries per key to prevent unbounded growth
        if len(entries) > 50:
            entries[:] = entries[:50]

        # Clean up old session caches periodically (lazy cleanup)
        self._cleanup_old_session_caches()

    def _fix_gmail_temporal_syntax(self, query: str) -> str:
        """
        Fix invalid Gmail date syntax like 'after:today' -> 'after:2026/01/18'.

        Gmail requires actual dates (YYYY/MM/DD), not temporal words.
        """
        from datetime import datetime, timedelta

        today = datetime.now().date()

        # Map temporal words to actual dates
        temporal_to_date = {
            "today": today,
            "yesterday": today - timedelta(days=1),
            "tomorrow": today + timedelta(days=1),
        }

        result = query
        for word, date in temporal_to_date.items():
            date_str = date.strftime("%Y/%m/%d")
            # Replace after:word, before:word patterns
            result = re.sub(
                rf"(after:|before:){word}\b",
                rf"\g<1>{date_str}",
                result,
                flags=re.IGNORECASE,
            )

        if result != query:
            logger.info(f"Gmail date fix: '{query}' -> '{result}'")

        return result

    def _cleanup_old_session_caches(self) -> None:
        """
        Clean up expired session cache entries.

        Called lazily during cache operations to prevent unbounded growth.
        """
        now = time.time()
        keys_to_delete = []

        for key, entries in _session_email_cache.items():
            if not entries:
                keys_to_delete.append(key)
                continue

            # Check if the most recent entry is expired
            _, most_recent_time = entries[0]
            if now - most_recent_time > SESSION_EMAIL_CACHE_TTL:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del _session_email_cache[key]

        if keys_to_delete:
            logger.debug(
                f"Cleaned up {len(keys_to_delete)} expired session cache entries"
            )

    def _lookup_from_session_cache(
        self,
        session_key: str | None,
        account_email: str,
        entity_type: str,
        query_text: str,
    ) -> dict | None:
        """
        Look up an email from the session-scoped cache.

        Returns the most recently viewed email matching the query.
        """
        # Build cache key matching _add_to_session_cache
        if session_key:
            cache_key = f"{session_key}:{account_email}:{entity_type}:{query_text}"
        else:
            cache_key = f"nosession:{account_email}:{entity_type}:{query_text}"

        entries = _session_email_cache.get(cache_key)
        if not entries:
            return None

        # Return the most recent entry (first in list)
        metadata, timestamp = entries[0]

        # Check if expired
        if time.time() - timestamp > SESSION_EMAIL_CACHE_TTL:
            return None

        return metadata

    def _lookup_cached_email(self, query: str) -> dict | None:
        """
        Look up a cached email by sender name, sender email, or subject.

        Uses session-scoped in-memory cache to:
        1. Prevent email matching from leaking across sessions
        2. Return the most recently viewed email when there are duplicates

        Handles both plain text queries and Gmail search syntax:
        - Plain: "The Hub on Verulam"
        - Gmail: 'from:"The Hub on Verulam" subject:Valentine'

        Returns dict with message_id, account_email, and other metadata if found.
        """
        try:
            query_lower = query.lower().strip()

            # Get session context
            session_key = getattr(self, "_session_key", None)
            account_email = self._get_account_email()

            # Extract sender and subject from Gmail query syntax if present
            sender_queries = []
            subject_queries = []

            # Parse from: field - handle both quoted and unquoted values
            from_quoted = re.findall(r'from:"([^"]+)"', query_lower)
            from_quoted += re.findall(r"from:'([^']+)'", query_lower)
            from_unquoted = re.findall(r'from:([^\s"\']+?)(?:\s|$)', query_lower)

            for match in from_quoted + from_unquoted:
                cleaned = match.strip()
                if cleaned and cleaned.lower() != "or":
                    sender_queries.append(cleaned)

            # Parse subject: field - same approach
            subject_quoted = re.findall(r'subject:"([^"]+)"', query_lower)
            subject_quoted += re.findall(r"subject:'([^']+)'", query_lower)
            subject_unquoted = re.findall(r'subject:([^\s"\']+?)(?:\s|$)', query_lower)

            for match in subject_quoted + subject_unquoted:
                cleaned = match.strip()
                if cleaned and cleaned.lower() != "or":
                    subject_queries.append(cleaned)

            # Only use cache if we found from: or subject: in the query
            if not sender_queries and not subject_queries:
                logger.debug(f"Gmail cache lookup skipped - no from:/subject: in query")
                return None

            logger.debug(
                f"Gmail session cache lookup: senders={sender_queries}, subjects={subject_queries}, "
                f"session={session_key[:8] if session_key else 'none'}"
            )

            # Try lookup by sender(s) - returns most recently viewed
            for sender_q in sender_queries:
                result = self._lookup_from_session_cache(
                    session_key, account_email, "sender", sender_q
                )
                if result:
                    logger.debug(
                        f"Session cache hit by sender: '{sender_q}' -> {result.get('message_id')}"
                    )
                    return result

            # Also try subject queries as sender lookups (designator sometimes confuses them)
            for subj_q in subject_queries:
                result = self._lookup_from_session_cache(
                    session_key, account_email, "sender", subj_q
                )
                if result:
                    logger.debug(
                        f"Session cache hit by sender (from subject query): '{subj_q}' -> {result.get('message_id')}"
                    )
                    return result

            # Try lookup by subject(s)
            for subject_q in subject_queries:
                result = self._lookup_from_session_cache(
                    session_key, account_email, "subject", subject_q[:50]
                )
                if result:
                    logger.debug(
                        f"Session cache hit by subject: '{subject_q}' -> {result.get('message_id')}"
                    )
                    return result

            return None

        except Exception as e:
            logger.debug(f"Failed to lookup cached email: {e}")
            return None

    def _get_account_email(self) -> str:
        """Get the email address for this Gmail account."""
        if self.account_id:
            from db.oauth_tokens import get_oauth_token_info

            token_info = get_oauth_token_info(self.account_id)
            if token_info:
                return token_info.get("account_email", "")
        return ""

    def _format_emails(self, emails: list[dict], action: str) -> str:
        """Format emails for LLM context."""
        account_email = self._get_account_email()

        if not emails:
            if action == "unread":
                return "### Unread Emails\nNo unread emails."
            elif action == "today":
                return "### Today's Emails\nNo emails received today."
            return f"### Emails ({action})\nNo emails found."

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

            # Check if unread
            is_unread = "UNREAD" in labels
            unread_marker = " (UNREAD)" if is_unread else ""

            lines.append(f"**{subject}**{unread_marker}")
            lines.append(f"  ID: {msg_id}")
            lines.append(f"  From: {sender}")
            lines.append(f"  Date: {date}")

            if snippet:
                # Clean up snippet
                snippet = snippet.replace("\n", " ").strip()
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                lines.append(f"  Preview: {snippet}")

            lines.append("")

        return "\n".join(lines)

    def list_tools(self) -> list[dict]:
        """List available Gmail tools for agentic mode."""
        return [
            {
                "name": "lookup_cached",
                "description": "Look up a recently seen email by sender name or subject. Fast - uses cache from previous queries. Try this FIRST before searching.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Sender name, sender email, or subject text (e.g., 'Myprotein', 'john@example.com', '99p delivery')",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_emails",
                "description": "Search emails using Gmail query syntax. Use if lookup_cached fails. Returns emails with IDs needed for reply/forward actions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Gmail search query (e.g., 'from:sender@example.com', 'subject:keyword', 'from:company after:2024/01/01')",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum emails to return (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_unread",
                "description": "Get unread emails",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum emails to return (default: 10)",
                            "default": 10,
                        },
                    },
                },
            },
            {
                "name": "get_today",
                "description": "Get emails received today",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum emails to return (default: 10)",
                            "default": 10,
                        },
                    },
                },
            },
            {
                "name": "get_recent",
                "description": "Get recent emails from the last 7 days",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum emails to return (default: 10)",
                            "default": 10,
                        },
                    },
                },
            },
        ]

    def call_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Call a Gmail tool and return the result."""
        # Handle cache lookup first (doesn't need OAuth)
        if tool_name == "lookup_cached":
            query = tool_args.get("query", "")
            cached = self._lookup_cached_email(query)
            if cached:
                return {
                    "found": True,
                    "message_id": cached.get("message_id"),
                    "account_email": cached.get("account_email"),
                    "subject": cached.get("subject"),
                    "from": cached.get("from"),
                    "date": cached.get("date"),
                    "thread_id": cached.get("thread_id"),
                }
            return {
                "found": False,
                "message": f"No cached email found for '{query}'. Try search_emails instead.",
            }

        access_token = self._get_access_token()
        if not access_token:
            return {"error": "No valid Google OAuth token"}

        max_results = tool_args.get("max_results", 10)

        if tool_name == "search_emails":
            query = tool_args.get("query", "")
            emails = self._fetch_emails(access_token, "search", query, max_results)
        elif tool_name == "get_unread":
            emails = self._fetch_emails(access_token, "unread", None, max_results)
        elif tool_name == "get_today":
            emails = self._fetch_emails(access_token, "today", None, max_results)
        elif tool_name == "get_recent":
            emails = self._fetch_emails(access_token, "recent", None, max_results)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

        return {"emails": emails, "count": len(emails)}

    def fetch_agentic(
        self,
        goal: str,
        context: dict | None = None,
        designator_model: str | None = None,
        max_iterations: int = 5,
    ) -> LiveDataResult:
        """
        Fetch email data using an agentic loop that can make multiple tool calls.

        This allows multi-step lookups, e.g., searching for an email and then
        refining the search if needed.
        """
        start_time = time.time()

        tools = self.list_tools()
        account_email = self._get_account_email()

        # Build tools description
        tools_desc = ""
        for tool in tools:
            tools_desc += f"\n- {tool['name']}: {tool['description']}"
            schema = tool.get("inputSchema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            if props:
                params = []
                for pname, pinfo in props.items():
                    req_marker = " (required)" if pname in required else ""
                    params.append(
                        f"{pname}: {pinfo.get('description', '')}{req_marker}"
                    )
                tools_desc += f"\n  Parameters: {', '.join(params)}"

        messages = []
        tool_results = []
        iterations = 0

        system_prompt = f"""You are an email retrieval agent for Gmail account: {account_email}.
Your goal is to fetch the requested email data by calling the appropriate tools.

AVAILABLE TOOLS:{tools_desc}

INSTRUCTIONS:
1. Analyze the user's goal and determine which tool to call
2. For finding specific emails to reply/forward, use search_emails with sender/subject query
3. After each tool call, check if you have the email ID needed. If not, refine your search.
4. When you have the data needed (including email IDs for actions), provide the final answer.

RESPONSE FORMAT:
- To call a tool: TOOL_CALL: {{"tool": "tool_name", "args": {{"param": "value"}}}}
- When done: FINAL_ANSWER: <formatted email data with IDs>

Always include the email ID (shown as "ID: xxx") in your final answer - it's required for reply/forward actions."""

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Goal: {goal}"})

        if context:
            ctx_parts = []
            if context.get("user_intelligence"):
                ctx_parts.append(f"User context: {context['user_intelligence']}")
            if ctx_parts:
                messages.append({"role": "user", "content": "\n".join(ctx_parts)})

        model = designator_model or os.environ.get(
            "DESIGNATOR_MODEL", "groq/llama-3.1-8b-instant"
        )

        try:
            from providers import registry

            if not registry._providers:
                from providers import register_all_providers

                register_all_providers()

            resolved = registry._resolve_actual_model(model)
            if not resolved or not resolved.provider:
                return LiveDataResult(
                    source_name=self.name,
                    success=False,
                    error=f"Designator model not found: {model}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            llm_provider = resolved.provider

            while iterations < max_iterations:
                iterations += 1

                response = llm_provider.chat_completion(
                    model=resolved.model_id,
                    messages=messages,
                    system=None,
                    options={"temperature": 0.1, "max_tokens": 1000},
                )

                assistant_msg = response.get("content", "")
                messages.append({"role": "assistant", "content": assistant_msg})

                logger.debug(
                    f"Gmail {self.name} agent iteration {iterations}: {assistant_msg[:200]}"
                )

                if "FINAL_ANSWER:" in assistant_msg:
                    final_answer = assistant_msg.split("FINAL_ANSWER:", 1)[1].strip()
                    return LiveDataResult(
                        source_name=self.name,
                        success=True,
                        data={"tool_results": tool_results, "iterations": iterations},
                        formatted=f"### Gmail ({account_email})\n{final_answer}",
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                if "TOOL_CALL:" in assistant_msg:
                    try:
                        tool_json = assistant_msg.split("TOOL_CALL:", 1)[1].strip()
                        if "\n" in tool_json:
                            tool_json = tool_json.split("\n")[0]
                        tool_call = json.loads(tool_json)
                        tool_name = tool_call.get("tool")
                        tool_args = tool_call.get("args", {})

                        logger.info(
                            f"Gmail {self.name} agent calling: {tool_name} with {tool_args}"
                        )

                        result = self.call_tool(tool_name, tool_args)
                        tool_results.append(
                            {"tool": tool_name, "args": tool_args, "result": result}
                        )

                        # Format result for LLM
                        if "error" in result:
                            result_msg = f"Error: {result['error']}"
                        elif tool_name == "lookup_cached":
                            if result.get("found"):
                                result_msg = (
                                    f"Found cached email:\n"
                                    f"- ID: {result.get('message_id')}\n"
                                    f"  Account: {result.get('account_email')}\n"
                                    f"  From: {result.get('from')}\n"
                                    f"  Subject: {result.get('subject')}\n"
                                    f"  Date: {result.get('date')}"
                                )
                            else:
                                result_msg = result.get(
                                    "message", "No cached email found."
                                )
                        else:
                            emails = result.get("emails", [])
                            if emails:
                                result_lines = [f"Found {len(emails)} email(s):"]
                                for email in emails:
                                    result_lines.append(f"- ID: {email.get('id')}")
                                    result_lines.append(f"  From: {email.get('from')}")
                                    result_lines.append(
                                        f"  Subject: {email.get('subject')}"
                                    )
                                    result_lines.append(f"  Date: {email.get('date')}")
                                result_msg = "\n".join(result_lines)
                            else:
                                result_msg = "No emails found matching the query."

                        messages.append(
                            {"role": "user", "content": f"Tool result:\n{result_msg}"}
                        )

                    except json.JSONDecodeError as e:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Error parsing tool call: {e}. Please use valid JSON.",
                            }
                        )
                else:
                    # No tool call or final answer - prompt to continue
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please call a tool or provide FINAL_ANSWER.",
                        }
                    )

            # Max iterations reached
            return LiveDataResult(
                source_name=self.name,
                success=True,
                data={"tool_results": tool_results, "iterations": iterations},
                formatted=f"### Gmail ({account_email})\nReached max iterations. Last results: {tool_results[-1] if tool_results else 'none'}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Gmail {self.name} agentic fetch error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def test_connection(self) -> tuple[bool, str]:
        """Test Gmail connection."""
        if not self.account_id:
            return False, "No Google account configured"

        result = self.fetch("test", {"action": "recent", "max_results": 1})
        if result.success:
            return True, f"Connected ({result.latency_ms:.0f}ms)"
        return False, result.error or "Unknown error"


class OuraLiveProvider(LiveDataProvider):
    """
    Live Oura Ring provider for health and fitness data.

    Requires Oura OAuth connection (OURA_CLIENT_ID, OURA_CLIENT_SECRET).

    Use cases:
    - "How did I sleep last night?"
    - "What's my readiness score today?"
    - "How active was I today?"
    - "What's my current heart rate?"

    Actions:
    - sleep: Daily sleep data (score, duration, stages, efficiency)
    - readiness: Readiness score and contributors (HRV, temperature, recovery)
    - activity: Activity data (steps, calories, active time)
    - heart_rate: Heart rate data (current/recent readings)
    - summary: Combined overview of sleep, readiness, and activity
    """

    API_BASE_URL = "https://api.ouraring.com/v2/usercollection"

    def __init__(self, source: Any = None):
        self.source = source
        self.name = source.name if source else "oura"
        self.cache_ttl = source.cache_ttl_seconds if source else 300
        self.last_endpoint = None
        self._access_token: str | None = None
        self._token_data: dict | None = None

        # Get account_id from auth_config
        auth_config = {}
        if source and source.auth_config_json:
            try:
                auth_config = json.loads(source.auth_config_json)
            except json.JSONDecodeError:
                pass

        self.account_id = auth_config.get("account_id")

    def _get_token_data(self) -> dict | None:
        """Get and cache the decrypted token data."""
        if self._token_data is None and self.account_id:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> str | None:
        """Get a valid access token, refreshing if necessary."""
        import requests as http_requests

        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"Oura OAuth token {self.account_id} not found")
            return None

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            logger.error("No access token in stored Oura credentials")
            return None

        # Try a simple API call to validate token
        test_response = http_requests.get(
            f"{self.API_BASE_URL}/personal_info",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )

        if test_response.status_code == 200:
            return access_token

        # Token expired, try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error(
                "Cannot refresh Oura token - missing refresh_token or client credentials"
            )
            return None

        logger.info("Oura access token expired, refreshing...")
        refresh_response = http_requests.post(
            "https://api.ouraring.com/oauth/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        if refresh_response.status_code != 200:
            logger.error(f"Oura token refresh failed: {refresh_response.text}")
            return None

        new_token_data = refresh_response.json()
        new_access_token = new_token_data.get("access_token")

        # Update stored token
        from db.oauth_tokens import update_oauth_token_data

        updated_data = {**token_data, "access_token": new_access_token}
        # Oura may return a new refresh token
        if new_token_data.get("refresh_token"):
            updated_data["refresh_token"] = new_token_data["refresh_token"]
        update_oauth_token_data(self.account_id, updated_data)

        self._token_data = updated_data
        logger.info("Oura token refreshed successfully")
        return new_access_token

    def is_available(self) -> bool:
        """Check if we have a valid OAuth account configured."""
        if not self.account_id:
            return False
        return self._get_access_token() is not None

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch Oura health data.

        Context params:
        - action: "sleep", "readiness", "activity", "heart_rate", "summary"
        - date: Specific date in YYYY-MM-DD format (default: today)
        - days: Number of days to look back (default: 1)
        """
        start_time = time.time()

        access_token = self._get_access_token()
        if not access_token:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No valid Oura OAuth token",
                latency_ms=(time.time() - start_time) * 1000,
            )

        action = "summary"
        target_date = None
        days = 1

        if context:
            action = context.get("action", "summary")
            target_date = context.get("date")
            days = context.get("days", 1)

        self.last_endpoint = f"oura:{action}"

        try:
            if action == "sleep":
                data = self._fetch_sleep(access_token, target_date, days)
                formatted = self._format_sleep(data)
            elif action == "readiness":
                data = self._fetch_readiness(access_token, target_date, days)
                formatted = self._format_readiness(data)
            elif action == "activity":
                data = self._fetch_activity(access_token, target_date, days)
                formatted = self._format_activity(data)
            elif action == "heart_rate":
                data = self._fetch_heart_rate(access_token, target_date)
                formatted = self._format_heart_rate(data)
            else:  # summary
                data = self._fetch_summary(access_token, target_date)
                formatted = self._format_summary(data)

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=data,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Oura API error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _get_date_range(self, target_date: str | None, days: int) -> tuple[str, str]:
        """Get start and end dates for API queries."""
        from datetime import datetime, timedelta

        if target_date:
            end_date = target_date
            end_dt = datetime.strptime(target_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()
            end_date = end_dt.strftime("%Y-%m-%d")

        start_dt = end_dt - timedelta(days=days - 1)
        start_date = start_dt.strftime("%Y-%m-%d")

        return start_date, end_date

    def _fetch_sleep(
        self, access_token: str, target_date: str | None, days: int
    ) -> dict:
        """Fetch daily sleep data."""
        import requests as http_requests

        start_date, end_date = self._get_date_range(target_date, days)

        response = http_requests.get(
            f"{self.API_BASE_URL}/daily_sleep",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"start_date": start_date, "end_date": end_date},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_readiness(
        self, access_token: str, target_date: str | None, days: int
    ) -> dict:
        """Fetch daily readiness data."""
        import requests as http_requests

        start_date, end_date = self._get_date_range(target_date, days)

        response = http_requests.get(
            f"{self.API_BASE_URL}/daily_readiness",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"start_date": start_date, "end_date": end_date},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_activity(
        self, access_token: str, target_date: str | None, days: int
    ) -> dict:
        """Fetch daily activity data."""
        import requests as http_requests

        start_date, end_date = self._get_date_range(target_date, days)

        response = http_requests.get(
            f"{self.API_BASE_URL}/daily_activity",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"start_date": start_date, "end_date": end_date},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_heart_rate(self, access_token: str, target_date: str | None) -> dict:
        """Fetch heart rate data."""
        from datetime import datetime, timedelta

        import requests as http_requests

        # Heart rate endpoint uses start_datetime and end_datetime
        if target_date:
            start_dt = datetime.strptime(target_date, "%Y-%m-%d")
        else:
            start_dt = datetime.now() - timedelta(hours=24)

        end_dt = start_dt + timedelta(hours=24)

        response = http_requests.get(
            f"{self.API_BASE_URL}/heartrate",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "start_datetime": start_dt.strftime("%Y-%m-%dT00:00:00+00:00"),
                "end_datetime": end_dt.strftime("%Y-%m-%dT23:59:59+00:00"),
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_summary(self, access_token: str, target_date: str | None) -> dict:
        """Fetch combined summary (sleep, readiness, activity)."""
        sleep = self._fetch_sleep(access_token, target_date, 1)
        readiness = self._fetch_readiness(access_token, target_date, 1)
        activity = self._fetch_activity(access_token, target_date, 1)

        return {
            "sleep": sleep,
            "readiness": readiness,
            "activity": activity,
        }

    def _format_sleep(self, data: dict) -> str:
        """Format sleep data for LLM context."""
        items = data.get("data", [])
        if not items:
            return "### Sleep Data\nNo sleep data available for this period."

        lines = ["### Sleep Data"]

        for item in items:
            day = item.get("day", "Unknown")
            score = item.get("score", "N/A")
            contributors = item.get("contributors", {})

            lines.append(f"\n**{day}**")
            lines.append(f"  Sleep Score: {score}/100")

            # Key contributors
            if contributors:
                deep_sleep = contributors.get("deep_sleep")
                rem_sleep = contributors.get("rem_sleep")
                efficiency = contributors.get("efficiency")
                latency = contributors.get("latency")
                timing = contributors.get("timing")

                if deep_sleep is not None:
                    lines.append(f"  Deep Sleep: {deep_sleep}/100")
                if rem_sleep is not None:
                    lines.append(f"  REM Sleep: {rem_sleep}/100")
                if efficiency is not None:
                    lines.append(f"  Efficiency: {efficiency}/100")
                if latency is not None:
                    lines.append(f"  Sleep Latency: {latency}/100")
                if timing is not None:
                    lines.append(f"  Timing: {timing}/100")

        return "\n".join(lines)

    def _format_readiness(self, data: dict) -> str:
        """Format readiness data for LLM context."""
        items = data.get("data", [])
        if not items:
            return "### Readiness Data\nNo readiness data available for this period."

        lines = ["### Readiness Data"]

        for item in items:
            day = item.get("day", "Unknown")
            score = item.get("score", "N/A")
            contributors = item.get("contributors", {})
            temperature_deviation = item.get("temperature_deviation")
            temperature_trend_deviation = item.get("temperature_trend_deviation")

            lines.append(f"\n**{day}**")
            lines.append(f"  Readiness Score: {score}/100")

            if temperature_deviation is not None:
                lines.append(f"  Temperature Deviation: {temperature_deviation:+.1f}°C")
            if temperature_trend_deviation is not None:
                lines.append(
                    f"  Temperature Trend: {temperature_trend_deviation:+.1f}°C"
                )

            # Key contributors
            if contributors:
                hrv_balance = contributors.get("hrv_balance")
                recovery_index = contributors.get("recovery_index")
                resting_hr = contributors.get("resting_heart_rate")
                sleep_balance = contributors.get("sleep_balance")

                if hrv_balance is not None:
                    lines.append(f"  HRV Balance: {hrv_balance}/100")
                if recovery_index is not None:
                    lines.append(f"  Recovery Index: {recovery_index}/100")
                if resting_hr is not None:
                    lines.append(f"  Resting Heart Rate: {resting_hr}/100")
                if sleep_balance is not None:
                    lines.append(f"  Sleep Balance: {sleep_balance}/100")

        return "\n".join(lines)

    def _format_activity(self, data: dict) -> str:
        """Format activity data for LLM context."""
        items = data.get("data", [])
        if not items:
            return "### Activity Data\nNo activity data available for this period."

        lines = ["### Activity Data"]

        for item in items:
            day = item.get("day", "Unknown")
            score = item.get("score", "N/A")
            steps = item.get("steps", 0)
            active_calories = item.get("active_calories", 0)
            total_calories = item.get("total_calories", 0)
            target_calories = item.get("target_calories", 0)
            equivalent_walking_distance = item.get("equivalent_walking_distance", 0)
            high_activity_time = item.get("high_activity_time", 0)
            medium_activity_time = item.get("medium_activity_time", 0)
            low_activity_time = item.get("low_activity_time", 0)

            lines.append(f"\n**{day}**")
            lines.append(f"  Activity Score: {score}/100")
            lines.append(f"  Steps: {steps:,}")

            if equivalent_walking_distance:
                km = equivalent_walking_distance / 1000
                lines.append(f"  Distance: {km:.1f} km")

            lines.append(
                f"  Calories: {active_calories:,} active / {total_calories:,} total (target: {target_calories:,})"
            )

            # Activity time breakdown
            if high_activity_time or medium_activity_time or low_activity_time:
                high_min = high_activity_time // 60
                med_min = medium_activity_time // 60
                low_min = low_activity_time // 60
                lines.append(
                    f"  Activity Time: {high_min}min high / {med_min}min medium / {low_min}min low"
                )

        return "\n".join(lines)

    def _format_heart_rate(self, data: dict) -> str:
        """Format heart rate data for LLM context."""
        items = data.get("data", [])
        if not items:
            return "### Heart Rate Data\nNo heart rate data available for this period."

        lines = ["### Heart Rate Data"]

        # Get summary statistics
        bpms = [item.get("bpm", 0) for item in items if item.get("bpm")]
        if bpms:
            avg_bpm = sum(bpms) / len(bpms)
            min_bpm = min(bpms)
            max_bpm = max(bpms)
            latest_bpm = bpms[-1] if bpms else 0

            lines.append(f"  Latest: {latest_bpm} bpm")
            lines.append(f"  Average: {avg_bpm:.0f} bpm")
            lines.append(f"  Range: {min_bpm} - {max_bpm} bpm")
            lines.append(f"  Readings: {len(bpms)}")

            # Show last few readings with timestamps
            lines.append("\n  Recent readings:")
            for item in items[-5:]:
                ts = item.get("timestamp", "")
                bpm = item.get("bpm", "N/A")
                source = item.get("source", "")
                if ts:
                    # Parse and format timestamp
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        time_str = dt.strftime("%H:%M")
                    except ValueError:
                        time_str = ts
                    lines.append(f"    {time_str}: {bpm} bpm ({source})")

        return "\n".join(lines)

    def _format_summary(self, data: dict) -> str:
        """Format combined summary for LLM context."""
        lines = ["### Oura Health Summary"]

        # Sleep
        sleep_items = data.get("sleep", {}).get("data", [])
        if sleep_items:
            latest = sleep_items[-1]
            score = latest.get("score", "N/A")
            day = latest.get("day", "")
            lines.append(f"\n**Sleep** ({day})")
            lines.append(f"  Score: {score}/100")

        # Readiness
        readiness_items = data.get("readiness", {}).get("data", [])
        if readiness_items:
            latest = readiness_items[-1]
            score = latest.get("score", "N/A")
            day = latest.get("day", "")
            temp = latest.get("temperature_deviation")
            lines.append(f"\n**Readiness** ({day})")
            lines.append(f"  Score: {score}/100")
            if temp is not None:
                lines.append(f"  Body Temperature: {temp:+.1f}°C from baseline")

        # Activity
        activity_items = data.get("activity", {}).get("data", [])
        if activity_items:
            latest = activity_items[-1]
            score = latest.get("score", "N/A")
            steps = latest.get("steps", 0)
            day = latest.get("day", "")
            lines.append(f"\n**Activity** ({day})")
            lines.append(f"  Score: {score}/100")
            lines.append(f"  Steps: {steps:,}")

        if len(lines) == 1:
            lines.append("\nNo health data available for today.")

        return "\n".join(lines)

    def test_connection(self) -> tuple[bool, str]:
        """Test Oura connection."""
        if not self.account_id:
            return False, "No Oura account configured (set account_id in auth_config)"
        token = self._get_access_token()
        if token:
            return True, "Connected to Oura Ring"
        return False, "Failed to get valid access token"


class WithingsLiveProvider(LiveDataProvider):
    """
    Live Withings provider for health and body composition data.

    Requires Withings OAuth connection (WITHINGS_CLIENT_ID, WITHINGS_CLIENT_SECRET).

    Use cases:
    - "What's my current weight?"
    - "What's my body composition?"
    - "How did I sleep last night?"
    - "How active was I today?"

    Actions:
    - weight: Latest weight and body composition (fat %, muscle mass, etc.)
    - measures: All body measurements (weight, fat, muscle, bone, hydration)
    - sleep: Sleep data (duration, phases, quality)
    - activity: Activity data (steps, calories, distance)
    - summary: Combined overview of weight, sleep, and activity
    """

    API_BASE_URL = "https://wbsapi.withings.net"

    # Withings measurement type codes
    MEAS_TYPES = {
        1: ("weight", "kg"),
        4: ("height", "m"),
        5: ("fat_free_mass", "kg"),
        6: ("fat_ratio", "%"),
        8: ("fat_mass", "kg"),
        9: ("diastolic_bp", "mmHg"),
        10: ("systolic_bp", "mmHg"),
        11: ("heart_pulse", "bpm"),
        12: ("temperature", "°C"),
        54: ("spo2", "%"),
        76: ("muscle_mass", "kg"),
        77: ("hydration", "kg"),
        88: ("bone_mass", "kg"),
        91: ("pulse_wave_velocity", "m/s"),
        123: ("vo2_max", "ml/min/kg"),
    }

    def __init__(self, source: Any = None):
        self.source = source
        self.name = source.name if source else "withings"
        self.cache_ttl = source.cache_ttl_seconds if source else 300
        self.last_endpoint = None
        self._access_token: str | None = None
        self._token_data: dict | None = None

        # Get account_id from auth_config
        auth_config = {}
        if source and source.auth_config_json:
            try:
                auth_config = json.loads(source.auth_config_json)
            except json.JSONDecodeError:
                pass

        self.account_id = auth_config.get("account_id")

    def _get_token_data(self) -> dict | None:
        """Get and cache the decrypted token data."""
        if self._token_data is None and self.account_id:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> str | None:
        """Get a valid access token, refreshing if necessary."""
        import hashlib
        import hmac

        import requests as http_requests

        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"Withings OAuth token {self.account_id} not found")
            return None

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            logger.error("No access token in stored Withings credentials")
            return None

        # Try a simple API call to validate token
        # Withings tokens expire after 3 hours
        nonce = str(int(time.time()))
        sign_data = f"getdevice,{client_id},{nonce}"
        signature = hmac.new(
            client_secret.encode() if client_secret else b"",
            sign_data.encode(),
            hashlib.sha256,
        ).hexdigest()

        test_response = http_requests.post(
            f"{self.API_BASE_URL}/v2/user",
            headers={"Authorization": f"Bearer {access_token}"},
            data={
                "action": "getdevice",
                "nonce": nonce,
                "signature": signature,
                "client_id": client_id,
            },
            timeout=10,
        )

        response_data = test_response.json()
        if response_data.get("status") == 0:
            return access_token

        # Token expired, try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error(
                "Cannot refresh Withings token - missing refresh_token or client credentials"
            )
            return None

        logger.info("Withings access token expired, refreshing...")

        # Withings OAuth2 token refresh - standard OAuth flow, no signature needed
        refresh_response = http_requests.post(
            "https://wbsapi.withings.net/v2/oauth2",
            data={
                "action": "requesttoken",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        response_data = refresh_response.json()
        if response_data.get("status") != 0:
            logger.error(f"Withings token refresh failed: {response_data}")
            return None

        new_token_data = response_data.get("body", {})
        new_access_token = new_token_data.get("access_token")

        # Update stored token
        from db.oauth_tokens import update_oauth_token_data

        updated_data = {
            **token_data,
            "access_token": new_access_token,
            "refresh_token": new_token_data.get("refresh_token", refresh_token),
        }
        update_oauth_token_data(self.account_id, updated_data)

        self._token_data = updated_data
        logger.info("Withings token refreshed successfully")
        return new_access_token

    def is_available(self) -> bool:
        """Check if we have a valid OAuth account configured."""
        if not self.account_id:
            return False
        return self._get_access_token() is not None

    def _make_api_call(
        self, endpoint: str, action: str, params: dict | None = None
    ) -> dict:
        """Make an authenticated API call to Withings."""
        import hashlib
        import hmac

        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            raise Exception("No valid access token")

        token_data = self._get_token_data()
        client_id = token_data.get("client_id", "")
        client_secret = token_data.get("client_secret", "")

        nonce = str(int(time.time()))
        sign_data = f"{action},{client_id},{nonce}"
        signature = hmac.new(
            client_secret.encode() if client_secret else b"",
            sign_data.encode(),
            hashlib.sha256,
        ).hexdigest()

        data = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "client_id": client_id,
            **(params or {}),
        }

        response = http_requests.post(
            f"{self.API_BASE_URL}/{endpoint}",
            headers={"Authorization": f"Bearer {access_token}"},
            data=data,
            timeout=15,
        )
        response.raise_for_status()

        result = response.json()
        if result.get("status") != 0:
            raise Exception(f"API error: {result.get('error', 'Unknown error')}")

        return result.get("body", {})

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch Withings health data.

        Context params:
        - action: "weight", "measures", "sleep", "activity", "summary"
        - startdate: Start date (Unix timestamp or YYYY-MM-DD)
        - enddate: End date (Unix timestamp or YYYY-MM-DD)
        """
        start_time = time.time()

        access_token = self._get_access_token()
        if not access_token:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="No valid Withings OAuth token",
                latency_ms=(time.time() - start_time) * 1000,
            )

        action = "summary"
        startdate = None
        enddate = None

        if context:
            action = context.get("action", "summary")
            startdate = context.get("startdate")
            enddate = context.get("enddate")

        self.last_endpoint = f"withings:{action}"

        try:
            if action == "weight":
                data = self._fetch_weight(startdate, enddate)
                formatted = self._format_weight(data)
            elif action == "measures":
                data = self._fetch_measures(startdate, enddate)
                formatted = self._format_measures(data)
            elif action == "sleep":
                data = self._fetch_sleep(startdate, enddate)
                formatted = self._format_sleep(data)
            elif action == "activity":
                data = self._fetch_activity(startdate, enddate)
                formatted = self._format_activity(data)
            else:  # summary
                data = self._fetch_summary()
                formatted = self._format_summary(data)

            return LiveDataResult(
                source_name=self.name,
                success=True,
                data=data,
                formatted=formatted,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Withings API error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _get_date_params(self, startdate: str | None, enddate: str | None) -> dict:
        """Convert date parameters to Unix timestamps."""
        from datetime import datetime, timedelta

        params = {}

        if startdate:
            if isinstance(startdate, str) and "-" in startdate:
                dt = datetime.strptime(startdate, "%Y-%m-%d")
                params["startdate"] = int(dt.timestamp())
            else:
                params["startdate"] = int(startdate)
        else:
            # Default to last 30 days
            params["startdate"] = int((datetime.now() - timedelta(days=30)).timestamp())

        if enddate:
            if isinstance(enddate, str) and "-" in enddate:
                dt = datetime.strptime(enddate, "%Y-%m-%d")
                params["enddate"] = int(dt.timestamp())
            else:
                params["enddate"] = int(enddate)
        else:
            params["enddate"] = int(datetime.now().timestamp())

        return params

    def _fetch_weight(self, startdate: str | None, enddate: str | None) -> dict:
        """Fetch weight measurements."""
        params = self._get_date_params(startdate, enddate)
        # Weight and body composition types
        params["meastypes"] = "1,5,6,8,76,77,88"
        params["category"] = 1  # Real measures only

        return self._make_api_call("measure", "getmeas", params)

    def _fetch_measures(self, startdate: str | None, enddate: str | None) -> dict:
        """Fetch all body measurements."""
        params = self._get_date_params(startdate, enddate)
        # All common measurement types
        params["meastypes"] = "1,4,5,6,8,9,10,11,54,76,77,88,91,123"
        params["category"] = 1

        return self._make_api_call("measure", "getmeas", params)

    def _fetch_sleep(self, startdate: str | None, enddate: str | None) -> dict:
        """Fetch sleep summary data."""
        params = self._get_date_params(startdate, enddate)
        # Convert to date strings for sleep API
        from datetime import datetime

        params["startdateymd"] = datetime.fromtimestamp(params["startdate"]).strftime(
            "%Y-%m-%d"
        )
        params["enddateymd"] = datetime.fromtimestamp(params["enddate"]).strftime(
            "%Y-%m-%d"
        )
        del params["startdate"]
        del params["enddate"]

        return self._make_api_call("v2/sleep", "getsummary", params)

    def _fetch_activity(self, startdate: str | None, enddate: str | None) -> dict:
        """Fetch activity data."""
        params = self._get_date_params(startdate, enddate)
        # Convert to date strings for activity API
        from datetime import datetime

        params["startdateymd"] = datetime.fromtimestamp(params["startdate"]).strftime(
            "%Y-%m-%d"
        )
        params["enddateymd"] = datetime.fromtimestamp(params["enddate"]).strftime(
            "%Y-%m-%d"
        )
        del params["startdate"]
        del params["enddate"]

        return self._make_api_call("v2/measure", "getactivity", params)

    def _fetch_summary(self) -> dict:
        """Fetch combined summary (weight, sleep, activity)."""
        from datetime import datetime, timedelta

        # Last 7 days for summary
        end = datetime.now()
        start = end - timedelta(days=7)
        startdate = start.strftime("%Y-%m-%d")
        enddate = end.strftime("%Y-%m-%d")

        weight = self._fetch_weight(startdate, enddate)
        sleep = self._fetch_sleep(startdate, enddate)
        activity = self._fetch_activity(startdate, enddate)

        return {
            "weight": weight,
            "sleep": sleep,
            "activity": activity,
        }

    def _parse_measure_groups(self, measuregrps: list) -> list[dict]:
        """Parse Withings measure groups into readable format."""
        from datetime import datetime

        results = []
        for grp in measuregrps:
            date = datetime.fromtimestamp(grp.get("date", 0)).strftime("%Y-%m-%d %H:%M")
            measures = {}
            for m in grp.get("measures", []):
                mtype = m.get("type")
                value = m.get("value", 0) * (10 ** m.get("unit", 0))
                if mtype in self.MEAS_TYPES:
                    name, unit = self.MEAS_TYPES[mtype]
                    measures[name] = {"value": round(value, 2), "unit": unit}
            if measures:
                results.append({"date": date, "measures": measures})
        return results

    def _format_weight(self, data: dict) -> str:
        """Format weight data for LLM context."""
        measuregrps = data.get("measuregrps", [])
        if not measuregrps:
            return "### Weight Data\nNo weight measurements available."

        parsed = self._parse_measure_groups(measuregrps)
        if not parsed:
            return "### Weight Data\nNo weight measurements available."

        lines = ["### Weight & Body Composition"]

        # Show most recent first
        for entry in parsed[:5]:  # Last 5 measurements
            lines.append(f"\n**{entry['date']}**")
            measures = entry["measures"]
            if "weight" in measures:
                lines.append(f"  Weight: {measures['weight']['value']} kg")
            if "fat_ratio" in measures:
                lines.append(f"  Body Fat: {measures['fat_ratio']['value']}%")
            if "muscle_mass" in measures:
                lines.append(f"  Muscle Mass: {measures['muscle_mass']['value']} kg")
            if "bone_mass" in measures:
                lines.append(f"  Bone Mass: {measures['bone_mass']['value']} kg")
            if "hydration" in measures:
                lines.append(f"  Hydration: {measures['hydration']['value']} kg")

        return "\n".join(lines)

    def _format_measures(self, data: dict) -> str:
        """Format all measures for LLM context."""
        measuregrps = data.get("measuregrps", [])
        if not measuregrps:
            return "### Body Measurements\nNo measurements available."

        parsed = self._parse_measure_groups(measuregrps)
        if not parsed:
            return "### Body Measurements\nNo measurements available."

        lines = ["### Body Measurements"]

        for entry in parsed[:10]:
            lines.append(f"\n**{entry['date']}**")
            for name, info in entry["measures"].items():
                display_name = name.replace("_", " ").title()
                lines.append(f"  {display_name}: {info['value']} {info['unit']}")

        return "\n".join(lines)

    def _format_sleep(self, data: dict) -> str:
        """Format sleep data for LLM context."""
        series = data.get("series", [])
        if not series:
            return "### Sleep Data\nNo sleep data available."

        lines = ["### Sleep Data"]

        for entry in series[:7]:  # Last 7 nights
            date = entry.get("date", "Unknown")
            lines.append(f"\n**{date}**")

            # Duration
            total_seconds = entry.get("data", {}).get("total_sleep_time", 0)
            if total_seconds:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                lines.append(f"  Total Sleep: {hours}h {minutes}m")

            # Sleep phases
            deep = entry.get("data", {}).get("deepsleepduration", 0)
            light = entry.get("data", {}).get("lightsleepduration", 0)
            rem = entry.get("data", {}).get("remsleepduration", 0)

            if deep or light or rem:
                deep_m = deep // 60
                light_m = light // 60
                rem_m = rem // 60
                lines.append(
                    f"  Phases: {deep_m}m deep / {light_m}m light / {rem_m}m REM"
                )

            # Sleep score if available
            score = entry.get("data", {}).get("sleep_score")
            if score:
                lines.append(f"  Sleep Score: {score}/100")

            # Wake count
            wakeup_count = entry.get("data", {}).get("wakeupcount", 0)
            if wakeup_count:
                lines.append(f"  Wake-ups: {wakeup_count}")

        return "\n".join(lines)

    def _format_activity(self, data: dict) -> str:
        """Format activity data for LLM context."""
        activities = data.get("activities", [])
        if not activities:
            return "### Activity Data\nNo activity data available."

        lines = ["### Activity Data"]

        for entry in activities[:7]:  # Last 7 days
            date = entry.get("date", "Unknown")
            lines.append(f"\n**{date}**")

            steps = entry.get("steps", 0)
            if steps:
                lines.append(f"  Steps: {steps:,}")

            distance = entry.get("distance", 0)
            if distance:
                km = distance / 1000
                lines.append(f"  Distance: {km:.1f} km")

            calories = entry.get("calories", 0)
            if calories:
                lines.append(f"  Calories: {calories:,}")

            active = entry.get("active", 0)
            if active:
                minutes = active // 60
                lines.append(f"  Active Time: {minutes} min")

            elevation = entry.get("elevation", 0)
            if elevation:
                lines.append(f"  Elevation: {elevation} m")

        return "\n".join(lines)

    def _format_summary(self, data: dict) -> str:
        """Format combined summary for LLM context."""
        lines = ["### Withings Health Summary (Last 7 Days)"]

        # Weight
        weight_grps = data.get("weight", {}).get("measuregrps", [])
        if weight_grps:
            parsed = self._parse_measure_groups(weight_grps)
            if parsed:
                latest = parsed[0]
                lines.append(f"\n**Latest Weight** ({latest['date']})")
                measures = latest["measures"]
                if "weight" in measures:
                    lines.append(f"  Weight: {measures['weight']['value']} kg")
                if "fat_ratio" in measures:
                    lines.append(f"  Body Fat: {measures['fat_ratio']['value']}%")

        # Sleep
        sleep_series = data.get("sleep", {}).get("series", [])
        if sleep_series:
            latest = sleep_series[0]
            date = latest.get("date", "Unknown")
            total = latest.get("data", {}).get("total_sleep_time", 0)
            if total:
                hours = total // 3600
                minutes = (total % 3600) // 60
                lines.append(f"\n**Last Night's Sleep** ({date})")
                lines.append(f"  Duration: {hours}h {minutes}m")
                score = latest.get("data", {}).get("sleep_score")
                if score:
                    lines.append(f"  Score: {score}/100")

        # Activity
        activities = data.get("activity", {}).get("activities", [])
        if activities:
            latest = activities[0]
            date = latest.get("date", "Unknown")
            steps = latest.get("steps", 0)
            lines.append(f"\n**Today's Activity** ({date})")
            lines.append(f"  Steps: {steps:,}")
            calories = latest.get("calories", 0)
            if calories:
                lines.append(f"  Calories: {calories:,}")

        if len(lines) == 1:
            lines.append("\nNo health data available.")

        return "\n".join(lines)

    def test_connection(self) -> tuple[bool, str]:
        """Test Withings connection."""
        if not self.account_id:
            return (
                False,
                "No Withings account configured (set account_id in auth_config)",
            )
        token = self._get_access_token()
        if token:
            return True, "Connected to Withings"
        return False, "Failed to get valid access token"


class GoogleMapsProvider(LiveDataProvider):
    """
    Built-in Google Maps provider for Places and Routes APIs.

    Requires GOOGLE_MAPS_API_KEY environment variable.

    Supports:
    - Places API (New): Find places, get details, search nearby
    - Routes API: Directions, travel time, distance
    - Geocoding: Address to coordinates (used internally)
    """

    PLACES_BASE_URL = "https://places.googleapis.com/v1/places"
    ROUTES_BASE_URL = "https://routes.googleapis.com/directions/v2"
    GEOCODING_BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(self, source: Any = None):
        self.source = source
        self.api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        self.name = source.name if source else "google_maps"
        self.cache_ttl = source.cache_ttl_seconds if source else 300
        self.last_endpoint = None

    def fetch(self, query: str, context: dict | None = None) -> LiveDataResult:
        """
        Fetch location data based on query.

        The designator should provide structured params in context:
        - For places: {"action": "search", "query": "...", "location": "lat,lng"}
                   or {"action": "details", "place_id": "..."}
                   or {"action": "nearby", "location": "lat,lng", "type": "restaurant"}
        - For routes: {"action": "directions", "origin": "...", "destination": "..."}
        """
        start_time = time.time()

        if not self.api_key:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="GOOGLE_MAPS_API_KEY not configured",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Get action from context or infer from query
        action = "search"  # default
        if context:
            action = context.get("action", "search")

        try:
            if action == "directions":
                return self._fetch_directions(query, context, start_time)
            elif action == "details":
                return self._fetch_place_details(query, context, start_time)
            elif action == "nearby":
                return self._fetch_nearby(query, context, start_time)
            else:
                return self._fetch_text_search(query, context, start_time)
        except Exception as e:
            logger.error(f"Google Maps API error: {e}")
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _fetch_text_search(
        self, query: str, context: dict | None, start_time: float
    ) -> LiveDataResult:
        """Search for places using text query (Places API New)."""
        self.last_endpoint = "places:searchText"

        search_query = query
        if context:
            search_query = context.get("query", query)

        # Check cache
        cache_key = f"places_search:{search_query}"
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_places(cached, search_query),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Build request for Places API (New)
        request_body = {
            "textQuery": search_query,
            "maxResultCount": 5,
        }

        # Add location bias if provided
        if context and context.get("location"):
            lat, lng = self._parse_location(context["location"])
            if lat and lng:
                request_body["locationBias"] = {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": 5000.0,  # 5km radius
                    }
                }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.currentOpeningHours,places.websiteUri,places.nationalPhoneNumber,places.types",
        }

        with httpx.Client(timeout=15) as client:
            response = client.post(
                f"{self.PLACES_BASE_URL}:searchText",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        places = data.get("places", [])

        # Cache result
        if self.cache_ttl > 0:
            self.set_cached(cache_key, places)

        return LiveDataResult(
            source_name=self.name,
            success=True,
            data=places,
            formatted=self._format_places(places, search_query),
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _fetch_nearby(
        self, query: str, context: dict | None, start_time: float
    ) -> LiveDataResult:
        """Search for nearby places (Places API New)."""
        self.last_endpoint = "places:searchNearby"

        if not context or not context.get("location"):
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="Location required for nearby search",
                latency_ms=(time.time() - start_time) * 1000,
            )

        lat, lng = self._parse_location(context["location"])
        if not lat or not lng:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error=f"Invalid location format: {context['location']}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        place_type = context.get("type", "restaurant")
        radius = context.get("radius", 1000)  # Default 1km

        cache_key = f"places_nearby:{lat},{lng}:{place_type}:{radius}"
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_places(cached, f"nearby {place_type}"),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        request_body = {
            "maxResultCount": 10,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(radius),
                }
            },
        }

        # Add included types if specified
        if place_type:
            request_body["includedTypes"] = [place_type]

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.currentOpeningHours,places.types",
        }

        with httpx.Client(timeout=15) as client:
            response = client.post(
                f"{self.PLACES_BASE_URL}:searchNearby",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        places = data.get("places", [])

        if self.cache_ttl > 0:
            self.set_cached(cache_key, places)

        return LiveDataResult(
            source_name=self.name,
            success=True,
            data=places,
            formatted=self._format_places(places, f"nearby {place_type}"),
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _fetch_place_details(
        self, query: str, context: dict | None, start_time: float
    ) -> LiveDataResult:
        """Get details for a specific place (Places API New)."""
        self.last_endpoint = "places:get"

        place_id = context.get("place_id") if context else None
        if not place_id:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="place_id required for details lookup",
                latency_ms=(time.time() - start_time) * 1000,
            )

        cache_key = f"place_details:{place_id}"
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_place_details(cached),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "id,displayName,formattedAddress,rating,userRatingCount,priceLevel,currentOpeningHours,regularOpeningHours,websiteUri,nationalPhoneNumber,types,reviews,editorialSummary",
        }

        with httpx.Client(timeout=15) as client:
            response = client.get(
                f"{self.PLACES_BASE_URL}/{place_id}",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        if self.cache_ttl > 0:
            self.set_cached(cache_key, data)

        return LiveDataResult(
            source_name=self.name,
            success=True,
            data=data,
            formatted=self._format_place_details(data),
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _fetch_directions(
        self, query: str, context: dict | None, start_time: float
    ) -> LiveDataResult:
        """Get directions between two locations (Routes API)."""
        self.last_endpoint = "routes:computeRoutes"

        if not context:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="Origin and destination required for directions",
                latency_ms=(time.time() - start_time) * 1000,
            )

        origin = context.get("origin")
        destination = context.get("destination")

        if not origin or not destination:
            return LiveDataResult(
                source_name=self.name,
                success=False,
                error="Both origin and destination are required",
                latency_ms=(time.time() - start_time) * 1000,
            )

        travel_mode = context.get("mode", "DRIVE").upper()
        if travel_mode not in ["DRIVE", "WALK", "BICYCLE", "TRANSIT", "TWO_WHEELER"]:
            travel_mode = "DRIVE"

        cache_key = f"directions:{origin}:{destination}:{travel_mode}"
        if self.cache_ttl > 0:
            cached = self.get_cached(cache_key, self.cache_ttl)
            if cached is not None:
                return LiveDataResult(
                    source_name=self.name,
                    success=True,
                    data=cached,
                    formatted=self._format_directions(cached, origin, destination),
                    cache_hit=True,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Build waypoints - can be addresses or lat/lng
        def build_waypoint(location: str) -> dict:
            # Check if it's coordinates
            lat, lng = self._parse_location(location)
            if lat and lng:
                return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}
            else:
                return {"address": location}

        request_body = {
            "origin": build_waypoint(origin),
            "destination": build_waypoint(destination),
            "travelMode": travel_mode,
            "routingPreference": "TRAFFIC_AWARE",
            "computeAlternativeRoutes": False,
            "languageCode": "en",
            "units": "METRIC",
        }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline,routes.legs,routes.travelAdvisory,routes.description",
        }

        with httpx.Client(timeout=15) as client:
            response = client.post(
                f"{self.ROUTES_BASE_URL}:computeRoutes",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        if self.cache_ttl > 0:
            self.set_cached(cache_key, data)

        return LiveDataResult(
            source_name=self.name,
            success=True,
            data=data,
            formatted=self._format_directions(data, origin, destination),
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _parse_location(self, location: str) -> tuple[float | None, float | None]:
        """Parse location string to lat/lng coordinates."""
        if not location:
            return None, None

        # Try parsing as "lat,lng" or "lat, lng"
        parts = location.replace(" ", "").split(",")
        if len(parts) == 2:
            try:
                lat = float(parts[0])
                lng = float(parts[1])
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    return lat, lng
            except ValueError:
                pass

        return None, None

    def _format_places(self, places: list, query: str) -> str:
        """Format places results for context injection."""
        if not places:
            return f"### Places Search: {query}\nNo places found."

        lines = [f"### Places Search: {query}"]
        lines.append(f"Found {len(places)} result(s):\n")

        for i, place in enumerate(places, 1):
            name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "")
            rating = place.get("rating")
            rating_count = place.get("userRatingCount", 0)
            price_level = place.get("priceLevel", "")
            place_id = place.get("id", "")

            lines.append(f"**{i}. {name}**")
            if address:
                lines.append(f"   Address: {address}")
            if rating:
                lines.append(f"   Rating: {rating}/5 ({rating_count} reviews)")
            if price_level:
                # Convert PRICE_LEVEL_* to $ symbols
                price_map = {
                    "PRICE_LEVEL_FREE": "Free",
                    "PRICE_LEVEL_INEXPENSIVE": "$",
                    "PRICE_LEVEL_MODERATE": "$$",
                    "PRICE_LEVEL_EXPENSIVE": "$$$",
                    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
                }
                lines.append(f"   Price: {price_map.get(price_level, price_level)}")

            # Opening hours
            hours = place.get("currentOpeningHours", {})
            if hours.get("openNow") is not None:
                status = "Open now" if hours["openNow"] else "Closed"
                lines.append(f"   Status: {status}")

            if place_id:
                lines.append(f"   Place ID: {place_id}")
            lines.append("")

        return "\n".join(lines)

    def _format_place_details(self, place: dict) -> str:
        """Format place details for context injection."""
        name = place.get("displayName", {}).get("text", "Unknown Place")
        lines = [f"### {name}"]

        if place.get("editorialSummary", {}).get("text"):
            lines.append(f"\n_{place['editorialSummary']['text']}_\n")

        address = place.get("formattedAddress")
        if address:
            lines.append(f"**Address:** {address}")

        phone = place.get("nationalPhoneNumber")
        if phone:
            lines.append(f"**Phone:** {phone}")

        website = place.get("websiteUri")
        if website:
            lines.append(f"**Website:** {website}")

        rating = place.get("rating")
        rating_count = place.get("userRatingCount", 0)
        if rating:
            lines.append(f"**Rating:** {rating}/5 ({rating_count} reviews)")

        price_level = place.get("priceLevel")
        if price_level:
            price_map = {
                "PRICE_LEVEL_FREE": "Free",
                "PRICE_LEVEL_INEXPENSIVE": "$",
                "PRICE_LEVEL_MODERATE": "$$",
                "PRICE_LEVEL_EXPENSIVE": "$$$",
                "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
            }
            lines.append(f"**Price Level:** {price_map.get(price_level, price_level)}")

        # Opening hours
        hours = place.get("currentOpeningHours", {})
        if hours:
            status = "Open now" if hours.get("openNow") else "Closed"
            lines.append(f"**Status:** {status}")

            weekday_text = hours.get("weekdayDescriptions", [])
            if weekday_text:
                lines.append("\n**Hours:**")
                for day in weekday_text:
                    lines.append(f"  {day}")

        # Reviews
        reviews = place.get("reviews", [])
        if reviews:
            lines.append(f"\n**Recent Reviews ({len(reviews)}):**")
            for review in reviews[:3]:  # Show top 3 reviews
                author = review.get("authorAttribution", {}).get(
                    "displayName", "Anonymous"
                )
                rating = review.get("rating", "?")
                text = review.get("text", {}).get("text", "")
                if text:
                    # Truncate long reviews
                    if len(text) > 200:
                        text = text[:200] + "..."
                    lines.append(f'  - {author} ({rating}/5): "{text}"')

        return "\n".join(lines)

    def _format_directions(self, data: dict, origin: str, destination: str) -> str:
        """Format directions for context injection."""
        routes = data.get("routes", [])
        if not routes:
            return f"### Directions: {origin} → {destination}\nNo route found."

        route = routes[0]
        lines = [f"### Directions: {origin} → {destination}"]

        # Duration and distance
        duration = route.get("duration", "")
        if duration:
            # Duration is in seconds with 's' suffix, e.g., "3600s"
            seconds = int(duration.rstrip("s"))
            hours, remainder = divmod(seconds, 3600)
            minutes = remainder // 60
            if hours > 0:
                time_str = f"{hours}h {minutes}min"
            else:
                time_str = f"{minutes} min"
            lines.append(f"**Duration:** {time_str}")

        distance = route.get("distanceMeters", 0)
        if distance:
            if distance >= 1000:
                lines.append(f"**Distance:** {distance / 1000:.1f} km")
            else:
                lines.append(f"**Distance:** {distance} m")

        # Description (route name like "M1")
        description = route.get("description")
        if description:
            lines.append(f"**Via:** {description}")

        # Traffic/travel advisory
        advisory = route.get("travelAdvisory", {})
        if advisory.get("tollInfo"):
            lines.append("**Note:** Route includes tolls")

        # Legs with step-by-step directions
        legs = route.get("legs", [])
        if legs:
            lines.append("\n**Route:**")
            for leg in legs:
                steps = leg.get("steps", [])
                for i, step in enumerate(steps[:10], 1):  # Limit to 10 steps
                    instruction = step.get("navigationInstruction", {})
                    maneuver = instruction.get("maneuver", "")
                    instructions_text = instruction.get("instructions", "")

                    step_distance = step.get("distanceMeters", 0)
                    if step_distance >= 1000:
                        dist_str = f"{step_distance / 1000:.1f} km"
                    else:
                        dist_str = f"{step_distance} m"

                    if instructions_text:
                        lines.append(f"  {i}. {instructions_text} ({dist_str})")
                    elif maneuver:
                        lines.append(f"  {i}. {maneuver} ({dist_str})")

                if len(steps) > 10:
                    lines.append(f"  ... and {len(steps) - 10} more steps")

        return "\n".join(lines)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GOOGLE_MAPS_API_KEY not set"

        # Test with a simple places search
        result = self.fetch(
            "Eiffel Tower", {"action": "search", "query": "Eiffel Tower"}
        )
        if result.success:
            places = result.data or []
            return (
                True,
                f"Connected. Found {len(places)} place(s) ({result.latency_ms:.0f}ms)",
            )
        return False, result.error or "Unknown error"


def get_provider_for_source(source: Any) -> LiveDataProvider:
    """
    Get the appropriate provider for a LiveDataSource.

    Args:
        source: LiveDataSource or DetachedLiveDataSource

    Returns:
        Configured LiveDataProvider instance
    """
    source_type = source.source_type

    if source_type == "builtin_stocks":
        return StocksProvider(source)
    elif source_type == "builtin_alpha_vantage":
        return AlphaVantageProvider(source)
    elif source_type == "builtin_weather":
        return WeatherProvider(source)
    elif source_type == "builtin_open_meteo":
        return OpenMeteoProvider(source)
    elif source_type == "builtin_transport":
        return TransportProvider(source)
    elif source_type == "builtin_google_maps":
        return GoogleMapsProvider(source)
    elif source_type == "google_calendar_live":
        return GoogleCalendarLiveProvider(source)
    elif source_type == "google_tasks_live":
        return GoogleTasksLiveProvider(source)
    elif source_type == "google_gmail_live":
        return GmailLiveProvider(source)
    elif source_type == "oura_live":
        return OuraLiveProvider(source)
    elif source_type == "withings_live":
        return WithingsLiveProvider(source)
    elif source_type == "mcp_server":
        return MCPProvider(source)
    else:
        # MCP provider is the new default for RapidAPI sources
        # (replaces the old RestApiProvider approach)
        return MCPProvider(source)


def fetch_live_data(
    sources: list[Any],
    query: str,
    context: dict | None = None,
) -> list[LiveDataResult]:
    """
    Fetch live data from multiple sources.

    Args:
        sources: List of LiveDataSource/DetachedLiveDataSource objects
        query: The user's query
        context: Optional context dict

    Returns:
        List of LiveDataResult objects
    """
    results = []

    for source in sources:
        if not source.enabled:
            continue

        try:
            provider = get_provider_for_source(source)
            if provider.is_available():
                result = provider.fetch(query, context)
                results.append(result)

                # Update source status (if it's a real DB object)
                from db.live_data_sources import update_live_data_source_status

                if result.success:
                    update_live_data_source_status(source.id, success=True)
                else:
                    update_live_data_source_status(
                        source.id, success=False, error=result.error
                    )

        except Exception as e:
            logger.error(f"Error fetching from {source.name}: {e}")
            results.append(
                LiveDataResult(
                    source_name=source.name,
                    success=False,
                    error=str(e),
                )
            )

    return results


# Module-level helper for session cache lookups (used by email action handler)
def lookup_email_from_session_cache(
    session_key: str | None,
    account_email: str,
    subject_hint: str | None = None,
    sender_hint: str | None = None,
) -> dict | None:
    """
    Look up an email from the session-scoped cache by subject or sender.

    This is used by the email action handler to resolve message IDs when
    the LLM doesn't provide them or provides hallucinated ones.

    Args:
        session_key: Session identifier for scoped caching
        account_email: Email account to search in
        subject_hint: Subject text to search for
        sender_hint: Sender email/name to search for

    Returns:
        Dict with message_id, subject, from, etc. or None if not found
    """
    import time

    def lookup(entity_type: str, query_text: str) -> dict | None:
        if session_key:
            cache_key = f"{session_key}:{account_email}:{entity_type}:{query_text}"
        else:
            cache_key = f"nosession:{account_email}:{entity_type}:{query_text}"

        entries = _session_email_cache.get(cache_key)
        if not entries:
            return None

        # Return the most recent entry (first in list)
        metadata, timestamp = entries[0]

        # Check if expired
        if time.time() - timestamp > SESSION_EMAIL_CACHE_TTL:
            return None

        return metadata

    # Try lookup by subject first
    if subject_hint:
        subject_clean = subject_hint.lower().strip()
        # Remove common prefixes
        for prefix in ["fwd:", "re:", "fw:"]:
            if subject_clean.startswith(prefix):
                subject_clean = subject_clean[len(prefix) :].strip()

        result = lookup("subject", subject_clean[:50])
        if result:
            _session_cache_stats["email_cache_hits"] += 1
            logger.info(
                f"Session email cache HIT by subject: '{subject_clean[:30]}' -> {result.get('message_id')}"
            )
            return result

    # Try lookup by sender
    if sender_hint:
        sender_clean = sender_hint.lower().strip()
        # Extract email if in "Name <email>" format
        if "<" in sender_clean and ">" in sender_clean:
            sender_clean = sender_clean.split("<")[1].split(">")[0]

        result = lookup("sender", sender_clean)
        if result:
            _session_cache_stats["email_cache_hits"] += 1
            logger.info(
                f"Session email cache HIT by sender: '{sender_clean}' -> {result.get('message_id')}"
            )
            return result

    _session_cache_stats["email_cache_misses"] += 1
    logger.debug(
        f"Session email cache MISS: subject='{subject_hint[:30] if subject_hint else 'N/A'}', "
        f"sender='{sender_hint[:30] if sender_hint else 'N/A'}'"
    )
    return None


def store_session_live_context(
    session_key: str, source_name: str, formatted_data: str
) -> None:
    """
    Store formatted live data in session context for LLM reference.

    Accumulates results from multiple sources/queries within a session,
    replacing older results from the same source.

    Args:
        session_key: Session identifier
        source_name: Unique identifier for the data source (e.g., "gmail:user@example.com", "weather:london")
        formatted_data: The formatted data to store
    """
    import time

    now = time.time()

    if session_key not in _session_live_context:
        _session_live_context[session_key] = []

    entries = _session_live_context[session_key]

    # Remove expired entries and any existing entry for this source
    entries[:] = [
        (src, fmt, ts)
        for src, fmt, ts in entries
        if ts > now - SESSION_LIVE_CONTEXT_TTL and src != source_name
    ]

    # Add new entry
    entries.append((source_name, formatted_data, now))

    logger.info(
        f"Stored live context for session {session_key[:8]}..., "
        f"source {source_name}, {len(entries)} total entries"
    )


def get_session_live_context(session_key: str | None) -> str | None:
    """
    Retrieve accumulated live data context for a session.

    Returns formatted string with all recent live data results,
    or None if no context available.
    """
    import time

    if not session_key or session_key not in _session_live_context:
        _session_cache_stats["live_context_misses"] += 1
        return None

    now = time.time()
    entries = _session_live_context[session_key]

    # Filter to non-expired entries
    valid_entries = [
        (src, fmt, ts)
        for src, fmt, ts in entries
        if ts > now - SESSION_LIVE_CONTEXT_TTL
    ]

    if not valid_entries:
        _session_cache_stats["live_context_misses"] += 1
        return None

    _session_cache_stats["live_context_hits"] += 1

    # Combine all results
    parts = []
    for src, formatted, ts in valid_entries:
        parts.append(formatted)

    return "\n\n".join(parts)


# Backwards compatibility alias
def get_session_email_context(session_key: str | None) -> str | None:
    """Alias for get_session_live_context for backwards compatibility."""
    return get_session_live_context(session_key)


def get_session_cache_stats() -> dict:
    """
    Get session cache statistics.

    Returns dict with:
    - email_cache_hits/misses: Lookups for email message IDs
    - live_context_hits/misses: Injections of prior live data context
    - email_cache_entries: Current number of cached email entries
    - live_context_sessions: Current number of sessions with live context
    """
    import time

    now = time.time()

    # Count non-expired email cache entries
    email_entries = 0
    for entries in _session_email_cache.values():
        email_entries += sum(
            1 for _, ts in entries if ts > now - SESSION_EMAIL_CACHE_TTL
        )

    # Count sessions with non-expired live context
    live_sessions = 0
    for entries in _session_live_context.values():
        if any(ts > now - SESSION_LIVE_CONTEXT_TTL for _, _, ts in entries):
            live_sessions += 1

    return {
        **_session_cache_stats,
        "email_cache_entries": email_entries,
        "live_context_sessions": live_sessions,
    }


def clear_session_cache_stats() -> None:
    """Reset session cache statistics."""
    _session_cache_stats["email_cache_hits"] = 0
    _session_cache_stats["email_cache_misses"] = 0
    _session_cache_stats["live_context_hits"] = 0
    _session_cache_stats["live_context_misses"] = 0
