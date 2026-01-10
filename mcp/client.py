"""
MCP Client for connecting to MCP servers and retrieving resources.

Supports both stdio (subprocess) and HTTP transports.
"""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# MCP Protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""

    name: str
    transport: str  # "stdio" or "http"

    # For stdio transport
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # For HTTP transport
    url: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)

    # Resource filtering
    uri_patterns: list[str] = field(default_factory=list)  # e.g., ["file://**/*.pdf"]

    # Tool-based discovery (for servers that use tools instead of resources)
    discovery_tool: Optional[str] = None  # e.g., "search"
    discovery_args: dict = field(default_factory=dict)  # e.g., {"filter": {"value": "page"}}
    content_tool: Optional[str] = None  # e.g., "retrieve-page-content"
    content_id_field: Optional[str] = None  # e.g., "id" - field in discovery results

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "url": self.url,
            "headers": self.headers,
            "uri_patterns": self.uri_patterns,
            "discovery_tool": self.discovery_tool,
            "discovery_args": self.discovery_args,
            "content_tool": self.content_tool,
            "content_id_field": self.content_id_field,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MCPServerConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            transport=data.get("transport", "stdio"),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url"),
            headers=data.get("headers", {}),
            uri_patterns=data.get("uri_patterns", []),
            discovery_tool=data.get("discovery_tool"),
            discovery_args=data.get("discovery_args", {}),
            content_tool=data.get("content_tool"),
            content_id_field=data.get("content_id_field"),
        )


@dataclass
class MCPResource:
    """Represents a resource from an MCP server."""

    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class MCPResourceContent:
    """Content of an MCP resource."""

    uri: str
    mime_type: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[bytes] = None  # Base64 decoded


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""

    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None


class MCPClient:
    """
    Client for connecting to MCP servers.

    Supports stdio and HTTP transports for listing and reading resources.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._message_id = 0
        self._process: Optional[subprocess.Popen] = None
        self._session_id: Optional[str] = None
        self._initialized = False

    def _next_id(self) -> int:
        """Get the next message ID."""
        self._message_id += 1
        return self._message_id

    def connect(self) -> bool:
        """
        Connect to the MCP server and initialize.

        Returns:
            True if connection successful
        """
        try:
            if self.config.transport == "stdio":
                return self._connect_stdio()
            elif self.config.transport == "http":
                return self._connect_http()
            else:
                logger.error(f"Unknown transport: {self.config.transport}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.config.name}': {e}")
            return False

    def disconnect(self):
        """Disconnect from the MCP server."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
        self._initialized = False
        self._session_id = None

    def _connect_stdio(self) -> bool:
        """Connect via stdio transport."""
        if not self.config.command:
            raise ValueError("Command required for stdio transport")

        import os

        # Build command
        cmd = [self.config.command] + self.config.args

        # Merge environment - always inherit parent env, then overlay config env
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        logger.info(f"Starting MCP server: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        # Initialize the connection
        return self._initialize()

    def _connect_http(self) -> bool:
        """Connect via HTTP transport."""
        if not self.config.url:
            raise ValueError("URL required for HTTP transport")

        # Initialize the connection
        return self._initialize()

    def _initialize(self) -> bool:
        """Send initialize request to the server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "llm-relay",
                    "version": "1.6.0",
                },
            },
        }

        response = self._send_request(request)
        if response and "result" in response:
            self._initialized = True
            logger.info(
                f"MCP server '{self.config.name}' initialized: "
                f"protocol={response['result'].get('protocolVersion')}"
            )

            # Send initialized notification
            self._send_notification({"jsonrpc": "2.0", "method": "notifications/initialized"})
            return True

        return False

    def _send_request(self, request: dict) -> Optional[dict]:
        """Send a JSON-RPC request and wait for response."""
        if self.config.transport == "stdio":
            return self._send_stdio(request)
        else:
            return self._send_http(request)

    def _send_notification(self, notification: dict):
        """Send a JSON-RPC notification (no response expected)."""
        if self.config.transport == "stdio":
            self._send_stdio(notification, expect_response=False)
        else:
            self._send_http(notification, expect_response=False)

    def _send_stdio(
        self, message: dict, expect_response: bool = True
    ) -> Optional[dict]:
        """Send message via stdio."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Not connected to MCP server")

        # Send message
        line = json.dumps(message) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        if not expect_response:
            return None

        # Read response
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from MCP server")

        return json.loads(response_line)

    def _send_http(
        self, message: dict, expect_response: bool = True
    ) -> Optional[dict]:
        """Send message via HTTP."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            **self.config.headers,
        }

        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        response = requests.post(
            self.config.url,
            json=message,
            headers=headers,
            timeout=30,
        )

        # Check for session ID
        if "Mcp-Session-Id" in response.headers:
            self._session_id = response.headers["Mcp-Session-Id"]

        if not expect_response:
            return None

        if response.status_code == 202:
            return None

        response.raise_for_status()
        return response.json()

    def list_resources(self, cursor: Optional[str] = None) -> list[MCPResource]:
        """
        List available resources from the server.

        Args:
            cursor: Pagination cursor

        Returns:
            List of MCPResource objects
        """
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")

        params: dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/list",
            "params": params,
        }

        response = self._send_request(request)
        if not response or "result" not in response:
            return []

        result = response["result"]
        resources = []

        for r in result.get("resources", []):
            resources.append(
                MCPResource(
                    uri=r["uri"],
                    name=r.get("name", ""),
                    description=r.get("description"),
                    mime_type=r.get("mimeType"),
                )
            )

        # Handle pagination
        next_cursor = result.get("nextCursor")
        if next_cursor:
            resources.extend(self.list_resources(cursor=next_cursor))

        return resources

    def read_resource(self, uri: str) -> Optional[MCPResourceContent]:
        """
        Read a specific resource.

        Args:
            uri: Resource URI

        Returns:
            MCPResourceContent or None if not found
        """
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/read",
            "params": {"uri": uri},
        }

        response = self._send_request(request)
        if not response or "result" not in response:
            return None

        contents = response["result"].get("contents", [])
        if not contents:
            return None

        # Return first content block
        content = contents[0]

        # Handle binary content
        blob = None
        if "blob" in content:
            import base64

            blob = base64.b64decode(content["blob"])

        return MCPResourceContent(
            uri=content.get("uri", uri),
            mime_type=content.get("mimeType"),
            text=content.get("text"),
            blob=blob,
        )

    def list_tools(self) -> list[MCPTool]:
        """
        List available tools from the server.

        Returns:
            List of MCPTool objects
        """
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }

        response = self._send_request(request)
        if not response or "result" not in response:
            return []

        result = response["result"]
        tools = []

        for t in result.get("tools", []):
            tools.append(
                MCPTool(
                    name=t["name"],
                    description=t.get("description"),
                    input_schema=t.get("inputSchema"),
                )
            )

        return tools

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        """
        Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments (optional)

        Returns:
            Tool result (structure depends on the tool)
        """
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }

        response = self._send_request(request)
        if not response:
            return None

        # Check for error
        if "error" in response:
            error = response["error"]
            logger.error(f"Tool call '{name}' failed: {error.get('message', error)}")
            return None

        result = response.get("result", {})

        # MCP tool results have a 'content' array with text/image blocks
        content = result.get("content", [])
        if not content:
            return result

        # Extract text content (most common case)
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        if text_parts:
            return "\n".join(text_parts)

        # Return raw result if no text content
        return result

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
