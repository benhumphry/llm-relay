"""
MCP (Model Context Protocol) client functionality.

Provides client functionality to connect to MCP servers.
This module is retained for potential future use with generic MCP servers.
"""

from .client import MCPClient, MCPServerConfig

__all__ = [
    "MCPClient",
    "MCPServerConfig",
]
