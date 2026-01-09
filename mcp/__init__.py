"""
MCP (Model Context Protocol) integration for document sources.

Provides client functionality to connect to MCP servers and retrieve
resources for use in Smart RAG document indexing.
"""

from .client import MCPClient, MCPServerConfig
from .sources import DocumentSource, LocalDocumentSource, MCPDocumentSource

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "DocumentSource",
    "LocalDocumentSource",
    "MCPDocumentSource",
]
