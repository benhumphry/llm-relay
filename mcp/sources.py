"""
Document source abstractions for Smart RAG.

Provides a unified interface for retrieving documents from:
- Local filesystem (existing functionality)
- MCP servers (new functionality)
"""

import fnmatch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)

# Supported file extensions (same as RAG indexer)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".asciidoc",
    ".adoc",
    # Images (OCR)
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}


@dataclass
class DocumentInfo:
    """Information about a document available for indexing."""

    uri: str  # Unique identifier (file path or MCP URI)
    name: str  # Display name
    mime_type: Optional[str] = None
    size: Optional[int] = None


@dataclass
class DocumentContent:
    """Content of a document."""

    uri: str
    name: str
    mime_type: Optional[str] = None
    text: Optional[str] = None  # Text content
    binary: Optional[bytes] = None  # Binary content (for PDFs, images, etc.)

    @property
    def is_binary(self) -> bool:
        """Check if this is binary content."""
        return self.binary is not None


class DocumentSource(ABC):
    """Abstract base class for document sources."""

    @abstractmethod
    def list_documents(self) -> Iterator[DocumentInfo]:
        """
        List all available documents.

        Yields:
            DocumentInfo for each available document
        """
        pass

    @abstractmethod
    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """
        Read a specific document.

        Args:
            uri: Document URI

        Returns:
            DocumentContent or None if not found
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is available."""
        pass


class LocalDocumentSource(DocumentSource):
    """
    Document source for local filesystem.

    This wraps the existing local file functionality from the RAG indexer.
    """

    def __init__(self, source_path: str):
        """
        Initialize with a local directory path.

        Args:
            source_path: Path to directory containing documents
        """
        self.source_path = Path(source_path)

    def is_available(self) -> bool:
        """Check if the source path exists and is a directory."""
        return self.source_path.exists() and self.source_path.is_dir()

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all supported documents in the directory."""
        if not self.is_available():
            logger.warning(f"Source path not available: {self.source_path}")
            return

        for file_path in self.source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                relative_path = file_path.relative_to(self.source_path)
                yield DocumentInfo(
                    uri=str(file_path),
                    name=str(relative_path),
                    mime_type=self._get_mime_type(file_path.suffix),
                    size=file_path.stat().st_size,
                )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from the filesystem."""
        file_path = Path(uri)

        if not file_path.exists():
            logger.warning(f"File not found: {uri}")
            return None

        # Determine if binary or text
        suffix = file_path.suffix.lower()
        mime_type = self._get_mime_type(suffix)

        # Binary formats
        binary_formats = {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
        }

        if suffix in binary_formats:
            try:
                binary = file_path.read_bytes()
                return DocumentContent(
                    uri=uri,
                    name=file_path.name,
                    mime_type=mime_type,
                    binary=binary,
                )
            except Exception as e:
                logger.error(f"Failed to read binary file {uri}: {e}")
                return None
        else:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                return DocumentContent(
                    uri=uri,
                    name=file_path.name,
                    mime_type=mime_type,
                    text=text,
                )
            except Exception as e:
                logger.error(f"Failed to read text file {uri}: {e}")
                return None

    def _get_mime_type(self, suffix: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".html": "text/html",
            ".htm": "text/html",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".asciidoc": "text/asciidoc",
            ".adoc": "text/asciidoc",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".bmp": "image/bmp",
        }
        return mime_types.get(suffix.lower(), "application/octet-stream")


class MCPDocumentSource(DocumentSource):
    """
    Document source for MCP servers.

    Connects to an MCP server and retrieves resources for indexing.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize with MCP server configuration.

        Args:
            config: MCPServerConfig with connection details
        """
        self.config = config
        self._client: Optional[MCPClient] = None

    def is_available(self) -> bool:
        """Check if we can connect to the MCP server."""
        try:
            client = MCPClient(self.config)
            if client.connect():
                client.disconnect()
                return True
            return False
        except Exception as e:
            logger.warning(f"MCP server '{self.config.name}' not available: {e}")
            return False

    def _get_client(self) -> MCPClient:
        """Get or create a connected client."""
        if self._client is None:
            self._client = MCPClient(self.config)
            if not self._client.connect():
                raise RuntimeError(
                    f"Failed to connect to MCP server '{self.config.name}'"
                )
        return self._client

    def _close_client(self):
        """Close the client connection."""
        if self._client:
            self._client.disconnect()
            self._client = None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List documents from the MCP server (resources or tools)."""
        try:
            client = self._get_client()

            # First try resource-based listing
            resources = client.list_resources()
            resource_count = 0

            for resource in resources:
                # Apply URI pattern filtering if configured
                if self.config.uri_patterns:
                    if not self._matches_patterns(
                        resource.uri, self.config.uri_patterns
                    ):
                        continue

                # Check if it's a supported document type
                if not self._is_supported_resource(resource):
                    continue

                resource_count += 1
                yield DocumentInfo(
                    uri=resource.uri,
                    name=resource.name or resource.uri,
                    mime_type=resource.mime_type,
                )

            # If no resources found and discovery_tool is configured, try tool-based discovery
            if resource_count == 0 and self.config.discovery_tool:
                logger.info(
                    f"No resources found, trying tool-based discovery with '{self.config.discovery_tool}'"
                )
                yield from self._list_documents_via_tools(client)

        except Exception as e:
            logger.error(f"Failed to list documents from MCP server: {e}")
        finally:
            self._close_client()

    def _list_documents_via_tools(self, client: MCPClient) -> Iterator[DocumentInfo]:
        """List documents using tool-based discovery with pagination support."""
        import json

        try:
            next_cursor = None
            page_count = 0
            max_pages = 50  # Safety limit

            while page_count < max_pages:
                page_count += 1

                # Build args with pagination cursor if present
                args = (
                    dict(self.config.discovery_args)
                    if self.config.discovery_args
                    else {}
                )
                if next_cursor:
                    args["start_cursor"] = next_cursor

                # Call the discovery tool
                result = client.call_tool(self.config.discovery_tool, args)

                if not result:
                    logger.warning(
                        f"Discovery tool '{self.config.discovery_tool}' returned no results"
                    )
                    return

                # Parse response to check for pagination
                parsed = None
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except json.JSONDecodeError:
                        pass

                # Extract items from the result
                items = self._parse_discovery_result(result)

                for item in items:
                    # Extract ID for content retrieval
                    item_id = self._extract_id(item)
                    if not item_id:
                        continue

                    # Create a tool:// URI for tool-based content retrieval
                    uri = f"tool://{self.config.name}/{item_id}"
                    name = self._extract_name(item) or item_id

                    yield DocumentInfo(
                        uri=uri,
                        name=name,
                        mime_type="text/markdown",  # Tool-based servers typically return markdown
                    )

                # Check for more pages (Notion-style pagination)
                if parsed and isinstance(parsed, dict):
                    has_more = parsed.get("has_more", False)
                    next_cursor = parsed.get("next_cursor")
                    if not has_more or not next_cursor:
                        break
                    logger.debug(f"Fetching next page (cursor: {next_cursor[:20]}...)")
                else:
                    break  # No pagination info, assume single page

        except Exception as e:
            logger.error(f"Tool-based discovery failed: {e}")

    def _parse_discovery_result(self, result: Any) -> list:
        """Parse the discovery tool result into a list of items."""
        import json

        # If result is a string, try to parse as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    # Look for common list fields
                    for key in ["results", "items", "pages", "data", "objects"]:
                        if key in parsed and isinstance(parsed[key], list):
                            return parsed[key]
                    # Return as single-item list
                    return [parsed]
            except json.JSONDecodeError:
                # Not JSON, might be plain text - can't extract items
                logger.warning("Discovery result is not valid JSON")
                return []

        # If already a list, return as-is
        if isinstance(result, list):
            return result

        # If it's a dict, look for list fields
        if isinstance(result, dict):
            for key in ["results", "items", "pages", "data", "objects"]:
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]

        return []

    def _extract_id(self, item: Any) -> Optional[str]:
        """Extract the ID from a discovery result item."""
        if isinstance(item, dict):
            # Try common ID fields first
            for key in ["id", "page_id", "pageId", "uuid", "uri", "url"]:
                if key in item:
                    return str(item[key])

            # Fall back to configured field if common fields not found
            if self.config.content_id_field and self.config.content_id_field in item:
                return str(item[self.config.content_id_field])

        if isinstance(item, str):
            return item

        return None

    def _extract_name(self, item: Any) -> Optional[str]:
        """Extract a display name from a discovery result item."""
        if isinstance(item, dict):
            # First check for Notion-style properties with title type
            if "properties" in item and isinstance(item["properties"], dict):
                for prop_name, prop_val in item["properties"].items():
                    if isinstance(prop_val, dict) and prop_val.get("type") == "title":
                        title_arr = prop_val.get("title", [])
                        if title_arr and isinstance(title_arr[0], dict):
                            return title_arr[0].get("plain_text", "")

            # Fall back to top-level fields
            for key in ["title", "name", "label", "heading"]:
                if key in item:
                    val = item[key]
                    # Handle Notion-style title objects
                    if isinstance(val, list) and val:
                        if isinstance(val[0], dict) and "plain_text" in val[0]:
                            return val[0]["plain_text"]
                        return str(val[0])
                    return str(val)
        return None

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from the MCP server (resource or tool-based)."""
        try:
            client = self._get_client()

            # Check if this is a tool-based URI
            if uri.startswith("tool://"):
                return self._read_document_via_tool(client, uri)

            # Standard resource-based reading
            content = client.read_resource(uri)

            if not content:
                return None

            return DocumentContent(
                uri=uri,
                name=uri.split("/")[-1],
                mime_type=content.mime_type,
                text=content.text,
                binary=content.blob,
            )

        except Exception as e:
            logger.error(f"Failed to read document {uri}: {e}")
            return None
        finally:
            self._close_client()

    def _read_document_via_tool(
        self, client: MCPClient, uri: str
    ) -> Optional[DocumentContent]:
        """Read document content using the configured content tool."""
        if not self.config.content_tool:
            logger.error(f"No content_tool configured for tool-based URI: {uri}")
            return None

        # Extract the ID from the URI (tool://server_name/item_id)
        parts = uri.replace("tool://", "").split("/", 1)
        if len(parts) < 2:
            logger.error(f"Invalid tool URI format: {uri}")
            return None

        item_id = parts[1]

        # Call the content tool with the page_id argument
        # Most tools expect an argument like "page_id" or "id"
        arg_name = self.config.content_id_field or "page_id"
        result = client.call_tool(self.config.content_tool, {arg_name: item_id})

        if not result:
            logger.warning(
                f"Content tool '{self.config.content_tool}' returned no content for {item_id}"
            )
            return None

        # Parse and extract text from the result
        text_content = self._extract_text_from_result(result)

        return DocumentContent(
            uri=uri,
            name=item_id,
            mime_type="text/markdown",
            text=text_content,
        )

    def _extract_text_from_result(self, result: Any) -> str:
        """Extract plain text from tool result, handling Notion block structures."""
        import json

        # Parse JSON if string
        if isinstance(result, str):
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                # Not JSON, return as-is
                return result
        else:
            data = result

        # Handle Notion block list response
        if isinstance(data, dict) and data.get("object") == "list":
            blocks = data.get("results", [])
            return self._extract_text_from_blocks(blocks)

        # Handle plain dict/list
        if isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)

        return str(data)

    def _extract_text_from_blocks(self, blocks: list) -> str:
        """Extract text from Notion blocks into markdown."""
        lines = []

        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            # Extract rich text
            rich_text = block_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)

            # Format based on block type
            if block_type == "paragraph":
                if text:
                    lines.append(text)
                    lines.append("")
            elif block_type.startswith("heading_"):
                level = block_type[-1]  # heading_1, heading_2, heading_3
                prefix = "#" * int(level)
                lines.append(f"{prefix} {text}")
                lines.append("")
            elif block_type == "bulleted_list_item":
                lines.append(f"- {text}")
            elif block_type == "numbered_list_item":
                lines.append(f"1. {text}")
            elif block_type == "to_do":
                checked = block_data.get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                lines.append(f"- {checkbox} {text}")
            elif block_type == "toggle":
                lines.append(f"<details><summary>{text}</summary></details>")
                lines.append("")
            elif block_type == "code":
                language = block_data.get("language", "")
                lines.append(f"```{language}")
                lines.append(text)
                lines.append("```")
                lines.append("")
            elif block_type == "quote":
                lines.append(f"> {text}")
                lines.append("")
            elif block_type == "callout":
                icon = block_data.get("icon", {}).get("emoji", "")
                lines.append(f"> {icon} {text}")
                lines.append("")
            elif block_type == "divider":
                lines.append("---")
                lines.append("")
            elif block_type == "table_row":
                cells = block_data.get("cells", [])
                cell_texts = []
                for cell in cells:
                    cell_text = "".join(rt.get("plain_text", "") for rt in cell)
                    cell_texts.append(cell_text)
                lines.append("| " + " | ".join(cell_texts) + " |")
            elif text:
                # Fallback for other types with text
                lines.append(text)
                lines.append("")

        return "\n".join(lines).strip()

    def _matches_patterns(self, uri: str, patterns: list[str]) -> bool:
        """Check if a URI matches any of the configured patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(uri, pattern):
                return True
        return False

    def _is_supported_resource(self, resource) -> bool:
        """Check if a resource is a supported document type."""
        # Check MIME type first
        if resource.mime_type:
            supported_mimes = {
                "application/pdf",
                "text/plain",
                "text/markdown",
                "text/html",
                "image/png",
                "image/jpeg",
            }
            if any(resource.mime_type.startswith(m) for m in supported_mimes):
                return True

        # Fallback to checking URI extension
        uri_lower = resource.uri.lower()
        for ext in SUPPORTED_EXTENSIONS:
            if uri_lower.endswith(ext):
                return True

        # Google Docs exports (no extension, but MIME type indicates doc)
        if resource.mime_type and "google" in resource.mime_type.lower():
            return True

        return False


def _inject_mcp_env_vars(mcp_config: dict) -> dict:
    """
    Inject required environment variables into MCP config based on server name.

    Different MCP servers expect different environment variable names:
    - Notion: NOTION_TOKEN
    - GitHub: GITHUB_TOKEN
    - Slack: SLACK_BOT_TOKEN, SLACK_TEAM_ID
    """
    import os

    config = mcp_config.copy()
    env = config.get("env", {}) or {}
    server_name = config.get("name", "").lower()

    # Notion MCP server expects NOTION_TOKEN
    if server_name == "notion":
        if "NOTION_TOKEN" not in env:
            token = os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
            if token:
                env["NOTION_TOKEN"] = token

    # GitHub MCP server expects GITHUB_TOKEN or GITHUB_PERSONAL_ACCESS_TOKEN
    elif server_name == "github":
        if "GITHUB_TOKEN" not in env and "GITHUB_PERSONAL_ACCESS_TOKEN" not in env:
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get(
                "GITHUB_PERSONAL_ACCESS_TOKEN"
            )
            if token:
                env["GITHUB_TOKEN"] = token

    # Slack MCP server expects SLACK_BOT_TOKEN and SLACK_TEAM_ID
    elif server_name == "slack":
        if "SLACK_BOT_TOKEN" not in env:
            token = os.environ.get("SLACK_BOT_TOKEN")
            if token:
                env["SLACK_BOT_TOKEN"] = token
        if "SLACK_TEAM_ID" not in env:
            team_id = os.environ.get("SLACK_TEAM_ID")
            if team_id:
                env["SLACK_TEAM_ID"] = team_id

    config["env"] = env
    return config


def get_document_source(
    source_type: str,
    source_path: Optional[str] = None,
    mcp_config: Optional[dict] = None,
) -> DocumentSource:
    """
    Factory function to create the appropriate document source.

    Args:
        source_type: "local" or "mcp"
        source_path: Path for local sources
        mcp_config: Configuration dict for MCP sources

    Returns:
        Appropriate DocumentSource instance
    """
    if source_type == "local":
        if not source_path:
            raise ValueError("source_path required for local document source")
        return LocalDocumentSource(source_path)

    elif source_type == "mcp":
        if not mcp_config:
            raise ValueError("mcp_config required for MCP document source")

        # Inject required environment variables based on server name
        mcp_config = _inject_mcp_env_vars(mcp_config)

        config = MCPServerConfig.from_dict(mcp_config)
        return MCPDocumentSource(config)

    else:
        raise ValueError(f"Unknown source type: {source_type}")
