"""
Document source abstractions for Smart RAG.

Provides a unified interface for retrieving documents from:
- Local filesystem (existing functionality)
- MCP servers (new functionality)
"""

import fnmatch
import logging
import os
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
    modified_time: Optional[str] = None  # ISO format timestamp for incremental indexing


@dataclass
class DocumentContent:
    """Content of a document."""

    uri: str
    name: str
    mime_type: Optional[str] = None
    text: Optional[str] = None  # Text content
    binary: Optional[bytes] = None  # Binary content (for PDFs, images, etc.)
    metadata: Optional[dict] = None  # Additional metadata (dates, etc.) for filtering

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

    # Live lookup capability for unified sources

    @property
    def supports_live_lookup(self) -> bool:
        """
        Whether this source supports on-demand document fetch.

        Most sources can support this since they implement read_document().
        Override to return False for sources that can't fetch individual docs.
        """
        return True

    def resolve_document(
        self,
        query: str,
        indexed_metadata: Optional[list[dict]] = None,
    ) -> Optional[str]:
        """
        Resolve a natural language reference to a document URI.

        Uses fuzzy matching against indexed document metadata to find the
        best matching document for a user's reference like "the October invoice"
        or "INVOICE-2022.docx".

        Args:
            query: User's reference to a document (filename, title, description)
            indexed_metadata: Metadata from RAG chunks for this store, each dict
                              may contain 'source_path', 'title', 'name', etc.

        Returns:
            Document URI if resolved, None if ambiguous or not found
        """
        return self._fuzzy_match_document(query, indexed_metadata or [])

    def _fuzzy_match_document(
        self,
        query: str,
        indexed_metadata: list[dict],
    ) -> Optional[str]:
        """
        Default fuzzy matching implementation using rapidfuzz.

        Builds a candidate list from metadata and finds the best match.
        """
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.warning("rapidfuzz not installed, falling back to exact match")
            return self._exact_match_document(query, indexed_metadata)

        # Build candidates: {match_target: uri}
        candidates: dict[str, str] = {}
        for meta in indexed_metadata:
            uri = meta.get("source_path") or meta.get("uri") or meta.get("doc_id")
            if not uri:
                continue

            # Use multiple fields as match targets
            targets = []
            if meta.get("title"):
                targets.append(meta["title"])
            if meta.get("name"):
                targets.append(meta["name"])
            if uri:
                # Also match against filename portion of path
                targets.append(Path(uri).name)

            for target in targets:
                target_lower = target.lower()
                if target_lower not in candidates:
                    candidates[target_lower] = uri

        if not candidates:
            logger.debug(f"No candidates found for document resolution: {query}")
            return None

        # Fuzzy match with partial_ratio for better substring matching
        query_lower = query.lower()
        match = process.extractOne(
            query_lower,
            candidates.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=70,
        )

        if match:
            matched_target, score, _ = match
            uri = candidates[matched_target]
            logger.info(
                f"Resolved document query '{query}' -> '{uri}' (score={score:.0f})"
            )
            return uri

        logger.debug(f"No fuzzy match found for document query: {query}")
        return None

    def _exact_match_document(
        self,
        query: str,
        indexed_metadata: list[dict],
    ) -> Optional[str]:
        """Fallback exact substring matching when rapidfuzz is not available."""
        query_lower = query.lower()

        for meta in indexed_metadata:
            uri = meta.get("source_path") or meta.get("uri") or meta.get("doc_id")
            if not uri:
                continue

            # Check title, name, and filename
            targets = [
                meta.get("title", ""),
                meta.get("name", ""),
                Path(uri).name,
            ]

            for target in targets:
                if query_lower in target.lower() or target.lower() in query_lower:
                    logger.info(f"Exact match resolved '{query}' -> '{uri}'")
                    return uri

        return None


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
        from datetime import datetime

        if not self.is_available():
            logger.warning(f"Source path not available: {self.source_path}")
            return

        for file_path in self.source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                relative_path = file_path.relative_to(self.source_path)
                stat = file_path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
                yield DocumentInfo(
                    uri=str(file_path),
                    name=str(relative_path),
                    mime_type=self._get_mime_type(file_path.suffix),
                    size=stat.st_size,
                    modified_time=modified_time,
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
        import re

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
                    # Support different pagination param names
                    if self.config.name == "gdrive":
                        args["pageToken"] = next_cursor
                    else:
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
                page_token = None
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except json.JSONDecodeError:
                        # Check for Google Drive style pagination in text
                        # "More results available. Use pageToken: <token>"
                        match = re.search(r"Use pageToken:\s*(\S+)", result)
                        if match:
                            page_token = match.group(1)

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

                    # Get mime type if available
                    mime_type = (
                        item.get("mimeType", "text/markdown")
                        if isinstance(item, dict)
                        else "text/markdown"
                    )

                    # Extract modified time (Notion: last_edited_time, others: modifiedTime)
                    modified_time = None
                    if isinstance(item, dict):
                        modified_time = (
                            item.get("last_edited_time")
                            or item.get("modifiedTime")
                            or item.get("updated_at")
                            or item.get("modified")
                        )

                    yield DocumentInfo(
                        uri=uri,
                        name=name,
                        mime_type=mime_type,
                        modified_time=modified_time,
                    )

                # Check for more pages
                if parsed and isinstance(parsed, dict):
                    # Notion-style pagination
                    has_more = parsed.get("has_more", False)
                    next_cursor = parsed.get("next_cursor")
                    if not has_more or not next_cursor:
                        break
                    logger.debug(f"Fetching next page (cursor: {next_cursor[:20]}...)")
                elif page_token:
                    # Google Drive style pagination
                    next_cursor = page_token
                    logger.debug(
                        f"Fetching next page (pageToken: {next_cursor[:20]}...)"
                    )
                else:
                    break  # No pagination info, assume single page

        except Exception as e:
            logger.error(f"Tool-based discovery failed: {e}")

    def _parse_discovery_result(self, result: Any) -> list:
        """Parse the discovery tool result into a list of items."""
        import json
        import re

        # If result is a string, try to parse as JSON first
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
                # Not JSON - try Google Drive text format parsing
                # Format: "fileId fileName (mimeType)"
                items = []
                for line in result.split("\n"):
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("Found ")
                        or line.startswith("More results")
                    ):
                        continue

                    # Parse: "fileId name (mimeType)"
                    match = re.match(r"^(\S+)\s+(.+?)\s+\(([^)]+)\)$", line)
                    if match:
                        file_id, name, mime_type = match.groups()
                        items.append(
                            {
                                "id": file_id,
                                "name": name,
                                "mimeType": mime_type,
                            }
                        )

                if items:
                    return items

                logger.warning(
                    "Discovery result is not valid JSON or known text format"
                )
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


def _setup_google_oauth_files(account_id: int) -> Optional[str]:
    """
    Set up credential files for mcp-gsuite based on stored OAuth token.

    Creates .gauth.json, .accounts.json, and .oauth.{email}.json files
    in a temporary directory for mcp-gsuite to use.

    Args:
        account_id: ID of the stored OAuth token

    Returns:
        Path to the credentials directory, or None if setup failed
    """
    import json

    from db.oauth_tokens import get_oauth_token_by_id, list_oauth_tokens

    # Get the token by ID
    tokens = list_oauth_tokens(provider="google")
    token_meta = next((t for t in tokens if t["id"] == account_id), None)
    if not token_meta:
        logger.error(f"OAuth token {account_id} not found")
        return None

    token_data = get_oauth_token_by_id(account_id)
    if not token_data:
        logger.error(f"Failed to decrypt OAuth token {account_id}")
        return None

    account_email = token_meta["account_email"]

    # Create a persistent temp directory for credentials
    # Use a fixed path based on account ID so files persist across calls
    creds_dir = f"/tmp/llm-relay-oauth-{account_id}"
    os.makedirs(creds_dir, exist_ok=True)

    # Create .gauth.json with client credentials
    gauth_path = os.path.join(creds_dir, ".gauth.json")
    gauth_data = {
        "web": {
            "client_id": token_data.get("client_id"),
            "client_secret": token_data.get("client_secret"),
            "redirect_uris": ["http://localhost:4100/code"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    with open(gauth_path, "w") as f:
        json.dump(gauth_data, f)

    # Create .accounts.json with account info
    accounts_path = os.path.join(creds_dir, ".accounts.json")
    accounts_data = {
        "accounts": [
            {
                "email": account_email,
                "account_type": "personal",
            }
        ]
    }
    with open(accounts_path, "w") as f:
        json.dump(accounts_data, f)

    # Create .oauth.{email}.json with the actual OAuth tokens
    oauth_path = os.path.join(creds_dir, f".oauth.{account_email}.json")
    oauth_data = {
        "token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": token_data.get("client_id"),
        "client_secret": token_data.get("client_secret"),
        "scopes": token_meta.get("scopes", []),
        "expiry": token_data.get("expiry"),
    }
    with open(oauth_path, "w") as f:
        json.dump(oauth_data, f)

    logger.info(f"Set up OAuth credentials for {account_email} in {creds_dir}")
    return creds_dir


def _setup_gdrive_oauth_files(account_id: int) -> Optional[dict]:
    """
    Set up credential files for @isaacphi/mcp-gdrive based on stored OAuth token.

    Creates gcp-oauth.keys.json and credentials.json files in the format
    expected by the mcp-gdrive package.

    Args:
        account_id: ID of the stored OAuth token

    Returns:
        Dict with creds_dir, client_id, client_secret, or None if setup failed
    """
    import json

    from db.oauth_tokens import get_oauth_token_by_id, list_oauth_tokens

    # Get the token by ID
    tokens = list_oauth_tokens(provider="google")
    token_meta = next((t for t in tokens if t["id"] == account_id), None)
    if not token_meta:
        logger.error(f"OAuth token {account_id} not found")
        return None

    token_data = get_oauth_token_by_id(account_id)
    if not token_data:
        logger.error(f"Failed to decrypt OAuth token {account_id}")
        return None

    client_id = token_data.get("client_id")
    client_secret = token_data.get("client_secret")

    # Create a persistent temp directory for credentials
    # Use a fixed path based on account ID so files persist across calls
    creds_dir = f"/tmp/llm-relay-gdrive-{account_id}"
    os.makedirs(creds_dir, exist_ok=True)

    # Create gcp-oauth.keys.json with client credentials
    # @isaacphi/mcp-gdrive expects this specific filename
    keys_path = os.path.join(creds_dir, "gcp-oauth.keys.json")
    keys_data = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    with open(keys_path, "w") as f:
        json.dump(keys_data, f)

    # Create .gdrive-server-credentials.json with the OAuth tokens
    # This is the token file that mcp-gdrive expects after authentication
    creds_path = os.path.join(creds_dir, ".gdrive-server-credentials.json")
    creds_data = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "scope": " ".join(token_meta.get("scopes", [])),
        "token_type": "Bearer",
        "expiry_date": token_data.get("expiry"),
    }
    with open(creds_path, "w") as f:
        json.dump(creds_data, f)

    account_email = token_meta["account_email"]
    logger.info(f"Set up Google Drive credentials for {account_email} in {creds_dir}")
    return {
        "creds_dir": creds_dir,
        "client_id": client_id,
        "client_secret": client_secret,
    }


class GoogleDriveDocumentSource(DocumentSource):
    """
    Document source for Google Drive using direct API calls.

    Uses Google Drive API v3 directly with stored OAuth tokens,
    bypassing MCP for better performance and proper folder filtering.

    Supports Docling processing with optional vision models for PDFs
    and other binary document formats.
    """

    # Google Workspace MIME types that need export
    GOOGLE_EXPORT_TYPES = {
        "application/vnd.google-apps.document": ("text/plain", ".txt"),
        "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
        "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
        "application/vnd.google-apps.drawing": ("image/png", ".png"),
    }

    # Binary file types that should be processed through Docling
    # Maps MIME type to file extension for temp file creation
    DOCLING_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }

    def __init__(
        self,
        account_id: int,
        folder_id: Optional[str] = None,
        vision_provider: str = "local",
        vision_model: Optional[str] = None,
        vision_ollama_url: Optional[str] = None,
    ):
        """
        Initialize with OAuth account ID and optional folder filter.

        Args:
            account_id: ID of the stored OAuth token
            folder_id: Optional folder ID to restrict indexing scope
            vision_provider: Vision model provider for Docling ("local", "ollama", etc.)
            vision_model: Vision model name (e.g., "granite3.2-vision:latest")
            vision_ollama_url: Ollama URL when using ollama provider
        """
        self.account_id = account_id
        self.folder_id = folder_id
        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_ollama_url = vision_ollama_url
        self._access_token: Optional[str] = None
        self._token_data: Optional[dict] = None
        self._document_converter = None

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        import requests as http_requests

        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"OAuth token {self.account_id} not found")
            return None

        # Check if token needs refresh (simplified - always try to refresh if we have refresh_token)
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            logger.error("No access token in stored credentials")
            return None

        # Try the access token first
        test_response = http_requests.get(
            "https://www.googleapis.com/drive/v3/about",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "user"},
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

        # Update cache
        self._token_data = updated_data
        logger.info("Token refreshed successfully")
        return new_access_token

    def is_available(self) -> bool:
        """Check if we can access Google Drive."""
        return self._get_access_token() is not None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all documents in Drive or specified folder."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list documents - no valid access token")
            return

        # Build query - filter by folder if specified
        if self.folder_id:
            query = f"'{self.folder_id}' in parents and trashed = false"
            logger.info(f"Listing documents in folder: {self.folder_id}")
        else:
            query = "trashed = false"
            logger.info("Listing all documents in Drive")

        # Fields to retrieve
        fields = "nextPageToken, files(id, name, mimeType, size, modifiedTime)"

        page_token = None
        total_files = 0

        while True:
            params = {
                "q": query,
                "fields": fields,
                "pageSize": 100,
                "orderBy": "modifiedTime desc",
            }
            if page_token:
                params["pageToken"] = page_token

            response = http_requests.get(
                "https://www.googleapis.com/drive/v3/files",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Drive API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            files = data.get("files", [])

            for file_info in files:
                mime_type = file_info.get("mimeType", "")
                file_id = file_info["id"]
                name = file_info["name"]

                # Skip folders
                if mime_type == "application/vnd.google-apps.folder":
                    continue

                # Check if it's a supported type
                if not self._is_supported_file(mime_type, name):
                    continue

                total_files += 1
                yield DocumentInfo(
                    uri=f"gdrive://{file_id}",
                    name=name,
                    mime_type=mime_type,
                    size=int(file_info.get("size", 0))
                    if file_info.get("size")
                    else None,
                    modified_time=file_info.get("modifiedTime"),
                )

            # Check for more pages
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Found {total_files} documents to index")

    def _is_supported_file(self, mime_type: str, name: str) -> bool:
        """Check if a file type is supported for indexing."""
        # Google Workspace types we can export
        if mime_type in self.GOOGLE_EXPORT_TYPES:
            return True

        # Binary types we can process through Docling
        if mime_type in self.DOCLING_MIME_TYPES:
            return True

        # Check file extension for text types
        name_lower = name.lower()
        text_extensions = {".txt", ".md", ".html", ".htm", ".asciidoc", ".adoc"}
        for ext in text_extensions:
            if name_lower.endswith(ext):
                return True

        return False

    def _get_document_converter(self):
        """Get or create a Docling DocumentConverter with vision support."""
        if self._document_converter is None:
            from rag.vision import get_document_converter

            self._document_converter = get_document_converter(
                vision_provider=self.vision_provider,
                vision_model=self.vision_model,
                vision_ollama_url=self.vision_ollama_url,
            )
        return self._document_converter

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from Google Drive."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read document - no valid access token")
            return None

        # Extract file ID from URI (gdrive://file_id)
        if not uri.startswith("gdrive://"):
            logger.error(f"Invalid Drive URI: {uri}")
            return None

        file_id = uri.replace("gdrive://", "")

        # First get file metadata to determine how to download
        meta_response = http_requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "id, name, mimeType, size"},
            timeout=30,
        )

        if meta_response.status_code != 200:
            logger.error(f"Failed to get file metadata: {meta_response.text}")
            return None

        file_meta = meta_response.json()
        mime_type = file_meta.get("mimeType", "")
        name = file_meta.get("name", file_id)

        # Handle Google Workspace files (need export)
        if mime_type in self.GOOGLE_EXPORT_TYPES:
            export_mime, _ = self.GOOGLE_EXPORT_TYPES[mime_type]
            return self._export_google_doc(access_token, file_id, name, export_mime)

        # Handle binary files through Docling (PDF, DOCX, images, etc.)
        if mime_type in self.DOCLING_MIME_TYPES:
            return self._download_and_process_with_docling(
                access_token, file_id, name, mime_type
            )

        # Handle text files
        return self._download_text(access_token, file_id, name, mime_type)

    def _export_google_doc(
        self, access_token: str, file_id: str, name: str, export_mime: str
    ) -> Optional[DocumentContent]:
        """Export a Google Workspace document to text/csv format."""
        import requests as http_requests

        response = http_requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"mimeType": export_mime},
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"Failed to export Google Doc {name}: {response.status_code}")
            return None

        return DocumentContent(
            uri=f"gdrive://{file_id}",
            name=name,
            mime_type=export_mime,
            text=response.text,
        )

    def _download_and_process_with_docling(
        self, access_token: str, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Download a binary file and process it through Docling with vision support."""
        import tempfile

        import requests as http_requests

        # Download the file
        response = http_requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"alt": "media"},
            timeout=120,
        )

        if response.status_code != 200:
            logger.error(f"Failed to download {name}: {response.status_code}")
            return None

        # Get file extension for temp file
        extension = self.DOCLING_MIME_TYPES.get(mime_type, ".bin")

        # Save to temp file and process with Docling
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                converter = self._get_document_converter()
                result = converter.convert(tmp_path)
                text = result.document.export_to_markdown()

                logger.debug(f"Processed {name} with Docling: {len(text)} chars")

                return DocumentContent(
                    uri=f"gdrive://{file_id}",
                    name=name,
                    mime_type="text/markdown",
                    text=text,
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Docling processing failed for {name}: {e}")
            # Return None on failure - let the indexer skip this document
            return None

    def _download_text(
        self, access_token: str, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Download a text file from Drive."""
        import requests as http_requests

        response = http_requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"alt": "media"},
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"Failed to download {name}: {response.status_code}")
            return None

        # Try to decode as text
        try:
            text = response.content.decode("utf-8", errors="ignore")
        except Exception:
            text = response.text

        return DocumentContent(
            uri=f"gdrive://{file_id}",
            name=name,
            mime_type=mime_type or "text/plain",
            text=text,
        )


class GmailDocumentSource(DocumentSource):
    """
    Document source for Gmail using direct API calls.

    Uses Gmail API directly with stored OAuth tokens,
    bypassing MCP for better performance and reliability.
    """

    def __init__(self, account_id: int, label_id: Optional[str] = None):
        """
        Initialize with OAuth account ID and optional label filter.

        Args:
            account_id: ID of the stored OAuth token
            label_id: Optional label ID to filter emails (e.g., "INBOX", "SENT")
        """
        self.account_id = account_id
        self.label_id = label_id
        self._access_token: Optional[str] = None
        self._token_data: Optional[dict] = None

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
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

        # Try the access token first
        test_response = http_requests.get(
            "https://www.googleapis.com/gmail/v1/users/me/profile",
            headers={"Authorization": f"Bearer {access_token}"},
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
        """Check if we can access Gmail."""
        return self._get_access_token() is not None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List emails from Gmail."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list emails - no valid access token")
            return

        # Build query parameters
        params = {
            "maxResults": 100,
        }

        # Filter by label if specified
        if self.label_id:
            params["labelIds"] = self.label_id
            logger.info(f"Listing emails with label: {self.label_id}")
        else:
            logger.info("Listing all emails")

        page_token = None
        total_emails = 0

        while True:
            if page_token:
                params["pageToken"] = page_token

            response = http_requests.get(
                "https://www.googleapis.com/gmail/v1/users/me/messages",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Gmail API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            messages = data.get("messages", [])

            for msg in messages:
                total_emails += 1
                yield DocumentInfo(
                    uri=f"gmail://{msg['id']}",
                    name=msg["id"],  # Will be replaced with subject when reading
                    mime_type="message/rfc822",
                )

            # Check for more pages
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Found {total_emails} emails to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read an email from Gmail."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read email - no valid access token")
            return None

        # Extract message ID from URI (gmail://message_id)
        if not uri.startswith("gmail://"):
            logger.error(f"Invalid Gmail URI: {uri}")
            return None

        message_id = uri.replace("gmail://", "")

        # Get full message
        response = http_requests.get(
            f"https://www.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"format": "full"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get email {message_id}: {response.status_code}")
            return None

        msg_data = response.json()

        # Extract headers
        headers = {
            h["name"].lower(): h["value"]
            for h in msg_data.get("payload", {}).get("headers", [])
        }
        subject = headers.get("subject", "(No Subject)")
        from_addr = headers.get("from", "")
        to_addr = headers.get("to", "")
        date = headers.get("date", "")

        # Extract body text
        body_text = self._extract_body(msg_data.get("payload", {}))

        # Format as readable text
        content = f"""Subject: {subject}
From: {from_addr}
To: {to_addr}
Date: {date}

{body_text}
"""

        # Parse email date to YYYY-MM-DD format for filtering
        email_date = None
        if date:
            try:
                from email.utils import parsedate_to_datetime

                parsed = parsedate_to_datetime(date)
                email_date = parsed.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Parse sender name from "Name <email@example.com>" format
        from_name = from_addr
        if "<" in from_addr:
            from_name = from_addr.split("<")[0].strip().strip('"')
        elif "@" in from_addr:
            from_name = from_addr.split("@")[0]

        # Get account email for action routing
        token_data = self._get_token_data()
        account_email = token_data.get("account_email", "") if token_data else ""

        # Get thread ID for reply threading
        thread_id = msg_data.get("threadId", "")

        return DocumentContent(
            uri=uri,
            name=subject[:100] if subject else message_id,
            mime_type="text/plain",
            text=content,
            metadata={
                "message_id": message_id,  # For replies/forwards
                "thread_id": thread_id,  # For threading context
                "account_id": self.account_id,  # Internal OAuth account ID
                "account_email": account_email,  # For action routing
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
        import base64

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

        # Multipart - look for text/plain first, then text/html
        if parts:
            # First pass: look for text/plain
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

            # Second pass: look for text/html and strip tags
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
                nested_parts = part.get("parts", [])
                if nested_parts:
                    result = self._extract_body(part)
                    if result:
                        return result

        return ""

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and decode entities."""
        import html as html_module
        import re

        # Remove script and style elements
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

        # Replace common block elements with newlines
        html = re.sub(r"<(br|p|div|tr|li)[^>]*>", "\n", html, flags=re.IGNORECASE)

        # Remove all remaining tags
        html = re.sub(r"<[^>]+>", "", html)

        # Decode HTML entities
        html = html_module.unescape(html)

        # Clean up whitespace
        html = re.sub(r"\n\s*\n", "\n\n", html)
        html = html.strip()

        return html


class GoogleCalendarDocumentSource(DocumentSource):
    """
    Document source for Google Calendar using direct API calls.

    Uses Google Calendar API directly with stored OAuth tokens,
    bypassing MCP for better performance and reliability.
    """

    def __init__(
        self,
        account_id: int,
        calendar_id: Optional[str] = None,
        days_back: int = 30,
        days_forward: int = 30,
    ):
        """
        Initialize with OAuth account ID and optional calendar filter.

        Args:
            account_id: ID of the stored OAuth token
            calendar_id: Optional calendar ID to filter (default: primary)
            days_back: How many days in the past to index
            days_forward: How many days in the future to index
        """
        self.account_id = account_id
        self.calendar_id = calendar_id or "primary"
        self.days_back = days_back
        self.days_forward = days_forward
        self._access_token: Optional[str] = None
        self._token_data: Optional[dict] = None

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
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

        # Try the access token first
        test_response = http_requests.get(
            "https://www.googleapis.com/calendar/v3/users/me/calendarList",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"maxResults": 1},
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
        """Check if we can access Google Calendar."""
        return self._get_access_token() is not None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List calendar events."""
        from datetime import datetime, timedelta

        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list events - no valid access token")
            return

        # Calculate time range
        now = datetime.utcnow()
        time_min = (now - timedelta(days=self.days_back)).isoformat() + "Z"
        time_max = (now + timedelta(days=self.days_forward)).isoformat() + "Z"

        logger.info(
            f"Listing calendar events from {self.calendar_id} ({self.days_back} days back, {self.days_forward} days forward)"
        )

        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": 250,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        page_token = None
        total_events = 0

        while True:
            if page_token:
                params["pageToken"] = page_token

            response = http_requests.get(
                f"https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}/events",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Calendar API error: {response.status_code} - {response.text}"
                )
                logger.error(
                    f"Calendar API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            events = data.get("items", [])

            for event in events:
                event_id = event.get("id")
                summary = event.get("summary", "(No title)")
                # Use updated timestamp for change detection
                modified_time = event.get("updated")

                total_events += 1
                yield DocumentInfo(
                    uri=f"gcal://{self.calendar_id}/{event_id}",
                    name=summary[:100],
                    mime_type="text/calendar",
                    modified_time=modified_time,
                )

            # Check for more pages
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Found {total_events} calendar events to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a calendar event."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read event - no valid access token")
            return None

        # Extract calendar ID and event ID from URI (gcal://calendar_id/event_id)
        if not uri.startswith("gcal://"):
            logger.error(f"Invalid Calendar URI: {uri}")
            return None

        parts = uri.replace("gcal://", "").split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid Calendar URI format: {uri}")
            return None

        calendar_id, event_id = parts

        response = http_requests.get(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get event {event_id}: {response.status_code}")
            return None

        event = response.json()

        # Extract event details
        summary = event.get("summary", "(No title)")
        description = event.get("description", "")
        location = event.get("location", "")

        # Parse start/end times
        start = event.get("start", {})
        end = event.get("end", {})
        start_time = start.get("dateTime") or start.get("date", "")
        end_time = end.get("dateTime") or end.get("date", "")

        # Get attendees
        attendees = event.get("attendees", [])
        attendee_list = ", ".join(
            a.get("email", "") for a in attendees if a.get("email")
        )

        # Get organizer
        organizer = event.get("organizer", {})
        organizer_email = organizer.get("email", "")

        # Format as readable text
        content = f"""Event: {summary}
Start: {start_time}
End: {end_time}
Location: {location}
Organizer: {organizer_email}
Attendees: {attendee_list}

{description}
"""

        # Extract date for metadata (YYYY-MM-DD format)
        event_date = None
        if start_time:
            # Handle both dateTime (2026-01-11T10:00:00Z) and date (2026-01-11) formats
            event_date = start_time[:10] if len(start_time) >= 10 else start_time

        return DocumentContent(
            uri=uri,
            name=summary[:100],
            mime_type="text/plain",
            text=content.strip(),
            metadata={
                "event_date": event_date,
                "start_time": start_time,
                "end_time": end_time,
                "location": location or None,
                "source_type": "calendar",
            },
        )


class GoogleTasksDocumentSource(DocumentSource):
    """
    Document source for Google Tasks using direct API calls.

    Uses Google Tasks API directly with stored OAuth tokens.
    Indexes task lists and their tasks for RAG retrieval.
    """

    def __init__(
        self,
        account_id: int,
        tasklist_id: Optional[str] = None,
        include_completed: bool = True,
    ):
        """
        Initialize with OAuth account ID and optional task list filter.

        Args:
            account_id: ID of the stored OAuth token
            tasklist_id: Optional task list ID to filter (default: all lists)
            include_completed: Whether to include completed tasks
        """
        self.account_id = account_id
        self.tasklist_id = tasklist_id
        self.include_completed = include_completed
        self._access_token: Optional[str] = None
        self._token_data: Optional[dict] = None

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
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

        # Try the access token first with a simple API call
        test_response = http_requests.get(
            "https://tasks.googleapis.com/tasks/v1/users/@me/lists",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"maxResults": 1},
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
        """Check if we can access Google Tasks."""
        return self._get_access_token() is not None

    def _get_task_lists(self, access_token: str) -> list[dict]:
        """Get all task lists for the user."""
        import requests as http_requests

        task_lists = []
        page_token = None

        while True:
            params = {"maxResults": 100}
            if page_token:
                params["pageToken"] = page_token

            response = http_requests.get(
                "https://tasks.googleapis.com/tasks/v1/users/@me/lists",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Tasks API error listing task lists: {response.status_code}"
                )
                break

            data = response.json()
            task_lists.extend(data.get("items", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return task_lists

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List tasks from Google Tasks."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list tasks - no valid access token")
            return

        # Get task lists
        if self.tasklist_id:
            task_lists = [{"id": self.tasklist_id, "title": "Selected List"}]
        else:
            task_lists = self._get_task_lists(access_token)

        logger.info(f"Listing tasks from {len(task_lists)} task list(s)")

        total_tasks = 0

        for tasklist in task_lists:
            tasklist_id = tasklist.get("id")
            tasklist_title = tasklist.get("title", "Untitled")

            params = {
                "maxResults": 100,
                "showCompleted": str(self.include_completed).lower(),
            }
            page_token = None

            while True:
                if page_token:
                    params["pageToken"] = page_token

                response = http_requests.get(
                    f"https://tasks.googleapis.com/tasks/v1/lists/{tasklist_id}/tasks",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=30,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Tasks API error: {response.status_code} - {response.text}"
                    )
                    break

                data = response.json()
                tasks = data.get("items", [])

                for task in tasks:
                    task_id = task.get("id")
                    title = task.get("title", "(No title)")
                    updated = task.get("updated")

                    if not title.strip():
                        continue  # Skip empty tasks

                    total_tasks += 1
                    yield DocumentInfo(
                        uri=f"gtasks://{tasklist_id}/{task_id}",
                        name=f"{tasklist_title}: {title[:80]}",
                        mime_type="text/plain",
                        modified_time=updated,
                    )

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        logger.info(f"Found {total_tasks} tasks to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a task from Google Tasks."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read task - no valid access token")
            return None

        # Extract task list ID and task ID from URI (gtasks://tasklist_id/task_id)
        if not uri.startswith("gtasks://"):
            logger.error(f"Invalid Tasks URI: {uri}")
            return None

        parts = uri.replace("gtasks://", "").split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid Tasks URI format: {uri}")
            return None

        tasklist_id, task_id = parts

        response = http_requests.get(
            f"https://tasks.googleapis.com/tasks/v1/lists/{tasklist_id}/tasks/{task_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get task {task_id}: {response.status_code}")
            return None

        task = response.json()

        # Extract task details
        title = task.get("title", "(No title)")
        notes = task.get("notes", "")
        status = task.get("status", "needsAction")
        due = task.get("due", "")
        completed = task.get("completed", "")
        updated = task.get("updated", "")

        # Format status nicely
        status_text = "Completed" if status == "completed" else "Pending"

        # Format as readable text - include task_id for action use
        content_parts = [
            f"Task: {title}",
            f"Task ID: {task_id}",
            f"Status: {status_text}",
        ]

        if due:
            # Due date is in RFC 3339 format
            due_date = due[:10] if len(due) >= 10 else due
            content_parts.append(f"Due: {due_date}")

        if completed:
            completed_date = completed[:10] if len(completed) >= 10 else completed
            content_parts.append(f"Completed: {completed_date}")

        if notes:
            content_parts.append(f"\nNotes:\n{notes}")

        content = "\n".join(content_parts)

        # Extract due date for metadata
        due_date = due[:10] if due and len(due) >= 10 else None

        return DocumentContent(
            uri=uri,
            name=title[:100],
            mime_type="text/plain",
            text=content.strip(),
            metadata={
                "task_id": task_id,
                "due_date": due_date,
                "status": status,
                "completed": status == "completed",
                "source_type": "task",
            },
        )


class GoogleContactsDocumentSource(DocumentSource):
    """
    Document source for Google Contacts using the People API.

    Uses Google People API directly with stored OAuth tokens.
    Indexes contacts for RAG retrieval (names, emails, phones, etc.).
    """

    def __init__(
        self,
        account_id: int,
        include_other_contacts: bool = False,
    ):
        """
        Initialize with OAuth account ID.

        Args:
            account_id: ID of the stored OAuth token
            include_other_contacts: Include "Other contacts" (auto-saved from interactions)
        """
        self.account_id = account_id
        self.include_other_contacts = include_other_contacts
        self._access_token: Optional[str] = None
        self._token_data: Optional[dict] = None

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
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

        # Try the access token first with a simple API call
        test_response = http_requests.get(
            "https://people.googleapis.com/v1/people/me",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"personFields": "names"},
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
        """Check if we can access Google Contacts."""
        return self._get_access_token() is not None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List contacts from Google People API."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list contacts - no valid access token")
            return

        logger.info("Listing Google Contacts")

        # Fields to request for each contact
        person_fields = "names,emailAddresses,phoneNumbers,organizations,metadata"

        total_contacts = 0
        page_token = None

        while True:
            params = {
                "pageSize": 100,
                "personFields": person_fields,
                "sortOrder": "LAST_MODIFIED_DESCENDING",
            }
            if page_token:
                params["pageToken"] = page_token

            response = http_requests.get(
                "https://people.googleapis.com/v1/people/me/connections",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"People API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            connections = data.get("connections", [])

            for person in connections:
                resource_name = person.get("resourceName", "")
                names = person.get("names", [])
                name = names[0].get("displayName", "Unknown") if names else "Unknown"

                # Get modification time from metadata
                metadata = person.get("metadata", {})
                sources = metadata.get("sources", [])
                modified_time = None
                if sources:
                    modified_time = sources[0].get("updateTime")

                total_contacts += 1
                yield DocumentInfo(
                    uri=f"gcontacts://{resource_name}",
                    name=name[:100],
                    mime_type="text/vcard",
                    modified_time=modified_time,
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        # Optionally include "Other contacts"
        if self.include_other_contacts:
            page_token = None
            while True:
                params = {
                    "pageSize": 100,
                    "readMask": "names,emailAddresses,phoneNumbers",
                }
                if page_token:
                    params["pageToken"] = page_token

                response = http_requests.get(
                    "https://people.googleapis.com/v1/otherContacts",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=30,
                )

                if response.status_code != 200:
                    logger.warning(f"Other contacts API error: {response.status_code}")
                    break

                data = response.json()
                other_contacts = data.get("otherContacts", [])

                for person in other_contacts:
                    resource_name = person.get("resourceName", "")
                    names = person.get("names", [])
                    name = (
                        names[0].get("displayName", "Unknown") if names else "Unknown"
                    )

                    total_contacts += 1
                    yield DocumentInfo(
                        uri=f"gcontacts://{resource_name}",
                        name=f"(Other) {name[:90]}",
                        mime_type="text/vcard",
                    )

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        logger.info(f"Found {total_contacts} contacts to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a contact from Google People API."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read contact - no valid access token")
            return None

        # Extract resource name from URI (gcontacts://people/c123456)
        if not uri.startswith("gcontacts://"):
            logger.error(f"Invalid Contacts URI: {uri}")
            return None

        resource_name = uri.replace("gcontacts://", "")

        # Request comprehensive fields
        person_fields = (
            "names,emailAddresses,phoneNumbers,organizations,addresses,"
            "birthdays,biographies,urls,relations,occupations,nicknames"
        )

        response = http_requests.get(
            f"https://people.googleapis.com/v1/{resource_name}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"personFields": person_fields},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(
                f"Failed to get contact {resource_name}: {response.status_code}"
            )
            return None

        person = response.json()

        # Extract contact details
        names = person.get("names", [])
        display_name = names[0].get("displayName", "Unknown") if names else "Unknown"

        emails = person.get("emailAddresses", [])
        phones = person.get("phoneNumbers", [])
        orgs = person.get("organizations", [])
        addresses = person.get("addresses", [])
        birthdays = person.get("birthdays", [])
        bios = person.get("biographies", [])
        urls = person.get("urls", [])
        nicknames = person.get("nicknames", [])

        # Format as readable text
        content_parts = [f"Contact: {display_name}"]

        if nicknames:
            nick_list = ", ".join(n.get("value", "") for n in nicknames)
            content_parts.append(f"Nicknames: {nick_list}")

        if emails:
            for email in emails:
                email_type = email.get("type", "other")
                email_val = email.get("value", "")
                content_parts.append(f"Email ({email_type}): {email_val}")

        if phones:
            for phone in phones:
                phone_type = phone.get("type", "other")
                phone_val = phone.get("value", "")
                content_parts.append(f"Phone ({phone_type}): {phone_val}")

        if orgs:
            for org in orgs:
                org_name = org.get("name", "")
                org_title = org.get("title", "")
                if org_name or org_title:
                    org_str = (
                        f"{org_title} at {org_name}"
                        if org_title and org_name
                        else (org_title or org_name)
                    )
                    content_parts.append(f"Organization: {org_str}")

        if addresses:
            for addr in addresses:
                addr_type = addr.get("type", "other")
                formatted = addr.get("formattedValue", "")
                if formatted:
                    content_parts.append(f"Address ({addr_type}): {formatted}")

        if birthdays:
            for bday in birthdays:
                date = bday.get("date", {})
                if date:
                    bday_str = f"{date.get('year', '????')}-{date.get('month', '??'):02d}-{date.get('day', '??'):02d}"
                    content_parts.append(f"Birthday: {bday_str}")

        if urls:
            for url in urls:
                url_type = url.get("type", "other")
                url_val = url.get("value", "")
                content_parts.append(f"URL ({url_type}): {url_val}")

        if bios:
            for bio in bios:
                bio_val = bio.get("value", "")
                if bio_val:
                    content_parts.append(f"\nNotes: {bio_val}")

        content = "\n".join(content_parts)

        # Primary email for metadata
        primary_email = emails[0].get("value") if emails else None

        return DocumentContent(
            uri=uri,
            name=display_name[:100],
            mime_type="text/plain",
            text=content.strip(),
            metadata={
                "email": primary_email,
                "source_type": "contact",
            },
        )


class PaperlessDocumentSource(DocumentSource):
    """
    Document source for Paperless-ngx using the REST API.

    Connects to a Paperless-ngx instance and retrieves documents for indexing.
    Uses the /api/documents/ endpoint which returns document metadata and
    extracted text content.

    Credentials are read from environment variables:
    - PAPERLESS_URL: Base URL (e.g., "http://paperless:8000")
    - PAPERLESS_TOKEN: API token for authentication
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        tag_id: Optional[int] = None,
    ):
        """
        Initialize with Paperless-ngx connection details.

        Args:
            base_url: Base URL of the Paperless instance (defaults to PAPERLESS_URL env var)
            api_token: API token for authentication (defaults to PAPERLESS_TOKEN env var)
            tag_id: Optional tag ID to filter documents (only index docs with this tag)
        """
        self.base_url = (base_url or os.environ.get("PAPERLESS_URL", "")).rstrip("/")
        self.api_token = api_token or os.environ.get("PAPERLESS_TOKEN", "")
        self.tag_id = tag_id

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Token {self.api_token}",
            "Accept": "application/json",
        }

    def is_available(self) -> bool:
        """Check if we can connect to Paperless."""
        import requests as http_requests

        try:
            # Use /api/documents/ with page_size=1 to verify authentication works
            response = http_requests.get(
                f"{self.base_url}/api/documents/",
                headers=self._get_headers(),
                params={"page_size": 1},
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Paperless not available at {self.base_url}: {e}")
            return False

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all documents from Paperless, optionally filtered by tag."""
        import requests as http_requests

        if self.tag_id:
            logger.info(
                f"Listing documents from Paperless at {self.base_url} (tag_id={self.tag_id})"
            )
        else:
            logger.info(f"Listing documents from Paperless at {self.base_url}")

        page = 1
        total_docs = 0

        while True:
            try:
                # Build params with optional tag filter
                params = {"page": page, "page_size": 100}
                if self.tag_id:
                    params["tags__id"] = self.tag_id

                response = http_requests.get(
                    f"{self.base_url}/api/documents/",
                    headers=self._get_headers(),
                    params=params,
                    timeout=30,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Paperless API error: {response.status_code} - {response.text}"
                    )
                    return

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                for doc in results:
                    doc_id = doc.get("id")
                    title = doc.get("title", f"Document {doc_id}")
                    modified = doc.get("modified")

                    total_docs += 1
                    yield DocumentInfo(
                        uri=f"paperless://{doc_id}",
                        name=title,
                        mime_type="text/plain",
                        modified_time=modified,
                    )

                # Check for more pages
                if not data.get("next"):
                    break

                page += 1

            except Exception as e:
                logger.error(f"Error listing Paperless documents: {e}")
                return

        logger.info(f"Found {total_docs} documents in Paperless")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from Paperless."""
        import requests as http_requests

        # Extract document ID from URI (paperless://doc_id)
        if not uri.startswith("paperless://"):
            logger.error(f"Invalid Paperless URI: {uri}")
            return None

        doc_id = uri.replace("paperless://", "")

        try:
            response = http_requests.get(
                f"{self.base_url}/api/documents/{doc_id}/",
                headers=self._get_headers(),
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Failed to get Paperless document {doc_id}: {response.status_code}"
                )
                return None

            doc = response.json()

            # Extract document fields
            title = doc.get("title", f"Document {doc_id}")
            content = doc.get("content", "")
            correspondent = doc.get("correspondent_name", "")
            document_type = doc.get("document_type_name", "")
            created = doc.get("created", "")
            added = doc.get("added", "")
            tags = doc.get("tags", [])

            # Get tag names if tags are IDs
            tag_names = []
            if tags and isinstance(tags[0], int):
                # Tags are IDs, would need separate API call to get names
                # For now, just note the count
                tag_names = [f"tag_{t}" for t in tags]
            else:
                tag_names = tags

            # Format as readable text with metadata
            text_parts = [f"Title: {title}"]
            if correspondent:
                text_parts.append(f"Correspondent: {correspondent}")
            if document_type:
                text_parts.append(f"Type: {document_type}")
            if created:
                text_parts.append(f"Created: {created}")
            if tag_names:
                text_parts.append(f"Tags: {', '.join(str(t) for t in tag_names)}")
            text_parts.append("")
            text_parts.append(content)

            full_text = "\n".join(text_parts)

            # Extract date for metadata (YYYY-MM-DD format)
            doc_date = None
            if created:
                doc_date = created[:10] if len(created) >= 10 else created

            return DocumentContent(
                uri=uri,
                name=title[:100],
                mime_type="text/plain",
                text=full_text,
                metadata={
                    "doc_date": doc_date,
                    "correspondent": correspondent or None,
                    "document_type": document_type or None,
                    "source_type": "paperless",
                },
            )

        except Exception as e:
            logger.error(f"Error reading Paperless document {doc_id}: {e}")
            return None


def _inject_mcp_env_vars(mcp_config: dict) -> dict:
    """
    Inject required environment variables into MCP config based on server name.

    Different MCP servers expect different environment variable names:
    - Notion: NOTION_TOKEN
    - GitHub: GITHUB_TOKEN
    - Slack: SLACK_BOT_TOKEN, SLACK_TEAM_ID
    - GWorkspace (mcp-gsuite): OAuth credential files via GAUTH_FILE, ACCOUNTS_FILE, CREDENTIALS_DIR
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

    # Google (mcp-gsuite) - set up OAuth credential files for Drive/Gmail/Calendar
    elif server_name == "google":
        google_account_id = config.get("google_account_id")
        google_service = config.get("google_service", "drive")
        if google_account_id:
            creds_dir = _setup_google_oauth_files(google_account_id)
            if creds_dir:
                env["GAUTH_FILE"] = os.path.join(creds_dir, ".gauth.json")
                env["ACCOUNTS_FILE"] = os.path.join(creds_dir, ".accounts.json")
                env["CREDENTIALS_DIR"] = creds_dir
                # Set the service type for mcp-gsuite to know which API to use
                env["GOOGLE_SERVICE"] = google_service
        else:
            logger.warning("Google MCP server configured without google_account_id")

    config["env"] = env
    return config


class NotionDocumentSource(DocumentSource):
    """
    Document source for Notion using the REST API directly.

    Uses Notion API v1 to list and read pages, bypassing MCP for
    simpler deployment (no Node.js required).

    Credentials are read from NOTION_TOKEN or NOTION_API_KEY environment variable.
    """

    def __init__(
        self,
        database_id: Optional[str] = None,
        root_page_id: Optional[str] = None,
    ):
        """
        Initialize with optional database or page filter.

        Args:
            database_id: Optional database ID to index pages from
            root_page_id: Optional root page ID to index (and children)
        """
        self.database_id = database_id
        self.root_page_id = root_page_id
        self.api_token = os.environ.get("NOTION_TOKEN") or os.environ.get(
            "NOTION_API_KEY", ""
        )
        self.base_url = "https://api.notion.com/v1"
        self.api_version = "2022-06-28"

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Notion-Version": self.api_version,
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        """Check if we can connect to Notion."""
        import requests as http_requests

        if not self.api_token:
            logger.warning("No Notion API token configured")
            return False

        try:
            response = http_requests.get(
                f"{self.base_url}/users/me",
                headers=self._get_headers(),
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Notion not available: {e}")
            return False

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List pages from Notion."""
        import requests as http_requests

        if self.database_id:
            # Query a specific database
            logger.info(f"Listing pages from Notion database: {self.database_id}")
            yield from self._list_database_pages(self.database_id)
        elif self.root_page_id:
            # List children of a specific page
            logger.info(f"Listing child pages from Notion page: {self.root_page_id}")
            yield from self._list_page_children(self.root_page_id)
        else:
            # Search all accessible pages
            logger.info("Searching all accessible Notion pages")
            yield from self._search_all_pages()

    def _list_database_pages(self, database_id: str) -> Iterator[DocumentInfo]:
        """Query pages from a Notion database."""
        import requests as http_requests

        next_cursor = None
        total_pages = 0

        while True:
            payload = {"page_size": 100}
            if next_cursor:
                payload["start_cursor"] = next_cursor

            response = http_requests.post(
                f"{self.base_url}/databases/{database_id}/query",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Notion API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            results = data.get("results", [])

            for page in results:
                page_id = page.get("id")
                title = self._extract_page_title(page)
                modified_time = page.get("last_edited_time")

                total_pages += 1
                yield DocumentInfo(
                    uri=f"notion://{page_id}",
                    name=title or page_id,
                    mime_type="text/markdown",
                    modified_time=modified_time,
                )

            # Check for more pages
            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")

        logger.info(f"Found {total_pages} pages in Notion database")

    def _list_page_children(self, page_id: str) -> Iterator[DocumentInfo]:
        """List child pages of a Notion page."""
        import requests as http_requests

        next_cursor = None
        total_pages = 0

        while True:
            params = {"page_size": 100}
            if next_cursor:
                params["start_cursor"] = next_cursor

            response = http_requests.get(
                f"{self.base_url}/blocks/{page_id}/children",
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Notion API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            results = data.get("results", [])

            for block in results:
                # Only yield child pages
                if block.get("type") == "child_page":
                    child_id = block.get("id")
                    title = block.get("child_page", {}).get("title", child_id)
                    modified_time = block.get("last_edited_time")

                    total_pages += 1
                    yield DocumentInfo(
                        uri=f"notion://{child_id}",
                        name=title,
                        mime_type="text/markdown",
                        modified_time=modified_time,
                    )

            # Check for more pages
            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")

        logger.info(f"Found {total_pages} child pages in Notion")

    def _search_all_pages(self) -> Iterator[DocumentInfo]:
        """Search all accessible Notion pages."""
        import requests as http_requests

        next_cursor = None
        total_pages = 0

        while True:
            payload = {
                "filter": {"property": "object", "value": "page"},
                "page_size": 100,
            }
            if next_cursor:
                payload["start_cursor"] = next_cursor

            response = http_requests.post(
                f"{self.base_url}/search",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Notion API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            results = data.get("results", [])

            for page in results:
                page_id = page.get("id")
                title = self._extract_page_title(page)
                modified_time = page.get("last_edited_time")

                total_pages += 1
                yield DocumentInfo(
                    uri=f"notion://{page_id}",
                    name=title or page_id,
                    mime_type="text/markdown",
                    modified_time=modified_time,
                )

            # Check for more pages
            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")

        logger.info(f"Found {total_pages} pages in Notion")

    def _extract_page_title(self, page: dict) -> Optional[str]:
        """Extract title from a Notion page object."""
        properties = page.get("properties", {})

        # Look for title property (could be named "title", "Name", etc.)
        for prop_name, prop_val in properties.items():
            if isinstance(prop_val, dict) and prop_val.get("type") == "title":
                title_arr = prop_val.get("title", [])
                if title_arr and isinstance(title_arr[0], dict):
                    return title_arr[0].get("plain_text", "")

        return None

    def _extract_page_properties(self, page: dict) -> dict[str, str]:
        """Extract all properties from a Notion page as displayable strings.

        Returns a dict of property_name -> displayable_value for non-empty properties.
        Skips system properties and the title (which is handled separately).
        """
        properties = page.get("properties", {})
        result = {}

        for prop_name, prop_val in properties.items():
            if not isinstance(prop_val, dict):
                continue

            prop_type = prop_val.get("type")

            # Skip system properties and title
            if prop_type in (
                "title",
                "created_by",
                "created_time",
                "last_edited_by",
                "last_edited_time",
                "formula",
                "rollup",
                "relation",
            ):
                continue

            value = None

            if prop_type == "select":
                select_data = prop_val.get("select")
                if select_data:
                    value = select_data.get("name")

            elif prop_type == "multi_select":
                multi_data = prop_val.get("multi_select", [])
                if multi_data:
                    value = ", ".join(item.get("name", "") for item in multi_data)

            elif prop_type == "status":
                status_data = prop_val.get("status")
                if status_data:
                    value = status_data.get("name")

            elif prop_type == "checkbox":
                value = "Yes" if prop_val.get("checkbox") else "No"

            elif prop_type == "number":
                num_val = prop_val.get("number")
                if num_val is not None:
                    value = str(num_val)

            elif prop_type == "date":
                date_data = prop_val.get("date")
                if date_data:
                    start = date_data.get("start", "")
                    end = date_data.get("end")
                    value = f"{start} - {end}" if end else start

            elif prop_type == "rich_text":
                rich_text = prop_val.get("rich_text", [])
                if rich_text:
                    value = "".join(rt.get("plain_text", "") for rt in rich_text)

            elif prop_type == "url":
                value = prop_val.get("url")

            elif prop_type == "email":
                value = prop_val.get("email")

            elif prop_type == "phone_number":
                value = prop_val.get("phone_number")

            elif prop_type == "people":
                people = prop_val.get("people", [])
                if people:
                    names = [p.get("name", p.get("id", "")) for p in people]
                    value = ", ".join(names)

            elif prop_type == "files":
                files = prop_val.get("files", [])
                if files:
                    value = ", ".join(f.get("name", "") for f in files)

            if value:
                result[prop_name] = value

        return result

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a Notion page and convert to markdown."""
        import requests as http_requests

        # Extract page ID from URI (notion://page_id)
        if not uri.startswith("notion://"):
            logger.error(f"Invalid Notion URI: {uri}")
            return None

        page_id = uri.replace("notion://", "")

        try:
            # First get page metadata for title
            page_response = http_requests.get(
                f"{self.base_url}/pages/{page_id}",
                headers=self._get_headers(),
                timeout=30,
            )

            title = page_id
            page_properties = {}
            if page_response.status_code == 200:
                page_data = page_response.json()
                title = self._extract_page_title(page_data) or page_id
                page_properties = self._extract_page_properties(page_data)

            # Get all blocks (content)
            blocks = self._get_all_blocks(page_id)

            # Convert blocks to markdown
            markdown = self._blocks_to_markdown(blocks)

            # Prepend properties as a metadata header if any exist
            if page_properties:
                props_lines = [f"**{k}:** {v}" for k, v in page_properties.items()]
                props_header = "\n".join(props_lines)
                markdown = (
                    f"{props_header}\n\n---\n\n{markdown}" if markdown else props_header
                )

            return DocumentContent(
                uri=uri,
                name=title[:100],
                mime_type="text/markdown",
                text=markdown,
            )

        except Exception as e:
            logger.error(f"Failed to read Notion page {page_id}: {e}")
            return None

    def _get_all_blocks(self, page_id: str) -> list:
        """Get all blocks from a page, handling pagination."""
        import requests as http_requests

        blocks = []
        next_cursor = None

        while True:
            params = {"page_size": 100}
            if next_cursor:
                params["start_cursor"] = next_cursor

            response = http_requests.get(
                f"{self.base_url}/blocks/{page_id}/children",
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"Failed to get blocks: {response.status_code}")
                break

            data = response.json()
            blocks.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")

        return blocks

    def _blocks_to_markdown(self, blocks: list) -> str:
        """Convert Notion blocks to markdown with full rich text support."""
        from utils.notion_markdown import notion_blocks_to_markdown

        return notion_blocks_to_markdown(blocks)


class GitHubDocumentSource(DocumentSource):
    """
    Document source for GitHub repositories using the REST API.

    Fetches files from a GitHub repository for indexing.
    Credentials are read from GITHUB_TOKEN environment variable.
    """

    # File extensions to index (code and docs)
    INDEXABLE_EXTENSIONS = {
        ".md",
        ".txt",
        ".rst",
        ".adoc",  # Documentation
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",  # Code
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".json",
        ".yaml",
        ".yml",
        ".toml",  # Config
        ".html",
        ".css",
        ".scss",  # Web
        ".sql",
        ".sh",
        ".bash",  # Scripts
    }

    def __init__(
        self,
        repo: str,
        branch: Optional[str] = None,
        path_filter: Optional[str] = None,
    ):
        """
        Initialize with GitHub repository details.

        Args:
            repo: Repository in "owner/repo" format
            branch: Branch name (defaults to repo default branch)
            path_filter: Optional path prefix to filter files (e.g., "docs/")
        """
        self.repo = repo
        self.branch = branch
        self.path_filter = path_filter
        self.api_token = os.environ.get("GITHUB_TOKEN") or os.environ.get(
            "GITHUB_PERSONAL_ACCESS_TOKEN", ""
        )
        self.base_url = "https://api.github.com"

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def is_available(self) -> bool:
        """Check if we can access the repository."""
        import requests as http_requests

        try:
            response = http_requests.get(
                f"{self.base_url}/repos/{self.repo}",
                headers=self._get_headers(),
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"GitHub repo {self.repo} not available: {e}")
            return False

    def _get_default_branch(self) -> Optional[str]:
        """Get the default branch for the repository."""
        import requests as http_requests

        try:
            response = http_requests.get(
                f"{self.base_url}/repos/{self.repo}",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 200:
                return response.json().get("default_branch", "main")
        except Exception as e:
            logger.warning(f"Failed to get default branch: {e}")
        return "main"

    def _should_index_file(self, path: str) -> bool:
        """Check if a file should be indexed based on extension and path filter."""
        # Check extension
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.INDEXABLE_EXTENSIONS:
            return False

        # Check path filter
        if self.path_filter:
            if not path.startswith(self.path_filter.rstrip("/")):
                return False

        return True

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all indexable files in the repository."""
        import requests as http_requests

        branch = self.branch or self._get_default_branch()
        logger.info(f"Listing files from GitHub {self.repo} (branch: {branch})")

        try:
            # Get the tree recursively
            response = http_requests.get(
                f"{self.base_url}/repos/{self.repo}/git/trees/{branch}",
                headers=self._get_headers(),
                params={"recursive": "1"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"GitHub API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            tree = data.get("tree", [])

            for item in tree:
                if item.get("type") != "blob":
                    continue

                path = item.get("path", "")
                if not self._should_index_file(path):
                    continue

                yield DocumentInfo(
                    uri=f"github://{self.repo}/{branch}/{path}",
                    name=path,
                    mime_type=self._get_mime_type(path),
                    size=item.get("size"),
                )

        except Exception as e:
            logger.error(f"Failed to list GitHub files: {e}")

    def _get_mime_type(self, path: str) -> str:
        """Get MIME type based on file extension."""
        ext = os.path.splitext(path)[1].lower()
        mime_types = {
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".ts": "text/typescript",
            ".json": "application/json",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".html": "text/html",
            ".css": "text/css",
        }
        return mime_types.get(ext, "text/plain")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a file from the repository."""
        import requests as http_requests

        # Parse URI: github://owner/repo/branch/path
        if not uri.startswith("github://"):
            return None

        parts = uri[9:].split("/", 3)  # Remove "github://"
        if len(parts) < 4:
            return None

        owner, repo, branch, path = parts[0], parts[1], parts[2], parts[3]

        try:
            # Get file content
            response = http_requests.get(
                f"{self.base_url}/repos/{owner}/{repo}/contents/{path}",
                headers=self._get_headers(),
                params={"ref": branch},
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"Failed to read {path}: {response.status_code}")
                return None

            data = response.json()

            # Content is base64 encoded
            import base64

            content = base64.b64decode(data.get("content", "")).decode(
                "utf-8", errors="replace"
            )

            return DocumentContent(
                uri=uri,
                name=path,
                mime_type=self._get_mime_type(path),
                text=content,
            )

        except Exception as e:
            logger.error(f"Failed to read GitHub file {path}: {e}")
            return None


class NextcloudDocumentSource(DocumentSource):
    """
    Document source for Nextcloud using WebDAV API.

    Uses Nextcloud's WebDAV endpoint to list and retrieve files for indexing.
    Supports Docling processing with optional vision models for PDFs
    and other binary document formats.

    Credentials are read from environment variables:
    - NEXTCLOUD_URL: Base URL (e.g., "https://nextcloud.example.com")
    - NEXTCLOUD_USER: Username for authentication
    - NEXTCLOUD_PASSWORD: Password or app password for authentication
    """

    # Binary file types that should be processed through Docling
    DOCLING_EXTENSIONS = {
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

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        folder_path: Optional[str] = None,
        vision_provider: str = "local",
        vision_model: Optional[str] = None,
        vision_ollama_url: Optional[str] = None,
    ):
        """
        Initialize with Nextcloud connection details.

        Args:
            base_url: Base URL of the Nextcloud instance (defaults to NEXTCLOUD_URL env var)
            username: Username for authentication (defaults to NEXTCLOUD_USER env var)
            password: Password or app password (defaults to NEXTCLOUD_PASSWORD env var)
            folder_path: Optional folder path to restrict indexing scope (e.g., "/Documents")
            vision_provider: Vision model provider for Docling ("local", "ollama", etc.)
            vision_model: Vision model name (e.g., "granite3.2-vision:latest")
            vision_ollama_url: Ollama URL when using ollama provider
        """
        self.base_url = (base_url or os.environ.get("NEXTCLOUD_URL", "")).rstrip("/")
        self.username = username or os.environ.get("NEXTCLOUD_USER", "")
        self.password = password or os.environ.get("NEXTCLOUD_PASSWORD", "")
        self.folder_path = (folder_path or "").strip("/")
        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_ollama_url = vision_ollama_url
        self._document_converter = None

    def _get_webdav_url(self, path: str = "") -> str:
        """Get the full WebDAV URL for a path."""
        # Nextcloud WebDAV endpoint: /remote.php/dav/files/{username}/
        base = f"{self.base_url}/remote.php/dav/files/{self.username}"
        if path:
            # Ensure path starts with /
            if not path.startswith("/"):
                path = "/" + path
            return base + path
        return base + "/"

    def _get_auth(self) -> tuple:
        """Get authentication tuple for requests."""
        return (self.username, self.password)

    def is_available(self) -> bool:
        """Check if we can connect to Nextcloud."""
        import requests as http_requests

        if not self.base_url or not self.username or not self.password:
            logger.warning("Nextcloud credentials not configured")
            return False

        try:
            # Use PROPFIND with Depth: 0 to check access
            response = http_requests.request(
                "PROPFIND",
                self._get_webdav_url(),
                auth=self._get_auth(),
                headers={"Depth": "0"},
                timeout=10,
            )
            return response.status_code in (200, 207)
        except Exception as e:
            logger.warning(f"Nextcloud not available at {self.base_url}: {e}")
            return False

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all supported documents from Nextcloud."""
        import xml.etree.ElementTree as ET

        import requests as http_requests

        start_path = self.folder_path if self.folder_path else ""
        logger.info(
            f"Listing documents from Nextcloud at {self.base_url}"
            + (f" (folder: /{start_path})" if start_path else " (all files)")
        )

        # Use recursive PROPFIND with Depth: infinity
        # Note: Some servers limit this, so we may need to handle pagination
        propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:">
    <d:prop>
        <d:resourcetype/>
        <d:getcontentlength/>
        <d:getcontenttype/>
        <d:getlastmodified/>
    </d:prop>
</d:propfind>"""

        try:
            response = http_requests.request(
                "PROPFIND",
                self._get_webdav_url(start_path),
                auth=self._get_auth(),
                headers={
                    "Depth": "infinity",
                    "Content-Type": "application/xml",
                },
                data=propfind_body,
                timeout=60,
            )

            if response.status_code not in (200, 207):
                logger.error(
                    f"Nextcloud PROPFIND error: {response.status_code} - {response.text[:200]}"
                )
                return

            # Parse XML response
            root = ET.fromstring(response.content)

            # Define namespaces
            ns = {"d": "DAV:"}

            total_files = 0
            for response_elem in root.findall(".//d:response", ns):
                href = response_elem.find("d:href", ns)
                if href is None:
                    continue

                href_text = href.text or ""

                # Check if it's a collection (folder)
                resourcetype = response_elem.find(".//d:resourcetype", ns)
                if resourcetype is not None:
                    collection = resourcetype.find("d:collection", ns)
                    if collection is not None:
                        continue  # Skip folders

                # Get file properties
                propstat = response_elem.find("d:propstat", ns)
                if propstat is None:
                    continue

                prop = propstat.find("d:prop", ns)
                if prop is None:
                    continue

                # Extract file path from href
                # href is typically: /remote.php/dav/files/username/path/to/file
                file_path = href_text
                dav_prefix = f"/remote.php/dav/files/{self.username}/"
                if dav_prefix in file_path:
                    file_path = file_path.split(dav_prefix, 1)[1]

                # URL decode the path
                from urllib.parse import unquote

                file_path = unquote(file_path)

                # Check if it's a supported file type
                if not self._is_supported_file(file_path):
                    continue

                # Get content type
                content_type_elem = prop.find("d:getcontenttype", ns)
                content_type = (
                    content_type_elem.text if content_type_elem is not None else None
                )

                # Get file size
                content_length_elem = prop.find("d:getcontentlength", ns)
                size = None
                if content_length_elem is not None and content_length_elem.text:
                    try:
                        size = int(content_length_elem.text)
                    except ValueError:
                        pass

                # Get last modified
                lastmod_elem = prop.find("d:getlastmodified", ns)
                modified_time = None
                if lastmod_elem is not None and lastmod_elem.text:
                    # Convert RFC 2822 date to ISO format
                    try:
                        from email.utils import parsedate_to_datetime

                        dt = parsedate_to_datetime(lastmod_elem.text)
                        modified_time = dt.isoformat()
                    except Exception:
                        modified_time = lastmod_elem.text

                total_files += 1
                yield DocumentInfo(
                    uri=f"nextcloud://{file_path}",
                    name=file_path,
                    mime_type=content_type,
                    size=size,
                    modified_time=modified_time,
                )

            logger.info(f"Found {total_files} documents in Nextcloud")

        except Exception as e:
            logger.error(f"Failed to list Nextcloud documents: {e}")

    def _is_supported_file(self, path: str) -> bool:
        """Check if a file type is supported for indexing."""
        path_lower = path.lower()

        # Check against supported extensions
        for ext in SUPPORTED_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        return False

    def _get_document_converter(self):
        """Get or create a Docling DocumentConverter with vision support."""
        if self._document_converter is None:
            from rag.vision import get_document_converter

            self._document_converter = get_document_converter(
                vision_provider=self.vision_provider,
                vision_model=self.vision_model,
                vision_ollama_url=self.vision_ollama_url,
            )
        return self._document_converter

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from Nextcloud."""
        import requests as http_requests

        # Extract file path from URI (nextcloud://path/to/file)
        if not uri.startswith("nextcloud://"):
            logger.error(f"Invalid Nextcloud URI: {uri}")
            return None

        file_path = uri.replace("nextcloud://", "")

        try:
            # GET request to download the file
            response = http_requests.get(
                self._get_webdav_url(file_path),
                auth=self._get_auth(),
                timeout=120,
            )

            if response.status_code != 200:
                logger.error(f"Failed to download {file_path}: {response.status_code}")
                return None

            # Determine file type and process accordingly
            file_name = os.path.basename(file_path)
            ext = os.path.splitext(file_name)[1].lower()

            # Binary files - process through Docling
            if ext in self.DOCLING_EXTENSIONS:
                return self._process_with_docling(uri, file_name, ext, response.content)

            # Text files - return directly
            try:
                text = response.content.decode("utf-8", errors="ignore")
            except Exception:
                text = response.text

            return DocumentContent(
                uri=uri,
                name=file_name,
                mime_type=self._get_mime_type(ext),
                text=text,
            )

        except Exception as e:
            logger.error(f"Failed to read Nextcloud document {file_path}: {e}")
            return None

    def _process_with_docling(
        self, uri: str, name: str, ext: str, content: bytes
    ) -> Optional[DocumentContent]:
        """Process a binary file through Docling with vision support."""
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                converter = self._get_document_converter()
                result = converter.convert(tmp_path)
                text = result.document.export_to_markdown()

                logger.debug(f"Processed {name} with Docling: {len(text)} chars")

                return DocumentContent(
                    uri=uri,
                    name=name,
                    mime_type="text/markdown",
                    text=text,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Docling processing failed for {name}: {e}")
            return None

    def _get_mime_type(self, ext: str) -> str:
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
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        return mime_types.get(ext.lower(), "text/plain")


class WebsiteDocumentSource(DocumentSource):
    """
    Document source for crawled websites.

    Uses either builtin trafilatura crawler or Jina Reader API
    (configured globally in Web Config settings, or overridden per-store).
    """

    def __init__(
        self,
        url: str,
        crawl_depth: int = 1,
        max_pages: int = 50,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        crawler_override: Optional[str] = None,
    ):
        """
        Initialize with website URL and crawl settings.

        Args:
            url: Starting URL for the crawl
            crawl_depth: Maximum depth of links to follow (0 = start page only)
            max_pages: Maximum number of pages to crawl
            include_pattern: Regex pattern - only crawl URLs matching this
            exclude_pattern: Regex pattern - skip URLs matching this
            crawler_override: Override global crawler setting (None, "builtin", or "jina")
        """
        self.url = url
        self.crawl_depth = crawl_depth
        self.max_pages = max_pages
        self.include_pattern = include_pattern
        self.exclude_pattern = exclude_pattern
        self.crawler_override = crawler_override
        self._crawled_pages: Optional[list] = None
        self._crawl_result = None

    def is_available(self) -> bool:
        """Check if the website is reachable.

        When using Jina crawler, skip local reachability check since Jina
        will handle it from their servers (different IP, bypasses bot protection).
        """
        provider = self._get_crawl_provider()

        # When using Jina, assume available - Jina will handle from their servers
        if provider in ("jina", "jina-api"):
            logger.debug(
                f"Skipping local reachability check for {self.url} (using Jina)"
            )
            return True

        # For builtin crawler, check reachability from our server
        import random

        import httpx

        from rag.crawler import BROWSER_HEADERS, USER_AGENTS

        try:
            headers = BROWSER_HEADERS.copy()
            headers["User-Agent"] = random.choice(USER_AGENTS)
            with httpx.Client(
                timeout=10, follow_redirects=True, headers=headers
            ) as client:
                response = client.head(self.url)
                return response.status_code < 400
        except Exception as e:
            logger.warning(f"Website not reachable: {self.url} - {e}")
            return False

    def _get_crawl_provider(self) -> str:
        """Get the crawl provider, checking per-store override first."""
        # Check for per-store override
        if self.crawler_override:
            return self.crawler_override

        # Fall back to global setting
        try:
            from db.connection import get_db_context
            from db.models import Setting

            with get_db_context() as db:
                setting = (
                    db.query(Setting)
                    .filter(Setting.key == Setting.KEY_WEB_CRAWL_PROVIDER)
                    .first()
                )
                return setting.value if setting else "builtin"
        except Exception:
            return "builtin"

    def _crawl_if_needed(self):
        """Crawl the website if not already done."""
        if self._crawled_pages is not None:
            return

        from rag.crawler import get_crawler

        provider = self._get_crawl_provider()
        crawler = get_crawler(provider)

        logger.info(f"Using {provider} crawler for {self.url}")

        result = crawler.crawl(
            start_url=self.url,
            max_depth=self.crawl_depth,
            max_pages=self.max_pages,
            include_pattern=self.include_pattern,
            exclude_pattern=self.exclude_pattern,
            same_domain_only=True,
        )

        self._crawl_result = result
        self._crawled_pages = result.pages
        logger.info(
            f"Crawled {result.pages_crawled} pages from {self.url} "
            f"(depth={self.crawl_depth}, max={self.max_pages}, provider={provider})"
        )

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all crawled pages as documents."""
        self._crawl_if_needed()

        for page in self._crawled_pages:
            if not page.success or not page.content:
                continue

            yield DocumentInfo(
                uri=page.url,
                name=page.title or page.url,
                mime_type="text/html",
            )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read content from a crawled page."""
        self._crawl_if_needed()

        # Find the page by URI
        for page in self._crawled_pages:
            if page.url == uri and page.success:
                return DocumentContent(
                    uri=uri,
                    name=page.title or uri,
                    mime_type="text/plain",
                    text=page.content,
                )

        logger.warning(f"Page not found in crawl results: {uri}")
        return None


class SlackDocumentSource(DocumentSource):
    """
    Document source for Slack using the REST API.

    Indexes messages from Slack channels. Requires a Bot User OAuth Token
    with appropriate scopes (channels:history, channels:read, users:read).

    Credentials are read from SLACK_BOT_TOKEN environment variable.
    """

    def __init__(
        self,
        channel_id: Optional[str] = None,
        channel_types: str = "public_channel",
        days_back: int = 90,
    ):
        """
        Initialize Slack document source.

        Args:
            channel_id: Optional specific channel ID to index (indexes all accessible if None)
            channel_types: Comma-separated channel types to index (public_channel, private_channel)
            days_back: How many days of history to fetch (max 90 for free plan)
        """
        self.channel_id = channel_id
        self.channel_types = channel_types
        self.days_back = min(days_back or 90, 90)  # Free plan limit
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        self.base_url = "https://slack.com/api"
        self._user_cache: dict = {}  # Cache user info to reduce API calls

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def is_available(self) -> bool:
        """Check if we can connect to Slack."""
        import requests as http_requests

        if not self.bot_token:
            logger.warning("No Slack bot token configured (SLACK_BOT_TOKEN)")
            return False

        try:
            response = http_requests.get(
                f"{self.base_url}/auth.test",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    logger.debug(f"Slack connected as: {data.get('user')}")
                    return True
                else:
                    logger.warning(f"Slack auth failed: {data.get('error')}")
                    return False
            return False
        except Exception as e:
            logger.warning(f"Slack not available: {e}")
            return False

    def _get_user_name(self, user_id: str) -> str:
        """Get username for a user ID (cached)."""
        import requests as http_requests

        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            response = http_requests.get(
                f"{self.base_url}/users.info",
                headers=self._get_headers(),
                params={"user": user_id},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    user = data.get("user", {})
                    # Prefer display name, fall back to real name, then username
                    name = (
                        user.get("profile", {}).get("display_name")
                        or user.get("real_name")
                        or user.get("name")
                        or user_id
                    )
                    self._user_cache[user_id] = name
                    return name
        except Exception as e:
            logger.debug(f"Failed to get user info for {user_id}: {e}")

        self._user_cache[user_id] = user_id
        return user_id

    def _list_channels(self) -> list:
        """List accessible channels."""
        import requests as http_requests

        channels = []
        cursor = None

        while True:
            params = {
                "types": self.channel_types,
                "limit": 200,
                "exclude_archived": "true",
            }
            if cursor:
                params["cursor"] = cursor

            response = http_requests.get(
                f"{self.base_url}/conversations.list",
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"Slack API error: {response.status_code}")
                break

            data = response.json()
            if not data.get("ok"):
                logger.error(f"Slack API error: {data.get('error')}")
                break

            channels.extend(data.get("channels", []))

            # Check for pagination
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return channels

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List Slack channels as documents (each channel = one document)."""
        from datetime import datetime, timedelta

        if self.channel_id:
            # Single channel mode
            channels = [{"id": self.channel_id, "name": self.channel_id}]
        else:
            channels = self._list_channels()

        logger.info(f"Found {len(channels)} Slack channels to index")

        oldest_ts = (datetime.utcnow() - timedelta(days=self.days_back)).timestamp()

        for channel in channels:
            channel_id = channel.get("id")
            channel_name = channel.get("name", channel_id)

            # Use channel's latest message timestamp as modified time if available
            # For simplicity, we'll use current time
            yield DocumentInfo(
                uri=f"slack://{channel_id}",
                name=f"#{channel_name}",
                mime_type="text/plain",
                modified_time=datetime.utcnow().isoformat(),
            )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read messages from a Slack channel."""
        from datetime import datetime, timedelta

        import requests as http_requests

        # Parse channel ID from URI
        if not uri.startswith("slack://"):
            logger.error(f"Invalid Slack URI: {uri}")
            return None

        channel_id = uri.replace("slack://", "")

        # Calculate oldest timestamp
        oldest_ts = (datetime.utcnow() - timedelta(days=self.days_back)).timestamp()

        messages = []
        cursor = None
        total_fetched = 0

        while True:
            params = {
                "channel": channel_id,
                "oldest": str(oldest_ts),
                "limit": 200,
                "inclusive": "true",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                response = http_requests.get(
                    f"{self.base_url}/conversations.history",
                    headers=self._get_headers(),
                    params=params,
                    timeout=30,
                )
            except Exception as e:
                logger.error(f"Failed to fetch Slack messages: {e}")
                break

            if response.status_code != 200:
                logger.error(
                    f"Slack API error: {response.status_code} - {response.text}"
                )
                break

            data = response.json()
            if not data.get("ok"):
                error = data.get("error", "unknown")
                if error == "channel_not_found":
                    logger.warning(f"Channel not found: {channel_id}")
                elif error == "not_in_channel":
                    logger.warning(
                        f"Bot not in channel {channel_id} - invite bot to channel"
                    )
                else:
                    logger.error(f"Slack API error: {error}")
                break

            batch = data.get("messages", [])
            messages.extend(batch)
            total_fetched += len(batch)

            # Check for pagination
            if not data.get("has_more"):
                break
            cursor = data.get("response_metadata", {}).get("next_cursor")

            # Safety limit
            if total_fetched >= 10000:
                logger.warning(f"Hit message limit (10000) for channel {channel_id}")
                break

        if not messages:
            logger.info(f"No messages found in channel {channel_id}")
            return DocumentContent(
                uri=uri,
                name=f"#{channel_id}",
                mime_type="text/plain",
                text="(No messages in the specified time range)",
            )

        # Sort messages by timestamp (oldest first)
        messages.sort(key=lambda m: float(m.get("ts", 0)))

        # Format messages as readable text
        formatted_lines = []
        for msg in messages:
            # Skip bot messages, join/leave messages, etc. unless they have text
            msg_type = msg.get("subtype", "")
            if msg_type in (
                "channel_join",
                "channel_leave",
                "channel_topic",
                "channel_purpose",
            ):
                continue

            text = msg.get("text", "").strip()
            if not text:
                continue

            # Get timestamp and format it
            ts = float(msg.get("ts", 0))
            dt = datetime.fromtimestamp(ts)
            date_str = dt.strftime("%Y-%m-%d %H:%M")

            # Get username
            user_id = msg.get("user", "unknown")
            username = self._get_user_name(user_id)

            # Format the message
            formatted_lines.append(f"[{date_str}] {username}: {text}")

        # Get channel name for the document
        channel_name = channel_id
        try:
            info_response = http_requests.get(
                f"{self.base_url}/conversations.info",
                headers=self._get_headers(),
                params={"channel": channel_id},
                timeout=10,
            )
            if info_response.status_code == 200:
                info_data = info_response.json()
                if info_data.get("ok"):
                    channel_name = info_data.get("channel", {}).get("name", channel_id)
        except Exception:
            pass

        content = f"# Slack Channel: #{channel_name}\n"
        content += f"# Messages from last {self.days_back} days\n"
        content += f"# Total messages: {len(formatted_lines)}\n\n"
        content += "\n".join(formatted_lines)

        logger.info(f"Indexed {len(formatted_lines)} messages from #{channel_name}")

        return DocumentContent(
            uri=uri,
            name=f"#{channel_name}",
            mime_type="text/plain",
            text=content,
        )


class TodoistDocumentSource(DocumentSource):
    """
    Document source for Todoist using the REST API.

    Indexes tasks from Todoist. Requires TODOIST_API_TOKEN environment variable.
    Uses personal API tokens from https://app.todoist.com/prefs/integrations
    """

    BASE_URL = "https://api.todoist.com/rest/v2"

    def __init__(
        self,
        project_id: Optional[str] = None,
        filter_expr: Optional[str] = None,
        include_completed: bool = False,
    ):
        """
        Initialize with optional project filter.

        Args:
            project_id: Optional project ID to filter tasks
            filter_expr: Optional Todoist filter expression (e.g., "today", "priority 1")
            include_completed: Whether to include completed tasks
        """
        self.project_id = project_id
        self.filter_expr = filter_expr
        self.include_completed = include_completed
        # Support both TODOIST_API_TOKEN and TODOIST_API_KEY for compatibility
        self.api_token = os.environ.get("TODOIST_API_TOKEN") or os.environ.get(
            "TODOIST_API_KEY", ""
        )
        self._projects_cache: dict[str, str] = {}  # id -> name

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        """Check if Todoist API is configured and accessible."""
        if not self.api_token:
            return False

        import requests as http_requests

        try:
            response = http_requests.get(
                f"{self.BASE_URL}/projects",
                headers=self._get_headers(),
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _get_projects(self) -> dict[str, str]:
        """Get all projects and cache them (id -> name mapping)."""
        if self._projects_cache:
            return self._projects_cache

        import requests as http_requests

        try:
            response = http_requests.get(
                f"{self.BASE_URL}/projects",
                headers=self._get_headers(),
                timeout=30,
            )
            if response.status_code == 200:
                projects = response.json()
                self._projects_cache = {p["id"]: p["name"] for p in projects}
        except Exception as e:
            logger.error(f"Failed to fetch Todoist projects: {e}")

        return self._projects_cache

    def _get_project_name(self, project_id: str) -> str:
        """Get project name from cache or fetch."""
        projects = self._get_projects()
        return projects.get(project_id, "Unknown Project")

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List tasks from Todoist."""
        import requests as http_requests

        if not self.api_token:
            logger.error("Cannot list tasks - TODOIST_API_TOKEN not set")
            return

        # Build query parameters
        params = {}
        if self.project_id:
            params["project_id"] = self.project_id
        if self.filter_expr:
            params["filter"] = self.filter_expr

        try:
            # Get active tasks
            response = http_requests.get(
                f"{self.BASE_URL}/tasks",
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Todoist API error: {response.status_code} - {response.text}"
                )
                return

            tasks = response.json()
            logger.info(f"Found {len(tasks)} active tasks from Todoist")

            for task in tasks:
                task_id = task.get("id")
                content = task.get("content", "(No title)")
                project_id = task.get("project_id", "")
                project_name = self._get_project_name(project_id)

                # Use created_at as modified time (Todoist doesn't have updated_at in REST API)
                created_at = task.get("created_at")

                yield DocumentInfo(
                    uri=f"todoist://{task_id}",
                    name=f"{project_name}: {content[:80]}",
                    mime_type="text/plain",
                    modified_time=created_at,
                )

            # Get completed tasks if requested
            if self.include_completed:
                # Completed tasks require Sync API, but we can use a filter
                completed_params = {"filter": "completed"}
                if self.project_id:
                    completed_params["project_id"] = self.project_id

                # Note: REST API v2 doesn't support completed tasks directly
                # For now, we skip this - would need Sync API for completed tasks
                logger.debug("Completed tasks require Sync API - skipping")

        except Exception as e:
            logger.error(f"Error listing Todoist tasks: {e}")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a task from Todoist."""
        import requests as http_requests

        if not self.api_token:
            logger.error("Cannot read task - TODOIST_API_TOKEN not set")
            return None

        # Extract task ID from URI (todoist://task_id)
        if not uri.startswith("todoist://"):
            logger.error(f"Invalid Todoist URI: {uri}")
            return None

        task_id = uri.replace("todoist://", "")

        try:
            response = http_requests.get(
                f"{self.BASE_URL}/tasks/{task_id}",
                headers=self._get_headers(),
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Failed to get Todoist task {task_id}: {response.status_code}"
                )
                return None

            task = response.json()

            # Extract task details
            content = task.get("content", "(No title)")
            description = task.get("description", "")
            project_id = task.get("project_id", "")
            project_name = self._get_project_name(project_id)
            priority = task.get("priority", 1)  # 1-4, where 4 is highest
            labels = task.get("labels", [])
            due = task.get("due")
            created_at = task.get("created_at", "")

            # Format priority
            priority_names = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}
            priority_text = priority_names.get(priority, "Normal")

            # Format as readable text
            content_parts = [
                f"Task: {content}",
                f"Project: {project_name}",
                f"Priority: {priority_text}",
            ]

            if labels:
                content_parts.append(f"Labels: {', '.join(labels)}")

            if due:
                due_date = due.get("date", "")
                due_string = due.get("string", "")
                if due_string:
                    content_parts.append(f"Due: {due_string}")
                elif due_date:
                    content_parts.append(f"Due: {due_date}")

            if description:
                content_parts.append(f"\nDescription:\n{description}")

            text_content = "\n".join(content_parts)

            # Extract due date for metadata
            due_date = due.get("date") if due else None

            return DocumentContent(
                uri=uri,
                name=content[:100],
                mime_type="text/plain",
                text=text_content.strip(),
                metadata={
                    "project": project_name,
                    "project_id": project_id,
                    "priority": priority,
                    # ChromaDB doesn't accept lists - convert to comma-separated string
                    "labels": ", ".join(labels) if labels else "",
                    "due_date": due_date or "",
                    "source_type": "task",
                },
            )

        except Exception as e:
            logger.error(f"Error reading Todoist task {task_id}: {e}")
            return None


# =============================================================================
# Microsoft Graph API Document Sources
# =============================================================================


class MicrosoftGraphMixin:
    """
    Mixin providing common Microsoft Graph API authentication and token refresh.

    Used by OneDrive, Outlook, and OneNote document sources.
    """

    account_id: int
    _token_data: Optional[dict]

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        import requests as http_requests

        token_data = self._get_token_data()
        if not token_data:
            logger.error(f"Microsoft OAuth token {self.account_id} not found")
            return None

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            logger.error("No access token in stored Microsoft credentials")
            return None

        # Try the access token first with a simple Graph API call
        test_response = http_requests.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )

        if test_response.status_code == 200:
            return access_token

        # Token expired, try to refresh
        if not refresh_token or not client_id or not client_secret:
            logger.error(
                "Cannot refresh Microsoft token - missing refresh_token or client credentials"
            )
            return None

        logger.info("Microsoft access token expired, refreshing...")
        refresh_response = http_requests.post(
            "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        if refresh_response.status_code != 200:
            logger.error(f"Microsoft token refresh failed: {refresh_response.text}")
            return None

        new_token_data = refresh_response.json()
        new_access_token = new_token_data.get("access_token")

        # Update stored token (Microsoft may return a new refresh token)
        from db.oauth_tokens import update_oauth_token_data

        updated_data = {**token_data, "access_token": new_access_token}
        if new_token_data.get("refresh_token"):
            updated_data["refresh_token"] = new_token_data["refresh_token"]
        update_oauth_token_data(self.account_id, updated_data)

        # Update cache
        self._token_data = updated_data
        logger.info("Microsoft token refreshed successfully")
        return new_access_token


class OneDriveDocumentSource(MicrosoftGraphMixin, DocumentSource):
    """
    Document source for OneDrive using Microsoft Graph API.

    Uses Graph API directly with stored OAuth tokens for file access.
    Supports Docling processing with optional vision models for PDFs
    and other binary document formats.
    """

    # Binary file types that should be processed through Docling
    DOCLING_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
    }

    def __init__(
        self,
        account_id: int,
        folder_id: Optional[str] = None,
        vision_provider: str = "local",
        vision_model: Optional[str] = None,
        vision_ollama_url: Optional[str] = None,
    ):
        """
        Initialize with OAuth account ID and optional folder filter.

        Args:
            account_id: ID of the stored OAuth token
            folder_id: Optional folder ID to restrict indexing scope
            vision_provider: Vision model provider for Docling ("local", "ollama", etc.)
            vision_model: Vision model name (e.g., "granite3.2-vision:latest")
            vision_ollama_url: Ollama URL when using ollama provider
        """
        self.account_id = account_id
        self.folder_id = folder_id
        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_ollama_url = vision_ollama_url
        self._token_data: Optional[dict] = None
        self._document_converter = None

    def is_available(self) -> bool:
        """Check if we can access OneDrive."""
        return self._get_access_token() is not None

    def _get_document_converter(self):
        """Get or create a Docling DocumentConverter with vision support."""
        if self._document_converter is None:
            from rag.vision import get_document_converter

            self._document_converter = get_document_converter(
                vision_provider=self.vision_provider,
                vision_model=self.vision_model,
                vision_ollama_url=self.vision_ollama_url,
            )
        return self._document_converter

    def _is_supported_file(self, mime_type: str, name: str) -> bool:
        """Check if a file type is supported for indexing."""
        # Binary types we can process through Docling
        if mime_type in self.DOCLING_MIME_TYPES:
            return True

        # Check file extension for text types
        name_lower = name.lower()
        text_extensions = {".txt", ".md", ".html", ".htm", ".json", ".xml", ".csv"}
        for ext in text_extensions:
            if name_lower.endswith(ext):
                return True

        return False

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all documents in OneDrive or specified folder."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list OneDrive documents - no valid access token")
            return

        # Build the Graph API URL
        if self.folder_id:
            url = f"https://graph.microsoft.com/v1.0/me/drive/items/{self.folder_id}/children"
            logger.info(f"Listing documents in OneDrive folder: {self.folder_id}")
        else:
            url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
            logger.info("Listing all documents in OneDrive root")

        total_files = 0

        while url:
            response = http_requests.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "$select": "id,name,file,folder,size,lastModifiedDateTime",
                    "$top": 100,
                },
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"OneDrive API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            items = data.get("value", [])

            for item in items:
                # Skip folders
                if "folder" in item:
                    continue

                # Get file info
                file_info = item.get("file", {})
                mime_type = file_info.get("mimeType", "application/octet-stream")
                name = item["name"]

                # Check if it's a supported type
                if not self._is_supported_file(mime_type, name):
                    continue

                total_files += 1
                yield DocumentInfo(
                    uri=f"onedrive://{item['id']}",
                    name=name,
                    mime_type=mime_type,
                    size=item.get("size"),
                    modified_time=item.get("lastModifiedDateTime"),
                )

            # Check for next page
            url = data.get("@odata.nextLink")

        logger.info(f"Found {total_files} OneDrive documents to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a document from OneDrive."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read OneDrive document - no valid access token")
            return None

        # Extract file ID from URI (onedrive://file_id)
        if not uri.startswith("onedrive://"):
            logger.error(f"Invalid OneDrive URI: {uri}")
            return None

        file_id = uri.replace("onedrive://", "")

        # Get file metadata first
        meta_response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"$select": "id,name,file,size"},
            timeout=30,
        )

        if meta_response.status_code != 200:
            logger.error(f"Failed to get OneDrive file metadata: {meta_response.text}")
            return None

        file_meta = meta_response.json()
        file_info = file_meta.get("file", {})
        mime_type = file_info.get("mimeType", "application/octet-stream")
        name = file_meta.get("name", file_id)

        # Handle binary files through Docling (PDF, DOCX, images, etc.)
        if mime_type in self.DOCLING_MIME_TYPES:
            return self._download_and_process_with_docling(
                access_token, file_id, name, mime_type
            )

        # Handle text files
        return self._download_text(access_token, file_id, name, mime_type)

    def _download_and_process_with_docling(
        self, access_token: str, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Download a binary file and process it through Docling with vision support."""
        import tempfile

        import requests as http_requests

        # Download the file content
        response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=120,
        )

        if response.status_code != 200:
            logger.error(
                f"Failed to download OneDrive file {name}: {response.status_code}"
            )
            return None

        # Get file extension for temp file
        extension = self.DOCLING_MIME_TYPES.get(mime_type, ".bin")

        # Save to temp file and process with Docling
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                converter = self._get_document_converter()
                result = converter.convert(tmp_path)
                text = result.document.export_to_markdown()

                logger.debug(
                    f"Processed OneDrive file {name} with Docling: {len(text)} chars"
                )

                return DocumentContent(
                    uri=f"onedrive://{file_id}",
                    name=name,
                    mime_type="text/markdown",
                    text=text,
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Docling processing failed for OneDrive file {name}: {e}")
            return None

    def _download_text(
        self, access_token: str, file_id: str, name: str, mime_type: str
    ) -> Optional[DocumentContent]:
        """Download a text file from OneDrive."""
        import requests as http_requests

        response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(
                f"Failed to download OneDrive file {name}: {response.status_code}"
            )
            return None

        # Try to decode as text
        try:
            text = response.content.decode("utf-8", errors="ignore")
        except Exception:
            text = response.text

        return DocumentContent(
            uri=f"onedrive://{file_id}",
            name=name,
            mime_type=mime_type or "text/plain",
            text=text,
        )


class OutlookMailDocumentSource(MicrosoftGraphMixin, DocumentSource):
    """
    Document source for Outlook Mail using Microsoft Graph API.

    Indexes emails from user's mailbox with optional folder filtering.
    """

    def __init__(
        self,
        account_id: int,
        folder_id: Optional[str] = None,
        days_back: int = 90,
    ):
        """
        Initialize with OAuth account ID and optional folder filter.

        Args:
            account_id: ID of the stored OAuth token
            folder_id: Optional folder ID to filter emails (e.g., "inbox", "sentitems")
            days_back: Number of days of email history to index (default: 90)
        """
        self.account_id = account_id
        self.folder_id = folder_id
        self.days_back = days_back
        self._token_data: Optional[dict] = None

    def is_available(self) -> bool:
        """Check if we can access Outlook."""
        return self._get_access_token() is not None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List emails in the mailbox or specified folder."""
        from datetime import datetime, timedelta

        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list Outlook emails - no valid access token")
            return

        # Calculate date filter
        since_date = datetime.utcnow() - timedelta(days=self.days_back)
        date_filter = f"receivedDateTime ge {since_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"

        # Build the Graph API URL
        if self.folder_id:
            url = f"https://graph.microsoft.com/v1.0/me/mailFolders/{self.folder_id}/messages"
            logger.info(f"Listing emails in Outlook folder: {self.folder_id}")
        else:
            url = "https://graph.microsoft.com/v1.0/me/messages"
            logger.info("Listing all emails in Outlook")

        total_emails = 0

        while url:
            response = http_requests.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "$select": "id,subject,from,receivedDateTime,hasAttachments",
                    "$filter": date_filter,
                    "$orderby": "receivedDateTime desc",
                    "$top": 100,
                },
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Outlook API error: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            messages = data.get("value", [])

            for msg in messages:
                total_emails += 1
                from_addr = msg.get("from", {}).get("emailAddress", {})
                from_name = from_addr.get("name", from_addr.get("address", "Unknown"))

                yield DocumentInfo(
                    uri=f"outlook://{msg['id']}",
                    name=f"{msg.get('subject', 'No Subject')} - {from_name}",
                    mime_type="message/rfc822",
                    modified_time=msg.get("receivedDateTime"),
                )

            # Check for next page
            url = data.get("@odata.nextLink")

        logger.info(f"Found {total_emails} Outlook emails to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read an email from Outlook."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read Outlook email - no valid access token")
            return None

        # Extract message ID from URI (outlook://message_id)
        if not uri.startswith("outlook://"):
            logger.error(f"Invalid Outlook URI: {uri}")
            return None

        message_id = uri.replace("outlook://", "")

        # Get full message content
        response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/me/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,hasAttachments"
            },
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get Outlook email: {response.text}")
            return None

        msg = response.json()

        # Build email content
        from_addr = msg.get("from", {}).get("emailAddress", {})
        from_str = (
            f"{from_addr.get('name', '')} <{from_addr.get('address', '')}>".strip()
        )

        to_list = []
        for recipient in msg.get("toRecipients", []):
            addr = recipient.get("emailAddress", {})
            to_list.append(
                f"{addr.get('name', '')} <{addr.get('address', '')}>".strip()
            )

        cc_list = []
        for recipient in msg.get("ccRecipients", []):
            addr = recipient.get("emailAddress", {})
            cc_list.append(
                f"{addr.get('name', '')} <{addr.get('address', '')}>".strip()
            )

        # Get body content (prefer text, fall back to HTML stripped)
        body = msg.get("body", {})
        body_content = body.get("content", "")
        body_type = body.get("contentType", "text")

        if body_type.lower() == "html":
            # Strip HTML using trafilatura
            try:
                import trafilatura

                body_content = trafilatura.extract(body_content) or body_content
            except Exception:
                # Fallback: basic HTML tag stripping
                import re

                body_content = re.sub(r"<[^>]+>", "", body_content)

        # Format the email
        content = f"Subject: {msg.get('subject', 'No Subject')}\n"
        content += f"From: {from_str}\n"
        content += f"To: {', '.join(to_list)}\n"
        if cc_list:
            content += f"Cc: {', '.join(cc_list)}\n"
        content += f"Date: {msg.get('receivedDateTime', '')}\n"
        content += f"Has Attachments: {msg.get('hasAttachments', False)}\n"
        content += "\n---\n\n"
        content += body_content

        # Get account email for action routing
        token_data = self._get_token_data()
        account_email = token_data.get("account_email", "") if token_data else ""

        # Get conversation ID for threading (Outlook's equivalent of thread_id)
        conversation_id = msg.get("conversationId", "")

        # Parse email date to YYYY-MM-DD format
        email_date = None
        received_dt = msg.get("receivedDateTime", "")
        if received_dt:
            try:
                email_date = received_dt[:10]  # Already ISO format
            except Exception:
                pass

        return DocumentContent(
            uri=uri,
            name=msg.get("subject", "No Subject"),
            mime_type="text/plain",
            text=content,
            metadata={
                "message_id": message_id,  # For replies/forwards
                "conversation_id": conversation_id,  # Outlook's thread equivalent
                "account_id": self.account_id,  # Internal OAuth account ID
                "account_email": account_email,  # For action routing
                "email_date": email_date,
                "from": from_str,
                "from_name": from_addr.get("name", ""),
                "to": ", ".join(to_list),
                "cc": ", ".join(cc_list) if cc_list else "",
                "subject": msg.get("subject", ""),
                "source_type": "email",
            },
        )


class OneNoteDocumentSource(MicrosoftGraphMixin, DocumentSource):
    """
    Document source for OneNote using Microsoft Graph API.

    Indexes notebooks and pages from user's OneNote.
    """

    def __init__(
        self,
        account_id: int,
        notebook_id: Optional[str] = None,
    ):
        """
        Initialize with OAuth account ID and optional notebook filter.

        Args:
            account_id: ID of the stored OAuth token
            notebook_id: Optional notebook ID to restrict indexing scope
        """
        self.account_id = account_id
        self.notebook_id = notebook_id
        self._token_data: Optional[dict] = None

    def is_available(self) -> bool:
        """Check if we can access OneNote."""
        return self._get_access_token() is not None

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all pages in OneNote or specified notebook."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list OneNote pages - no valid access token")
            return

        # If notebook specified, list pages in that notebook's sections
        # Otherwise, list all pages across all notebooks
        if self.notebook_id:
            # First get sections in the notebook
            sections_url = f"https://graph.microsoft.com/v1.0/me/onenote/notebooks/{self.notebook_id}/sections"
            logger.info(f"Listing pages in OneNote notebook: {self.notebook_id}")
        else:
            sections_url = "https://graph.microsoft.com/v1.0/me/onenote/sections"
            logger.info("Listing all OneNote pages")

        total_pages = 0

        # Get all sections
        sections_response = http_requests.get(
            sections_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if sections_response.status_code != 200:
            logger.error(
                f"OneNote sections API error: {sections_response.status_code} - {sections_response.text}"
            )
            return

        sections_data = sections_response.json()
        sections = sections_data.get("value", [])
        logger.info(f"Found {len(sections)} OneNote sections to scan")

        # For each section, get pages
        for section_idx, section in enumerate(sections):
            section_id = section["id"]
            section_name = section.get("displayName", "Unknown Section")
            logger.info(
                f"Scanning section {section_idx + 1}/{len(sections)}: {section_name}"
            )

            pages_url = f"https://graph.microsoft.com/v1.0/me/onenote/sections/{section_id}/pages"

            while pages_url:
                pages_response = http_requests.get(
                    pages_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={
                        "$select": "id,title,createdDateTime,lastModifiedDateTime",
                        "$top": 100,
                    },
                    timeout=30,
                )

                if pages_response.status_code != 200:
                    logger.warning(
                        f"Failed to list pages in section {section_name}: {pages_response.status_code}"
                    )
                    break

                pages_data = pages_response.json()
                pages = pages_data.get("value", [])
                logger.debug(f"Found {len(pages)} pages in section {section_name}")

                for page in pages:
                    total_pages += 1
                    yield DocumentInfo(
                        uri=f"onenote://{page['id']}",
                        name=f"{section_name} / {page.get('title', 'Untitled')}",
                        mime_type="text/html",
                        modified_time=page.get("lastModifiedDateTime"),
                    )

                # Check for next page
                pages_url = pages_data.get("@odata.nextLink")

        logger.info(f"Found {total_pages} OneNote pages to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a page from OneNote."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read OneNote page - no valid access token")
            return None

        # Extract page ID from URI (onenote://page_id)
        if not uri.startswith("onenote://"):
            logger.error(f"Invalid OneNote URI: {uri}")
            return None

        page_id = uri.replace("onenote://", "")

        # Get page content (returns HTML)
        response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}/content",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get OneNote page content: {response.text}")
            return None

        html_content = response.text

        # Get page metadata for the title
        meta_response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/me/onenote/pages/{page_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"$select": "id,title,parentSection"},
            timeout=30,
        )

        title = "Untitled"
        section_name = ""
        if meta_response.status_code == 200:
            meta = meta_response.json()
            title = meta.get("title", "Untitled")
            section_name = meta.get("parentSection", {}).get("displayName", "")

        # Convert HTML to text using trafilatura
        try:
            import trafilatura

            text_content = trafilatura.extract(html_content)
            if not text_content:
                # Fallback: basic HTML tag stripping
                import re

                text_content = re.sub(r"<[^>]+>", " ", html_content)
                text_content = re.sub(r"\s+", " ", text_content).strip()
        except Exception as e:
            logger.warning(f"trafilatura extraction failed for OneNote page: {e}")
            # Fallback: basic HTML tag stripping
            import re

            text_content = re.sub(r"<[^>]+>", " ", html_content)
            text_content = re.sub(r"\s+", " ", text_content).strip()

        # Format with metadata
        full_name = f"{section_name} / {title}" if section_name else title
        content = f"# {title}\n\n{text_content}"

        return DocumentContent(
            uri=uri,
            name=full_name,
            mime_type="text/plain",
            text=content,
        )


class TeamsDocumentSource(MicrosoftGraphMixin, DocumentSource):
    """
    Document source for Microsoft Teams channel messages using Graph API.

    Indexes messages from Teams channels with optional team/channel filtering.
    """

    def __init__(
        self,
        account_id: int,
        team_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        days_back: int = 90,
    ):
        """
        Initialize with OAuth account ID and optional team/channel filter.

        Args:
            account_id: ID of the stored OAuth token
            team_id: Optional team ID to filter (required if channel_id is set)
            channel_id: Optional channel ID to filter messages
            days_back: Number of days of message history to index (default: 90)
        """
        self.account_id = account_id
        self.team_id = team_id
        self.channel_id = channel_id
        self.days_back = days_back
        self._token_data: Optional[dict] = None

    def is_available(self) -> bool:
        """Check if we can access Teams."""
        return self._get_access_token() is not None

    def _get_teams(self, access_token: str) -> list[dict]:
        """Get list of teams the user is a member of."""
        import requests as http_requests

        teams = []
        url = "https://graph.microsoft.com/v1.0/me/joinedTeams"

        while url:
            response = http_requests.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"$select": "id,displayName,description"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Teams API error: {response.status_code} - {response.text}"
                )
                break

            data = response.json()
            teams.extend(data.get("value", []))
            url = data.get("@odata.nextLink")

        return teams

    def _get_channels(self, access_token: str, team_id: str) -> list[dict]:
        """Get list of channels for a team."""
        import requests as http_requests

        channels = []
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels"

        while url:
            response = http_requests.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"$select": "id,displayName,description"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Teams channels API error: {response.status_code} - {response.text}"
                )
                break

            data = response.json()
            channels.extend(data.get("value", []))
            url = data.get("@odata.nextLink")

        return channels

    def _get_channel_messages(
        self, access_token: str, team_id: str, channel_id: str, since_date: str
    ) -> Iterator[dict]:
        """Get messages from a channel."""
        import requests as http_requests

        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"

        while url:
            response = http_requests.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "$top": 50,
                    "$orderby": "createdDateTime desc",
                },
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Teams messages API error: {response.status_code} - {response.text}"
                )
                break

            data = response.json()
            messages = data.get("value", [])

            for msg in messages:
                # Check date filter (Graph API doesn't support $filter on messages)
                created = msg.get("createdDateTime", "")
                if created and created < since_date:
                    # Messages are ordered by date desc, so we can stop here
                    return

                yield msg

            url = data.get("@odata.nextLink")

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List messages from Teams channels."""
        from datetime import datetime, timedelta

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list Teams messages - no valid access token")
            return

        # Calculate date filter
        since_date = (datetime.utcnow() - timedelta(days=self.days_back)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        total_messages = 0

        # Determine which teams/channels to index
        if self.team_id and self.channel_id:
            # Specific channel
            teams_channels = [(self.team_id, self.channel_id)]
        elif self.team_id:
            # All channels in a specific team
            channels = self._get_channels(access_token, self.team_id)
            teams_channels = [(self.team_id, ch["id"]) for ch in channels]
        else:
            # All channels in all teams
            teams_channels = []
            teams = self._get_teams(access_token)
            for team in teams:
                channels = self._get_channels(access_token, team["id"])
                for ch in channels:
                    teams_channels.append((team["id"], ch["id"]))

        logger.info(f"Indexing messages from {len(teams_channels)} Teams channel(s)")

        for team_id, channel_id in teams_channels:
            for msg in self._get_channel_messages(
                access_token, team_id, channel_id, since_date
            ):
                # Skip system messages
                msg_type = msg.get("messageType", "")
                if msg_type != "message":
                    continue

                # Skip deleted messages
                if msg.get("deletedDateTime"):
                    continue

                total_messages += 1
                sender = msg.get("from", {}).get("user", {})
                sender_name = sender.get("displayName", "Unknown")

                # Build a readable name
                body = msg.get("body", {}).get("content", "")
                # Get first 50 chars of message for the name
                preview = body[:50].replace("\n", " ").strip()
                if len(body) > 50:
                    preview += "..."

                yield DocumentInfo(
                    uri=f"teams://{team_id}/{channel_id}/{msg['id']}",
                    name=f"{sender_name}: {preview}"
                    if preview
                    else f"{sender_name}: [message]",
                    mime_type="text/plain",
                    modified_time=msg.get("createdDateTime"),
                )

        logger.info(f"Found {total_messages} Teams messages to index")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a message from Teams."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read Teams message - no valid access token")
            return None

        # Extract IDs from URI (teams://team_id/channel_id/message_id)
        if not uri.startswith("teams://"):
            logger.error(f"Invalid Teams URI: {uri}")
            return None

        parts = uri.replace("teams://", "").split("/")
        if len(parts) != 3:
            logger.error(f"Invalid Teams URI format: {uri}")
            return None

        team_id, channel_id, message_id = parts

        # Get the message
        response = http_requests.get(
            f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages/{message_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Failed to get Teams message: {response.text}")
            return None

        msg = response.json()

        # Get sender info
        sender = msg.get("from", {}).get("user", {})
        sender_name = sender.get("displayName", "Unknown")
        sender_email = sender.get("email", "")

        # Get body content
        body = msg.get("body", {})
        body_content = body.get("content", "")
        body_type = body.get("contentType", "text")

        if body_type.lower() == "html":
            # Strip HTML
            try:
                import trafilatura

                body_content = trafilatura.extract(body_content) or body_content
            except Exception:
                import re

                body_content = re.sub(r"<[^>]+>", "", body_content)

        # Get replies count
        replies_url = msg.get("replies@odata.context", "")

        # Format the message
        content = f"From: {sender_name}"
        if sender_email:
            content += f" <{sender_email}>"
        content += f"\nDate: {msg.get('createdDateTime', '')}\n"

        # Add subject if present (usually for channel posts)
        subject = msg.get("subject")
        if subject:
            content += f"Subject: {subject}\n"

        # Add mentions if any
        mentions = msg.get("mentions", [])
        if mentions:
            mention_names = [
                m.get("mentioned", {}).get("user", {}).get("displayName", "")
                for m in mentions
            ]
            mention_names = [n for n in mention_names if n]
            if mention_names:
                content += f"Mentions: {', '.join(mention_names)}\n"

        # Add attachments info if any
        attachments = msg.get("attachments", [])
        if attachments:
            attachment_names = [a.get("name", "attachment") for a in attachments]
            content += f"Attachments: {', '.join(attachment_names)}\n"

        content += "\n---\n\n"
        content += body_content

        # Try to get replies
        try:
            replies_response = http_requests.get(
                f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages/{message_id}/replies",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"$top": 50},
                timeout=30,
            )

            if replies_response.status_code == 200:
                replies = replies_response.json().get("value", [])
                if replies:
                    content += f"\n\n--- Replies ({len(replies)}) ---\n"
                    for reply in replies:
                        reply_sender = reply.get("from", {}).get("user", {})
                        reply_sender_name = reply_sender.get("displayName", "Unknown")
                        reply_body = reply.get("body", {}).get("content", "")
                        reply_type = reply.get("body", {}).get("contentType", "text")

                        if reply_type.lower() == "html":
                            try:
                                import trafilatura

                                reply_body = (
                                    trafilatura.extract(reply_body) or reply_body
                                )
                            except Exception:
                                import re

                                reply_body = re.sub(r"<[^>]+>", "", reply_body)

                        content += f"\n[{reply_sender_name} @ {reply.get('createdDateTime', '')}]\n"
                        content += f"{reply_body}\n"
        except Exception as e:
            logger.warning(f"Failed to get replies for message: {e}")

        return DocumentContent(
            uri=uri,
            name=f"{sender_name}: {msg.get('subject', 'Teams message')}",
            mime_type="text/plain",
            text=content,
        )


class WebSearchDocumentSource(DocumentSource):
    """
    Document source for scheduled web searches.

    Runs a web search query, scrapes the top results, and indexes the content.
    Useful for keeping up-to-date with news or specific topics.
    """

    def __init__(
        self,
        query: str,
        max_results: int = 10,
        pages_to_scrape: int = 5,
        time_range: Optional[str] = None,
        category: Optional[str] = None,
    ):
        """
        Initialize with search query and settings.

        Args:
            query: Search query string (e.g., "uk latest news")
            max_results: Maximum number of search results to fetch
            pages_to_scrape: How many of those results to actually scrape
            time_range: Optional time filter (day, week, month, year)
            category: Optional search category (news, general, etc.)
        """
        self.query = query
        self.max_results = max_results
        self.pages_to_scrape = pages_to_scrape
        self.time_range = time_range
        self.category = category
        self._search_results: Optional[list] = None
        self._scraped_content: Optional[dict] = None

    def is_available(self) -> bool:
        """Check if search is available."""
        from augmentation.search import get_configured_search_provider

        provider = get_configured_search_provider()
        return provider is not None

    def _search_if_needed(self):
        """Run the search if not already done."""
        if self._search_results is not None:
            return

        from augmentation.search import get_configured_search_provider

        provider = get_configured_search_provider()
        if not provider:
            logger.error("No search provider configured for web search source")
            self._search_results = []
            return

        logger.info(
            f"Running web search: '{self.query}' (max_results={self.max_results}, "
            f"time_range={self.time_range}, category={self.category})"
        )

        try:
            self._search_results = provider.search(
                query=self.query,
                max_results=self.max_results,
                time_range=self.time_range,
                category=self.category,
            )
            logger.info(f"Found {len(self._search_results)} search results")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            self._search_results = []

    def _scrape_if_needed(self):
        """Scrape top results if not already done."""
        if self._scraped_content is not None:
            return

        self._search_if_needed()
        self._scraped_content = {}

        if not self._search_results:
            return

        # Get scraper
        from augmentation.scraper import WebScraper

        scraper = WebScraper()

        # Scrape top N results
        urls_to_scrape = [r.url for r in self._search_results[: self.pages_to_scrape]]
        logger.info(f"Scraping {len(urls_to_scrape)} pages from search results")

        scraped = scraper.scrape_multiple(urls_to_scrape, max_urls=self.pages_to_scrape)

        for result in scraped:
            if result.success and result.content:
                self._scraped_content[result.url] = result

        logger.info(
            f"Successfully scraped {len(self._scraped_content)}/{len(urls_to_scrape)} pages"
        )

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List all scraped search result pages as documents."""
        from datetime import datetime

        self._scrape_if_needed()

        # Use current timestamp for all results from this search run
        # This ensures old search results get replaced on re-index
        search_timestamp = datetime.utcnow().isoformat()

        for url, result in self._scraped_content.items():
            yield DocumentInfo(
                uri=url,
                name=result.title or url,
                mime_type="text/html",
                modified_time=search_timestamp,
            )

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read content from a scraped search result."""
        self._scrape_if_needed()

        if uri not in self._scraped_content:
            return None

        result = self._scraped_content[uri]
        return DocumentContent(
            uri=uri,
            name=result.title or uri,
            mime_type="text/plain",
            text=result.content,
        )


class GoogleKeepDocumentSource(DocumentSource):
    """
    Document source for Google Keep using the Keep API.

    Uses Google Keep API directly with stored OAuth tokens.
    Indexes notes (both text and list notes) for RAG retrieval.

    Note: Google Keep API is only available for Google Workspace accounts.
    """

    KEEP_API_BASE = "https://keep.googleapis.com/v1"

    def __init__(
        self,
        account_id: int,
        include_trashed: bool = False,
    ):
        """
        Initialize with OAuth account ID.

        Args:
            account_id: ID of the stored OAuth token
            include_trashed: Whether to include trashed notes
        """
        self.account_id = account_id
        self.include_trashed = include_trashed
        self._access_token: Optional[str] = None
        self._token_data: Optional[dict] = None

    def _get_token_data(self) -> Optional[dict]:
        """Get and cache the decrypted token data."""
        if self._token_data is None:
            from db.oauth_tokens import get_oauth_token_by_id

            self._token_data = get_oauth_token_by_id(self.account_id)
        return self._token_data

    def _get_access_token(self) -> Optional[str]:
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

        # Try the access token first with a simple API call
        test_response = http_requests.get(
            f"{self.KEEP_API_BASE}/notes",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"pageSize": 1},
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
        """Check if we can access Google Keep."""
        return self._get_access_token() is not None

    def _format_note_content(self, note: dict) -> str:
        """Format a Keep note's content for indexing."""
        parts = []

        title = note.get("title", "")
        if title:
            parts.append(f"# {title}\n")

        body = note.get("body", {})

        # Handle text notes
        if "text" in body:
            text_content = body["text"].get("text", "")
            if text_content:
                parts.append(text_content)

        # Handle list notes
        if "list" in body:
            list_content = body["list"]
            list_items = list_content.get("listItems", [])
            for item in list_items:
                text_item = item.get("text", {})
                item_text = text_item.get("text", "")
                checked = item.get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                parts.append(f"- {checkbox} {item_text}")

                # Handle nested items (one level of nesting)
                child_items = item.get("childListItems", [])
                for child in child_items:
                    child_text_item = child.get("text", {})
                    child_text = child_text_item.get("text", "")
                    child_checked = child.get("checked", False)
                    child_checkbox = "[x]" if child_checked else "[ ]"
                    parts.append(f"  - {child_checkbox} {child_text}")

        return "\n".join(parts)

    def list_documents(self) -> Iterator[DocumentInfo]:
        """List notes from Google Keep."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot list notes - no valid access token")
            return

        logger.info("Listing Google Keep notes")

        total_notes = 0
        page_token = None

        # Build filter - by default exclude trashed notes
        filter_str = None if self.include_trashed else "trashed=false"

        while True:
            params = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            if filter_str:
                params["filter"] = filter_str

            response = http_requests.get(
                f"{self.KEEP_API_BASE}/notes",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Keep API error listing notes: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            notes = data.get("notes", [])

            for note in notes:
                note_name = note.get("name", "")  # e.g., "notes/abc123"
                note_id = note_name.split("/")[-1] if "/" in note_name else note_name
                title = note.get("title", "Untitled Note")
                update_time = note.get("updateTime")

                total_notes += 1
                yield DocumentInfo(
                    uri=f"gkeep://{note_id}",
                    name=title[:100] if title else f"Note {note_id[:8]}",
                    mime_type="text/plain",
                    modified_time=update_time,
                    metadata={
                        "note_id": note_id,
                        "note_name": note_name,
                        "source_type": "note",
                    },
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Listed {total_notes} Google Keep notes")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read a specific note from Google Keep."""
        import requests as http_requests

        access_token = self._get_access_token()
        if not access_token:
            logger.error("Cannot read note - no valid access token")
            return None

        # Extract note ID from URI (gkeep://note_id)
        if uri.startswith("gkeep://"):
            note_id = uri[8:]
        else:
            note_id = uri

        response = http_requests.get(
            f"{self.KEEP_API_BASE}/notes/{note_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(
                f"Keep API error reading note {note_id}: {response.status_code} - {response.text}"
            )
            return None

        note = response.json()
        title = note.get("title", "Untitled Note")
        content = self._format_note_content(note)

        return DocumentContent(
            uri=uri,
            name=title,
            mime_type="text/plain",
            text=content,
        )


def get_document_source(
    source_type: str,
    source_path: Optional[str] = None,
    mcp_config: Optional[dict] = None,
    google_account_id: Optional[int] = None,
    gdrive_folder_id: Optional[str] = None,
    gmail_label_id: Optional[str] = None,
    gcalendar_calendar_id: Optional[str] = None,
    gtasks_tasklist_id: Optional[str] = None,
    gcontacts_group_id: Optional[str] = None,
    microsoft_account_id: Optional[int] = None,
    onedrive_folder_id: Optional[str] = None,
    outlook_folder_id: Optional[str] = None,
    outlook_days_back: int = 90,
    onenote_notebook_id: Optional[str] = None,
    teams_team_id: Optional[str] = None,
    teams_channel_id: Optional[str] = None,
    teams_days_back: int = 90,
    paperless_url: Optional[str] = None,  # Deprecated - use PAPERLESS_URL env var
    paperless_token: Optional[str] = None,  # Deprecated - use PAPERLESS_TOKEN env var
    paperless_tag_id: Optional[int] = None,
    github_repo: Optional[str] = None,
    github_branch: Optional[str] = None,
    github_path: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    notion_page_id: Optional[str] = None,
    nextcloud_folder: Optional[str] = None,
    website_url: Optional[str] = None,
    website_crawl_depth: int = 1,
    website_max_pages: int = 50,
    website_include_pattern: Optional[str] = None,
    website_exclude_pattern: Optional[str] = None,
    website_crawler_override: Optional[str] = None,
    slack_channel_id: Optional[str] = None,
    slack_channel_types: str = "public_channel",
    slack_days_back: int = 90,
    todoist_project_id: Optional[str] = None,
    todoist_filter: Optional[str] = None,
    todoist_include_completed: bool = False,
    websearch_query: Optional[str] = None,
    websearch_max_results: int = 10,
    websearch_pages_to_scrape: int = 5,
    websearch_time_range: Optional[str] = None,
    websearch_category: Optional[str] = None,
    vision_provider: str = "local",
    vision_model: Optional[str] = None,
    vision_ollama_url: Optional[str] = None,
) -> DocumentSource:
    """
    Factory function to create the appropriate document source.

    Args:
        source_type: "local", "mcp", "mcp:gdrive", "mcp:gmail", "mcp:gcalendar",
            "mcp:gtasks", "mcp:gcontacts", "mcp:onedrive", "mcp:outlook",
            "mcp:onenote", "mcp:github", "slack", "notion", "nextcloud",
            "paperless", or "website"
        source_path: Path for local sources
        mcp_config: Configuration dict for MCP sources
        google_account_id: OAuth account ID for Google sources
        gdrive_folder_id: Optional folder ID to filter Google Drive indexing
        gmail_label_id: Optional label ID to filter Gmail indexing (e.g., "INBOX", "SENT")
        gcalendar_calendar_id: Optional calendar ID to filter Calendar indexing
        gtasks_tasklist_id: Optional task list ID to filter Tasks indexing
        gcontacts_group_id: Optional contact group ID to filter Contacts indexing
        microsoft_account_id: OAuth account ID for Microsoft sources
        onedrive_folder_id: Optional folder ID to filter OneDrive indexing
        outlook_folder_id: Optional folder ID to filter Outlook indexing (e.g., "inbox")
        outlook_days_back: Days of email history to index (default: 90)
        onenote_notebook_id: Optional notebook ID to filter OneNote indexing
        paperless_url: Deprecated - credentials read from PAPERLESS_URL env var
        paperless_token: Deprecated - credentials read from PAPERLESS_TOKEN env var
        github_repo: Repository in "owner/repo" format for GitHub sources
        github_branch: Branch name for GitHub sources
        github_path: Path filter for GitHub sources
        notion_database_id: Optional Notion database ID to index
        notion_page_id: Optional Notion page ID (indexes children)
        nextcloud_folder: Optional folder path to restrict Nextcloud indexing (e.g., "/Documents")
        website_url: Starting URL for website crawling
        website_crawl_depth: How many levels of links to follow (0 = start page only)
        website_max_pages: Maximum number of pages to crawl
        website_include_pattern: Regex pattern - only crawl matching URLs
        website_exclude_pattern: Regex pattern - skip matching URLs
        slack_channel_id: Optional specific Slack channel ID to index
        slack_channel_types: Channel types to index (public_channel, private_channel)
        slack_days_back: How many days of Slack history to fetch (max 90 for free plan)
        vision_provider: Vision model provider for Docling ("local", "ollama", etc.)
        vision_model: Vision model name (e.g., "granite3.2-vision:latest")
        vision_ollama_url: Ollama URL when using ollama provider

    Returns:
        Appropriate DocumentSource instance
    """
    if source_type == "notion":
        # Notion direct API integration (no MCP/Node.js required)
        source = NotionDocumentSource(
            database_id=notion_database_id,
            root_page_id=notion_page_id,
        )
        if not source.api_token:
            raise ValueError(
                "NOTION_TOKEN or NOTION_API_KEY environment variable required for Notion source"
            )
        logger.info(
            f"Creating Notion source (database={notion_database_id or 'none'}, "
            f"page={notion_page_id or 'none'})"
        )
        return source

    elif source_type == "paperless":
        # Credentials from environment variables (PAPERLESS_URL, PAPERLESS_TOKEN)
        source = PaperlessDocumentSource(tag_id=paperless_tag_id)
        if not source.base_url or not source.api_token:
            raise ValueError(
                "PAPERLESS_URL and PAPERLESS_TOKEN environment variables required for paperless source"
            )
        tag_info = f", tag_id={paperless_tag_id}" if paperless_tag_id else ""
        logger.info(f"Creating Paperless source (url={source.base_url}{tag_info})")
        return source

    elif source_type == "nextcloud":
        # Credentials from environment variables (NEXTCLOUD_URL, NEXTCLOUD_USER, NEXTCLOUD_PASSWORD)
        source = NextcloudDocumentSource(
            folder_path=nextcloud_folder,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_ollama_url=vision_ollama_url,
        )
        if not source.base_url or not source.username or not source.password:
            raise ValueError(
                "NEXTCLOUD_URL, NEXTCLOUD_USER, and NEXTCLOUD_PASSWORD environment variables required for nextcloud source"
            )
        logger.info(
            f"Creating Nextcloud source (url={source.base_url}, "
            f"folder={nextcloud_folder or 'all'}, vision={vision_provider})"
        )
        return source

    elif source_type == "mcp:github":
        if not github_repo:
            raise ValueError("github_repo required for GitHub source")
        logger.info(
            f"Creating GitHub source (repo={github_repo}, branch={github_branch or 'default'}, path={github_path or 'all'})"
        )
        return GitHubDocumentSource(
            repo=github_repo,
            branch=github_branch,
            path_filter=github_path,
        )

    elif source_type == "local":
        if not source_path:
            raise ValueError("source_path required for local document source")
        return LocalDocumentSource(source_path)

    elif source_type == "website":
        if not website_url:
            raise ValueError("website_url required for website document source")
        logger.info(
            f"Creating Website source (url={website_url}, "
            f"depth={website_crawl_depth}, max_pages={website_max_pages})"
        )
        return WebsiteDocumentSource(
            url=website_url,
            crawl_depth=website_crawl_depth,
            max_pages=website_max_pages,
            include_pattern=website_include_pattern,
            exclude_pattern=website_exclude_pattern,
            crawler_override=website_crawler_override,
        )

    elif source_type == "websearch":
        if not websearch_query:
            raise ValueError("websearch_query required for web search document source")
        logger.info(
            f"Creating Web Search source (query='{websearch_query}', "
            f"max_results={websearch_max_results}, pages_to_scrape={websearch_pages_to_scrape}, "
            f"time_range={websearch_time_range}, category={websearch_category})"
        )
        return WebSearchDocumentSource(
            query=websearch_query,
            max_results=websearch_max_results,
            pages_to_scrape=websearch_pages_to_scrape,
            time_range=websearch_time_range,
            category=websearch_category,
        )

    elif source_type == "mcp:gdrive":
        # Google Drive - use direct API for better performance and proper folder filtering
        if not google_account_id:
            raise ValueError(f"google_account_id required for {source_type} source")

        logger.info(
            f"Creating Google Drive source (account={google_account_id}, "
            f"folder={gdrive_folder_id or 'all'}, vision={vision_provider})"
        )
        return GoogleDriveDocumentSource(
            account_id=google_account_id,
            folder_id=gdrive_folder_id,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_ollama_url=vision_ollama_url,
        )

    elif source_type == "mcp:gmail":
        # Gmail - use direct API for better performance and reliability
        if not google_account_id:
            raise ValueError(f"google_account_id required for {source_type} source")

        logger.info(
            f"Creating Gmail source (account={google_account_id}, "
            f"label={gmail_label_id or 'all'})"
        )
        return GmailDocumentSource(
            account_id=google_account_id,
            label_id=gmail_label_id,
        )

    elif source_type == "mcp:gcalendar":
        # Calendar - use direct API for better performance and reliability
        if not google_account_id:
            raise ValueError(f"google_account_id required for {source_type} source")

        logger.info(
            f"Creating Calendar source (account={google_account_id}, "
            f"calendar={gcalendar_calendar_id or 'primary'})"
        )
        return GoogleCalendarDocumentSource(
            account_id=google_account_id,
            calendar_id=gcalendar_calendar_id,
            # Default: 30 days back, 30 days forward
        )

    elif source_type == "mcp:gtasks":
        # Google Tasks - use direct API
        if not google_account_id:
            raise ValueError(f"google_account_id required for {source_type} source")

        logger.info(
            f"Creating Google Tasks source (account={google_account_id}, "
            f"tasklist={gtasks_tasklist_id or '@default'})"
        )
        return GoogleTasksDocumentSource(
            account_id=google_account_id,
            tasklist_id=gtasks_tasklist_id,
        )

    elif source_type == "mcp:gcontacts":
        # Google Contacts - use direct People API
        if not google_account_id:
            raise ValueError(f"google_account_id required for {source_type} source")

        logger.info(
            f"Creating Google Contacts source (account={google_account_id}, "
            f"group={gcontacts_group_id or 'all'})"
        )
        return GoogleContactsDocumentSource(
            account_id=google_account_id,
            group_id=gcontacts_group_id,
        )

    elif source_type == "mcp:gkeep":
        # Google Keep - use Keep API (Workspace accounts only)
        if not google_account_id:
            raise ValueError(f"google_account_id required for {source_type} source")

        logger.info(f"Creating Google Keep source (account={google_account_id})")
        return GoogleKeepDocumentSource(
            account_id=google_account_id,
        )

    elif source_type == "mcp:onedrive":
        # OneDrive - use Microsoft Graph API
        if not microsoft_account_id:
            raise ValueError(f"microsoft_account_id required for {source_type} source")

        logger.info(
            f"Creating OneDrive source (account={microsoft_account_id}, "
            f"folder={onedrive_folder_id or 'root'}, vision={vision_provider})"
        )
        return OneDriveDocumentSource(
            account_id=microsoft_account_id,
            folder_id=onedrive_folder_id,
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_ollama_url=vision_ollama_url,
        )

    elif source_type == "mcp:outlook":
        # Outlook Mail - use Microsoft Graph API
        if not microsoft_account_id:
            raise ValueError(f"microsoft_account_id required for {source_type} source")

        logger.info(
            f"Creating Outlook source (account={microsoft_account_id}, "
            f"folder={outlook_folder_id or 'all'}, days_back={outlook_days_back})"
        )
        return OutlookMailDocumentSource(
            account_id=microsoft_account_id,
            folder_id=outlook_folder_id,
            days_back=outlook_days_back,
        )

    elif source_type == "mcp:onenote":
        # OneNote - use Microsoft Graph API
        if not microsoft_account_id:
            raise ValueError(f"microsoft_account_id required for {source_type} source")

        logger.info(
            f"Creating OneNote source (account={microsoft_account_id}, "
            f"notebook={onenote_notebook_id or 'all'})"
        )
        return OneNoteDocumentSource(
            account_id=microsoft_account_id,
            notebook_id=onenote_notebook_id,
        )

    elif source_type == "mcp:teams":
        # Microsoft Teams - use Microsoft Graph API
        if not microsoft_account_id:
            raise ValueError(f"microsoft_account_id required for {source_type} source")

        logger.info(
            f"Creating Teams source (account={microsoft_account_id}, "
            f"team={teams_team_id or 'all'}, channel={teams_channel_id or 'all'}, "
            f"days_back={teams_days_back})"
        )
        return TeamsDocumentSource(
            account_id=microsoft_account_id,
            team_id=teams_team_id,
            channel_id=teams_channel_id,
            days_back=teams_days_back,
        )

    elif source_type == "slack":
        # Slack - use direct REST API
        source = SlackDocumentSource(
            channel_id=slack_channel_id,
            channel_types=slack_channel_types,
            days_back=slack_days_back,
        )
        if not source.bot_token:
            raise ValueError(
                "SLACK_BOT_TOKEN environment variable required for Slack source"
            )
        logger.info(
            f"Creating Slack source (channel={slack_channel_id or 'all'}, "
            f"types={slack_channel_types}, days_back={slack_days_back})"
        )
        return source

    elif source_type == "todoist":
        # Todoist - use direct REST API
        source = TodoistDocumentSource(
            project_id=todoist_project_id,
            filter_expr=todoist_filter,
            include_completed=todoist_include_completed,
        )
        if not source.api_token:
            raise ValueError(
                "TODOIST_API_TOKEN or TODOIST_API_KEY environment variable required for Todoist source"
            )
        logger.info(
            f"Creating Todoist source (project={todoist_project_id or 'all'}, "
            f"filter={todoist_filter or 'none'}, include_completed={todoist_include_completed})"
        )
        return source

    elif source_type == "mcp" or source_type.startswith("mcp:"):
        if not mcp_config:
            raise ValueError("mcp_config required for MCP document source")

        # Inject required environment variables based on server name
        mcp_config = _inject_mcp_env_vars(mcp_config)

        config = MCPServerConfig.from_dict(mcp_config)
        return MCPDocumentSource(config)

    else:
        raise ValueError(f"Unknown source type: {source_type}")
