"""
GitHub Unified Source Plugin.

Combines GitHub repository indexing (for RAG) and live querying (for real-time data)
into a single plugin with intelligent query routing.

Features:
- Document side: Index issues, PRs, discussions, code for semantic search
- Live side: Query recent activity, open issues, PR status
- Smart routing: Analyze queries to choose optimal data source

Query routing examples:
- "open issues assigned to me" -> Live only (real-time status)
- "discussion about feature X last year" -> RAG only (historical)
- "PRs related to authentication" -> Both, merge results
"""

import logging
import os
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


class GitHubUnifiedSource(PluginUnifiedSource):
    """
    Unified GitHub source - RAG for repository content, Live for activity.

    Single configuration provides:
    - Document indexing: Issues, PRs, discussions indexed for RAG
    - Live queries: Recent activity, open issues, PR status
    - Intelligent routing: System decides RAG vs Live based on query characteristics
    """

    source_type = "github"
    display_name = "GitHub"
    description = (
        "GitHub repositories with code/issue search (RAG) and real-time activity"
    )
    category = "development"
    icon = "🐙"

    # Document store types this unified source handles
    handles_doc_source_types = ["mcp:github"]

    supports_rag = True
    supports_live = True
    supports_actions = False  # Could add issue creation later
    supports_incremental = True

    default_cache_ttl = 300  # 5 minutes for live results

    _abstract = False

    @classmethod
    def build_config_from_store(cls, store) -> dict:
        """Build unified source config from a document store."""
        return {
            "api_token": os.environ.get("GITHUB_TOKEN", ""),
            "repositories": store.github_repo or "",
            "branch": store.github_branch or "",
            "path_filter": store.github_path or "",
        }

    # GitHub API base
    GITHUB_API_BASE = "https://api.github.com"

    @classmethod
    def get_config_fields(cls) -> list[FieldDefinition]:
        """Configuration for admin UI."""
        return [
            FieldDefinition(
                name="api_token",
                label="Personal Access Token",
                field_type=FieldType.PASSWORD,
                required=False,
                help_text="GitHub PAT (or set GITHUB_TOKEN env var). Needs repo scope for private repos.",
            ),
            FieldDefinition(
                name="repositories",
                label="Repositories",
                field_type=FieldType.TEXT,
                required=True,
                help_text="Comma-separated repo names: owner/repo,owner/repo2",
            ),
            FieldDefinition(
                name="index_issues",
                label="Index Issues",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index repository issues",
            ),
            FieldDefinition(
                name="index_prs",
                label="Index Pull Requests",
                field_type=FieldType.BOOLEAN,
                default=True,
                help_text="Index pull requests",
            ),
            FieldDefinition(
                name="index_discussions",
                label="Index Discussions",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Index GitHub Discussions (if enabled on repo)",
            ),
            FieldDefinition(
                name="index_code",
                label="Index Code Files",
                field_type=FieldType.BOOLEAN,
                default=False,
                help_text="Index code files (README, docs, etc.)",
            ),
            FieldDefinition(
                name="code_paths",
                label="Code Paths to Index",
                field_type=FieldType.TEXT,
                required=False,
                help_text="Paths to index (comma-separated). Example: README.md,docs/,src/",
            ),
            FieldDefinition(
                name="index_days",
                label="Days to Index",
                field_type=FieldType.INTEGER,
                default=365,
                help_text="How many days of issues/PRs to index",
            ),
            FieldDefinition(
                name="index_schedule",
                label="Index Schedule",
                field_type=FieldType.SELECT,
                required=False,
                default="",
                options=[
                    {"value": "", "label": "Manual only"},
                    {"value": "0 */6 * * *", "label": "Every 6 hours"},
                    {"value": "0 0 * * *", "label": "Daily"},
                    {"value": "0 0 * * 0", "label": "Weekly"},
                ],
                help_text="How often to re-index",
            ),
            FieldDefinition(
                name="live_max_results",
                label="Live Query Max Results",
                field_type=FieldType.INTEGER,
                default=20,
                help_text="Maximum items to return in live queries",
            ),
        ]

    @classmethod
    def get_live_params(cls) -> list[ParamDefinition]:
        """Parameters the designator can provide for live queries."""
        return [
            ParamDefinition(
                name="action",
                description="Query type: issues, prs, activity, search",
                param_type="string",
                required=False,
                default="activity",
                examples=["issues", "prs", "activity", "search"],
            ),
            ParamDefinition(
                name="query",
                description="Search query for issue/PR content",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="state",
                description="Filter by state: open, closed, all",
                param_type="string",
                required=False,
                default="open",
            ),
            ParamDefinition(
                name="repo",
                description="Specific repository (owner/repo)",
                param_type="string",
                required=False,
            ),
            ParamDefinition(
                name="max_results",
                description="Maximum items to return",
                param_type="integer",
                required=False,
                default=20,
            ),
        ]

    def __init__(self, config: dict):
        """Initialize with configuration."""
        # Get token from config or environment
        self.api_token = config.get("api_token") or os.environ.get("GITHUB_TOKEN", "")

        # Parse repositories
        repo_str = config.get("repositories", "")
        self.repositories = [r.strip() for r in repo_str.split(",") if r.strip()]

        self.index_issues = config.get("index_issues", True)
        self.index_prs = config.get("index_prs", True)
        self.index_discussions = config.get("index_discussions", False)
        self.index_code = config.get("index_code", False)

        # Parse code paths
        path_str = config.get("code_paths", "")
        self.code_paths = (
            [p.strip() for p in path_str.split(",") if p.strip()]
            if path_str
            else ["README.md", "docs/"]
        )

        self.index_days = config.get("index_days", 365)
        self.index_schedule = config.get("index_schedule", "")
        self.live_max_results = config.get("live_max_results", 20)

        self._client = httpx.Client(timeout=30)

    def _get_headers(self) -> dict:
        """Get API headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _github_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a GitHub API request."""
        url = f"{self.GITHUB_API_BASE}/{endpoint.lstrip('/')}"
        response = self._client.get(
            url, headers=self._get_headers(), params=params or {}
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Document Side (for RAG indexing)
    # =========================================================================

    def list_documents(self) -> Iterator[DocumentInfo]:
        """Enumerate items for indexing."""
        logger.info(f"Listing GitHub content from {len(self.repositories)} repos")

        cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=self.index_days)
        ).isoformat()

        for repo in self.repositories:
            if self.index_issues:
                yield from self._list_repo_issues(repo, cutoff_date)

            if self.index_prs:
                yield from self._list_repo_prs(repo, cutoff_date)

            if self.index_code:
                yield from self._list_repo_code(repo)

    def _list_repo_issues(self, repo: str, since: str) -> Iterator[DocumentInfo]:
        """List issues in a repository."""
        page = 1
        while True:
            try:
                issues = self._github_request(
                    f"/repos/{repo}/issues",
                    {
                        "state": "all",
                        "since": since,
                        "per_page": 100,
                        "page": page,
                        "sort": "updated",
                        "direction": "desc",
                    },
                )

                if not issues:
                    break

                for issue in issues:
                    # Skip PRs (they show up in issues endpoint too)
                    if "pull_request" in issue:
                        continue

                    yield DocumentInfo(
                        uri=f"github://{repo}/issues/{issue['number']}",
                        title=f"[{repo}] Issue #{issue['number']}: {issue['title']}",
                        mime_type="text/markdown",
                        modified_at=issue.get("updated_at", ""),
                        metadata={
                            "repo": repo,
                            "type": "issue",
                            "number": issue["number"],
                            "state": issue.get("state", ""),
                        },
                    )

                page += 1
                if len(issues) < 100:
                    break

            except Exception as e:
                logger.error(f"Failed to list issues for {repo}: {e}")
                break

    def _list_repo_prs(self, repo: str, since: str) -> Iterator[DocumentInfo]:
        """List pull requests in a repository."""
        page = 1
        while True:
            try:
                prs = self._github_request(
                    f"/repos/{repo}/pulls",
                    {
                        "state": "all",
                        "per_page": 100,
                        "page": page,
                        "sort": "updated",
                        "direction": "desc",
                    },
                )

                if not prs:
                    break

                for pr in prs:
                    # Check if within date range
                    updated = pr.get("updated_at", "")
                    if updated and updated < since:
                        continue

                    yield DocumentInfo(
                        uri=f"github://{repo}/pulls/{pr['number']}",
                        title=f"[{repo}] PR #{pr['number']}: {pr['title']}",
                        mime_type="text/markdown",
                        modified_at=updated,
                        metadata={
                            "repo": repo,
                            "type": "pull_request",
                            "number": pr["number"],
                            "state": pr.get("state", ""),
                        },
                    )

                page += 1
                if len(prs) < 100:
                    break

            except Exception as e:
                logger.error(f"Failed to list PRs for {repo}: {e}")
                break

    def _list_repo_code(self, repo: str) -> Iterator[DocumentInfo]:
        """List code files in a repository."""
        for path in self.code_paths:
            try:
                # Handle directory vs file
                if path.endswith("/"):
                    # List directory contents
                    contents = self._github_request(
                        f"/repos/{repo}/contents/{path.rstrip('/')}"
                    )
                    if isinstance(contents, list):
                        for item in contents:
                            if item.get("type") == "file":
                                yield DocumentInfo(
                                    uri=f"github://{repo}/code/{item['path']}",
                                    title=f"[{repo}] {item['path']}",
                                    mime_type="text/plain",
                                    modified_at="",
                                    metadata={
                                        "repo": repo,
                                        "type": "code",
                                        "path": item["path"],
                                    },
                                )
                else:
                    # Single file
                    yield DocumentInfo(
                        uri=f"github://{repo}/code/{path}",
                        title=f"[{repo}] {path}",
                        mime_type="text/plain",
                        modified_at="",
                        metadata={
                            "repo": repo,
                            "type": "code",
                            "path": path,
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to list code at {repo}/{path}: {e}")

    def read_document(self, uri: str) -> Optional[DocumentContent]:
        """Read document content for indexing."""
        if not uri.startswith("github://"):
            logger.error(f"Invalid GitHub URI: {uri}")
            return None

        path = uri.replace("github://", "")
        parts = path.split("/")

        if len(parts) < 4:
            return None

        repo = f"{parts[0]}/{parts[1]}"
        content_type = parts[2]
        identifier = "/".join(parts[3:])

        try:
            if content_type == "issues":
                return self._read_issue(repo, int(identifier))
            elif content_type == "pulls":
                return self._read_pr(repo, int(identifier))
            elif content_type == "code":
                return self._read_code(repo, identifier)
        except Exception as e:
            logger.error(f"Failed to read {uri}: {e}")
            return None

        return None

    def _read_issue(self, repo: str, number: int) -> DocumentContent:
        """Read an issue."""
        issue = self._github_request(f"/repos/{repo}/issues/{number}")

        # Get comments
        comments = []
        try:
            comments_data = self._github_request(
                f"/repos/{repo}/issues/{number}/comments",
                {"per_page": 50},
            )
            for comment in comments_data:
                author = comment.get("user", {}).get("login", "Unknown")
                body = comment.get("body", "")
                comments.append(f"**{author}**: {body}")
        except Exception:
            pass

        labels = [l.get("name", "") for l in issue.get("labels", [])]
        assignees = [a.get("login", "") for a in issue.get("assignees", [])]

        content = f"""# Issue #{number}: {issue.get("title", "")}

**Repository**: {repo}
**State**: {issue.get("state", "")}
**Author**: {issue.get("user", {}).get("login", "Unknown")}
**Labels**: {", ".join(labels) if labels else "None"}
**Assignees**: {", ".join(assignees) if assignees else "None"}
**Created**: {issue.get("created_at", "")}
**Updated**: {issue.get("updated_at", "")}

## Description

{issue.get("body", "") or "No description"}

## Comments ({len(comments)})

{chr(10).join(comments) if comments else "No comments"}
"""

        return DocumentContent(
            content=content,
            mime_type="text/markdown",
            metadata={
                "repo": repo,
                "type": "issue",
                "number": number,
                "state": issue.get("state", ""),
                "author": issue.get("user", {}).get("login", ""),
                "labels": ", ".join(labels)
                if labels
                else "",  # ChromaDB requires scalar values
                "source_type": "issue",
            },
        )

    def _read_pr(self, repo: str, number: int) -> DocumentContent:
        """Read a pull request."""
        pr = self._github_request(f"/repos/{repo}/pulls/{number}")

        # Get review comments
        reviews = []
        try:
            reviews_data = self._github_request(
                f"/repos/{repo}/pulls/{number}/reviews",
                {"per_page": 50},
            )
            for review in reviews_data:
                author = review.get("user", {}).get("login", "Unknown")
                state = review.get("state", "")
                body = review.get("body", "")
                if body:
                    reviews.append(f"**{author}** ({state}): {body}")
        except Exception:
            pass

        labels = [l.get("name", "") for l in pr.get("labels", [])]

        content = f"""# PR #{number}: {pr.get("title", "")}

**Repository**: {repo}
**State**: {pr.get("state", "")} {"(merged)" if pr.get("merged") else ""}
**Author**: {pr.get("user", {}).get("login", "Unknown")}
**Labels**: {", ".join(labels) if labels else "None"}
**Base**: {pr.get("base", {}).get("ref", "")} <- **Head**: {pr.get("head", {}).get("ref", "")}
**Created**: {pr.get("created_at", "")}
**Updated**: {pr.get("updated_at", "")}

## Description

{pr.get("body", "") or "No description"}

## Reviews ({len(reviews)})

{chr(10).join(reviews) if reviews else "No reviews"}
"""

        return DocumentContent(
            content=content,
            mime_type="text/markdown",
            metadata={
                "repo": repo,
                "type": "pull_request",
                "number": number,
                "state": pr.get("state", ""),
                "merged": pr.get("merged", False),
                "author": pr.get("user", {}).get("login", ""),
                "labels": ", ".join(labels)
                if labels
                else "",  # ChromaDB requires scalar values
                "source_type": "pull_request",
            },
        )

    def _read_code(self, repo: str, path: str) -> DocumentContent:
        """Read a code file."""
        content_data = self._github_request(f"/repos/{repo}/contents/{path}")

        if content_data.get("type") != "file":
            return None

        import base64

        content_b64 = content_data.get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8", errors="replace")

        return DocumentContent(
            content=content,
            mime_type="text/plain",
            metadata={
                "repo": repo,
                "type": "code",
                "path": path,
                "sha": content_data.get("sha", ""),
                "source_type": "code",
            },
        )

    # =========================================================================
    # Live Side (for real-time queries)
    # =========================================================================

    def fetch(self, params: dict) -> LiveDataResult:
        """Fetch live GitHub data."""
        action = params.get("action", "activity")
        search_query = params.get("query", "")
        state_filter = params.get("state", "open")
        repo_filter = params.get("repo", "")
        max_results = params.get("max_results", self.live_max_results)

        try:
            if action == "issues":
                items = self._get_issues(repo_filter, state_filter, max_results)
                formatted = self._format_issues(items, state_filter)
            elif action == "prs":
                items = self._get_prs(repo_filter, state_filter, max_results)
                formatted = self._format_prs(items, state_filter)
            elif action == "search" and search_query:
                items = self._search_issues(search_query, repo_filter, max_results)
                formatted = self._format_search_results(items)
            else:
                items = self._get_activity(repo_filter, max_results)
                formatted = self._format_activity(items)

            return LiveDataResult(
                success=True,
                data=items,
                formatted=formatted,
                cache_ttl=self.default_cache_ttl,
            )

        except Exception as e:
            logger.error(f"GitHub live query error: {e}")
            return LiveDataResult(
                success=False,
                error=str(e),
            )

    def _get_issues(self, repo_filter: str, state: str, max_results: int) -> list[dict]:
        """Get issues from repositories."""
        issues = []
        repos = [repo_filter] if repo_filter else self.repositories

        for repo in repos:
            try:
                data = self._github_request(
                    f"/repos/{repo}/issues",
                    {"state": state, "per_page": min(max_results, 30)},
                )
                for issue in data:
                    if "pull_request" in issue:
                        continue
                    issues.append(
                        {
                            "repo": repo,
                            "number": issue["number"],
                            "title": issue["title"],
                            "state": issue["state"],
                            "author": issue.get("user", {}).get("login", ""),
                            "labels": [l["name"] for l in issue.get("labels", [])],
                            "updated": issue.get("updated_at", ""),
                            "url": issue.get("html_url", ""),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to get issues from {repo}: {e}")

        return issues[:max_results]

    def _get_prs(self, repo_filter: str, state: str, max_results: int) -> list[dict]:
        """Get pull requests from repositories."""
        prs = []
        repos = [repo_filter] if repo_filter else self.repositories

        for repo in repos:
            try:
                data = self._github_request(
                    f"/repos/{repo}/pulls",
                    {"state": state, "per_page": min(max_results, 30)},
                )
                for pr in data:
                    prs.append(
                        {
                            "repo": repo,
                            "number": pr["number"],
                            "title": pr["title"],
                            "state": pr["state"],
                            "author": pr.get("user", {}).get("login", ""),
                            "labels": [l["name"] for l in pr.get("labels", [])],
                            "draft": pr.get("draft", False),
                            "updated": pr.get("updated_at", ""),
                            "url": pr.get("html_url", ""),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to get PRs from {repo}: {e}")

        return prs[:max_results]

    def _get_activity(self, repo_filter: str, max_results: int) -> list[dict]:
        """Get recent activity from repositories."""
        activity = []
        repos = [repo_filter] if repo_filter else self.repositories[:3]

        for repo in repos:
            try:
                events = self._github_request(
                    f"/repos/{repo}/events",
                    {"per_page": 30},
                )
                for event in events:
                    event_type = event.get("type", "")
                    payload = event.get("payload", {})
                    actor = event.get("actor", {}).get("login", "")

                    description = ""
                    if event_type == "PushEvent":
                        commits = payload.get("commits", [])
                        description = f"Pushed {len(commits)} commit(s)"
                    elif event_type == "IssuesEvent":
                        action = payload.get("action", "")
                        issue = payload.get("issue", {})
                        description = f"{action.title()} issue #{issue.get('number')}: {issue.get('title', '')}"
                    elif event_type == "PullRequestEvent":
                        action = payload.get("action", "")
                        pr = payload.get("pull_request", {})
                        description = f"{action.title()} PR #{pr.get('number')}: {pr.get('title', '')}"
                    elif event_type == "CreateEvent":
                        ref_type = payload.get("ref_type", "")
                        ref = payload.get("ref", "")
                        description = f"Created {ref_type}: {ref}"
                    else:
                        continue

                    activity.append(
                        {
                            "repo": repo,
                            "type": event_type,
                            "actor": actor,
                            "description": description,
                            "date": event.get("created_at", ""),
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to get activity from {repo}: {e}")

        return activity[:max_results]

    def _search_issues(
        self, query: str, repo_filter: str, max_results: int
    ) -> list[dict]:
        """Search issues and PRs."""
        # Build search query
        q = query
        if repo_filter:
            q += f" repo:{repo_filter}"
        elif self.repositories:
            repo_queries = " ".join(f"repo:{r}" for r in self.repositories[:5])
            q += f" ({repo_queries})"

        try:
            data = self._github_request(
                "/search/issues",
                {"q": q, "per_page": min(max_results, 30)},
            )
            items = data.get("items", [])
            return [
                {
                    "repo": item.get("repository_url", "").split("/")[-2:],
                    "number": item["number"],
                    "title": item["title"],
                    "type": "PR" if "pull_request" in item else "Issue",
                    "state": item["state"],
                    "author": item.get("user", {}).get("login", ""),
                    "updated": item.get("updated_at", ""),
                    "url": item.get("html_url", ""),
                }
                for item in items
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _format_issues(self, issues: list[dict], state: str) -> str:
        """Format issues for LLM context."""
        if not issues:
            return f"### GitHub Issues ({state})\nNo {state} issues found."

        lines = [f"### GitHub Issues ({state})", f"Found {len(issues)} issue(s):\n"]

        for issue in issues:
            labels = ", ".join(issue.get("labels", [])) or "None"
            lines.append(f"**[{issue['repo']}] #{issue['number']}: {issue['title']}**")
            lines.append(f"  State: {issue['state']} | Author: {issue['author']}")
            lines.append(f"  Labels: {labels}")
            lines.append(f"  Updated: {issue['updated']}")
            lines.append("")

        return "\n".join(lines)

    def _format_prs(self, prs: list[dict], state: str) -> str:
        """Format PRs for LLM context."""
        if not prs:
            return f"### GitHub Pull Requests ({state})\nNo {state} PRs found."

        lines = [f"### GitHub Pull Requests ({state})", f"Found {len(prs)} PR(s):\n"]

        for pr in prs:
            draft = " [DRAFT]" if pr.get("draft") else ""
            labels = ", ".join(pr.get("labels", [])) or "None"
            lines.append(f"**[{pr['repo']}] #{pr['number']}: {pr['title']}**{draft}")
            lines.append(f"  State: {pr['state']} | Author: {pr['author']}")
            lines.append(f"  Labels: {labels}")
            lines.append(f"  Updated: {pr['updated']}")
            lines.append("")

        return "\n".join(lines)

    def _format_activity(self, activity: list[dict]) -> str:
        """Format activity for LLM context."""
        if not activity:
            return "### GitHub Activity\nNo recent activity."

        lines = ["### GitHub Activity", f"Recent activity ({len(activity)} events):\n"]

        for event in activity:
            lines.append(f"**{event['actor']}** in {event['repo']}")
            lines.append(f"  {event['description']}")
            lines.append(f"  Date: {event['date']}")
            lines.append("")

        return "\n".join(lines)

    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results for LLM context."""
        if not results:
            return "### GitHub Search\nNo results found."

        lines = ["### GitHub Search Results", f"Found {len(results)} result(s):\n"]

        for item in results:
            repo = "/".join(item.get("repo", []))
            lines.append(
                f"**[{repo}] {item['type']} #{item['number']}: {item['title']}**"
            )
            lines.append(f"  State: {item['state']} | Author: {item['author']}")
            lines.append(f"  Updated: {item['updated']}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Smart Router
    # =========================================================================

    def analyze_query(self, query: str, params: dict) -> QueryAnalysis:
        """Analyze query to determine optimal routing."""
        query_lower = query.lower()
        action = params.get("action", "")

        # Real-time status queries -> Live only
        realtime_patterns = [
            "open issues",
            "open prs",
            "open pull requests",
            "my issues",
            "assigned to me",
            "recent activity",
            "what's new",
            "status",
            "current",
        ]
        if any(p in query_lower for p in realtime_patterns):
            return QueryAnalysis(
                routing=QueryRouting.LIVE_ONLY,
                live_params={**params, "action": action or "issues"},
                reason="Real-time status query - using live API only",
                max_live_results=self.live_max_results,
            )

        # Historical queries -> RAG only
        historical_patterns = [
            r"last year",
            r"20\d{2}",
            r"months ago",
            r"closed",
            r"merged",
            r"old",
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
        """Check if GitHub is accessible."""
        try:
            self._github_request("/user")
            return True
        except Exception:
            # Try unauthenticated
            try:
                self._github_request("/rate_limit")
                return True
            except Exception:
                return False

    def test_connection(self) -> tuple[bool, str]:
        """Test GitHub API connection."""
        results = []
        overall_success = True

        try:
            # Test authentication
            try:
                user = self._github_request("/user")
                results.append(f"Authenticated as: {user.get('login', 'Unknown')}")
            except Exception:
                results.append("Warning: Not authenticated (limited rate)")

            # Test repository access
            for repo in self.repositories[:3]:
                try:
                    self._github_request(f"/repos/{repo}")
                    results.append(f"Repo {repo}: Accessible")
                except Exception as e:
                    results.append(f"Repo {repo}: Error - {e}")
                    overall_success = False

            # Test document listing
            if self.supports_rag:
                try:
                    doc_count = 0
                    for _ in self.list_documents():
                        doc_count += 1
                        if doc_count >= 10:
                            break
                    results.append(f"Documents: Found items to index")
                except Exception as e:
                    results.append(f"Documents: Error - {e}")
                    overall_success = False

            # Test live query
            if self.supports_live:
                try:
                    live_result = self.fetch({"action": "activity", "max_results": 5})
                    if live_result.success:
                        count = len(live_result.data) if live_result.data else 0
                        results.append(f"Live: Found {count} recent events")
                    else:
                        results.append(f"Live: Error - {live_result.error}")
                        overall_success = False
                except Exception as e:
                    results.append(f"Live: Error - {e}")
                    overall_success = False

        except Exception as e:
            return False, f"Connection failed: {e}"

        return overall_success, "\n".join(results)

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
