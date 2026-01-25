"""
Context building utilities for injecting retrieved content into prompts.

Used by Smart Cache and Smart Augmentor to format and inject
context into LLM requests.
"""


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate based on character count.

    This is an approximation - actual token counts vary by model and tokenizer.
    Uses ~4 characters per token as a rough average.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated text with notice if truncated
    """
    if not text:
        return ""

    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to truncate at a sentence or paragraph boundary
    last_period = truncated.rfind(". ")
    last_newline = truncated.rfind("\n")
    cut_point = max(last_period, last_newline)

    if cut_point > max_chars * 0.8:  # Only use if we keep at least 80%
        truncated = truncated[: cut_point + 1]

    return truncated + "\n\n[Content truncated due to length limit]"


def format_context_header(source: str) -> str:
    """
    Format a header for injected context.

    Args:
        source: Source description (e.g., "Web Search", "Documents")

    Returns:
        Formatted header string
    """
    return f"## Context from {source}\n\n"


def format_context_footer() -> str:
    """
    Format a footer for injected context.

    Returns:
        Formatted footer string
    """
    return "\n---\n"


def inject_context_to_system(
    system: str | None,
    context: str,
    position: str = "prepend",
) -> str:
    """
    Inject context into system prompt.

    Args:
        system: Original system prompt (may be None)
        context: Context to inject
        position: Where to inject - "prepend" or "append"

    Returns:
        Modified system prompt with context
    """
    if not context:
        return system or ""

    if position == "prepend":
        if system:
            return f"{context}\n\n---\n\n{system}"
        return context
    else:  # append
        if system:
            return f"{system}\n\n---\n\n{context}"
        return context


def format_as_context_message(context: str, role: str = "user") -> dict:
    """
    Format context as a message for injection into conversation.

    Args:
        context: Context content
        role: Message role (usually "user" or "system")

    Returns:
        Message dict with role and content
    """
    return {
        "role": role,
        "content": f"[Additional Context]\n\n{context}\n\n[End Context]",
    }


def format_search_results(
    results: list[dict],
    include_urls: bool = True,
    include_snippets: bool = True,
    max_results: int | None = None,
) -> str:
    """
    Format search results for context injection.

    Args:
        results: List of search result dicts with keys: title, url, content
        include_urls: Whether to include URLs
        include_snippets: Whether to include content snippets
        max_results: Maximum results to include (None for all)

    Returns:
        Formatted search results string
    """
    if not results:
        return ""

    if max_results:
        results = results[:max_results]

    lines = [format_context_header("Web Search")]

    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")

        lines.append(f"**{i}. {title}**")
        if include_urls and url:
            lines.append(f"   URL: {url}")
        if include_snippets and content:
            # Truncate long snippets
            snippet = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines)


def format_scraped_content(
    pages: list[dict],
    max_chars_per_page: int = 2000,
) -> str:
    """
    Format scraped web page content for context injection.

    Args:
        pages: List of page dicts with keys: url, title, content, success
        max_chars_per_page: Maximum characters per page

    Returns:
        Formatted page content string
    """
    if not pages:
        return ""

    lines = [format_context_header("Web Pages")]

    for page in pages:
        if not page.get("success", False):
            continue

        title = page.get("title") or page.get("url", "Unknown")
        url = page.get("url", "")
        content = page.get("content", "")

        lines.append(f"### {title}")
        if url:
            lines.append(f"Source: {url}")
        lines.append("")

        if content:
            if len(content) > max_chars_per_page:
                content = content[:max_chars_per_page] + "\n[Content truncated]"
            lines.append(content)
        lines.append("")

    return "\n".join(lines)


def format_document_chunks(
    chunks: list[dict],
    include_sources: bool = True,
) -> str:
    """
    Format document chunks for RAG context injection.

    Args:
        chunks: List of chunk dicts with keys: content, source_path, chunk_index
        include_sources: Whether to include source file paths

    Returns:
        Formatted chunks string
    """
    if not chunks:
        return ""

    lines = [format_context_header("Documents")]

    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        source = chunk.get("source_path", "")
        chunk_idx = chunk.get("chunk_index", 0)

        if include_sources and source:
            lines.append(f"**[{i}] From: {source} (chunk {chunk_idx})**")
        else:
            lines.append(f"**[{i}]**")

        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def merge_contexts(
    contexts: list[str],
    max_tokens: int,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Merge multiple context strings within a token limit.

    Contexts are added in order until the limit is reached.

    Args:
        contexts: List of context strings to merge
        max_tokens: Maximum total tokens
        separator: String to separate contexts

    Returns:
        Merged context string
    """
    if not contexts:
        return ""

    merged = []
    total_tokens = 0

    for ctx in contexts:
        if not ctx:
            continue

        ctx_tokens = estimate_tokens(ctx)
        sep_tokens = estimate_tokens(separator) if merged else 0

        if total_tokens + ctx_tokens + sep_tokens <= max_tokens:
            merged.append(ctx)
            total_tokens += ctx_tokens + sep_tokens
        else:
            # Try to fit a truncated version
            remaining = max_tokens - total_tokens - sep_tokens
            if remaining > 100:  # Only include if we can fit something meaningful
                truncated = truncate_to_tokens(ctx, remaining)
                merged.append(truncated)
            break

    return separator.join(merged)
