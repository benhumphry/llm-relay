"""
Notion Markdown Converter.

Bidirectional conversion between Markdown and Notion API blocks using mistune.
"""

import re

import mistune


class NotionBlockBuilder:
    """
    Helper class for building Notion API block objects.

    Handles:
    - Headings (h1, h2, h3)
    - Paragraphs with rich text (bold, italic, code, links)
    - Bullet and numbered lists (with nesting)
    - Code blocks with language
    - Blockquotes
    - Horizontal rules (dividers)
    - Tables
    - Images
    """

    def _rich_text(self, text: str, annotations: dict | None = None) -> dict:
        """Create a Notion rich_text object."""
        rt = {
            "type": "text",
            "text": {"content": text},
        }
        if annotations:
            rt["annotations"] = annotations
        return rt

    def _rich_text_link(
        self, text: str, url: str, annotations: dict | None = None
    ) -> dict:
        """Create a Notion rich_text object with a link."""
        rt = {
            "type": "text",
            "text": {"content": text, "link": {"url": url}},
        }
        if annotations:
            rt["annotations"] = annotations
        return rt

    def _parse_inline(self, text: str) -> list[dict]:
        """
        Parse inline markdown and return Notion rich_text array.

        Handles: **bold**, *italic*, `code`, [links](url), ~~strikethrough~~
        """
        if not text:
            return []

        rich_text = []

        # Combined pattern for inline elements
        # Order matters: bold before italic (** before *)
        pattern = re.compile(
            r"(\*\*\*(.+?)\*\*\*)"  # Bold + italic
            r"|(\*\*(.+?)\*\*)"  # Bold
            r"|(\*(.+?)\*)"  # Italic
            r"|(__(.+?)__)"  # Bold (underscore)
            r"|(_(.+?)_)"  # Italic (underscore)
            r"|(~~(.+?)~~)"  # Strikethrough
            r"|(`(.+?)`)"  # Inline code
            r"|(\[([^\]]+)\]\(([^)]+)\))"  # Links
        )

        last_end = 0
        for match in pattern.finditer(text):
            # Add text before this match
            if match.start() > last_end:
                plain_text = text[last_end : match.start()]
                if plain_text:
                    rich_text.append(self._rich_text(plain_text))

            # Determine which group matched
            if match.group(2):  # Bold + italic ***text***
                rich_text.append(
                    self._rich_text(match.group(2), {"bold": True, "italic": True})
                )
            elif match.group(4):  # Bold **text**
                rich_text.append(self._rich_text(match.group(4), {"bold": True}))
            elif match.group(6):  # Italic *text*
                rich_text.append(self._rich_text(match.group(6), {"italic": True}))
            elif match.group(8):  # Bold __text__
                rich_text.append(self._rich_text(match.group(8), {"bold": True}))
            elif match.group(10):  # Italic _text_
                rich_text.append(self._rich_text(match.group(10), {"italic": True}))
            elif match.group(12):  # Strikethrough ~~text~~
                rich_text.append(
                    self._rich_text(match.group(12), {"strikethrough": True})
                )
            elif match.group(14):  # Inline code `text`
                rich_text.append(self._rich_text(match.group(14), {"code": True}))
            elif match.group(16):  # Link [text](url)
                link_text = match.group(16)
                link_url = match.group(17)
                rich_text.append(self._rich_text_link(link_text, link_url))

            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining:
                rich_text.append(self._rich_text(remaining))

        return rich_text if rich_text else [self._rich_text(text)]

    def paragraph(self, text: str) -> dict:
        """Build a paragraph block."""
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": self._parse_inline(text)},
        }

    def heading(self, text: str, level: int) -> dict:
        """Build a heading block (h1, h2, h3 - Notion only supports up to h3)."""
        level = min(level, 3)  # Notion max is h3
        heading_type = f"heading_{level}"
        return {
            "object": "block",
            "type": heading_type,
            heading_type: {"rich_text": self._parse_inline(text)},
        }

    def code_block(self, code: str, language: str | None = None) -> dict:
        """Build a code block."""
        lang = language.strip() if language else "plain text"
        # Map common language aliases
        lang_map = {
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "rb": "ruby",
            "yml": "yaml",
            "sh": "shell",
            "bash": "shell",
        }
        lang = lang_map.get(lang.lower(), lang.lower())

        return {
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [self._rich_text(code.rstrip())],
                "language": lang,
            },
        }

    def quote(self, text: str) -> dict:
        """Build a blockquote."""
        return {
            "object": "block",
            "type": "quote",
            "quote": {"rich_text": self._parse_inline(text)},
        }

    def divider(self) -> dict:
        """Build a horizontal rule (divider)."""
        return {
            "object": "block",
            "type": "divider",
            "divider": {},
        }


def markdown_to_notion_blocks(markdown: str) -> list[dict]:
    """
    Convert Markdown text to Notion API block objects.

    Args:
        markdown: Markdown formatted text

    Returns:
        List of Notion block objects ready for the API
    """
    if not markdown or not markdown.strip():
        return []

    builder = NotionBlockBuilder()

    # Use mistune's AST mode to get structured data
    # Enable table plugin for table support
    md_ast = mistune.create_markdown(renderer="ast", plugins=["table"])
    ast = md_ast(markdown)

    # Convert AST to Notion blocks
    return _ast_to_notion_blocks(ast, builder)


def _ast_to_notion_blocks(ast: list, builder: NotionBlockBuilder) -> list[dict]:
    """Convert mistune AST to Notion blocks."""
    blocks = []

    for node in ast:
        node_type = node.get("type")
        children = node.get("children", [])

        if node_type == "paragraph":
            text = _children_to_text(children)
            if text.strip():
                blocks.append(builder.paragraph(text))

        elif node_type == "heading":
            level = node.get("attrs", {}).get("level", 1)
            text = _children_to_text(children)
            blocks.append(builder.heading(text, level))

        elif node_type == "block_code":
            code = node.get("raw", "")
            info = node.get("attrs", {}).get("info", "")
            blocks.append(builder.code_block(code, info))

        elif node_type == "block_quote":
            # Recursively process blockquote children
            inner_text = []
            for child in children:
                if child.get("type") == "paragraph":
                    inner_text.append(_children_to_text(child.get("children", [])))
            blocks.append(builder.quote("\n".join(inner_text)))

        elif node_type == "thematic_break":
            blocks.append(builder.divider())

        elif node_type == "list":
            ordered = node.get("attrs", {}).get("ordered", False)
            list_blocks = _process_list_items(children, builder, ordered)
            blocks.extend(list_blocks)

        elif node_type == "table":
            table_block = _process_table(node, builder)
            if table_block:
                blocks.append(table_block)

        elif node_type == "blank_line":
            pass  # Skip blank lines

    return blocks


def _process_table(node: dict, builder: NotionBlockBuilder) -> dict | None:
    """Convert a table AST node to a Notion table block."""
    children = node.get("children", [])
    if not children:
        return None

    rows = []
    has_header = False

    for child in children:
        child_type = child.get("type")

        if child_type == "table_head":
            has_header = True
            # table_head contains cells directly (not wrapped in table_row)
            head_cells = child.get("children", [])
            if head_cells:
                cells = _process_table_cells(head_cells, builder)
                rows.append(
                    {
                        "object": "block",
                        "type": "table_row",
                        "table_row": {"cells": cells},
                    }
                )

        elif child_type == "table_body":
            for row in child.get("children", []):
                if row.get("type") == "table_row":
                    cells = _process_table_cells(row.get("children", []), builder)
                    rows.append(
                        {
                            "object": "block",
                            "type": "table_row",
                            "table_row": {"cells": cells},
                        }
                    )

    if not rows:
        return None

    # Determine column count from first row
    col_count = len(rows[0]["table_row"]["cells"]) if rows else 1

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": col_count,
            "has_column_header": has_header,
            "has_row_header": False,
            "children": rows,
        },
    }


def _process_table_cells(
    cell_nodes: list, builder: NotionBlockBuilder
) -> list[list[dict]]:
    """Process table cells and return as rich_text arrays."""
    cells = []
    for cell in cell_nodes:
        if cell.get("type") == "table_cell":
            text = _children_to_text(cell.get("children", []))
            cells.append(builder._parse_inline(text))
    return cells


def _process_list_items(
    items: list, builder: NotionBlockBuilder, ordered: bool = False
) -> list[dict]:
    """Process list items, handling nesting."""
    blocks = []

    for item in items:
        if item.get("type") != "list_item":
            continue

        children = item.get("children", [])

        # Get the text content of this item
        text_parts = []
        nested_list = None

        for child in children:
            if child.get("type") == "paragraph":
                text_parts.append(_children_to_text(child.get("children", [])))
            elif child.get("type") == "list":
                nested_list = child
            elif child.get("type") == "block_text":
                text_parts.append(_children_to_text(child.get("children", [])))

        text = " ".join(text_parts).strip()

        # Check for checkbox
        checkbox_match = re.match(r"^\[([ xX])\]\s*", text)
        if checkbox_match:
            checked = checkbox_match.group(1).lower() == "x"
            text = text[checkbox_match.end() :]
            block = {
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": builder._parse_inline(text),
                    "checked": checked,
                },
            }
        elif ordered:
            block = {
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": builder._parse_inline(text)},
            }
        else:
            block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": builder._parse_inline(text)},
            }

        # Handle nested lists by adding as children
        if nested_list:
            nested_ordered = nested_list.get("attrs", {}).get("ordered", False)
            nested_blocks = _process_list_items(
                nested_list.get("children", []), builder, nested_ordered
            )
            # Notion supports nested blocks via "children" key
            block_type = block["type"]
            block[block_type]["children"] = nested_blocks

        blocks.append(block)

    return blocks


def _children_to_text(children: list) -> str:
    """Convert AST children to text with inline markdown markers."""
    parts = []

    for child in children:
        child_type = child.get("type")

        if child_type == "text":
            parts.append(child.get("raw", ""))
        elif child_type == "strong":
            inner = _children_to_text(child.get("children", []))
            parts.append(f"**{inner}**")
        elif child_type == "emphasis":
            inner = _children_to_text(child.get("children", []))
            parts.append(f"*{inner}*")
        elif child_type == "codespan":
            parts.append(f"`{child.get('raw', '')}`")
        elif child_type == "link":
            inner = _children_to_text(child.get("children", []))
            url = child.get("attrs", {}).get("url", "")
            parts.append(f"[{inner}]({url})")
        elif child_type == "strikethrough":
            inner = _children_to_text(child.get("children", []))
            parts.append(f"~~{inner}~~")
        elif child_type == "softbreak":
            parts.append(" ")
        elif child_type == "linebreak":
            parts.append("\n")
        elif child_type == "image":
            alt = child.get("attrs", {}).get("alt", "")
            url = child.get("attrs", {}).get("url", "")
            parts.append(f"![{alt}]({url})")

    return "".join(parts)


def notion_blocks_to_markdown(blocks: list[dict]) -> str:
    """
    Convert Notion API blocks to Markdown text.

    Args:
        blocks: List of Notion block objects

    Returns:
        Markdown formatted text
    """
    lines = []

    for block in blocks:
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        if block_type == "paragraph":
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            if text:
                lines.append(text)
                lines.append("")

        elif block_type.startswith("heading_"):
            level = int(block_type[-1])
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            lines.append(f"{'#' * level} {text}")
            lines.append("")

        elif block_type == "bulleted_list_item":
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            lines.append(f"- {text}")
            # Handle nested children
            if "children" in block_data:
                for child in block_data["children"]:
                    child_md = notion_blocks_to_markdown([child])
                    for line in child_md.split("\n"):
                        if line.strip():
                            lines.append(f"  {line}")

        elif block_type == "numbered_list_item":
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            lines.append(f"1. {text}")

        elif block_type == "to_do":
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            checked = block_data.get("checked", False)
            checkbox = "[x]" if checked else "[ ]"
            lines.append(f"- {checkbox} {text}")

        elif block_type == "code":
            code = _rich_text_to_markdown(block_data.get("rich_text", []))
            language = block_data.get("language", "")
            lines.append(f"```{language}")
            lines.append(code)
            lines.append("```")
            lines.append("")

        elif block_type == "quote":
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            for line in text.split("\n"):
                lines.append(f"> {line}")
            lines.append("")

        elif block_type == "divider":
            lines.append("---")
            lines.append("")

        elif block_type == "callout":
            icon = block_data.get("icon", {}).get("emoji", "")
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            lines.append(f"> {icon} {text}")
            lines.append("")

        elif block_type == "image":
            image_data = block_data
            if image_data.get("type") == "external":
                url = image_data.get("external", {}).get("url", "")
            else:
                url = image_data.get("file", {}).get("url", "")
            caption = _rich_text_to_markdown(image_data.get("caption", []))
            lines.append(f"![{caption}]({url})")
            lines.append("")

        elif block_type == "toggle":
            text = _rich_text_to_markdown(block_data.get("rich_text", []))
            lines.append(f"<details><summary>{text}</summary>")
            if "children" in block_data:
                child_md = notion_blocks_to_markdown(block_data["children"])
                lines.append(child_md)
            lines.append("</details>")
            lines.append("")

        elif block_type == "table":
            table_md = _table_to_markdown(block_data)
            if table_md:
                lines.append(table_md)
                lines.append("")

    return "\n".join(lines).strip()


def _table_to_markdown(table_data: dict) -> str:
    """Convert a Notion table block to markdown."""
    children = table_data.get("children", [])
    if not children:
        return ""

    rows = []
    for row_block in children:
        if row_block.get("type") == "table_row":
            cells = row_block.get("table_row", {}).get("cells", [])
            row_text = [_rich_text_to_markdown(cell) for cell in cells]
            rows.append(row_text)

    if not rows:
        return ""

    # Build markdown table
    lines = []
    col_count = len(rows[0]) if rows else 0

    # Header row
    lines.append("| " + " | ".join(rows[0]) + " |")
    # Separator
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    # Data rows
    for row in rows[1:]:
        # Pad row if needed
        while len(row) < col_count:
            row.append("")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _rich_text_to_markdown(rich_text: list[dict]) -> str:
    """Convert Notion rich_text array to markdown string."""
    parts = []

    for rt in rich_text:
        text = rt.get("text", {}).get("content", "") or rt.get("plain_text", "")
        annotations = rt.get("annotations", {})
        link = rt.get("text", {}).get("link")

        # Apply annotations
        if annotations.get("code"):
            text = f"`{text}`"
        if annotations.get("bold"):
            text = f"**{text}**"
        if annotations.get("italic"):
            text = f"*{text}*"
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
        if link:
            text = f"[{text}]({link.get('url', '')})"

        parts.append(text)

    return "".join(parts)
