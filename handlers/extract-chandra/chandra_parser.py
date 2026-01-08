"""
Parse Chandra OCR Markdown output into structured format.

Chandra outputs documents as structured Markdown with:
- Headers (H1, H2, etc.) for sections
- Bold key-value pairs (e.g., **Account Number:** ****4521)
- Tables for structured data (transactions, etc.)
- Lists for summary items
"""

import re
from typing import Any


def parse_chandra_output(markdown_output: str) -> dict:
    """
    Parse Chandra Markdown output into structured format.

    Args:
        markdown_output: Raw Markdown string from Chandra

    Returns:
        Dict with sections, key_value_pairs, tables, lists, and raw_text
    """
    if not markdown_output or not markdown_output.strip():
        return {
            "sections": [],
            "key_value_pairs": [],
            "tables": [],
            "lists": [],
            "raw_text": ""
        }

    return {
        "sections": parse_markdown_sections(markdown_output),
        "key_value_pairs": parse_key_value_pairs(markdown_output),
        "tables": parse_markdown_tables(markdown_output),
        "lists": parse_lists(markdown_output),
        "raw_text": markdown_output
    }


def parse_markdown_sections(markdown: str) -> list[dict]:
    """
    Extract headers and their content from Markdown.

    Args:
        markdown: Raw Markdown string

    Returns:
        List of dicts with level, title, and content keys
    """
    sections = []
    lines = markdown.split('\n')
    current_section = None

    for line in lines:
        # Match headers (# Header, ## Header, etc.)
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if header_match:
            # Save previous section
            if current_section:
                sections.append(current_section)

            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_section = {
                "level": level,
                "title": title,
                "content": ""
            }
        elif current_section:
            # Add content to current section
            if current_section["content"]:
                current_section["content"] += "\n" + line
            else:
                current_section["content"] = line

    # Don't forget the last section
    if current_section:
        sections.append(current_section)

    return sections


def parse_key_value_pairs(markdown: str) -> list[dict]:
    """
    Extract key-value pairs from Markdown.

    Matches patterns like:
    - **Key:** Value
    - **Key**: Value
    - Key: Value (at start of line)

    Args:
        markdown: Raw Markdown string

    Returns:
        List of dicts with key and value
    """
    pairs = []

    # Pattern for **Key:** Value or **Key**: Value
    bold_pattern = r'\*\*([^*]+)\*\*:\s*(.+?)(?=\n|$)'
    for match in re.finditer(bold_pattern, markdown):
        key = match.group(1).strip()
        value = match.group(2).strip()
        pairs.append({"key": key, "value": value})

    # Pattern for Key: Value at start of line (not in table)
    line_pattern = r'^([A-Za-z][A-Za-z\s]+):\s+(.+?)$'
    for line in markdown.split('\n'):
        line = line.strip()
        # Skip table rows and already matched bold patterns
        if line.startswith('|') or '**' in line:
            continue
        match = re.match(line_pattern, line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            # Avoid duplicates
            if not any(p["key"].lower() == key.lower() for p in pairs):
                pairs.append({"key": key, "value": value})

    return pairs


def parse_markdown_tables(markdown: str) -> list[dict]:
    """
    Extract tables from Markdown.

    Args:
        markdown: Raw Markdown string

    Returns:
        List of dicts with headers and rows keys
    """
    tables = []
    lines = markdown.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a table header row
        if line.startswith('|') and line.endswith('|'):
            # Parse header row
            headers = [cell.strip() for cell in line.split('|')[1:-1]]

            # Check for separator row
            if i + 1 < len(lines):
                separator = lines[i + 1].strip()
                if re.match(r'^\|[\s\-:|]+\|$', separator):
                    # This is a valid table
                    rows = []
                    i += 2  # Skip header and separator

                    # Parse data rows
                    while i < len(lines):
                        row_line = lines[i].strip()
                        if row_line.startswith('|') and row_line.endswith('|'):
                            cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                            if len(cells) == len(headers):
                                rows.append(cells)
                            i += 1
                        else:
                            break

                    tables.append({
                        "headers": headers,
                        "rows": rows
                    })
                    continue

        i += 1

    return tables


def parse_lists(markdown: str) -> list[dict]:
    """
    Extract bullet and numbered lists from Markdown.

    Args:
        markdown: Raw Markdown string

    Returns:
        List of dicts with items key
    """
    lists = []
    lines = markdown.split('\n')
    current_list = None

    for line in lines:
        stripped = line.strip()

        # Match bullet points (-, *, +)
        bullet_match = re.match(r'^[-*+]\s+(.+)$', stripped)
        # Match numbered lists (1., 2., etc.)
        numbered_match = re.match(r'^\d+\.\s+(.+)$', stripped)

        if bullet_match or numbered_match:
            item = (bullet_match or numbered_match).group(1)
            if current_list is None:
                current_list = {"items": []}
            current_list["items"].append(item)
        else:
            # End of list
            if current_list and current_list["items"]:
                lists.append(current_list)
                current_list = None

    # Don't forget the last list
    if current_list and current_list["items"]:
        lists.append(current_list)

    return lists


def combine_page_results(page_results: list[dict]) -> dict:
    """
    Combine parsed results from multiple pages into a single structure.

    Args:
        page_results: List of parsed page dicts

    Returns:
        Combined parsed structure
    """
    combined = {
        "sections": [],
        "key_value_pairs": [],
        "tables": [],
        "lists": [],
        "raw_text": "",
        "page_count": len(page_results)
    }

    for page in page_results:
        combined["sections"].extend(page.get("sections", []))
        combined["key_value_pairs"].extend(page.get("key_value_pairs", []))
        combined["tables"].extend(page.get("tables", []))
        combined["lists"].extend(page.get("lists", []))

        if combined["raw_text"]:
            combined["raw_text"] += "\n\n---\n\n"
        combined["raw_text"] += page.get("raw_text", "")

    return combined
