"""Document parser for chunking Markdown, text, and other doc files."""

import os
import re


def parse_document(content, file_path):
    """Parse a document file into semantic chunks."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".md":
        return parse_markdown(content, file_path)
    elif ext == ".rst":
        return parse_rst(content, file_path)
    else:
        return parse_text(content, file_path)


def parse_markdown(content, file_path):
    """Split Markdown by headings into semantic sections."""
    chunks = []
    lines = content.split("\n")

    current_section = None
    current_lines = []
    section_start = 1

    for i, line in enumerate(lines, 1):
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            # Save previous section
            if current_lines:
                section_content = "\n".join(current_lines).strip()
                if section_content:
                    chunks.append({
                        "file_path": file_path,
                        "chunk_type": "doc_section",
                        "name": current_section or os.path.basename(file_path),
                        "content": section_content,
                        "start_line": section_start,
                        "end_line": i - 1,
                        "language": "markdown",
                        "metadata": {
                            "heading_level": len(heading_match.group(1)) if current_section else 0,
                        },
                    })

            current_section = heading_match.group(2).strip()
            current_lines = [line]
            section_start = i
        else:
            current_lines.append(line)

    # Save the last section
    if current_lines:
        section_content = "\n".join(current_lines).strip()
        if section_content:
            chunks.append({
                "file_path": file_path,
                "chunk_type": "doc_section",
                "name": current_section or os.path.basename(file_path),
                "content": section_content,
                "start_line": section_start,
                "end_line": len(lines),
                "language": "markdown",
                "metadata": {},
            })

    if not chunks:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "doc_section",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": "markdown",
            "metadata": {},
        })

    return chunks


def parse_text(content, file_path):
    """Split plain text into paragraph-based chunks with overlap."""
    chunks = []
    lines = content.split("\n")
    chunk_size = 30  # lines per chunk
    overlap = 5

    i = 0
    chunk_num = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        block = "\n".join(lines[i:end]).strip()
        if block:
            chunk_num += 1
            chunks.append({
                "file_path": file_path,
                "chunk_type": "doc_section",
                "name": f"section_{chunk_num}",
                "content": block,
                "start_line": i + 1,
                "end_line": end,
                "language": "text",
                "metadata": {"chunk_number": chunk_num},
            })
        i = end - overlap if end < len(lines) else end

    if not chunks:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "doc_section",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": "text",
            "metadata": {},
        })

    return chunks


def parse_rst(content, file_path):
    """Parse reStructuredText by section dividers."""
    chunks = []
    lines = content.split("\n")
    section_pattern = re.compile(r"^[=\-~^\"]+$")

    current_lines = []
    section_name = os.path.basename(file_path)
    section_start = 1

    for i, line in enumerate(lines, 1):
        if section_pattern.match(line.strip()) and current_lines:
            # The line before the underline is the heading
            if len(current_lines) >= 2:
                heading = current_lines[-1].strip()
                section_content = "\n".join(current_lines[:-1]).strip()
                if section_content:
                    chunks.append({
                        "file_path": file_path,
                        "chunk_type": "doc_section",
                        "name": section_name,
                        "content": section_content,
                        "start_line": section_start,
                        "end_line": i - 2,
                        "language": "rst",
                        "metadata": {},
                    })
                section_name = heading
                section_start = i - 1
                current_lines = [current_lines[-1], line]
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        section_content = "\n".join(current_lines).strip()
        if section_content:
            chunks.append({
                "file_path": file_path,
                "chunk_type": "doc_section",
                "name": section_name,
                "content": section_content,
                "start_line": section_start,
                "end_line": len(lines),
                "language": "rst",
                "metadata": {},
            })

    return chunks or [{
        "file_path": file_path,
        "chunk_type": "doc_section",
        "name": os.path.basename(file_path),
        "content": content,
        "start_line": 1,
        "end_line": len(lines),
        "language": "rst",
        "metadata": {},
    }]
