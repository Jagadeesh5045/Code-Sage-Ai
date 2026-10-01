"""Citation engine - maps LLM response claims back to source code/doc chunks."""

import json
from core.openrouter import quick_llm


def extract_citations(response_text, used_chunks):
    """Map claims in the response to source chunks.

    Returns a list of citation objects linking response segments
    to their source files, line numbers, and content.
    """
    if not used_chunks:
        return []

    # Build chunk reference list
    chunk_refs = []
    for i, chunk in enumerate(used_chunks):
        meta = chunk.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        chunk_refs.append({
            "index": i,
            "file_path": meta.get("file_path", "") or chunk.get("file_path", ""),
            "name": meta.get("name", "") or chunk.get("name", ""),
            "start_line": meta.get("start_line", 0) or chunk.get("start_line", 0),
            "end_line": meta.get("end_line", 0) or chunk.get("end_line", 0),
            "chunk_type": meta.get("chunk_type", "") or chunk.get("chunk_type", ""),
            "preview": chunk.get("content", "")[:200],
        })

    refs_text = json.dumps(chunk_refs, indent=2)

    prompt = f"""Analyze the following AI-generated response and map each key claim or code reference back to the source chunks provided.

Response:
\"\"\"{response_text[:2000]}\"\"\"

Available Source Chunks:
{refs_text}

For each identifiable claim or code reference in the response, output a JSON array of citation objects:
[
  {{
    "claim": "brief description of the claim",
    "chunk_index": 0,
    "file_path": "path/to/file.py",
    "line_range": "10-25",
    "confidence": 0.9
  }}
]

Only return the JSON array. If a claim cannot be traced to any chunk, omit it."""

    result = quick_llm(prompt, system_prompt="You are a precise citation mapper. Return only valid JSON.")

    if result.get("error"):
        # Fallback: create basic citations from used chunks
        return _fallback_citations(used_chunks)

    try:
        content = result["content"].strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            citations = json.loads(content[start:end])
        else:
            citations = _fallback_citations(used_chunks)
    except (json.JSONDecodeError, ValueError):
        citations = _fallback_citations(used_chunks)

    # Enrich citations with full chunk data
    enriched = []
    for cit in citations:
        idx = cit.get("chunk_index", -1)
        if 0 <= idx < len(used_chunks):
            chunk = used_chunks[idx]
            meta = chunk.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            cit["file_path"] = meta.get("file_path", "") or chunk.get("file_path", "")
            cit["name"] = meta.get("name", "") or chunk.get("name", "")
            cit["chunk_type"] = meta.get("chunk_type", "") or chunk.get("chunk_type", "")
        enriched.append(cit)

    return enriched


def _fallback_citations(used_chunks):
    """Create basic citations when LLM-based citation fails."""
    citations = []
    for i, chunk in enumerate(used_chunks[:5]):
        meta = chunk.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        fpath = meta.get("file_path", "") or chunk.get("file_path", "")
        name = meta.get("name", "") or chunk.get("name", "")
        start = meta.get("start_line", 0) or chunk.get("start_line", 0)
        end = meta.get("end_line", 0) or chunk.get("end_line", 0)

        citations.append({
            "claim": f"Reference to {name or fpath}",
            "chunk_index": i,
            "file_path": fpath,
            "name": name,
            "line_range": f"{start}-{end}" if start else "",
            "chunk_type": meta.get("chunk_type", ""),
            "confidence": 0.7,
        })
    return citations


def format_citations_html(citations):
    """Format citations as HTML for display in the response."""
    if not citations:
        return ""

    html = '<div class="citations-panel"><h4>Sources</h4><ul>'
    for cit in citations:
        fpath = cit.get("file_path", "unknown")
        line_range = cit.get("line_range", "")
        name = cit.get("name", "")
        confidence = cit.get("confidence", 0)
        conf_pct = int(confidence * 100) if isinstance(confidence, (int, float)) else 70

        label = f"{fpath}"
        if line_range:
            label += f":{line_range}"
        if name:
            label += f" ({name})"

        html += f'<li><span class="citation-badge">{conf_pct}%</span> <code>{label}</code>'
        if cit.get("claim"):
            html += f' — {cit["claim"]}'
        html += "</li>"

    html += "</ul></div>"
    return html
