"""Cross-encoder re-ranking of retrieved chunks.

Primary: sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2).
Fallback: LLM-based relevance scoring via OpenRouter.
"""

import json
import os

import config

# ── Cross-Encoder setup ───────────────────────────────────────────────────
_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        # Prefer fine-tuned model if available, else use pretrained
        model_path = config.FINETUNED_RERANKER_MODEL
        if not os.path.isdir(model_path):
            model_path = config.RERANKER_MODEL

        _cross_encoder = CrossEncoder(model_path, max_length=512)
    return _cross_encoder


def rerank(query, chunks, top_k=5):
    """Re-rank retrieved chunks using cross-encoder relevance scoring.

    Each (query, chunk) pair is scored by the cross-encoder.  Falls back
    to LLM-based scoring if the cross-encoder is unavailable.

    Returns the top_k most relevant chunks sorted by score.
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        for c in chunks:
            c["rerank_score"] = 7.0
        return chunks

    # ── Primary: Cross-encoder ─────────────────────────────────────────
    try:
        return _rerank_cross_encoder(query, chunks, top_k)
    except Exception:
        pass  # fall through to LLM

    # ── Fallback: LLM-based scoring ────────────────────────────────────
    return _rerank_llm(query, chunks, top_k)


def _rerank_cross_encoder(query, chunks, top_k):
    """Score each chunk against the query with a cross-encoder model."""
    model = _get_cross_encoder()

    # Build (query, document) pairs — limit to top 15 for efficiency
    candidates = chunks[:15]
    pairs = []
    for chunk in candidates:
        text = chunk.get("content", "")[:1024]
        name = chunk.get("name", "") or chunk.get("metadata", {}).get("name", "")
        if name:
            text = f"{name}\n{text}"
        pairs.append((query, text))

    scores = model.predict(pairs)

    # Attach scores and sort
    for i, chunk in enumerate(candidates):
        # Normalise to 1-10 scale for consistency with LLM fallback
        raw = float(scores[i])
        chunk["rerank_score"] = max(1.0, min(10.0, raw * 10))

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


def _rerank_llm(query, chunks, top_k):
    """Fallback: re-rank using LLM relevance scoring via OpenRouter."""
    from core.openrouter import quick_llm

    chunks_text = ""
    for i, chunk in enumerate(chunks[:15]):
        name = chunk.get("metadata", {}).get("name", "") or chunk.get("name", "")
        fpath = chunk.get("metadata", {}).get("file_path", "") or chunk.get("file_path", "")
        preview = chunk.get("content", "")[:400]
        chunks_text += f"\n--- Chunk {i} [{fpath}:{name}] ---\n{preview}\n"

    prompt = (
        f"You are a code relevance scorer. Given a developer's query and retrieved "
        f"code/doc chunks, score each chunk's relevance from 1-10.\n\n"
        f'Query: "{query}"\n\nRetrieved Chunks:\n{chunks_text}\n\n'
        f"Return a JSON array of objects with \"index\" and \"score\" fields, "
        f"sorted by score descending.\nOnly return the JSON array, no other text.\n"
        f'Example: [{{"index": 2, "score": 9}}, {{"index": 0, "score": 7}}]'
    )

    result = quick_llm(
        prompt,
        system_prompt="You are a precise code relevance scorer. Return only valid JSON.",
    )

    if result.get("error"):
        for c in chunks[:top_k]:
            c["rerank_score"] = 5.0
        return chunks[:top_k]

    try:
        content = result["content"].strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            scores = json.loads(content[start:end])
        else:
            scores = []
    except (json.JSONDecodeError, ValueError):
        scores = []

    if not scores:
        for c in chunks[:top_k]:
            c["rerank_score"] = 5.0
        return chunks[:top_k]

    score_map = {
        s["index"]: s["score"]
        for s in scores
        if "index" in s and "score" in s
    }
    scored_chunks = []
    for i, chunk in enumerate(chunks[:15]):
        chunk["rerank_score"] = score_map.get(i, 3.0)
        scored_chunks.append(chunk)

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_chunks[:top_k]
