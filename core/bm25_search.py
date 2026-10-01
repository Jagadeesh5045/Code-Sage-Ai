"""BM25 sparse keyword search for code chunks."""

import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Simple tokenizer for code and natural language."""
    text = text.lower()
    # Split on non-alphanumeric, keep underscores (common in code)
    tokens = re.findall(r"[a-z_][a-z0-9_]*", text)
    # Also split camelCase
    expanded = []
    for token in tokens:
        parts = re.findall(r"[a-z]+|[0-9]+", token)
        expanded.extend(parts)
        if len(parts) > 1:
            expanded.append(token)  # keep original compound token too
    return expanded


class BM25Index:
    """BM25 index for sparse keyword search over chunks."""

    def __init__(self):
        self.chunks = []
        self.index = None

    def build(self, chunks):
        """Build BM25 index from a list of chunk dicts."""
        self.chunks = chunks
        corpus = [tokenize(c["content"]) for c in chunks]
        if corpus:
            self.index = BM25Okapi(corpus)

    def search(self, query, top_k=20):
        """Search the BM25 index with a text query."""
        if not self.index or not self.chunks:
            return []

        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)

        # Pair scores with chunks and sort
        scored = list(zip(scores, self.chunks))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            if score > 0:
                results.append({
                    **chunk,
                    "bm25_score": float(score),
                })

        return results


def hybrid_search(dense_results, sparse_results, dense_weight=0.7, sparse_weight=0.3):
    """Combine dense (vector) and sparse (BM25) search results using RRF."""
    k = 60  # RRF constant
    scores = {}

    for rank, chunk in enumerate(dense_results):
        key = chunk.get("chromadb_id") or chunk.get("content", "")[:100]
        rrf = dense_weight * (1 / (k + rank + 1))
        if key in scores:
            scores[key]["score"] += rrf
        else:
            scores[key] = {"chunk": chunk, "score": rrf}

    for rank, chunk in enumerate(sparse_results):
        key = chunk.get("chromadb_id") or chunk.get("content", "")[:100]
        rrf = sparse_weight * (1 / (k + rank + 1))
        if key in scores:
            scores[key]["score"] += rrf
        else:
            scores[key] = {"chunk": chunk, "score": rrf}

    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [{"hybrid_score": item["score"], **item["chunk"]} for item in merged]
