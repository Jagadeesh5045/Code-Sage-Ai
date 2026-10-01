"""Retrieval agent - executes search across vector store and BM25."""

from core import embedder, vector_store, bm25_search
from core.database import get_chunks_for_project
import config


def execute_retrieval(plan, project_id, strategy="hybrid"):
    """Execute retrieval based on the query plan.

    Args:
        plan: query plan from query_planner
        project_id: project to search within
        strategy: 'dense', 'sparse', or 'hybrid'

    Returns:
        dict with:
            - chunks: list of retrieved chunks
            - strategy_used: which strategy was actually used
            - stats: retrieval statistics
    """
    all_sub_queries = plan.get("sub_queries", [])
    if not all_sub_queries:
        return {"chunks": [], "strategy_used": strategy, "stats": {}}

    all_dense_results = []
    all_sparse_results = []

    # Load chunks from SQLite for BM25
    db_chunks = get_chunks_for_project(project_id)
    chunk_dicts = [dict(row) for row in db_chunks] if db_chunks else []

    # Build BM25 index
    bm25_idx = bm25_search.BM25Index()
    if chunk_dicts:
        bm25_idx.build(chunk_dicts)

    for sub_query in all_sub_queries:
        # Dense (vector) retrieval
        if strategy in ("dense", "hybrid"):
            query_emb = embedder.get_query_embedding(sub_query)
            dense_results = vector_store.search(
                query_emb, project_id, top_k=config.TOP_K_RETRIEVAL
            )
            all_dense_results.extend(dense_results)

        # Sparse (BM25) retrieval
        if strategy in ("sparse", "hybrid"):
            sparse_results = bm25_idx.search(sub_query, top_k=config.TOP_K_RETRIEVAL)
            all_sparse_results.extend(sparse_results)

    # Merge and deduplicate
    if strategy == "hybrid" and all_dense_results and all_sparse_results:
        merged = bm25_search.hybrid_search(
            all_dense_results, all_sparse_results,
            dense_weight=config.DENSE_WEIGHT,
            sparse_weight=config.BM25_WEIGHT,
        )
    elif all_dense_results:
        merged = all_dense_results
    elif all_sparse_results:
        merged = all_sparse_results
    else:
        merged = []

    # Deduplicate by content
    seen = set()
    unique = []
    for chunk in merged:
        content_key = chunk.get("content", "")[:200]
        if content_key not in seen:
            seen.add(content_key)
            unique.append(chunk)

    stats = {
        "total_sub_queries": len(all_sub_queries),
        "dense_results": len(all_dense_results),
        "sparse_results": len(all_sparse_results),
        "merged_results": len(merged),
        "unique_results": len(unique),
    }

    return {
        "chunks": unique[:config.TOP_K_RETRIEVAL],
        "strategy_used": strategy,
        "stats": stats,
    }
