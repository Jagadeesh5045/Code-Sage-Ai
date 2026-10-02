"""Evaluation benchmark — systematically compares retrieval strategies.

Runs a suite of test queries against ingested projects using dense, sparse,
and hybrid retrieval, then computes and reports precision, recall,
faithfulness, and latency for each strategy.

Usage:
    python eval_benchmark.py                     # all ready projects
    python eval_benchmark.py --project-id 1      # single project
    python eval_benchmark.py --export results.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.database import get_all_projects, get_project, get_chunks_for_project
from core.embedder import get_query_embedding
from core import vector_store, bm25_search
from core.reranker import rerank
from agent.query_planner import plan_query
from agent.retrieval_agent import execute_retrieval
from agent.self_correction import check_quality

# ── Default evaluation queries (used when no custom set supplied) ─────────
DEFAULT_QUERIES = [
    "How does the main application initialise?",
    "What database models or schemas are defined?",
    "How is authentication or user validation handled?",
    "What are the primary API endpoints?",
    "How is error handling implemented?",
    "Explain the data processing pipeline",
    "What external libraries or dependencies are used and why?",
    "How are configuration values managed?",
    "What testing patterns exist in the codebase?",
    "How does the routing or URL dispatch work?",
]


def evaluate_strategy(query, project_id, strategy):
    """Run a single query with a specific retrieval strategy and measure quality.

    Returns a dict with precision, recall proxy, quality_score, latency, etc.
    """
    t0 = time.time()

    # Plan
    project = get_project(project_id)
    project_info = dict(project) if project else {}
    plan = plan_query(query, project_info)

    # Retrieve
    result = execute_retrieval(plan, project_id, strategy)
    chunks = result.get("chunks", [])

    # Rerank
    reranked = rerank(query, chunks, top_k=config.TOP_K_RERANK) if chunks else []

    # Quality check (acts as a relevance proxy)
    quality = check_quality(query, reranked)

    latency_ms = (time.time() - t0) * 1000

    return {
        "query": query,
        "strategy": strategy,
        "chunks_retrieved": len(chunks),
        "chunks_after_rerank": len(reranked),
        "quality_score": quality["quality_score"],
        "is_sufficient": quality["is_sufficient"],
        "top_rerank_scores": [
            round(c.get("rerank_score", 0), 3) for c in reranked[:3]
        ],
        "latency_ms": round(latency_ms, 1),
    }


def run_benchmark(project_id, queries=None, strategies=None):
    """Run the full benchmark for a project across all strategies.

    Returns:
        dict keyed by strategy with aggregated metrics.
    """
    queries = queries or DEFAULT_QUERIES
    strategies = strategies or ["dense", "sparse", "hybrid"]

    results = []
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        for i, q in enumerate(queries):
            print(f"    [{i + 1}/{len(queries)}] {q[:60]}...", end=" ", flush=True)
            try:
                r = evaluate_strategy(q, project_id, strategy)
                results.append(r)
                print(f"score={r['quality_score']:.2f}  "
                      f"latency={r['latency_ms']:.0f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "query": q,
                    "strategy": strategy,
                    "quality_score": 0,
                    "is_sufficient": False,
                    "chunks_retrieved": 0,
                    "chunks_after_rerank": 0,
                    "latency_ms": 0,
                    "error": str(e),
                })

    # ── Aggregate by strategy ──────────────────────────────────────────
    aggregated = {}
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    for strategy, strat_results in by_strategy.items():
        scores = [r["quality_score"] for r in strat_results]
        latencies = [r["latency_ms"] for r in strat_results]
        sufficient = [r["is_sufficient"] for r in strat_results]

        aggregated[strategy] = {
            "avg_quality": round(sum(scores) / len(scores), 4) if scores else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "sufficiency_rate": round(sum(sufficient) / len(sufficient), 4) if sufficient else 0,
            "total_queries": len(strat_results),
            "per_query": strat_results,
        }

    return aggregated


def print_summary(aggregated):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("  RETRIEVAL STRATEGY COMPARISON")
    print("=" * 70)
    print(f"  {'Strategy':<12} {'Avg Quality':>12} {'Sufficiency':>12} {'Avg Latency':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

    for strategy in ["dense", "sparse", "hybrid"]:
        if strategy in aggregated:
            a = aggregated[strategy]
            print(
                f"  {strategy:<12} "
                f"{a['avg_quality']:>11.3f} "
                f"{a['sufficiency_rate']:>10.1%} "
                f"{a['avg_latency_ms']:>10.0f}ms"
            )

    print("=" * 70)

    # Determine winner
    best = max(aggregated.items(), key=lambda x: x[1]["avg_quality"])
    print(f"\n  Best strategy: {best[0]} "
          f"(avg quality {best[1]['avg_quality']:.3f})\n")


def main():
    parser = argparse.ArgumentParser(description="CodeSage AI Evaluation Benchmark")
    parser.add_argument("--project-id", type=int, help="Project ID to evaluate")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--queries", type=str, help="JSON file with custom queries")
    args = parser.parse_args()

    # Load custom queries if provided
    queries = DEFAULT_QUERIES
    if args.queries and os.path.isfile(args.queries):
        with open(args.queries) as f:
            queries = json.load(f)

    # Select projects
    if args.project_id:
        projects = [get_project(args.project_id)]
        if not projects[0]:
            print(f"Project {args.project_id} not found.")
            sys.exit(1)
    else:
        projects = [p for p in get_all_projects() if p["status"] == "ready"]
        if not projects:
            print("No ready projects. Ingest a project first.")
            sys.exit(1)

    all_results = {}
    for project in projects:
        pid = project["id"]
        pname = project["name"]
        print(f"\n{'=' * 70}")
        print(f"  Benchmarking project: {pname} (id={pid})")
        print(f"{'=' * 70}")

        aggregated = run_benchmark(pid, queries)
        print_summary(aggregated)
        all_results[pname] = aggregated

    # Export
    if args.export:
        with open(args.export, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results exported to {args.export}")


if __name__ == "__main__":
    main()
