"""Self-correction loop - evaluates retrieval quality and retries if needed."""

import json
from core.openrouter import quick_llm
import config


def check_quality(query, retrieved_chunks):
    """Evaluate whether retrieved chunks sufficiently answer the query.

    Returns:
        dict with:
            - is_sufficient: bool
            - quality_score: float (0-1)
            - reasoning: str
            - suggestions: list of alternative search terms
    """
    if not retrieved_chunks:
        return {
            "is_sufficient": False,
            "quality_score": 0.0,
            "reasoning": "No chunks were retrieved.",
            "suggestions": [query],
        }

    chunks_preview = ""
    for i, chunk in enumerate(retrieved_chunks[:5]):
        name = chunk.get("name", "") or chunk.get("metadata", {}).get("name", "")
        preview = chunk.get("content", "")[:300]
        chunks_preview += f"\n[Chunk {i}: {name}]\n{preview}\n"

    prompt = f"""You are a retrieval quality evaluator. Assess whether the retrieved code/doc chunks are sufficient to answer the developer's question.

Question: "{query}"

Retrieved Chunks:
{chunks_preview}

Evaluate and return a JSON object:
{{
  "is_sufficient": true/false,
  "quality_score": 0.0-1.0,
  "reasoning": "Why the retrieval is or isn't sufficient",
  "suggestions": ["alternative search term 1", "alternative search term 2"]
}}

Scoring guide:
- 0.0-0.3: Completely irrelevant results
- 0.3-0.5: Partially relevant but missing key information
- 0.5-0.7: Relevant but could be more comprehensive
- 0.7-1.0: Highly relevant and comprehensive

Only return the JSON object."""

    result = quick_llm(prompt, system_prompt="You are a retrieval quality evaluator. Return only valid JSON.")

    if result.get("error"):
        # Fallback: basic heuristic check
        avg_score = sum(
            c.get("score", 0) or c.get("rerank_score", 0) or c.get("bm25_score", 0)
            for c in retrieved_chunks[:5]
        ) / max(len(retrieved_chunks[:5]), 1)

        return {
            "is_sufficient": avg_score > config.QUALITY_THRESHOLD,
            "quality_score": min(avg_score, 1.0),
            "reasoning": f"Heuristic evaluation based on similarity scores (avg: {avg_score:.2f})",
            "suggestions": [],
        }

    try:
        content = result["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            evaluation = json.loads(content[start:end])
        else:
            evaluation = {}
    except (json.JSONDecodeError, ValueError):
        evaluation = {}

    quality_score = evaluation.get("quality_score", 0.5)
    if isinstance(quality_score, str):
        try:
            quality_score = float(quality_score)
        except ValueError:
            quality_score = 0.5

    return {
        "is_sufficient": evaluation.get("is_sufficient", quality_score >= config.QUALITY_THRESHOLD),
        "quality_score": min(max(quality_score, 0.0), 1.0),
        "reasoning": evaluation.get("reasoning", "Quality evaluation completed"),
        "suggestions": evaluation.get("suggestions", []),
    }


def reformulate_query(original_query, attempt_number, previous_suggestions=None):
    """Generate reformulated search queries for retry attempts.

    Args:
        original_query: the original user query
        attempt_number: which retry attempt this is (1-3)
        previous_suggestions: suggestions from previous quality check

    Returns:
        dict with:
            - sub_queries: list of new search queries
            - reasoning: why these new queries were chosen
    """
    if previous_suggestions:
        return {
            "sub_queries": previous_suggestions[:3],
            "reasoning": f"Using quality checker suggestions for retry attempt {attempt_number}",
        }

    strategies = [
        "Try using more specific technical terms",
        "Try using broader/more general terms",
        "Try focusing on related concepts and synonyms",
    ]

    strategy = strategies[min(attempt_number - 1, len(strategies) - 1)]

    prompt = f"""The following search query didn't return sufficient results. Generate alternative search queries.

Original Query: "{original_query}"
Retry Attempt: {attempt_number}/3
Strategy: {strategy}

Return a JSON object:
{{
  "sub_queries": ["refined query 1", "refined query 2", "refined query 3"],
  "reasoning": "Why these queries might find better results"
}}

Only return the JSON object."""

    result = quick_llm(prompt, system_prompt="You are a search query optimizer. Return only valid JSON.")

    if result.get("error"):
        return {
            "sub_queries": [original_query + " implementation", original_query + " example"],
            "reasoning": f"Fallback reformulation for attempt {attempt_number}",
        }

    try:
        content = result["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            return {
                "sub_queries": parsed.get("sub_queries", [original_query]),
                "reasoning": parsed.get("reasoning", "Query reformulated"),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    return {
        "sub_queries": [original_query],
        "reasoning": "Could not reformulate query",
    }
