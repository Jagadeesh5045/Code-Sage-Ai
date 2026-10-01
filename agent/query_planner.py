"""Query planner - decomposes complex queries into sub-queries."""

import json
from core.openrouter import quick_llm


def plan_query(query, project_info=None):
    """Analyze a user query and create an execution plan.

    Returns:
        dict with:
            - sub_queries: list of specific search queries
            - search_strategy: 'code', 'docs', 'both', 'web'
            - complexity: 'simple', 'moderate', 'complex'
            - reasoning: explanation of the plan
    """
    project_ctx = ""
    if project_info:
        project_ctx = f"\nProject: {project_info.get('name', 'Unknown')}"
        if project_info.get("languages"):
            project_ctx += f"\nLanguages: {project_info['languages']}"
        if project_info.get("total_files"):
            project_ctx += f"\nTotal files: {project_info['total_files']}"

    prompt = f"""You are a query planning agent for a code intelligence system. Analyze the developer's question and create a search plan.
{project_ctx}

Developer's Question: "{query}"

Create an execution plan as a JSON object:
{{
  "sub_queries": ["specific search query 1", "specific search query 2"],
  "search_strategy": "code|docs|both|web",
  "complexity": "simple|moderate|complex",
  "reasoning": "Explanation of why you chose this strategy and these sub-queries",
  "key_terms": ["important", "technical", "terms"],
  "expected_sources": "What types of files/sections are likely to contain the answer"
}}

Guidelines:
- For simple factual questions: 1-2 sub-queries, "simple" complexity
- For "how does X work" questions: 2-3 sub-queries targeting different aspects, "moderate"
- For architectural/cross-cutting questions: 3-5 sub-queries, "complex"
- Use "code" strategy for implementation questions
- Use "docs" for conceptual/usage questions
- Use "both" for questions that span code and documentation
- Use "web" only when the question is about external libraries/frameworks

Only return the JSON object."""

    result = quick_llm(prompt, system_prompt="You are an expert query planner. Return only valid JSON.")

    if result.get("error"):
        return _fallback_plan(query)

    try:
        content = result["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            plan = json.loads(content[start:end])
        else:
            plan = _fallback_plan(query)
    except (json.JSONDecodeError, ValueError):
        plan = _fallback_plan(query)

    # Ensure required fields
    plan.setdefault("sub_queries", [query])
    plan.setdefault("search_strategy", "both")
    plan.setdefault("complexity", "moderate")
    plan.setdefault("reasoning", "Default query plan")
    plan.setdefault("key_terms", [])
    plan.setdefault("expected_sources", "")

    return plan


def _fallback_plan(query):
    """Create a basic plan when LLM planning fails."""
    words = query.lower().split()
    is_code = any(w in words for w in ["function", "class", "method", "implement", "code", "return", "def", "bug", "error"])
    is_docs = any(w in words for w in ["documentation", "readme", "guide", "tutorial", "explain", "what", "how", "why"])

    if is_code and not is_docs:
        strategy = "code"
    elif is_docs and not is_code:
        strategy = "docs"
    else:
        strategy = "both"

    return {
        "sub_queries": [query],
        "search_strategy": strategy,
        "complexity": "simple",
        "reasoning": f"Fallback plan: direct search with strategy '{strategy}'",
        "key_terms": [w for w in words if len(w) > 3][:5],
        "expected_sources": "code files" if strategy == "code" else "documentation and code",
    }
