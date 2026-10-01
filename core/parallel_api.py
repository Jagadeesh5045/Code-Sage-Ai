"""Parallel System API client for web search and content extraction.

Provides multi-source web enrichment:
  1. Primary: Parallel System API for live web search
  2. Fallback: LLM-based documentation retrieval using OpenRouter
"""

import json
import requests
import config


PARALLEL_BASE = "https://api.parallel.ai/v1"


# ── Parallel System API (Primary) ───────────────────────────────────────

def web_search(objective, search_queries, mode="fast", max_chars=10000):
    """Search the web using Parallel System API."""
    try:
        headers = {
            "Authorization": f"Bearer {config.PARALLEL_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "objective": objective,
            "search_queries": search_queries,
            "mode": mode,
            "excerpts": {"max_chars_per_result": max_chars},
        }
        resp = requests.post(
            f"{PARALLEL_BASE}/search",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[Parallel API] Search failed: HTTP {resp.status_code} - {resp.text[:300]}")
            return {"results": [], "error": f"Search API error: {resp.status_code} - {resp.text[:200]}"}

        data = resp.json()
        print(f"[Parallel API] Search returned {len(data.get('results', []))} results")
        results = []
        for r in data.get("results", []):
            results.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "excerpts": r.get("excerpts", []),
                "publish_date": r.get("publish_date"),
            })
        return {"results": results, "error": None}

    except Exception as e:
        return {"results": [], "error": str(e)}


def extract_content(urls, objective, excerpts=True, full_content=False):
    """Extract content from URLs using Parallel System API."""
    try:
        headers = {
            "Authorization": f"Bearer {config.PARALLEL_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "urls": urls if isinstance(urls, list) else [urls],
            "objective": objective,
            "excerpts": excerpts,
            "full_content": full_content,
        }
        resp = requests.post(
            f"{PARALLEL_BASE}/extract",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            return {"results": [], "error": f"Extract API error: {resp.status_code}"}

        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "excerpts": r.get("excerpts", []),
                "full_content": r.get("full_content"),
            })
        return {"results": results, "error": None}

    except Exception as e:
        return {"results": [], "error": str(e)}


# ── LLM Documentation Oracle (Fallback) ─────────────────────────────────

def llm_documentation_search(query, language="Python"):
    """Use LLM to retrieve official documentation knowledge as fallback.

    When the web search API is unavailable, the LLM acts as a documentation
    oracle — it has extensive training knowledge of official docs for Python,
    Flask, Django, pandas, NumPy, scikit-learn, and other major frameworks.

    Returns the same format as web_search for compatibility.
    """
    from core.openrouter import chat_completion

    prompt_messages = [
        {
            "role": "system",
            "content": f"""You are a documentation expert. Provide accurate, official documentation knowledge for {language} programming.

Your response MUST be structured exactly as JSON with this format:
{{
  "sources": [
    {{
      "title": "Official Doc Page Title",
      "url": "https://docs.python.org/... (real official URL)",
      "content": "Accurate documentation content including syntax, parameters, return values, and examples"
    }}
  ]
}}

Rules:
- Only reference REAL official documentation URLs (docs.python.org, flask.palletsprojects.com, pandas.pydata.org, etc.)
- Provide accurate technical content matching what's in the official docs
- Include code examples from official documentation
- Return 2-3 sources maximum
- Content should be 200-400 words per source"""
        },
        {
            "role": "user",
            "content": f"Provide official {language} documentation for: {query}"
        }
    ]

    try:
        result = chat_completion(prompt_messages, temperature=0.1)
        content = result.get("content", "")

        # Parse the JSON response
        # Try to extract JSON from the response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(content[json_start:json_end])
        else:
            return {"results": [], "error": "LLM response not in expected format", "source": "llm_fallback"}

        results = []
        for s in data.get("sources", []):
            results.append({
                "url": s.get("url", ""),
                "title": s.get("title", ""),
                "excerpts": [s.get("content", "")],
                "source_type": "llm_documentation",
            })

        print(f"[LLM Docs] Retrieved {len(results)} documentation sources")
        return {"results": results, "error": None, "source": "llm_fallback"}

    except json.JSONDecodeError:
        # If JSON parsing fails, treat the whole response as a single doc source
        return {
            "results": [{
                "url": f"https://docs.python.org/3/search.html?q={query.replace(' ', '+')}",
                "title": f"{language} Documentation: {query}",
                "excerpts": [content[:1000]],
                "source_type": "llm_documentation",
            }],
            "error": None,
            "source": "llm_fallback",
        }
    except Exception as e:
        return {"results": [], "error": str(e), "source": "llm_fallback"}


# ── Unified Search (tries Parallel first, falls back to LLM) ────────────

def search_programming_docs(query, language="Python"):
    """Search for programming documentation — tries web API first, then LLM fallback."""
    # Try Parallel API first
    result = web_search(
        objective=f"Find official {language} documentation and examples for: {query}",
        search_queries=[
            f"{query} {language} official documentation",
            f"{query} {language} example code",
        ],
    )

    # If web search succeeded, return it
    if result.get("results"):
        result["source"] = "parallel_api"
        return result

    # Fallback: use LLM as documentation oracle
    print(f"[Parallel API] Falling back to LLM documentation oracle")
    return llm_documentation_search(query, language)


def enrichment_search(query, language="Python", search_queries=None):
    """Full enrichment search — tries Parallel API, then LLM fallback.

    Used by the orchestrator's WEB_ENRICHMENT state.
    Returns: {results: [...], error: str|None, source: str}
    """
    # Step 1: Try Parallel System API
    if search_queries:
        result = web_search(
            objective=f"Find official {language} documentation and best practices for: {query}",
            search_queries=search_queries,
        )
        if result.get("results"):
            result["source"] = "parallel_api"
            return result

    # Step 2: Fallback to LLM documentation oracle
    print(f"[Web Enrichment] Parallel API unavailable, using LLM documentation oracle for: {query}")
    fallback = llm_documentation_search(query, language)
    return fallback
