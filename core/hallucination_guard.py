"""Hallucination detection - verifies LLM claims against retrieved context."""

import json
from core.openrouter import quick_llm


def check_hallucination(response_text, context_chunks):
    """Verify that every claim in the response is supported by the context.

    Returns:
        dict with keys:
            - score (float): 0.0 = fully hallucinated, 1.0 = fully grounded
            - verified_claims (list): claims with supporting evidence
            - flagged_claims (list): claims not supported by context
            - clean_response (str): response with flagged claims annotated
            - reasoning (str): LLM explanation of the verification
    """
    if not context_chunks or not response_text:
        return {
            "score": 0.0,
            "verified_claims": [],
            "flagged_claims": [],
            "clean_response": response_text,
            "reasoning": "No context provided for verification.",
        }

    # Build context summary
    context_text = ""
    for i, chunk in enumerate(context_chunks[:8]):
        meta = chunk.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        fpath = meta.get("file_path", "") or chunk.get("file_path", "")
        context_text += f"\n--- Context {i} [{fpath}] ---\n{chunk.get('content', '')[:500]}\n"

    prompt = f"""You are a hallucination detector for a code documentation assistant. Analyze the AI response and verify each claim against the provided source context.

AI Response:
\"\"\"{response_text[:2000]}\"\"\"

Source Context:
{context_text}

For each factual claim in the response, determine if it is:
1. VERIFIED - directly supported by the source context
2. FLAGGED - not supported or contradicted by the context

Return a JSON object:
{{
  "score": 0.85,
  "verified_claims": ["claim 1 text", "claim 2 text"],
  "flagged_claims": ["unsupported claim text"],
  "reasoning": "Brief explanation of the verification process"
}}

Only return the JSON object, no other text."""

    result = quick_llm(prompt, system_prompt="You are a precise hallucination detector. Return only valid JSON.")

    if result.get("error"):
        return {
            "score": 0.7,
            "verified_claims": [],
            "flagged_claims": [],
            "clean_response": response_text,
            "reasoning": "Hallucination check unavailable - API error.",
        }

    try:
        content = result["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
        else:
            parsed = {}
    except (json.JSONDecodeError, ValueError):
        parsed = {}

    score = parsed.get("score", 0.7)
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            score = 0.7

    verified = parsed.get("verified_claims", [])
    flagged = parsed.get("flagged_claims", [])
    reasoning = parsed.get("reasoning", "")

    # Annotate flagged claims in the response
    clean_response = response_text
    for claim in flagged:
        if claim in clean_response:
            clean_response = clean_response.replace(
                claim, f'<span class="hallucination-flag" title="This claim could not be verified against the source code">{claim}</span>'
            )

    return {
        "score": min(max(score, 0.0), 1.0),
        "verified_claims": verified if isinstance(verified, list) else [],
        "flagged_claims": flagged if isinstance(flagged, list) else [],
        "clean_response": clean_response,
        "reasoning": reasoning,
    }
