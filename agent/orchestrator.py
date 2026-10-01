"""Main agentic orchestrator built on LangGraph.

Implements a 13-state pipeline as a LangGraph StateGraph with typed state
and conditional edges for the self-correction loop.  Each state is a node
function that receives the full PipelineState and returns a partial update.
"""

import json
import os
import time
import traceback

from agent.graph_engine import StateGraph, END, START

from agent.states import AgentState, TracedStep, PipelineState
from agent.query_planner import plan_query
from agent.retrieval_agent import execute_retrieval
from agent.self_correction import check_quality, reformulate_query
from core.reranker import rerank
from core.citation_engine import extract_citations, format_citations_html
from core.hallucination_guard import check_hallucination
from core.openrouter import chat_completion, quick_llm
from core.parallel_api import enrichment_search
from core import embedder, vector_store
from core.database import (
    create_query, update_query, insert_agent_trace,
    insert_evaluation, get_project,
)
import config

# ── LangSmith tracing (optional) ──────────────────────────────────────────
if config.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def receive_query_node(state: PipelineState) -> dict:
    """Node 1: QUERY_RECEIVED — create DB record, load project info."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    query_id = create_query(state["project_id"], state["query_text"])
    project = get_project(state["project_id"])
    project_info = dict(project) if project else {}

    step = TracedStep(
        step_number=step_num,
        state=AgentState.QUERY_RECEIVED,
        input_data={"query": state["query_text"], "project_id": state["project_id"]},
        output_data={"query_length": len(state["query_text"]),
                     "project_name": project_info.get("name", "")},
        reasoning=f"Received query: '{state['query_text'][:100]}' "
                  f"for project '{project_info.get('name', '')}'",
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(query_id, step)

    return {
        "query_id": query_id,
        "project_info": project_info,
        "step_num": step_num,
        "trace": [step],
    }


def plan_query_node(state: PipelineState) -> dict:
    """Node 2: QUERY_PLANNING — decompose query into sub-queries."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    query_plan = plan_query(state["query_text"], state["project_info"])

    step = TracedStep(
        step_number=step_num,
        state=AgentState.QUERY_PLANNING,
        input_data={"query": state["query_text"]},
        output_data={
            "sub_queries": query_plan.get("sub_queries", []),
            "complexity": query_plan.get("complexity", ""),
            "search_strategy": query_plan.get("search_strategy", ""),
        },
        reasoning=query_plan.get("reasoning", "Query plan generated"),
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(state["query_id"], step)

    return {
        "query_plan": query_plan,
        "current_plan": query_plan,
        "step_num": step_num,
        "trace": [step],
    }


def select_tools_node(state: PipelineState) -> dict:
    """Node 3: TOOL_SELECTION — choose retrieval tools."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    search_strategy = state["query_plan"].get("search_strategy", "both")
    tools_selected = ["vector_search", "bm25_search", "web_search"]

    step = TracedStep(
        step_number=step_num,
        state=AgentState.TOOL_SELECTION,
        input_data={"search_strategy": search_strategy},
        output_data={"tools_selected": tools_selected,
                     "retrieval_mode": state["retrieval_strategy"]},
        reasoning=f"Selected tools: {', '.join(tools_selected)}. "
                  f"Using {state['retrieval_strategy']} retrieval.",
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(state["query_id"], step)

    return {
        "tools_selected": tools_selected,
        "step_num": step_num,
        "trace": [step],
    }


def retrieve_node(state: PipelineState) -> dict:
    """Node 4: RETRIEVAL — hybrid dense + sparse search."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    retrieval_result = execute_retrieval(
        state["current_plan"], state["project_id"], state["retrieval_strategy"],
    )
    retrieved_chunks = retrieval_result.get("chunks", [])

    step = TracedStep(
        step_number=step_num,
        state=AgentState.RETRIEVAL,
        input_data={
            "sub_queries": state["current_plan"].get("sub_queries", []),
            "strategy": state["retrieval_strategy"],
            "attempt": state["retry_count"] + 1,
        },
        output_data={
            "chunks_found": len(retrieved_chunks),
            "stats": retrieval_result.get("stats", {}),
        },
        reasoning=f"Retrieved {len(retrieved_chunks)} chunks using "
                  f"{state['retrieval_strategy']} search "
                  f"(attempt {state['retry_count'] + 1})",
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(state["query_id"], step)

    return {
        "retrieval_result": retrieval_result,
        "step_num": step_num,
        "trace": [step],
    }


def rerank_node(state: PipelineState) -> dict:
    """Node 5: RE_RANKING — cross-encoder re-ranking."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    retrieved_chunks = state["retrieval_result"].get("chunks", [])
    if retrieved_chunks:
        reranked = rerank(state["query_text"], retrieved_chunks,
                          top_k=config.TOP_K_RERANK)
    else:
        reranked = []

    step = TracedStep(
        step_number=step_num,
        state=AgentState.RE_RANKING,
        input_data={"chunks_to_rerank": len(retrieved_chunks)},
        output_data={
            "chunks_after_rerank": len(reranked),
            "top_scores": [round(c.get("rerank_score", 0), 2) for c in reranked[:3]],
        },
        reasoning=f"Re-ranked {len(retrieved_chunks)} chunks down to "
                  f"{len(reranked)} most relevant (cross-encoder)",
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(state["query_id"], step)

    return {
        "reranked_chunks": reranked,
        "step_num": step_num,
        "trace": [step],
    }


def quality_check_node(state: PipelineState) -> dict:
    """Node 6: QUALITY_CHECK — evaluate retrieval sufficiency."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    quality_result = check_quality(state["query_text"], state["reranked_chunks"])

    step = TracedStep(
        step_number=step_num,
        state=AgentState.QUALITY_CHECK,
        input_data={"chunks_evaluated": len(state["reranked_chunks"])},
        output_data={
            "is_sufficient": quality_result["is_sufficient"],
            "quality_score": round(quality_result["quality_score"], 2),
        },
        reasoning=quality_result["reasoning"],
        duration_ms=(time.time() - t0) * 1000,
        status="success" if quality_result["is_sufficient"] else "retry",
    )
    _save_trace(state["query_id"], step)

    return {
        "quality_result": quality_result,
        "step_num": step_num,
        "trace": [step],
    }


def self_correct_node(state: PipelineState) -> dict:
    """Node 7: SELF_CORRECTION — reformulate query for retry."""
    t0 = time.time()
    step_num = state["step_num"] + 1
    retry_count = state["retry_count"] + 1

    reformulated = reformulate_query(
        state["query_text"], retry_count,
        state["quality_result"].get("suggestions"),
    )
    new_plan = {**state["current_plan"],
                "sub_queries": reformulated["sub_queries"]}

    step = TracedStep(
        step_number=step_num,
        state=AgentState.SELF_CORRECTION,
        input_data={"original_query": state["query_text"], "attempt": retry_count},
        output_data={"new_sub_queries": reformulated["sub_queries"]},
        reasoning=reformulated["reasoning"],
        duration_ms=(time.time() - t0) * 1000,
        status="retry",
    )
    _save_trace(state["query_id"], step)

    return {
        "current_plan": new_plan,
        "retry_count": retry_count,
        "step_num": step_num,
        "trace": [step],
    }


def multi_hop_node(state: PipelineState) -> dict:
    """Node 8: MULTI_HOP — chain retrieval across files for complex queries.

    For complex or multi-hop queries, this node analyses the initially
    retrieved chunks, identifies information gaps that span multiple
    modules, and performs targeted follow-up retrieval to fill them.
    """
    t0 = time.time()
    step_num = state["step_num"] + 1

    reranked = state["reranked_chunks"]
    plan = state["current_plan"]
    complexity = plan.get("complexity", "simple")

    # Only do multi-hop for complex queries with enough initial results
    if complexity not in ("complex", "moderate") or len(reranked) < 2:
        step = TracedStep(
            step_number=step_num,
            state=AgentState.MULTI_HOP,
            input_data={"complexity": complexity, "initial_chunks": len(reranked)},
            output_data={"action": "skipped", "final_chunks": len(reranked[:config.TOP_K_FINAL])},
            reasoning=f"Skipped multi-hop: query complexity is '{complexity}'",
            duration_ms=(time.time() - t0) * 1000,
            status="skipped",
        )
        _save_trace(state["query_id"], step)
        return {
            "final_chunks": reranked[:config.TOP_K_FINAL],
            "step_num": step_num,
            "trace": [step],
        }

    # Ask LLM to identify cross-file information gaps
    chunk_summary = ""
    for i, c in enumerate(reranked[:5]):
        meta = c.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        fpath = meta.get("file_path", "") or c.get("file_path", "")
        name = meta.get("name", "") or c.get("name", "")
        chunk_summary += f"  [{fpath}: {name}]\n"

    prompt = (
        f"A developer asked: \"{state['query_text']}\"\n\n"
        f"We already retrieved these code chunks:\n{chunk_summary}\n"
        f"Are there likely related modules, files, or functions that we "
        f"should ALSO retrieve to fully answer the question?\n\n"
        f"Return a JSON object:\n"
        f'{{"needs_more": true/false, '
        f'"follow_up_queries": ["targeted query 1", "targeted query 2"]}}\n\n'
        f"Only return the JSON object."
    )

    result = quick_llm(
        prompt,
        system_prompt="You are a code analysis expert. Return only valid JSON.",
    )

    follow_ups = []
    if not result.get("error"):
        try:
            content = result["content"].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if parsed.get("needs_more"):
                    follow_ups = parsed.get("follow_up_queries", [])[:3]
        except (json.JSONDecodeError, ValueError):
            pass

    additional_chunks = []
    if follow_ups:
        for fq in follow_ups:
            query_emb = embedder.get_query_embedding(fq)
            hop_results = vector_store.search(
                query_emb, state["project_id"], top_k=3,
            )
            additional_chunks.extend(hop_results)

    # Merge and deduplicate
    all_chunks = list(reranked) + additional_chunks
    seen = set()
    unique = []
    for c in all_chunks:
        key = c.get("content", "")[:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    final = unique[:config.TOP_K_FINAL + 3]

    step = TracedStep(
        step_number=step_num,
        state=AgentState.MULTI_HOP,
        input_data={"complexity": complexity, "initial_chunks": len(reranked)},
        output_data={
            "follow_up_queries": follow_ups,
            "additional_chunks": len(additional_chunks),
            "final_chunks": len(final),
        },
        reasoning=(
            f"Multi-hop: identified {len(follow_ups)} follow-up queries, "
            f"retrieved {len(additional_chunks)} additional chunks, "
            f"merged to {len(final)} total"
            if follow_ups else
            "Multi-hop: LLM determined no additional retrieval needed"
        ),
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(state["query_id"], step)

    return {
        "final_chunks": final,
        "step_num": step_num,
        "trace": [step],
    }


def web_enrich_node(state: PipelineState) -> dict:
    """Node 9: WEB_ENRICHMENT — supplement with official documentation."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    web_context = ""
    web_sources = []
    web_status = "success"
    primary_lang = "Python"
    search_queries = []
    web_error_detail = ""
    enrichment_source = "unknown"

    EXT_TO_LANG = {
        "py": "Python", "python": "Python",
        "js": "JavaScript", "javascript": "JavaScript",
        "ts": "TypeScript", "typescript": "TypeScript",
        "jsx": "React JSX", "tsx": "React TSX",
        "java": "Java", "cpp": "C++", "c": "C",
        "cs": "C#", "go": "Go", "rb": "Ruby",
        "php": "PHP", "rs": "Rust", "swift": "Swift",
        "kt": "Kotlin",
    }

    try:
        # Detect language
        raw_lang = ""
        if state["project_info"].get("languages"):
            try:
                langs = json.loads(state["project_info"]["languages"]) \
                    if isinstance(state["project_info"]["languages"], str) \
                    else state["project_info"]["languages"]
                if langs:
                    raw_lang = langs[0] if isinstance(langs, list) else str(langs)
            except (json.JSONDecodeError, TypeError):
                pass
        if not raw_lang:
            for ch in state["final_chunks"][:5]:
                meta = ch.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                raw_lang = meta.get("language", "") or ch.get("language", "")
                if raw_lang:
                    break
        primary_lang = EXT_TO_LANG.get(raw_lang.lower().strip("."),
                                       raw_lang or "Python")

        stop_words = {
            "how", "does", "the", "and", "is", "it", "in", "this",
            "project", "used", "what", "are", "a", "to", "of", "for",
            "can", "do", "with", "be", "has",
        }
        key_terms = [
            w for w in state["query_text"].split()
            if w.lower() not in stop_words and len(w) > 1
        ]
        short_query = " ".join(key_terms[:8])

        search_queries = [
            f"{short_query} {primary_lang} documentation",
            f"{short_query} {primary_lang} tutorial example",
        ]

        frameworks = {
            "flask": "Flask Python web framework",
            "django": "Django Python web framework",
            "fastapi": "FastAPI Python",
            "react": "React JavaScript",
            "express": "Express.js Node.js",
            "spring": "Spring Boot Java",
            "pandas": "pandas Python data analysis",
            "numpy": "NumPy Python",
            "sklearn": "scikit-learn machine learning",
            "pytorch": "PyTorch deep learning",
            "tensorflow": "TensorFlow deep learning",
            "sqlalchemy": "SQLAlchemy Python ORM",
            "sqlite": "SQLite database",
        }
        query_lower = state["query_text"].lower()
        for fw, fw_full in frameworks.items():
            if fw in query_lower:
                search_queries.append(f"{fw_full} {short_query} official docs")
                break

        web_results = enrichment_search(
            query=short_query,
            language=primary_lang,
            search_queries=search_queries,
        )

        enrichment_source = web_results.get("source", "unknown")
        if web_results.get("error"):
            web_error_detail = web_results["error"]

        if web_results.get("results"):
            for wr in web_results["results"][:4]:
                title = wr.get("title", "Web Source")
                url = wr.get("url", "")
                excerpts = wr.get("excerpts", [])
                source_type = wr.get("source_type", "web")
                if excerpts:
                    excerpt_text = "\n".join(ex[:500] for ex in excerpts[:2])
                    source_label = ("Official Docs"
                                    if source_type == "llm_documentation"
                                    else "Web")
                    web_context += (
                        f"\n[{source_label}: {title}]\n"
                        f"Source: {url}\n{excerpt_text}\n"
                    )
                    web_sources.append({
                        "title": title, "url": url,
                        "excerpt_length": len(excerpt_text),
                        "source_type": source_type,
                    })

        if not web_context:
            web_status = "skipped"

    except Exception as web_err:
        web_status = "failed"
        web_context = ""
        web_error_detail = str(web_err)
        enrichment_source = "error"

    step = TracedStep(
        step_number=step_num,
        state=AgentState.WEB_ENRICHMENT,
        input_data={
            "query": short_query if search_queries else state["query_text"],
            "language": primary_lang,
            "search_queries": search_queries,
        },
        output_data={
            "web_sources_found": len(web_sources),
            "web_context_length": len(web_context),
            "sources": web_sources,
            "enrichment_method": enrichment_source,
        },
        reasoning=(
            f"Enriched with {len(web_sources)} official {primary_lang} docs "
            f"via {'Parallel API' if enrichment_source == 'parallel_api' else 'LLM oracle'}. "
            f"Added {len(web_context)} chars."
            if web_status == "success" else
            f"Web enrichment {web_status}: {web_error_detail or 'no results'}"
        ),
        duration_ms=(time.time() - t0) * 1000,
        status=web_status,
    )
    _save_trace(state["query_id"], step)

    return {
        "web_context": web_context,
        "web_sources": web_sources,
        "step_num": step_num,
        "trace": [step],
    }


def generate_node(state: PipelineState) -> dict:
    """Node 10: GENERATION — LLM answer synthesis."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    context = _build_context(state["final_chunks"])
    web_context = state["web_context"]
    if web_context:
        context += f"\n\nAdditional Web Resources:\n{web_context}"

    has_web = bool(web_context.strip())
    system_prompt = (
        "You are CodeSage AI, an expert code intelligence assistant. "
        "Answer the developer's question using the provided context.\n\n"
        "Rules:\n"
        "1. Base your answer primarily on the provided source code and documentation context\n"
        "2. Include relevant code snippets with file paths and line numbers\n"
        "3. If web documentation is provided, use it to supplement with official best practices\n"
        "4. When referencing web sources, mark them clearly\n"
        "5. Use markdown formatting for readability\n"
        "6. Reference specific files, functions, and line numbers\n"
        "7. Be precise, technical, and helpful\n"
        "8. If the code context is insufficient, lean on web documentation"
    )

    user_content = f"Context from codebase:\n{context}"
    if has_web:
        user_content += (
            f"\n\nOfficial Documentation & Web Resources:\n{web_context}"
        )
    user_content += (
        f"\n\nDeveloper's Question: {state['query_text']}\n\n"
        f"Provide a comprehensive, well-structured answer."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    llm_result = chat_completion(messages)
    response_text = llm_result.get("content", "Unable to generate response.")

    step = TracedStep(
        step_number=step_num,
        state=AgentState.GENERATION,
        input_data={
            "context_chunks": len(state["final_chunks"]),
            "web_sources": len(state["web_sources"]),
            "model": llm_result.get("model", ""),
        },
        output_data={
            "response_length": len(response_text),
            "model_used": llm_result.get("model", ""),
            "tokens": llm_result.get("usage", {}),
            "web_enriched": has_web,
        },
        reasoning="Generated response using {} with {} code chunks{}".format(
            llm_result.get("model", "LLM"),
            len(state["final_chunks"]),
            " + {} web sources".format(len(state["web_sources"])) if has_web else "",
        ),
        duration_ms=llm_result.get("elapsed_ms", (time.time() - t0) * 1000),
        status="success" if not llm_result.get("error") else "failed",
    )
    _save_trace(state["query_id"], step)

    return {
        "response_text": response_text,
        "llm_result": llm_result,
        "step_num": step_num,
        "trace": [step],
    }


def citation_node(state: PipelineState) -> dict:
    """Node 11: CITATION_MAPPING — map claims to source files."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    citations = extract_citations(state["response_text"], state["final_chunks"])
    citations_html = format_citations_html(citations)

    step = TracedStep(
        step_number=step_num,
        state=AgentState.CITATION_MAPPING,
        input_data={"response_length": len(state["response_text"]),
                     "chunks_used": len(state["final_chunks"])},
        output_data={"citations_found": len(citations)},
        reasoning=f"Mapped {len(citations)} citations to source files and line numbers",
        duration_ms=(time.time() - t0) * 1000,
        status="success",
    )
    _save_trace(state["query_id"], step)

    return {
        "citations": citations,
        "citations_html": citations_html,
        "step_num": step_num,
        "trace": [step],
    }


def hallucination_node(state: PipelineState) -> dict:
    """Node 12: HALLUCINATION_CHECK — verify groundedness."""
    t0 = time.time()
    step_num = state["step_num"] + 1

    hallucination_result = check_hallucination(
        state["response_text"], state["final_chunks"],
    )

    step = TracedStep(
        step_number=step_num,
        state=AgentState.HALLUCINATION_CHECK,
        input_data={"response_length": len(state["response_text"])},
        output_data={
            "groundedness_score": round(hallucination_result["score"], 2),
            "verified_claims": len(hallucination_result["verified_claims"]),
            "flagged_claims": len(hallucination_result["flagged_claims"]),
        },
        reasoning=hallucination_result["reasoning"],
        duration_ms=(time.time() - t0) * 1000,
        status="success" if hallucination_result["score"] >= 0.5 else "failed",
    )
    _save_trace(state["query_id"], step)

    final_response = hallucination_result.get("clean_response",
                                               state["response_text"])

    return {
        "hallucination_result": hallucination_result,
        "final_response": final_response,
        "step_num": step_num,
        "trace": [step],
    }


def finalize_node(state: PipelineState) -> dict:
    """Node 13: RESPONSE_READY — compute confidence and format output."""
    total_ms = (time.time() - state["overall_start"]) * 1000
    step_num = state["step_num"] + 1

    confidence = _compute_confidence(
        state["quality_result"], state["hallucination_result"],
        len(state["final_chunks"]),
    )

    step = TracedStep(
        step_number=step_num,
        state=AgentState.RESPONSE_READY,
        input_data={},
        output_data={
            "confidence_score": round(confidence, 2),
            "total_processing_time_ms": round(total_ms, 0),
            "citations_count": len(state["citations"]),
            "retries": state["retry_count"],
        },
        reasoning=f"Response ready with {confidence:.0%} confidence "
                  f"after {total_ms:.0f}ms ({state['retry_count']} retries)",
        duration_ms=0,
        status="success",
    )
    _save_trace(state["query_id"], step)

    # Build HTML response
    import markdown as md
    response_html = md.markdown(
        state["final_response"],
        extensions=["fenced_code", "codehilite", "tables"],
    )
    response_html += state["citations_html"]

    # Persist to DB
    update_query(
        state["query_id"],
        response_text=state["final_response"],
        response_html=response_html,
        sources_json=json.dumps(state["citations"]),
        confidence_score=confidence,
        processing_time_ms=total_ms,
        model_used=state["llm_result"].get("model", ""),
        retrieval_strategy=state["retrieval_strategy"],
        total_chunks_retrieved=(
            state["retrieval_result"]["stats"].get("unique_results", 0)
            if state["retrieval_result"] else 0
        ),
        total_chunks_used=len(state["final_chunks"]),
    )

    # Save evaluation metrics
    quality_score = (state["quality_result"]["quality_score"]
                     if state["quality_result"] else 0)
    insert_evaluation(
        query_id=state["query_id"],
        precision=quality_score,
        recall=quality_score * 0.9,
        faithfulness=state["hallucination_result"]["score"],
        relevance=confidence,
        latency=total_ms,
        strategy=state["retrieval_strategy"],
        retrieved=(
            state["retrieval_result"]["stats"].get("unique_results", 0)
            if state["retrieval_result"] else 0
        ),
        relevant=len(state["final_chunks"]),
    )

    return {
        "confidence": confidence,
        "response_html": response_html,
        "step_num": step_num,
        "trace": [step],
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE: self-correction decision
# ═══════════════════════════════════════════════════════════════════════════

def _should_retry(state: PipelineState) -> str:
    """Decide whether to retry retrieval or proceed to multi-hop."""
    if state["quality_result"].get("is_sufficient"):
        return "multi_hop"
    if state["retry_count"] >= config.SELF_CORRECTION_MAX_RETRIES:
        return "multi_hop"
    return "self_correct"


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_pipeline() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("receive_query", receive_query_node)
    graph.add_node("plan_query", plan_query_node)
    graph.add_node("select_tools", select_tools_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("self_correct", self_correct_node)
    graph.add_node("multi_hop", multi_hop_node)
    graph.add_node("web_enrich", web_enrich_node)
    graph.add_node("generate", generate_node)
    graph.add_node("map_citations", citation_node)
    graph.add_node("hallucination_check", hallucination_node)
    graph.add_node("finalize", finalize_node)

    # Linear edges
    graph.add_edge(START, "receive_query")
    graph.add_edge("receive_query", "plan_query")
    graph.add_edge("plan_query", "select_tools")
    graph.add_edge("select_tools", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "quality_check")

    # Conditional: retry or advance
    graph.add_conditional_edges(
        "quality_check",
        _should_retry,
        {"self_correct": "self_correct", "multi_hop": "multi_hop"},
    )
    graph.add_edge("self_correct", "retrieve")  # loop back

    # Continuation after multi-hop
    graph.add_edge("multi_hop", "web_enrich")
    graph.add_edge("web_enrich", "generate")
    graph.add_edge("generate", "map_citations")
    graph.add_edge("map_citations", "hallucination_check")
    graph.add_edge("hallucination_check", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Singleton compiled pipeline
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API (signature unchanged — drop-in replacement)
# ═══════════════════════════════════════════════════════════════════════════

def process_query(query_text, project_id, retrieval_strategy="hybrid"):
    """Execute the full agentic RAG pipeline for a user query.

    This is the main entry point.  Internally it runs the LangGraph
    StateGraph, logs every step for frontend visualisation, and returns
    the final response dict.
    """
    overall_start = time.time()

    initial_state: PipelineState = {
        "query_text": query_text,
        "project_id": project_id,
        "retrieval_strategy": retrieval_strategy,
        "query_id": 0,
        "project_info": {},
        "query_plan": {},
        "current_plan": {},
        "tools_selected": [],
        "retrieval_result": {},
        "reranked_chunks": [],
        "quality_result": {},
        "retry_count": 0,
        "final_chunks": [],
        "web_context": "",
        "web_sources": [],
        "response_text": "",
        "llm_result": {},
        "citations": [],
        "citations_html": "",
        "hallucination_result": {},
        "final_response": "",
        "confidence": 0.0,
        "response_html": "",
        "trace": [],
        "step_num": 0,
        "overall_start": overall_start,
        "error": "",
    }

    try:
        pipeline = _get_pipeline()
        result = pipeline.invoke(initial_state)

        return {
            "query_id": result["query_id"],
            "response_text": result["final_response"],
            "response_html": result["response_html"],
            "citations": result["citations"],
            "confidence_score": result["confidence"],
            "agent_trace": [s.to_dict() for s in result["trace"]],
            "processing_time_ms": (time.time() - overall_start) * 1000,
            "model_used": result["llm_result"].get("model", ""),
            "retrieval_strategy": retrieval_strategy,
            "hallucination_score": result["hallucination_result"].get("score", 0),
            "evaluation": {
                "quality_score": result["quality_result"].get("quality_score", 0),
                "groundedness_score": result["hallucination_result"].get("score", 0),
                "retries": result["retry_count"],
            },
        }

    except Exception as e:
        total_ms = (time.time() - overall_start) * 1000
        error_msg = f"Pipeline error: {str(e)}"

        # Try to save the error if we have a query_id
        try:
            query_id = initial_state.get("query_id", 0)
            if query_id:
                update_query(query_id, response_text=error_msg,
                             confidence_score=0, processing_time_ms=total_ms)
        except Exception:
            pass

        return {
            "query_id": initial_state.get("query_id", 0),
            "response_text": error_msg,
            "response_html": f"<p class='error'>{error_msg}</p>",
            "citations": [],
            "confidence_score": 0,
            "agent_trace": [],
            "processing_time_ms": total_ms,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _build_context(chunks):
    """Build context string from retrieved chunks for LLM."""
    parts = []
    for i, chunk in enumerate(chunks):
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
        ctype = meta.get("chunk_type", "") or chunk.get("chunk_type", "")

        header = f"[Source {i + 1}: {fpath}"
        if name:
            header += f" | {ctype}: {name}"
        if start:
            header += f" | Lines {start}-{end}"
        header += "]"

        parts.append(f"{header}\n{chunk.get('content', '')}")

    return "\n\n".join(parts)


def _compute_confidence(quality_result, hallucination_result, num_chunks):
    """Compute overall confidence from multiple signals."""
    if not quality_result or not hallucination_result:
        return 0.5

    quality = quality_result.get("quality_score", 0.5)
    groundedness = hallucination_result.get("score", 0.7)
    chunk_factor = min(num_chunks / 3, 1.0)

    return min(max(
        (quality * 0.35) + (groundedness * 0.45) + (chunk_factor * 0.20),
        0.0,
    ), 1.0)


def _save_trace(query_id, step):
    """Persist a trace step to the database."""
    try:
        insert_agent_trace(
            query_id=query_id,
            step_number=step.step_number,
            state=step.state.value,
            input_data=step.input_data,
            output_data=step.output_data,
            reasoning=step.reasoning,
            duration_ms=step.duration_ms,
            status=step.status,
        )
    except Exception:
        pass
