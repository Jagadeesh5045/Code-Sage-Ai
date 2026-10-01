"""Agent state definitions, LangGraph typed state, and transition types."""

import operator
import sys
from enum import Enum
from dataclasses import dataclass, field

if sys.version_info >= (3, 9):
    from typing import Annotated, TypedDict
else:
    from typing import TypedDict
    from typing_extensions import Annotated


# ═══════════════════════════════════════════════════════════════════════════
# AGENT STATE ENUM (labels, icons, colours for frontend)
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(Enum):
    """States in the agentic RAG pipeline."""
    QUERY_RECEIVED = "query_received"
    QUERY_PLANNING = "query_planning"
    TOOL_SELECTION = "tool_selection"
    RETRIEVAL = "retrieval"
    RE_RANKING = "re_ranking"
    QUALITY_CHECK = "quality_check"
    SELF_CORRECTION = "self_correction"
    MULTI_HOP = "multi_hop"
    WEB_ENRICHMENT = "web_enrichment"
    GENERATION = "generation"
    CITATION_MAPPING = "citation_mapping"
    HALLUCINATION_CHECK = "hallucination_check"
    RESPONSE_READY = "response_ready"


STATE_INFO = {
    AgentState.QUERY_RECEIVED: {
        "label": "Query Received",
        "icon": "message-square",
        "color": "#6366f1",
        "description": "User query received and preprocessed",
    },
    AgentState.QUERY_PLANNING: {
        "label": "Query Planning",
        "icon": "git-branch",
        "color": "#8b5cf6",
        "description": "Decomposing query into sub-queries and planning search strategy",
    },
    AgentState.TOOL_SELECTION: {
        "label": "Tool Selection",
        "icon": "wrench",
        "color": "#a855f7",
        "description": "Selecting optimal search tools (code search, doc search, web search)",
    },
    AgentState.RETRIEVAL: {
        "label": "Retrieval",
        "icon": "search",
        "color": "#3b82f6",
        "description": "Executing hybrid search (BM25 + vector similarity)",
    },
    AgentState.RE_RANKING: {
        "label": "Re-Ranking",
        "icon": "bar-chart-2",
        "color": "#0ea5e9",
        "description": "Cross-encoder re-ranking of retrieved chunks by relevance",
    },
    AgentState.QUALITY_CHECK: {
        "label": "Quality Check",
        "icon": "check-circle",
        "color": "#10b981",
        "description": "Evaluating retrieval quality and context sufficiency",
    },
    AgentState.SELF_CORRECTION: {
        "label": "Self-Correction",
        "icon": "refresh-cw",
        "color": "#f59e0b",
        "description": "Reformulating query and retrying retrieval (quality was insufficient)",
    },
    AgentState.MULTI_HOP: {
        "label": "Multi-Hop Reasoning",
        "icon": "git-merge",
        "color": "#14b8a6",
        "description": "Chaining retrieval across multiple files and modules",
    },
    AgentState.WEB_ENRICHMENT: {
        "label": "Web Enrichment",
        "icon": "globe",
        "color": "#0891b2",
        "description": "Searching official documentation and web resources to supplement code context",
    },
    AgentState.GENERATION: {
        "label": "Answer Generation",
        "icon": "cpu",
        "color": "#6366f1",
        "description": "Synthesizing answer from retrieved context using LLM",
    },
    AgentState.CITATION_MAPPING: {
        "label": "Citation Mapping",
        "icon": "link",
        "color": "#8b5cf6",
        "description": "Mapping response claims to source file paths and line numbers",
    },
    AgentState.HALLUCINATION_CHECK: {
        "label": "Hallucination Check",
        "icon": "shield",
        "color": "#ef4444",
        "description": "Verifying all claims are grounded in retrieved context",
    },
    AgentState.RESPONSE_READY: {
        "label": "Response Ready",
        "icon": "check",
        "color": "#22c55e",
        "description": "Final response assembled with citations and confidence score",
    },
}

# Valid state transitions (includes multi-hop)
TRANSITIONS = {
    AgentState.QUERY_RECEIVED: [AgentState.QUERY_PLANNING],
    AgentState.QUERY_PLANNING: [AgentState.TOOL_SELECTION],
    AgentState.TOOL_SELECTION: [AgentState.RETRIEVAL],
    AgentState.RETRIEVAL: [AgentState.RE_RANKING],
    AgentState.RE_RANKING: [AgentState.QUALITY_CHECK],
    AgentState.QUALITY_CHECK: [AgentState.SELF_CORRECTION, AgentState.MULTI_HOP],
    AgentState.SELF_CORRECTION: [AgentState.RETRIEVAL],
    AgentState.MULTI_HOP: [AgentState.WEB_ENRICHMENT],
    AgentState.WEB_ENRICHMENT: [AgentState.GENERATION],
    AgentState.GENERATION: [AgentState.CITATION_MAPPING],
    AgentState.CITATION_MAPPING: [AgentState.HALLUCINATION_CHECK],
    AgentState.HALLUCINATION_CHECK: [AgentState.RESPONSE_READY],
    AgentState.RESPONSE_READY: [],
}


# ═══════════════════════════════════════════════════════════════════════════
# TRACED STEP (logged to DB and rendered on frontend)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TracedStep:
    """A single step in the agent trace."""
    step_number: int
    state: AgentState
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    reasoning: str = ""
    duration_ms: float = 0.0
    status: str = "success"  # success, retry, failed, skipped

    def to_dict(self):
        info = STATE_INFO.get(self.state, {})
        return {
            "step_number": self.step_number,
            "state": self.state.value,
            "label": info.get("label", self.state.value),
            "icon": info.get("icon", "circle"),
            "color": info.get("color", "#6b7280"),
            "description": info.get("description", ""),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning": self.reasoning,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH PIPELINE STATE (TypedDict used by the StateGraph)
# ═══════════════════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    """Typed state that flows through the LangGraph pipeline."""
    # ── Inputs ──
    query_text: str
    project_id: int
    retrieval_strategy: str

    # ── Query record ──
    query_id: int
    project_info: dict

    # ── Planning ──
    query_plan: dict
    current_plan: dict
    tools_selected: list

    # ── Retrieval loop ──
    retrieval_result: dict
    reranked_chunks: list
    quality_result: dict
    retry_count: int

    # ── Final context ──
    final_chunks: list

    # ── Web enrichment ──
    web_context: str
    web_sources: list

    # ── Generation ──
    response_text: str
    llm_result: dict

    # ── Post-processing ──
    citations: list
    citations_html: str
    hallucination_result: dict
    final_response: str
    confidence: float
    response_html: str

    # ── Tracing (accumulates via operator.add) ──
    trace: Annotated[list, operator.add]
    step_num: int
    overall_start: float
    error: str
