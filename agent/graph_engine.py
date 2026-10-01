"""LangGraph-compatible StateGraph engine.

Tries to import the real ``langgraph`` package first.  If it is not
installed (e.g. Python < 3.9), a lightweight drop-in implementation
is used instead.  Both expose the same API::

    graph = StateGraph(PipelineState)
    graph.add_node("name", fn)
    graph.add_edge("a", "b")
    graph.add_conditional_edges("a", router_fn, {"x": "node_x", "y": "node_y"})
    compiled = graph.compile()
    result = compiled.invoke(initial_state)
"""

try:
    from langgraph.graph import StateGraph, END, START  # noqa: F401

    LANGGRAPH_NATIVE = True

except ImportError:
    # ── Lightweight fallback implementation ─────────────────────────────
    import copy
    import operator
    from typing import get_type_hints

    LANGGRAPH_NATIVE = False
    START = "__start__"
    END = "__end__"

    class _CompiledGraph:
        """Minimal compiled graph that can .invoke()."""

        def __init__(self, nodes, edges, cond_edges, entry, state_cls):
            self._nodes = nodes
            self._edges = edges
            self._cond_edges = cond_edges
            self._entry = entry
            self._state_cls = state_cls
            self._reducers = self._detect_reducers(state_cls)

        @staticmethod
        def _detect_reducers(state_cls):
            """Detect Annotated[list, operator.add] reducers."""
            reducers = {}
            try:
                hints = get_type_hints(state_cls, include_extras=True)
            except Exception:
                return reducers
            for field_name, hint in hints.items():
                meta = getattr(hint, "__metadata__", None)
                if meta:
                    for m in meta:
                        if callable(m):
                            reducers[field_name] = m
            return reducers

        def invoke(self, state):
            state = dict(state)  # mutable copy
            current = self._entry

            visited_count = {}
            max_visits = 50  # safety limit

            while current != END:
                visited_count[current] = visited_count.get(current, 0) + 1
                if visited_count[current] > max_visits:
                    raise RuntimeError(
                        f"Node '{current}' visited {max_visits} times — likely infinite loop"
                    )

                fn = self._nodes.get(current)
                if fn is None:
                    raise ValueError(f"Unknown node: {current}")

                # Run node
                updates = fn(state)
                if updates:
                    for k, v in updates.items():
                        if k in self._reducers:
                            state[k] = self._reducers[k](state.get(k, []), v)
                        else:
                            state[k] = v

                # Determine next node
                if current in self._cond_edges:
                    router_fn, mapping = self._cond_edges[current]
                    key = router_fn(state)
                    current = mapping.get(key, END)
                elif current in self._edges:
                    current = self._edges[current]
                else:
                    current = END

            return state

    class StateGraph:
        """LangGraph-compatible StateGraph (fallback implementation)."""

        def __init__(self, state_cls):
            self._state_cls = state_cls
            self._nodes = {}
            self._edges = {}
            self._cond_edges = {}
            self._entry = None

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, src, dst):
            if src == START:
                self._entry = dst
            else:
                self._edges[src] = dst

        def add_conditional_edges(self, src, router_fn, mapping):
            self._cond_edges[src] = (router_fn, mapping)

        def compile(self):
            if self._entry is None:
                raise ValueError("No START edge defined")
            return _CompiledGraph(
                self._nodes, self._edges, self._cond_edges,
                self._entry, self._state_cls,
            )
