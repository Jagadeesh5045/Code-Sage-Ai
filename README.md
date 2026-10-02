# Code Sage AI — Agentic RAG Code Assistant

An agentic retrieval-augmented generation (RAG) system that answers questions over codebases with **checkable citations** — built as an MSc Artificial Intelligence dissertation project at Aston University.

Instead of trusting an LLM's memory of your code, Code Sage AI ingests a repository, builds a hybrid retrieval index over it, and reasons over the results with a multi-step agent — every claim in an answer is traced back to exact file paths and line numbers, and a hallucination guard verifies groundedness before responding.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-agent_orchestration-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## What it does

- **Ingest** any codebase (upload a zip or pick a seeded test project) — Tree-sitter parses source files into AST-aware chunks across 12 languages
- **Ask questions** in natural language via a chat UI — e.g. *"Where is authentication handled and how does the token flow work?"*
- **Get grounded answers** with inline citations pointing to the exact files and lines the answer came from
- **Explore** via a dashboard: projects, query history, agent execution traces, ingestion logs and analytics
- **Evaluate** retrieval quality with a built-in 25-query benchmark across 7 categories

## Architecture

Five components work as one unified pipeline:

| Component | What it does |
|---|---|
| **Tree-sitter parsing engine** | AST decomposition of source files across 12 languages into semantically meaningful chunks |
| **Agent orchestration layer** | 13-state LangGraph state graph: query planning, retrieval, self-correction and multi-hop reasoning (`agent/`) |
| **Hybrid retrieval** | Dense vector search (ChromaDB, `all-MiniLM-L6-v2`) + BM25 sparse matching + cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`), with fine-tuning scripts in `scripts/` |
| **Citation engine** | Traces every claim in a generated answer to exact file paths and line numbers |
| **Hallucination guard** | Checks groundedness of the draft answer; triggers self-correction retries when evidence is thin |

```
User question
    → Query planner (decompose / reformulate)
    → Hybrid retrieval (dense + BM25 → cross-encoder re-rank, top-k)
    → Citation-aware generation (OpenRouter LLM)
    → Hallucination guard (groundedness check)
    → Self-correction loop (up to 3 retries) → Final answer + citations
```

## Evaluation highlights

- **25-query benchmark, 7 categories** (`eval_benchmark.py`): no single retrieval strategy dominated — dense search had the highest mean quality, hybrid had the fewest complete retrieval failures, and a generation-only baseline answered no query with project-specific evidence
- **Like-for-like vs ChatGPT** on an unseen repository with both systems holding the same source: both answered every question correctly, differing in whether answers carried checkable file-and-line citations

## Project structure

```
├── app.py                 # Flask application (dashboard, chat, ingest, evaluation pages)
├── config.py              # All configuration via environment variables
├── core/
│   ├── code_parser.py     # Tree-sitter AST parsing & chunking
│   ├── doc_parser.py      # Markdown/text/PDF parsing
│   ├── embedder.py        # Embedding generation (sentence-transformers)
│   ├── vector_store.py    # ChromaDB / numpy vector store backends
│   ├── bm25_search.py     # Sparse retrieval
│   ├── reranker.py        # Cross-encoder re-ranking
│   ├── citation_engine.py # Claim → file:line attribution
│   ├── hallucination_guard.py
│   ├── openrouter.py      # LLM client
│   └── database.py        # SQLite persistence (projects, queries, traces)
├── agent/                 # LangGraph orchestration
│   ├── states.py          # 13-state agent state definition
│   ├── graph_engine.py    # State graph construction
│   ├── query_planner.py
│   ├── retrieval_agent.py
│   ├── self_correction.py
│   └── orchestrator.py
├── scripts/
│   ├── finetune_embeddings.py
│   └── train_reranker.py
├── templates/ & static/   # Web UI
├── test_data/             # 10 seeded sample projects for demos & eval
├── eval_benchmark.py      # 25-query retrieval benchmark
├── init_db.py / seed_test_data.py
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/Jagadeesh5045/code-sage-ai.git
cd code-sage-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (at minimum, an OpenRouter key for generation)
cp .env.example .env
# edit .env -> OPENROUTER_API_KEY=...

# 3. Seed demo data and run
python seed_test_data.py
python app.py
# open http://localhost:5000
```

Or with Docker:

```bash
cp .env.example .env   # fill in OPENROUTER_API_KEY
docker compose up --build
```

## Tech stack

Python · Flask · LangGraph · Tree-sitter · ChromaDB · sentence-transformers · rank-bm25 · cross-encoders · SQLite · OpenRouter · Docker

## Background

Developed as the dissertation project for the MSc in Artificial Intelligence at Aston University (2026):
*"Gen AI Adoption in Programming Courses: An AI-Powered Code Assistant Using Agentic Retrieval-Augmented Generation."*

## License

MIT — see [LICENSE](LICENSE).
