"""Initialize the SQLite database for CodeSage AI."""

import os
import sqlite3

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    source_type TEXT,
    source_url TEXT,
    status TEXT DEFAULT 'pending',
    total_files INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    languages TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_type TEXT,
    name TEXT,
    content TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    language TEXT,
    metadata TEXT,
    chromadb_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id),
    query_text TEXT NOT NULL,
    response_text TEXT,
    response_html TEXT,
    sources_json TEXT,
    confidence_score REAL,
    processing_time_ms REAL,
    model_used TEXT,
    retrieval_strategy TEXT,
    total_chunks_retrieved INTEGER,
    total_chunks_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    step_number INTEGER,
    state TEXT,
    input_data TEXT,
    output_data TEXT,
    reasoning TEXT,
    duration_ms REAL,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER REFERENCES queries(id),
    retrieval_precision REAL,
    retrieval_recall REAL,
    answer_faithfulness REAL,
    answer_relevance REAL,
    response_latency_ms REAL,
    retrieval_strategy TEXT,
    chunks_retrieved INTEGER,
    chunks_relevant INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingestion_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    step TEXT,
    message TEXT,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chromadb ON chunks(chromadb_id);
CREATE INDEX IF NOT EXISTS idx_queries_project ON queries(project_id);
CREATE INDEX IF NOT EXISTS idx_traces_query ON agent_traces(query_id);
CREATE INDEX IF NOT EXISTS idx_evals_query ON evaluations(query_id);
CREATE INDEX IF NOT EXISTS idx_logs_project ON ingestion_logs(project_id);
"""


def init_database():
    os.makedirs(os.path.dirname(config.DATABASE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.executescript(SCHEMA)
    conn.close()
    print(f"Database initialized at {config.DATABASE_PATH}")


if __name__ == "__main__":
    init_database()
