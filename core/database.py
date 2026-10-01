"""SQLite database manager for CodeSage AI."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config


def get_connection():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Projects ─────────────────────────────────────────────────────────────

def create_project(name, description="", source_type="paste", source_url=""):
    with get_db() as db:
        cur = db.execute(
            """INSERT INTO projects (name, description, source_type, source_url)
               VALUES (?, ?, ?, ?)""",
            (name, description, source_type, source_url),
        )
        return cur.lastrowid


def get_project(project_id):
    with get_db() as db:
        return db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()


def get_all_projects():
    with get_db() as db:
        return db.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()


def update_project(project_id, **kwargs):
    allowed = {
        "name", "description", "status", "total_files",
        "total_chunks", "languages", "error_message",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    with get_db() as db:
        db.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?",
            (*fields.values(), project_id),
        )


def delete_project(project_id):
    with get_db() as db:
        db.execute("DELETE FROM projects WHERE id = ?", (project_id,))


# ── Chunks ────────────────────────────────────────────────────────────────

def insert_chunks(chunks_list):
    with get_db() as db:
        db.executemany(
            """INSERT INTO chunks
               (project_id, file_path, chunk_type, name, content,
                start_line, end_line, language, metadata, chromadb_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c["project_id"], c.get("file_path", ""), c.get("chunk_type", ""),
                    c.get("name", ""), c.get("content", ""),
                    c.get("start_line", 0), c.get("end_line", 0),
                    c.get("language", ""), json.dumps(c.get("metadata", {})),
                    c.get("chromadb_id", ""),
                )
                for c in chunks_list
            ],
        )


def get_chunks_for_project(project_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM chunks WHERE project_id = ? ORDER BY file_path, start_line",
            (project_id,),
        ).fetchall()


def get_chunk_by_chromadb_id(chromadb_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM chunks WHERE chromadb_id = ?", (chromadb_id,)
        ).fetchone()


def get_chunks_by_ids(chromadb_ids):
    if not chromadb_ids:
        return []
    placeholders = ",".join("?" for _ in chromadb_ids)
    with get_db() as db:
        return db.execute(
            f"SELECT * FROM chunks WHERE chromadb_id IN ({placeholders})",
            chromadb_ids,
        ).fetchall()


# ── Queries ───────────────────────────────────────────────────────────────

def create_query(project_id, query_text):
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO queries (project_id, query_text) VALUES (?, ?)",
            (project_id, query_text),
        )
        return cur.lastrowid


def update_query(query_id, **kwargs):
    allowed = {
        "response_text", "response_html", "sources_json",
        "confidence_score", "processing_time_ms", "model_used",
        "retrieval_strategy", "total_chunks_retrieved", "total_chunks_used",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    with get_db() as db:
        db.execute(
            f"UPDATE queries SET {set_clause} WHERE id = ?",
            (*fields.values(), query_id),
        )


def get_query(query_id):
    with get_db() as db:
        return db.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()


def get_queries_for_project(project_id, limit=50):
    with get_db() as db:
        return db.execute(
            """SELECT * FROM queries WHERE project_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (project_id, limit),
        ).fetchall()


def get_recent_queries(limit=20):
    with get_db() as db:
        return db.execute(
            """SELECT q.*, p.name as project_name
               FROM queries q JOIN projects p ON q.project_id = p.id
               ORDER BY q.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()


# ── Agent Traces ──────────────────────────────────────────────────────────

def insert_agent_trace(query_id, step_number, state, input_data,
                       output_data, reasoning, duration_ms, status):
    with get_db() as db:
        db.execute(
            """INSERT INTO agent_traces
               (query_id, step_number, state, input_data,
                output_data, reasoning, duration_ms, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query_id, step_number, state,
                json.dumps(input_data) if isinstance(input_data, (dict, list)) else input_data,
                json.dumps(output_data) if isinstance(output_data, (dict, list)) else output_data,
                reasoning, duration_ms, status,
            ),
        )


def get_agent_traces(query_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM agent_traces WHERE query_id = ? ORDER BY step_number",
            (query_id,),
        ).fetchall()


def get_recent_traces(limit=30):
    with get_db() as db:
        return db.execute(
            """SELECT at.*, q.query_text
               FROM agent_traces at
               JOIN queries q ON at.query_id = q.id
               ORDER BY at.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()


# ── Evaluations ───────────────────────────────────────────────────────────

def insert_evaluation(query_id, precision, recall, faithfulness,
                      relevance, latency, strategy, retrieved, relevant):
    with get_db() as db:
        db.execute(
            """INSERT INTO evaluations
               (query_id, retrieval_precision, retrieval_recall,
                answer_faithfulness, answer_relevance,
                response_latency_ms, retrieval_strategy,
                chunks_retrieved, chunks_relevant)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (query_id, precision, recall, faithfulness,
             relevance, latency, strategy, retrieved, relevant),
        )


def get_evaluations(limit=100):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


# ── Ingestion Logs ────────────────────────────────────────────────────────

def insert_ingestion_log(project_id, step, message, status="info"):
    with get_db() as db:
        db.execute(
            """INSERT INTO ingestion_logs (project_id, step, message, status)
               VALUES (?, ?, ?, ?)""",
            (project_id, step, message, status),
        )


def get_ingestion_logs(project_id):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM ingestion_logs WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ).fetchall()


# ── Dashboard Stats ───────────────────────────────────────────────────────

def get_dashboard_stats():
    with get_db() as db:
        projects = db.execute("SELECT COUNT(*) as c FROM projects WHERE status='ready'").fetchone()["c"]
        total_queries = db.execute("SELECT COUNT(*) as c FROM queries").fetchone()["c"]
        avg_conf = db.execute("SELECT AVG(confidence_score) as a FROM queries WHERE confidence_score IS NOT NULL").fetchone()["a"]
        total_chunks = db.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        return {
            "total_projects": projects,
            "total_queries": total_queries,
            "avg_confidence": round(avg_conf or 0, 2),
            "total_chunks": total_chunks,
        }
