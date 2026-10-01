"""CodeSage AI - Flask Application
An Agentic RAG system for programming course assistance.
Dissertation: 'Generative AI for Programming Course'
"""

import json
import os
import shutil
import threading
import zipfile
import tempfile

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    Response, stream_with_context,
)

import config
from init_db import init_database
from core.database import (
    create_project, get_project, get_all_projects, update_project,
    delete_project as db_delete_project,
    get_chunks_for_project, insert_chunks,
    get_query, get_queries_for_project, get_recent_queries,
    get_agent_traces, get_recent_traces,
    get_dashboard_stats, get_ingestion_logs, insert_ingestion_log,
    get_evaluations,
)
from core.code_parser import parse_file as parse_code_file
from core.doc_parser import parse_document
from core.embedder import get_embeddings_batch
from core.vector_store import (
    add_chunks as vs_add_chunks,
    delete_project as vs_delete_project,
)
from agent.orchestrator import process_query

# ── App Setup ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE_MB * 1024 * 1024

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.TEST_DATA_DIR, exist_ok=True)

init_database()

# ══════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    stats = get_dashboard_stats()
    projects = get_all_projects()
    recent_queries = get_recent_queries(limit=10)
    recent_traces = get_recent_traces(limit=15)
    return render_template(
        "dashboard.html",
        stats=stats,
        projects=projects,
        recent_queries=recent_queries,
        recent_traces=recent_traces,
    )


@app.route("/chat")
@app.route("/chat/<int:project_id>")
def chat(project_id=None):
    projects = get_all_projects()
    selected_project = None
    query_history = []

    if project_id:
        selected_project = get_project(project_id)
        query_history = get_queries_for_project(project_id, limit=20)

    return render_template(
        "chat.html",
        projects=projects,
        selected_project=selected_project,
        query_history=query_history,
        project_id=project_id,
    )


@app.route("/ingest")
def ingest():
    projects = get_all_projects()
    return render_template("ingest.html", projects=projects)


@app.route("/repositories")
def repositories():
    projects = get_all_projects()
    return render_template("repositories.html", projects=projects)


@app.route("/query/<int:query_id>")
def query_detail(query_id):
    query = get_query(query_id)
    if not query:
        return redirect(url_for("dashboard"))
    traces = get_agent_traces(query_id)
    project = get_project(query["project_id"]) if query else None
    return render_template(
        "query_detail.html",
        query=query,
        traces=traces,
        project=project,
    )


@app.route("/analytics")
def analytics():
    projects = get_all_projects()
    evaluations = [dict(e) for e in get_evaluations(limit=200)]
    recent_queries = [dict(q) for q in get_recent_queries(limit=50)]
    return render_template(
        "analytics.html",
        projects=projects,
        evaluations=evaluations,
        recent_queries=recent_queries,
    )


@app.route("/evaluation")
def evaluation():
    evaluations = [dict(e) for e in get_evaluations(limit=200)]
    return render_template("evaluation.html", evaluations=evaluations)


# ══════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/query", methods=["POST"])
def api_query():
    """Submit a query and get an AI-generated response with agent trace."""
    data = request.get_json()
    query_text = data.get("query", "").strip()
    project_id = data.get("project_id")
    strategy = data.get("retrieval_strategy", "hybrid")

    if not query_text or not project_id:
        return jsonify({"error": "Query text and project_id are required"}), 400

    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    if project["status"] != "ready":
        return jsonify({"error": "Project is not ready for queries"}), 400

    result = process_query(query_text, project_id, strategy)
    return jsonify(result)


@app.route("/api/query/<int:query_id>/trace")
def api_query_trace(query_id):
    """Get the agent trace for a query."""
    traces = get_agent_traces(query_id)
    return jsonify([dict(t) for t in traces])


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Start ingesting a codebase (GitHub URL, ZIP upload, or pasted code)."""
    source_type = request.form.get("source_type", "paste")
    name = request.form.get("name", "Untitled Project")
    description = request.form.get("description", "")

    project_id = create_project(name, description, source_type)

    if source_type == "paste":
        code = request.form.get("code", "")
        filename = request.form.get("filename", "main.py")
        if not code.strip():
            return jsonify({"error": "No code provided"}), 400
        # Process pasted code in background
        thread = threading.Thread(
            target=_ingest_paste, args=(project_id, code, filename)
        )
        thread.start()

    elif source_type == "zip":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return jsonify({"error": "Only ZIP files are supported"}), 400
        # Save and process
        zip_path = os.path.join(config.UPLOAD_FOLDER, f"project_{project_id}.zip")
        file.save(zip_path)
        thread = threading.Thread(
            target=_ingest_zip, args=(project_id, zip_path)
        )
        thread.start()

    elif source_type == "github":
        url = request.form.get("url", "").strip()
        if not url:
            return jsonify({"error": "GitHub URL is required"}), 400
        update_project(project_id, source_url=url)
        thread = threading.Thread(
            target=_ingest_github, args=(project_id, url)
        )
        thread.start()

    elif source_type == "test_project":
        project_name = request.form.get("test_project_name", "")
        project_dir = os.path.join(config.TEST_DATA_DIR, project_name)
        if not os.path.isdir(project_dir):
            return jsonify({"error": "Test project not found"}), 404
        thread = threading.Thread(
            target=_ingest_directory, args=(project_id, project_dir)
        )
        thread.start()

    return jsonify({"project_id": project_id, "status": "ingesting"})


@app.route("/api/ingest/status/<int:project_id>")
def api_ingest_status(project_id):
    """Get ingestion status (SSE stream)."""
    def generate():
        import time
        prev_count = -1
        for _ in range(300):  # max ~5 minutes
            project = get_project(project_id)
            logs = get_ingestion_logs(project_id)
            log_count = len(logs)

            if log_count != prev_count:
                prev_count = log_count
                data = {
                    "status": project["status"] if project else "error",
                    "total_files": project["total_files"] if project else 0,
                    "total_chunks": project["total_chunks"] if project else 0,
                    "logs": [
                        {"step": l["step"], "message": l["message"], "status": l["status"]}
                        for l in logs
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

            if project and project["status"] in ("ready", "error"):
                break
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/project/<int:project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    """Delete a project and its data."""
    vs_delete_project(project_id)
    db_delete_project(project_id)
    return jsonify({"status": "deleted"})


@app.route("/api/projects")
def api_projects():
    """List all projects."""
    projects = get_all_projects()
    return jsonify([dict(p) for p in projects])


@app.route("/api/dashboard/stats")
def api_dashboard_stats():
    """Dashboard statistics."""
    return jsonify(get_dashboard_stats())


@app.route("/api/analytics/<int:project_id>")
def api_analytics(project_id):
    """Analytics data for a project."""
    queries = get_queries_for_project(project_id, limit=100)
    return jsonify([dict(q) for q in queries])


# ══════════════════════════════════════════════════════════════════════════
# INGESTION WORKERS (run in background threads)
# ══════════════════════════════════════════════════════════════════════════

def _ingest_paste(project_id, code, filename):
    """Ingest pasted code."""
    try:
        update_project(project_id, status="ingesting")
        insert_ingestion_log(project_id, "parse", f"Parsing pasted code: {filename}", "info")

        chunks = _parse_single_file(code, filename)
        update_project(project_id, total_files=1)
        _embed_and_store(project_id, chunks, 1)

    except Exception as e:
        update_project(project_id, status="error", error_message=str(e))
        insert_ingestion_log(project_id, "error", str(e), "error")


def _ingest_zip(project_id, zip_path):
    """Ingest a ZIP file."""
    try:
        update_project(project_id, status="ingesting")
        insert_ingestion_log(project_id, "extract", "Extracting ZIP archive...", "info")

        extract_dir = os.path.join(config.UPLOAD_FOLDER, f"project_{project_id}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        insert_ingestion_log(project_id, "extract", "ZIP extracted successfully", "success")
        _ingest_directory(project_id, extract_dir)

    except Exception as e:
        update_project(project_id, status="error", error_message=str(e))
        insert_ingestion_log(project_id, "error", str(e), "error")


def _ingest_github(project_id, url):
    """Ingest from a GitHub repository URL."""
    try:
        update_project(project_id, status="ingesting")
        insert_ingestion_log(project_id, "clone", f"Cloning repository: {url}", "info")

        clone_dir = os.path.join(config.UPLOAD_FOLDER, f"project_{project_id}")
        os.makedirs(clone_dir, exist_ok=True)

        try:
            import git
            git.Repo.clone_from(url, clone_dir, depth=1)
            insert_ingestion_log(project_id, "clone", "Repository cloned successfully", "success")
        except Exception as e:
            insert_ingestion_log(project_id, "clone", f"Git clone failed: {e}. Trying URL download...", "warning")
            _download_github_zip(url, clone_dir)

        _ingest_directory(project_id, clone_dir)

    except Exception as e:
        update_project(project_id, status="error", error_message=str(e))
        insert_ingestion_log(project_id, "error", str(e), "error")


def _download_github_zip(url, target_dir):
    """Download GitHub repo as ZIP when git clone fails."""
    import requests
    # Convert github.com URL to ZIP download
    clean_url = url.rstrip("/").rstrip(".git")
    zip_url = f"{clean_url}/archive/refs/heads/main.zip"

    resp = requests.get(zip_url, timeout=60, stream=True)
    if resp.status_code != 200:
        zip_url = f"{clean_url}/archive/refs/heads/master.zip"
        resp = requests.get(zip_url, timeout=60, stream=True)

    if resp.status_code == 200:
        zip_path = target_dir + ".zip"
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        os.remove(zip_path)

        # Move files from nested directory if needed
        subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        if len(subdirs) == 1:
            nested = os.path.join(target_dir, subdirs[0])
            for item in os.listdir(nested):
                src = os.path.join(nested, item)
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    else:
        raise Exception(f"Could not download repository: HTTP {resp.status_code}")


def _ingest_directory(project_id, directory):
    """Ingest all files from a directory."""
    try:
        update_project(project_id, status="parsing")
        insert_ingestion_log(project_id, "parse", f"Scanning directory: {directory}", "info")

        all_chunks = []
        file_count = 0
        languages = set()

        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in config.IGNORED_DIRS]

            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, directory)
                ext = os.path.splitext(fname)[1].lower()

                if ext in config.SUPPORTED_CODE_EXTENSIONS:
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if content.strip():
                            chunks = parse_code_file(content, rel_path)
                            all_chunks.extend(chunks)
                            file_count += 1
                            languages.add(ext.lstrip("."))
                    except Exception:
                        continue

                elif ext in config.SUPPORTED_DOC_EXTENSIONS:
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if content.strip():
                            chunks = parse_document(content, rel_path)
                            all_chunks.extend(chunks)
                            file_count += 1
                    except Exception:
                        continue

        insert_ingestion_log(
            project_id, "parse",
            f"Parsed {file_count} files into {len(all_chunks)} chunks",
            "success",
        )

        update_project(
            project_id,
            total_files=file_count,
            languages=json.dumps(list(languages)),
        )

        _embed_and_store(project_id, all_chunks, file_count)

    except Exception as e:
        update_project(project_id, status="error", error_message=str(e))
        insert_ingestion_log(project_id, "error", str(e), "error")


def _parse_single_file(content, filename):
    """Parse a single file (for paste mode)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in config.SUPPORTED_CODE_EXTENSIONS:
        return parse_code_file(content, filename)
    elif ext in config.SUPPORTED_DOC_EXTENSIONS:
        return parse_document(content, filename)
    else:
        return parse_code_file(content, filename)


def _embed_and_store(project_id, chunks, file_count):
    """Generate embeddings and store in ChromaDB + SQLite."""
    if not chunks:
        update_project(project_id, status="ready", total_chunks=0)
        insert_ingestion_log(project_id, "embed", "No chunks to embed", "warning")
        return

    update_project(project_id, status="embedding")
    insert_ingestion_log(
        project_id, "embed",
        f"Generating embeddings for {len(chunks)} chunks...", "info",
    )

    # Generate embeddings
    texts = []
    for c in chunks:
        prefix = f"{c.get('chunk_type', '')}: {c.get('name', '')}\n" if c.get("name") else ""
        texts.append(prefix + c["content"][:2000])

    embeddings = get_embeddings_batch(texts)

    insert_ingestion_log(project_id, "embed", "Embeddings generated", "success")

    # Store in ChromaDB
    insert_ingestion_log(project_id, "index", "Indexing in vector store...", "info")
    chromadb_ids = vs_add_chunks(project_id, chunks, embeddings)

    # Store in SQLite
    for i, c in enumerate(chunks):
        c["project_id"] = project_id
        c["chromadb_id"] = chromadb_ids[i] if i < len(chromadb_ids) else ""

    insert_chunks(chunks)

    update_project(project_id, status="ready", total_chunks=len(chunks))
    insert_ingestion_log(
        project_id, "index",
        f"Indexing complete. {len(chunks)} chunks stored and ready.",
        "success",
    )


# ══════════════════════════════════════════════════════════════════════════
# TEMPLATE FILTERS
# ══════════════════════════════════════════════════════════════════════════

@app.template_filter("tojson_safe")
def tojson_safe(value):
    if value is None:
        return "null"
    if isinstance(value, str):
        try:
            return value
        except Exception:
            return json.dumps(value)
    return json.dumps(value)


@app.template_filter("from_json")
def from_json(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob

    print("=" * 60)
    print("  CodeSage AI - Generative AI for Programming Course")
    print("  Agentic RAG Code Intelligence System")
    print(f"  Running on http://localhost:{config.PORT}")
    print("=" * 60)

    # Prevent watchdog from restarting when uploaded files change
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        use_reloader=False,          # disable watchdog reloader
    )
