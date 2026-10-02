"""Create 10 sample Python projects for testing CodeSage AI end-to-end."""

import os
import sys

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

PROJECTS = {
    # ── 1. Flask Todo App ──────────────────────────────────────────────
    "flask_todo_app": {
        "app.py": '''"""Flask Todo Application - A simple CRUD task manager."""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from database import db, Todo

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///todos.db"
db.init_app(app)

@app.route("/")
def index():
    """Display all todo items sorted by creation date."""
    todos = Todo.query.order_by(Todo.created_at.desc()).all()
    return render_template("index.html", todos=todos)

@app.route("/add", methods=["POST"])
def add_todo():
    """Create a new todo item from form data."""
    title = request.form.get("title", "").strip()
    if title:
        todo = Todo(title=title, priority=request.form.get("priority", "medium"))
        db.session.add(todo)
        db.session.commit()
    return redirect(url_for("index"))

@app.route("/toggle/<int:todo_id>")
def toggle_todo(todo_id):
    """Toggle the completion status of a todo item."""
    todo = Todo.query.get_or_404(todo_id)
    todo.completed = not todo.completed
    db.session.commit()
    return redirect(url_for("index"))

@app.route("/delete/<int:todo_id>")
def delete_todo(todo_id):
    """Delete a todo item by its ID."""
    todo = Todo.query.get_or_404(todo_id)
    db.session.delete(todo)
    db.session.commit()
    return redirect(url_for("index"))

@app.route("/api/todos")
def api_list_todos():
    """REST API endpoint to list all todos as JSON."""
    todos = Todo.query.all()
    return jsonify([t.to_dict() for t in todos])

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
''',
        "database.py": '''"""Database models for the Todo application."""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Todo(db.Model):
    """Represents a single todo item with title, status, and priority."""

    __tablename__ = "todos"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    priority = db.Column(db.String(20), default="medium")  # low, medium, high
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Serialize todo item to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "completed": self.completed,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<Todo {self.id}: {self.title}>"
''',
        "README.md": '''# Flask Todo App
A simple CRUD task manager built with Flask and SQLAlchemy.

## Features
- Create, read, update, and delete todo items
- Toggle completion status
- Priority levels (low, medium, high)
- REST API endpoint for JSON access

## Setup
```bash
pip install flask flask-sqlalchemy
python app.py
```
''',
    },

    # ── 2. Calculator Library ──────────────────────────────────────────
    "calculator_lib": {
        "calculator.py": '''"""Advanced calculator library with mathematical operations."""

import math
from typing import List, Union

Number = Union[int, float]

class Calculator:
    """A full-featured calculator supporting basic and scientific operations."""

    def __init__(self):
        self.history: List[str] = []
        self.memory: float = 0.0

    def add(self, a: Number, b: Number) -> Number:
        """Add two numbers and return the result."""
        result = a + b
        self._record(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: Number, b: Number) -> Number:
        """Subtract b from a and return the result."""
        result = a - b
        self._record(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: Number, b: Number) -> Number:
        """Multiply two numbers and return the result."""
        result = a * b
        self._record(f"{a} * {b} = {result}")
        return result

    def divide(self, a: Number, b: Number) -> float:
        """Divide a by b. Raises ValueError if b is zero."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self._record(f"{a} / {b} = {result}")
        return result

    def power(self, base: Number, exponent: Number) -> Number:
        """Raise base to the power of exponent."""
        result = base ** exponent
        self._record(f"{base} ^ {exponent} = {result}")
        return result

    def sqrt(self, n: Number) -> float:
        """Calculate the square root of n. Raises ValueError if n < 0."""
        if n < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(n)
        self._record(f"sqrt({n}) = {result}")
        return result

    def factorial(self, n: int) -> int:
        """Calculate n factorial. Raises ValueError if n < 0."""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        result = math.factorial(n)
        self._record(f"{n}! = {result}")
        return result

    def memory_store(self, value: Number):
        """Store a value in calculator memory."""
        self.memory = float(value)

    def memory_recall(self) -> float:
        """Recall the value stored in calculator memory."""
        return self.memory

    def memory_clear(self):
        """Clear calculator memory."""
        self.memory = 0.0

    def get_history(self) -> List[str]:
        """Return list of all calculations performed."""
        return self.history.copy()

    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()

    def _record(self, entry: str):
        """Record a calculation in history."""
        self.history.append(entry)
''',
        "statistics_calc.py": '''"""Statistical calculator for data analysis."""

import math
from typing import List

def mean(data: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    if not data:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(data) / len(data)

def median(data: List[float]) -> float:
    """Calculate the median of a list of numbers."""
    if not data:
        raise ValueError("Cannot calculate median of empty list")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]

def variance(data: List[float], population: bool = True) -> float:
    """Calculate variance. Use population=False for sample variance."""
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")
    m = mean(data)
    ss = sum((x - m) ** 2 for x in data)
    divisor = len(data) if population else len(data) - 1
    return ss / divisor

def std_deviation(data: List[float], population: bool = True) -> float:
    """Calculate standard deviation."""
    return math.sqrt(variance(data, population))

def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile (0-100) of the data."""
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
''',
        "README.md": "# Calculator Library\\nAdvanced calculator with scientific operations and statistics.",
    },

    # ── 3. REST API Service ────────────────────────────────────────────
    "rest_api_service": {
        "app.py": '''"""REST API Service with authentication and user management."""

from flask import Flask, request, jsonify
from auth import require_auth, generate_token, hash_password, verify_password
from models import UserStore
from middleware import log_request, rate_limit

app = Flask(__name__)
users = UserStore()

@app.before_request
def before_request():
    """Run middleware before each request."""
    log_request(request)
    rate_limit(request)

@app.route("/api/register", methods=["POST"])
def register():
    """Register a new user with username and password."""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if users.get(username):
        return jsonify({"error": "Username already exists"}), 409

    hashed = hash_password(password)
    users.create(username, hashed, data.get("email", ""))
    token = generate_token(username)
    return jsonify({"token": token, "username": username}), 201

@app.route("/api/login", methods=["POST"])
def login():
    """Authenticate user and return JWT token."""
    data = request.get_json()
    username = data.get("username", "")
    password = data.get("password", "")

    user = users.get(username)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = generate_token(username)
    return jsonify({"token": token, "username": username})

@app.route("/api/profile")
@require_auth
def profile(current_user):
    """Get the authenticated user profile."""
    user = users.get(current_user)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"username": user["username"], "email": user["email"]})

@app.route("/api/users")
@require_auth
def list_users(current_user):
    """List all registered users (authenticated endpoint)."""
    all_users = users.list_all()
    return jsonify([{"username": u["username"], "email": u["email"]} for u in all_users])

if __name__ == "__main__":
    app.run(debug=True, port=5002)
''',
        "auth.py": '''"""Authentication module with JWT token handling."""

import hashlib
import hmac
import json
import time
import base64
from functools import wraps
from flask import request, jsonify

SECRET_KEY = "supersecret-api-key-2024"
TOKEN_EXPIRY = 3600  # 1 hour

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = "codesage-salt"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash

def generate_token(username: str) -> str:
    """Generate a simple JWT-like token."""
    payload = {"username": username, "exp": time.time() + TOKEN_EXPIRY}
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"

def decode_token(token: str) -> dict:
    """Decode and verify a token. Returns payload or raises ValueError."""
    parts = token.split(".")
    if len(parts) != 2:
        raise ValueError("Invalid token format")
    payload_b64, signature = parts
    expected = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid token signature")
    payload = json.loads(base64.b64decode(payload_b64))
    if payload.get("exp", 0) < time.time():
        raise ValueError("Token expired")
    return payload

def require_auth(f):
    """Decorator to require authentication for an endpoint."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing auth token"}), 401
        token = auth_header[7:]
        try:
            payload = decode_token(token)
            return f(current_user=payload["username"], *args, **kwargs)
        except ValueError as e:
            return jsonify({"error": str(e)}), 401
    return decorated
''',
        "models.py": '''"""In-memory user store for the API service."""

class UserStore:
    """Simple in-memory user storage."""

    def __init__(self):
        self._users = {}

    def create(self, username, password_hash, email=""):
        self._users[username] = {
            "username": username,
            "password_hash": password_hash,
            "email": email,
        }

    def get(self, username):
        return self._users.get(username)

    def list_all(self):
        return list(self._users.values())

    def delete(self, username):
        self._users.pop(username, None)
''',
        "middleware.py": '''"""Request middleware for logging and rate limiting."""

import time
from collections import defaultdict

request_log = []
rate_limits = defaultdict(list)

def log_request(request):
    """Log incoming request details."""
    entry = {
        "method": request.method,
        "path": request.path,
        "ip": request.remote_addr,
        "timestamp": time.time(),
    }
    request_log.append(entry)

def rate_limit(request, max_requests=100, window=60):
    """Simple rate limiter: max_requests per window (seconds)."""
    ip = request.remote_addr
    now = time.time()
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < window]
    if len(rate_limits[ip]) >= max_requests:
        from flask import jsonify
        return jsonify({"error": "Rate limit exceeded"}), 429
    rate_limits[ip].append(now)
''',
        "README.md": "# REST API Service\\nUser management API with JWT authentication, rate limiting, and middleware.",
    },

    # ── 4. Data Pipeline ──────────────────────────────────────────────
    "data_pipeline": {
        "pipeline.py": '''"""ETL data pipeline for processing CSV datasets."""

import csv
import os
from datetime import datetime
from transformers import clean_data, validate_record, enrich_record
from aggregator import aggregate_by_field

class DataPipeline:
    """Extract, Transform, Load pipeline for data processing."""

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.stats = {"total": 0, "valid": 0, "invalid": 0, "enriched": 0}

    def run(self):
        """Execute the full ETL pipeline."""
        raw_data = self.extract()
        cleaned = self.transform(raw_data)
        self.load(cleaned)
        return self.stats

    def extract(self):
        """Extract data from CSV file."""
        records = []
        with open(self.input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
                self.stats["total"] += 1
        return records

    def transform(self, records):
        """Clean, validate, and enrich records."""
        results = []
        for record in records:
            cleaned = clean_data(record)
            if validate_record(cleaned):
                enriched = enrich_record(cleaned)
                results.append(enriched)
                self.stats["valid"] += 1
                self.stats["enriched"] += 1
            else:
                self.stats["invalid"] += 1
        return results

    def load(self, records):
        """Write processed records to output CSV."""
        if not records:
            return
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
''',
        "transformers.py": '''"""Data transformation functions for the pipeline."""

import re
from datetime import datetime

def clean_data(record):
    """Clean a data record by stripping whitespace and normalizing fields."""
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, str):
            value = value.strip()
            value = re.sub(r"\\s+", " ", value)
        cleaned[key.lower().strip()] = value
    return cleaned

def validate_record(record):
    """Validate that a record has required fields and correct format."""
    required = ["name", "email"]
    for field in required:
        if field not in record or not record[field]:
            return False
    if "email" in record and not re.match(r"[^@]+@[^@]+\\.[^@]+", record["email"]):
        return False
    return True

def enrich_record(record):
    """Add computed fields to a record."""
    record["processed_at"] = datetime.utcnow().isoformat()
    if "name" in record:
        parts = record["name"].split()
        record["first_name"] = parts[0] if parts else ""
        record["last_name"] = parts[-1] if len(parts) > 1 else ""
    return record
''',
        "aggregator.py": '''"""Data aggregation functions."""

from collections import defaultdict

def aggregate_by_field(records, field):
    """Group records by a field and count occurrences."""
    groups = defaultdict(list)
    for record in records:
        key = record.get(field, "unknown")
        groups[key].append(record)
    return {k: len(v) for k, v in groups.items()}

def compute_summary(records):
    """Compute summary statistics for numeric fields."""
    summary = {}
    for key in records[0].keys() if records else []:
        values = []
        for r in records:
            try:
                values.append(float(r[key]))
            except (ValueError, TypeError):
                continue
        if values:
            summary[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    return summary
''',
        "README.md": "# Data Pipeline\\nETL pipeline for CSV data processing with validation and enrichment.",
    },

    # ── 5. CLI Tool ────────────────────────────────────────────────────
    "cli_tool": {
        "cli.py": '''"""Command-line file search and management tool."""

import argparse
import os
import sys
from file_ops import find_files, count_lines, search_in_file, get_file_stats

def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description="File search and management tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Find command
    find_parser = subparsers.add_parser("find", help="Find files by pattern")
    find_parser.add_argument("pattern", help="File name pattern (glob)")
    find_parser.add_argument("-d", "--directory", default=".", help="Directory to search")
    find_parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search text in files")
    search_parser.add_argument("text", help="Text to search for")
    search_parser.add_argument("path", help="File or directory to search")
    search_parser.add_argument("-i", "--ignore-case", action="store_true")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show file statistics")
    stats_parser.add_argument("path", help="File or directory path")

    args = parser.parse_args()

    if args.command == "find":
        results = find_files(args.directory, args.pattern, args.recursive)
        for f in results:
            print(f)
        print(f"Found {len(results)} files")

    elif args.command == "search":
        results = search_in_file(args.path, args.text, args.ignore_case)
        for filepath, line_num, line in results:
            print(f"{filepath}:{line_num}: {line.strip()}")

    elif args.command == "stats":
        stats = get_file_stats(args.path)
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
''',
        "file_ops.py": '''"""File operation utilities for the CLI tool."""

import os
import fnmatch

def find_files(directory, pattern, recursive=False):
    """Find files matching a glob pattern in a directory."""
    matches = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in fnmatch.filter(files, pattern):
                matches.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            if fnmatch.fnmatch(f, pattern) and os.path.isfile(os.path.join(directory, f)):
                matches.append(os.path.join(directory, f))
    return sorted(matches)

def count_lines(filepath):
    """Count the number of lines in a file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

def search_in_file(path, text, ignore_case=False):
    """Search for text in a file or directory. Returns list of (file, line_num, line)."""
    results = []
    files = []

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for f in filenames:
                files.append(os.path.join(root, f))

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    target = line.lower() if ignore_case else line
                    search = text.lower() if ignore_case else text
                    if search in target:
                        results.append((filepath, i, line))
        except (IOError, OSError):
            continue
    return results

def get_file_stats(path):
    """Get statistics about a file or directory."""
    if os.path.isfile(path):
        return {
            "type": "file",
            "size_bytes": os.path.getsize(path),
            "lines": count_lines(path),
            "extension": os.path.splitext(path)[1],
        }
    elif os.path.isdir(path):
        total_files = 0
        total_size = 0
        extensions = {}
        for root, dirs, files in os.walk(path):
            for f in files:
                fpath = os.path.join(root, f)
                total_files += 1
                total_size += os.path.getsize(fpath)
                ext = os.path.splitext(f)[1]
                extensions[ext] = extensions.get(ext, 0) + 1
        return {
            "type": "directory",
            "total_files": total_files,
            "total_size_bytes": total_size,
            "extensions": extensions,
        }
    return {"error": "Path not found"}
''',
        "README.md": "# CLI File Tool\\nCommand-line utility for file search, text search, and statistics.",
    },

    # ── 6. Chat Application ───────────────────────────────────────────
    "chat_app": {
        "server.py": '''"""Simple chat server with room-based messaging."""

from datetime import datetime

class ChatServer:
    """Multi-room chat server managing users and messages."""

    def __init__(self):
        self.rooms = {}  # room_name -> Room
        self.users = {}  # username -> User

    def create_room(self, room_name, created_by):
        """Create a new chat room."""
        if room_name in self.rooms:
            raise ValueError(f"Room '{room_name}' already exists")
        room = Room(room_name, created_by)
        self.rooms[room_name] = room
        return room

    def join_room(self, room_name, username):
        """Add a user to a chat room."""
        if room_name not in self.rooms:
            raise ValueError(f"Room '{room_name}' not found")
        self.rooms[room_name].add_member(username)

    def send_message(self, room_name, username, content):
        """Send a message to a room."""
        if room_name not in self.rooms:
            raise ValueError(f"Room '{room_name}' not found")
        room = self.rooms[room_name]
        if username not in room.members:
            raise ValueError(f"User '{username}' is not in room '{room_name}'")
        message = Message(username, content)
        room.messages.append(message)
        return message

    def get_messages(self, room_name, limit=50):
        """Get recent messages from a room."""
        if room_name not in self.rooms:
            return []
        return self.rooms[room_name].messages[-limit:]

class Room:
    """Represents a chat room with members and messages."""

    def __init__(self, name, created_by):
        self.name = name
        self.created_by = created_by
        self.members = set()
        self.messages = []
        self.created_at = datetime.utcnow()

    def add_member(self, username):
        self.members.add(username)

    def remove_member(self, username):
        self.members.discard(username)

class Message:
    """Represents a chat message."""

    def __init__(self, sender, content):
        self.sender = sender
        self.content = content
        self.timestamp = datetime.utcnow()

    def to_dict(self):
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
''',
        "README.md": "# Chat Application\\nRoom-based chat server with user and message management.",
    },

    # ── 7. ML Model Trainer ───────────────────────────────────────────
    "ml_trainer": {
        "trainer.py": '''"""Machine learning model training pipeline."""

import json
from data_loader import load_dataset, split_data
from model import create_model, evaluate_model

class ModelTrainer:
    """End-to-end ML training pipeline with evaluation."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.metrics = {}

    def run_pipeline(self, data_path):
        """Execute the full training pipeline."""
        # Step 1: Load data
        X, y = load_dataset(data_path)
        print(f"Loaded {len(X)} samples")

        # Step 2: Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=self.config.get("test_size", 0.2)
        )
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Step 3: Train model
        model_type = self.config.get("model_type", "random_forest")
        self.model = create_model(model_type, self.config.get("params", {}))
        self.model.fit(X_train, y_train)
        print(f"Model trained: {model_type}")

        # Step 4: Evaluate
        self.metrics = evaluate_model(self.model, X_test, y_test)
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        return self.metrics

    def save_results(self, output_path):
        """Save training results to JSON."""
        with open(output_path, "w") as f:
            json.dump({"config": self.config, "metrics": self.metrics}, f, indent=2)
''',
        "model.py": '''"""Model creation and evaluation utilities."""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_model(model_type, params=None):
    """Create an ML model by type name."""
    params = params or {}
    models = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "svm": SVC,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type](**params)

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return metrics."""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }
''',
        "data_loader.py": '''"""Dataset loading and preprocessing."""

import csv
import random

def load_dataset(filepath):
    """Load a CSV dataset and split into features and labels."""
    X, y = [], []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            X.append([float(v) for v in row[:-1]])
            y.append(row[-1])
    return X, y

def split_data(X, y, test_size=0.2, random_seed=42):
    """Split data into train and test sets."""
    random.seed(random_seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_point = int(len(indices) * (1 - test_size))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test
''',
        "README.md": "# ML Model Trainer\\nMachine learning training pipeline with multiple model types and evaluation.",
    },

    # ── 8. File Manager ───────────────────────────────────────────────
    "file_manager": {
        "manager.py": '''"""File manager for directory traversal and file operations."""

import os
import shutil
from pathlib import Path

class FileManager:
    """Manage files and directories with common operations."""

    def __init__(self, base_path="."):
        self.base_path = Path(base_path).resolve()

    def list_directory(self, relative_path=""):
        """List contents of a directory."""
        target = self.base_path / relative_path
        if not target.is_dir():
            raise FileNotFoundError(f"Directory not found: {target}")
        entries = []
        for item in sorted(target.iterdir()):
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
                "modified": item.stat().st_mtime,
            })
        return entries

    def search(self, pattern, recursive=True):
        """Search for files matching a glob pattern."""
        method = self.base_path.rglob if recursive else self.base_path.glob
        return [str(p.relative_to(self.base_path)) for p in method(pattern)]

    def copy_file(self, source, destination):
        """Copy a file from source to destination."""
        src = self.base_path / source
        dst = self.base_path / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def move_file(self, source, destination):
        """Move a file from source to destination."""
        src = self.base_path / source
        dst = self.base_path / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    def delete(self, path):
        """Delete a file or empty directory."""
        target = self.base_path / path
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            target.rmdir()
        else:
            raise FileNotFoundError(f"Not found: {target}")

    def get_tree(self, relative_path="", depth=3):
        """Get a tree representation of directory structure."""
        target = self.base_path / relative_path
        return self._build_tree(target, depth)

    def _build_tree(self, path, depth, prefix=""):
        """Recursively build directory tree string."""
        if depth <= 0:
            return ""
        lines = []
        entries = sorted(path.iterdir()) if path.is_dir() else []
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                lines.append(self._build_tree(entry, depth - 1, prefix + extension))
        return "\\n".join(filter(None, lines))
''',
        "README.md": "# File Manager\\nFile and directory management with search, copy, move, and tree display.",
    },

    # ── 9. Database ORM ───────────────────────────────────────────────
    "database_orm": {
        "orm.py": '''"""Custom lightweight ORM for SQLite databases."""

import sqlite3
from contextlib import contextmanager

class Database:
    """SQLite database connection manager."""

    def __init__(self, db_path):
        self.db_path = db_path

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

class Model:
    """Base model class for ORM entities."""
    _table_name = ""
    _fields = {}

    def __init__(self, db, **kwargs):
        self._db = db
        self.id = kwargs.get("id")
        for field, default in self._fields.items():
            setattr(self, field, kwargs.get(field, default))

    def save(self):
        """Insert or update the record."""
        if self.id:
            self._update()
        else:
            self._insert()

    def _insert(self):
        fields = list(self._fields.keys())
        values = [getattr(self, f) for f in fields]
        placeholders = ", ".join("?" for _ in fields)
        sql = f"INSERT INTO {self._table_name} ({', '.join(fields)}) VALUES ({placeholders})"
        with self._db.connection() as conn:
            cursor = conn.execute(sql, values)
            self.id = cursor.lastrowid

    def _update(self):
        fields = list(self._fields.keys())
        values = [getattr(self, f) for f in fields]
        set_clause = ", ".join(f"{f} = ?" for f in fields)
        sql = f"UPDATE {self._table_name} SET {set_clause} WHERE id = ?"
        with self._db.connection() as conn:
            conn.execute(sql, values + [self.id])

    def delete(self):
        """Delete this record from the database."""
        sql = f"DELETE FROM {self._table_name} WHERE id = ?"
        with self._db.connection() as conn:
            conn.execute(sql, [self.id])

    @classmethod
    def find(cls, db, record_id):
        """Find a record by ID."""
        sql = f"SELECT * FROM {cls._table_name} WHERE id = ?"
        with db.connection() as conn:
            row = conn.execute(sql, [record_id]).fetchone()
            if row:
                return cls(db, **dict(row))
        return None

    @classmethod
    def all(cls, db):
        """Get all records."""
        sql = f"SELECT * FROM {cls._table_name}"
        with db.connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [cls(db, **dict(r)) for r in rows]

    @classmethod
    def where(cls, db, **conditions):
        """Find records matching conditions."""
        clauses = " AND ".join(f"{k} = ?" for k in conditions)
        sql = f"SELECT * FROM {cls._table_name} WHERE {clauses}"
        with db.connection() as conn:
            rows = conn.execute(sql, list(conditions.values())).fetchall()
            return [cls(db, **dict(r)) for r in rows]

    @classmethod
    def create_table(cls, db):
        """Create the table for this model."""
        field_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        type_map = {str: "TEXT", int: "INTEGER", float: "REAL", bool: "INTEGER"}
        for name, default in cls._fields.items():
            sql_type = type_map.get(type(default), "TEXT")
            field_defs.append(f"{name} {sql_type}")
        sql = f"CREATE TABLE IF NOT EXISTS {cls._table_name} ({', '.join(field_defs)})"
        with db.connection() as conn:
            conn.execute(sql)
''',
        "README.md": "# Database ORM\\nCustom lightweight ORM for SQLite with model base class, CRUD, and query builder.",
    },

    # ── 10. Web Scraper ───────────────────────────────────────────────
    "web_scraper": {
        "scraper.py": '''"""Web scraper for extracting structured data from web pages."""

import re
import time
import urllib.request
from urllib.parse import urljoin, urlparse
from html_parser import parse_html, extract_links, extract_text

class WebScraper:
    """Configurable web scraper with rate limiting and content extraction."""

    def __init__(self, base_url, delay=1.0):
        self.base_url = base_url
        self.delay = delay
        self.visited = set()
        self.results = []

    def scrape_page(self, url):
        """Scrape a single page and extract content."""
        if url in self.visited:
            return None
        self.visited.add(url)

        try:
            html = self._fetch(url)
            title = self._extract_title(html)
            text = extract_text(html)
            links = extract_links(html, url)
            headings = self._extract_headings(html)

            result = {
                "url": url,
                "title": title,
                "text": text[:5000],
                "links": links,
                "headings": headings,
            }
            self.results.append(result)
            time.sleep(self.delay)
            return result

        except Exception as e:
            return {"url": url, "error": str(e)}

    def crawl(self, max_pages=10):
        """Crawl starting from base URL, following links up to max_pages."""
        queue = [self.base_url]
        while queue and len(self.visited) < max_pages:
            url = queue.pop(0)
            result = self.scrape_page(url)
            if result and "links" in result:
                for link in result["links"]:
                    if self._is_same_domain(link) and link not in self.visited:
                        queue.append(link)
        return self.results

    def _fetch(self, url):
        """Fetch HTML content from a URL."""
        req = urllib.request.Request(url, headers={"User-Agent": "CodeSage-Scraper/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _extract_title(self, html):
        """Extract page title from HTML."""
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_headings(self, html):
        """Extract all headings (h1-h6) from HTML."""
        headings = []
        for match in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\\1>", html, re.IGNORECASE | re.DOTALL):
            level = int(match.group(1))
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if text:
                headings.append({"level": level, "text": text})
        return headings

    def _is_same_domain(self, url):
        """Check if URL belongs to the same domain as base URL."""
        base_domain = urlparse(self.base_url).netloc
        url_domain = urlparse(url).netloc
        return base_domain == url_domain
''',
        "html_parser.py": '''"""HTML parsing utilities for the web scraper."""

import re
from urllib.parse import urljoin

def parse_html(html):
    """Parse HTML and return structured sections."""
    sections = []
    # Split by major block elements
    blocks = re.split(r"<(?:div|section|article|main)[^>]*>", html)
    for block in blocks:
        text = strip_tags(block).strip()
        if len(text) > 50:
            sections.append(text)
    return sections

def extract_links(html, base_url):
    """Extract all href links from HTML, resolved to absolute URLs."""
    links = []
    for match in re.finditer(r\'href=["\\\'](.*?)["\\\']\\', html):
        href = match.group(1).strip()
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute.startswith("http"):
            links.append(absolute)
    return list(set(links))

def extract_text(html):
    """Extract visible text from HTML, removing tags and scripts."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = strip_tags(text)
    # Clean whitespace
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def strip_tags(html):
    """Remove all HTML tags from a string."""
    return re.sub(r"<[^>]+>", " ", html)
''',
        "README.md": "# Web Scraper\\nConfigurable web scraper with crawling, content extraction, and rate limiting.",
    },
}


def create_test_projects():
    """Create all 10 test projects on disk."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for project_name, files in PROJECTS.items():
        project_dir = os.path.join(TEST_DATA_DIR, project_name)
        os.makedirs(project_dir, exist_ok=True)

        for filename, content in files.items():
            filepath = os.path.join(project_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        print(f"  Created: {project_name} ({len(files)} files)")

    print(f"\n  All 10 test projects created in {TEST_DATA_DIR}")


if __name__ == "__main__":
    print("Creating 10 sample test projects for CodeSage AI...\n")
    create_test_projects()
