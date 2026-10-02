"""REST API Service with authentication and user management."""

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
