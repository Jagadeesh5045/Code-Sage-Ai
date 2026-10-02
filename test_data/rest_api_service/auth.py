"""Authentication module with JWT token handling."""

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
