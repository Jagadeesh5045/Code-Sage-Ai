"""Request middleware for logging and rate limiting."""

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
