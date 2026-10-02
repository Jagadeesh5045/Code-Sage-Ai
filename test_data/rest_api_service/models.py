"""In-memory user store for the API service."""

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
