"""Simple chat server with room-based messaging."""

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
