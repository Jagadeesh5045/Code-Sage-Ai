"""File manager for directory traversal and file operations."""

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
        return "\n".join(filter(None, lines))
