"""File operation utilities for the CLI tool."""

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
