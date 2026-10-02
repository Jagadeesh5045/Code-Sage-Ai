"""Command-line file search and management tool."""

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
