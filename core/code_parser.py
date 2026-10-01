"""Code parser using Tree-sitter for structural AST extraction.

Primary: Tree-sitter (language-agnostic, supports 40+ languages).
Fallback: Python ast module / regex when Tree-sitter is unavailable.
"""

import ast
import re
import os

# ── Tree-sitter setup ──────────────────────────────────────────────────────
try:
    from tree_sitter_languages import get_parser as _ts_get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Map file extensions to Tree-sitter language names
_EXT_TO_TS_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp", ".c": "c", ".h": "c",
    ".cs": "c_sharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
}

# Tree-sitter node types for each language
_FUNCTION_TYPES = {
    "python": ["function_definition"],
    "javascript": ["function_declaration", "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "arrow_function", "method_definition"],
    "java": ["method_declaration", "constructor_declaration"],
    "cpp": ["function_definition"],
    "c": ["function_definition"],
    "c_sharp": ["method_declaration", "constructor_declaration"],
    "go": ["function_declaration", "method_declaration"],
    "ruby": ["method"],
    "php": ["function_definition", "method_declaration"],
    "rust": ["function_item"],
    "swift": ["function_declaration"],
    "kotlin": ["function_declaration"],
}

_CLASS_TYPES = {
    "python": ["class_definition"],
    "javascript": ["class_declaration"],
    "typescript": ["class_declaration"],
    "java": ["class_declaration", "interface_declaration"],
    "cpp": ["class_specifier", "struct_specifier"],
    "c_sharp": ["class_declaration", "interface_declaration"],
    "go": ["type_declaration"],
    "ruby": ["class", "module"],
    "php": ["class_declaration", "interface_declaration"],
    "rust": ["struct_item", "impl_item", "trait_item"],
    "swift": ["class_declaration", "struct_declaration", "protocol_declaration"],
    "kotlin": ["class_declaration", "object_declaration"],
}

_IMPORT_TYPES = {
    "python": ["import_statement", "import_from_statement"],
    "javascript": ["import_statement"],
    "typescript": ["import_statement"],
    "java": ["import_declaration"],
    "go": ["import_declaration"],
    "rust": ["use_declaration"],
}


def parse_file(content, file_path):
    """Parse a source file into structured chunks.

    Uses Tree-sitter as the primary parser for language-agnostic AST
    extraction. Falls back to Python ast / regex when Tree-sitter is
    unavailable or fails.
    """
    ext = os.path.splitext(file_path)[1].lower()
    ts_lang = _EXT_TO_TS_LANG.get(ext)

    # Try Tree-sitter first
    if TREE_SITTER_AVAILABLE and ts_lang:
        try:
            chunks = _parse_with_tree_sitter(content, file_path, ts_lang)
            if chunks:
                return chunks
        except Exception:
            pass  # fall through to legacy parsers

    # Fallback: language-specific legacy parsers
    if ext == ".py":
        return _parse_python_fallback(content, file_path)
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        return _parse_javascript_fallback(content, file_path)
    elif ext in (".java", ".cs", ".kt"):
        return _parse_java_like_fallback(content, file_path)
    else:
        return _parse_generic_file(content, file_path, ext)


# ═══════════════════════════════════════════════════════════════════════════
# TREE-SITTER PARSER (primary)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_with_tree_sitter(content, file_path, ts_lang):
    """Parse any supported language using Tree-sitter AST."""
    parser = _ts_get_parser(ts_lang)
    tree = parser.parse(bytes(content, "utf-8"))
    root = tree.root_node
    lines = content.split("\n")

    chunks = []
    func_types = set(_FUNCTION_TYPES.get(ts_lang, []))
    class_types = set(_CLASS_TYPES.get(ts_lang, []))
    import_types = set(_IMPORT_TYPES.get(ts_lang, []))

    import_lines = []

    def _walk(node):
        if node.type in func_types:
            chunk = _ts_extract_function(node, content, lines, file_path, ts_lang)
            if chunk:
                chunks.append(chunk)
        elif node.type in class_types:
            chunk = _ts_extract_class(node, content, lines, file_path, ts_lang)
            if chunk:
                chunks.append(chunk)
        elif node.type in import_types:
            start = node.start_point[0]
            end = node.end_point[0]
            import_lines.append("\n".join(lines[start:end + 1]))

        # Recurse into children (but NOT into function/class bodies
        # to avoid duplicate extraction of nested functions)
        if node.type not in func_types and node.type not in class_types:
            for child in node.children:
                _walk(child)

    _walk(root)

    # Add imports chunk
    if import_lines:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "import",
            "name": "imports",
            "content": "\n".join(import_lines),
            "start_line": 1,
            "end_line": len(import_lines),
            "language": ts_lang,
            "metadata": {"type": "imports", "parser": "tree-sitter"},
        })

    # If nothing was extracted, treat whole file as one chunk
    if not chunks:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": ts_lang,
            "metadata": {"parser": "tree-sitter"},
        })

    return chunks


def _ts_extract_function(node, content, lines, file_path, ts_lang):
    """Extract a function chunk from a Tree-sitter node."""
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    source = content[node.start_byte:node.end_byte]

    # Get function name
    name_node = node.child_by_field_name("name")
    if name_node:
        name = content[name_node.start_byte:name_node.end_byte]
    else:
        # Arrow functions / anonymous: try parent assignment
        name = _ts_infer_name(node, content) or f"anonymous_{start_line}"

    # Extract parameters
    params = []
    params_node = node.child_by_field_name("parameters")
    if params_node:
        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter", "parameter",
                              "formal_parameter", "simple_parameter"):
                params.append(content[child.start_byte:child.end_byte])

    # Detect decorators (Python)
    decorators = []
    if ts_lang == "python":
        prev = node.prev_named_sibling
        while prev and prev.type == "decorator":
            decorators.append(content[prev.start_byte:prev.end_byte])
            prev = prev.prev_named_sibling

    # Detect docstring (Python)
    docstring = ""
    if ts_lang == "python":
        body = node.child_by_field_name("body")
        if body and body.named_children:
            first_stmt = body.named_children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.named_children[0] if first_stmt.named_children else None
                if expr and expr.type == "string":
                    docstring = content[expr.start_byte:expr.end_byte].strip("\"'")[:500]

    return {
        "file_path": file_path,
        "chunk_type": "function",
        "name": name,
        "content": source,
        "start_line": start_line,
        "end_line": end_line,
        "language": ts_lang,
        "metadata": {
            "params": params,
            "decorators": decorators,
            "docstring": docstring,
            "is_async": "async" in (content[node.start_byte:node.start_byte + 10]),
            "parser": "tree-sitter",
        },
    }


def _ts_extract_class(node, content, lines, file_path, ts_lang):
    """Extract a class chunk from a Tree-sitter node."""
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    source = content[node.start_byte:node.end_byte]

    name_node = node.child_by_field_name("name")
    name = content[name_node.start_byte:name_node.end_byte] if name_node else f"Class_{start_line}"

    # Extract method names from class body
    methods = []
    body = node.child_by_field_name("body")
    if body:
        for child in body.named_children:
            if child.type in ("function_definition", "method_definition",
                              "method_declaration", "function_declaration"):
                mname_node = child.child_by_field_name("name")
                if mname_node:
                    methods.append(content[mname_node.start_byte:mname_node.end_byte])

    # Extract base classes (Python)
    bases = []
    if ts_lang == "python":
        arg_list = node.child_by_field_name("superclasses")
        if arg_list:
            for child in arg_list.named_children:
                bases.append(content[child.start_byte:child.end_byte])

    return {
        "file_path": file_path,
        "chunk_type": "class",
        "name": name,
        "content": source,
        "start_line": start_line,
        "end_line": end_line,
        "language": ts_lang,
        "metadata": {
            "methods": methods,
            "bases": bases,
            "parser": "tree-sitter",
        },
    }


def _ts_infer_name(node, content):
    """Infer the name of an anonymous function from its assignment context."""
    parent = node.parent
    if parent and parent.type in ("assignment", "variable_declarator",
                                   "lexical_declaration", "variable_declaration"):
        for child in parent.children:
            if child.type == "identifier":
                return content[child.start_byte:child.end_byte]
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return content[name_node.start_byte:name_node.end_byte]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK PARSERS (legacy — used when Tree-sitter is unavailable)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_python_fallback(content, file_path):
    """Parse Python source code using the ast module (fallback)."""
    chunks = []
    lines = content.split("\n")

    # Module-level docstring
    module_doc = _extract_module_docstring(content)
    if module_doc:
        chunks.append({
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": module_doc,
            "start_line": 1,
            "end_line": module_doc.count("\n") + 1,
            "language": "python",
            "metadata": {"type": "module_docstring", "parser": "ast"},
        })

    try:
        tree = ast.parse(content)
    except SyntaxError:
        chunks.append({
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": "python",
            "metadata": {"parse_error": True, "parser": "ast"},
        })
        return chunks

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _extract_function_chunk(node, lines, file_path)
            if chunk:
                chunks.append(chunk)
        elif isinstance(node, ast.ClassDef):
            chunk = _extract_class_chunk(node, lines, file_path)
            if chunk:
                chunks.append(chunk)

    imports = _extract_imports(tree, lines)
    if imports:
        chunks.append({
            "chunk_type": "import",
            "name": "imports",
            "content": imports,
            "start_line": 1,
            "end_line": imports.count("\n") + 1,
            "language": "python",
            "metadata": {"type": "imports", "parser": "ast"},
        })

    if not chunks:
        chunks.append({
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": "python",
            "metadata": {"parser": "ast"},
        })

    for c in chunks:
        c["file_path"] = file_path
    return chunks


def _extract_function_chunk(node, lines, file_path):
    start = node.lineno
    end = node.end_lineno or start
    source = "\n".join(lines[start - 1:end])
    args = [arg.arg for arg in node.args.args]
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(ast.dump(dec))
    docstring = ast.get_docstring(node) or ""
    returns = ""
    if node.returns:
        try:
            returns = ast.dump(node.returns)
        except Exception:
            pass
    return {
        "chunk_type": "function",
        "name": node.name,
        "content": source,
        "start_line": start,
        "end_line": end,
        "language": "python",
        "metadata": {
            "params": args,
            "decorators": decorators,
            "docstring": docstring[:500],
            "returns": returns,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "parser": "ast",
        },
    }


def _extract_class_chunk(node, lines, file_path):
    start = node.lineno
    end = node.end_lineno or start
    source = "\n".join(lines[start - 1:end])
    docstring = ast.get_docstring(node) or ""
    methods = [
        item.name for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
    return {
        "chunk_type": "class",
        "name": node.name,
        "content": source,
        "start_line": start,
        "end_line": end,
        "language": "python",
        "metadata": {
            "methods": methods,
            "bases": bases,
            "docstring": docstring[:500],
            "parser": "ast",
        },
    }


def _extract_imports(tree, lines):
    import_lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno
            end = node.end_lineno or start
            import_lines.append("\n".join(lines[start - 1:end]))
    return "\n".join(import_lines) if import_lines else ""


def _extract_module_docstring(content):
    try:
        tree = ast.parse(content)
        return ast.get_docstring(tree) or ""
    except SyntaxError:
        return ""


# ── Non-Python fallback parsers (regex-based) ─────────────────────────────

def _parse_javascript_fallback(content, file_path):
    """Parse JavaScript/TypeScript files using regex patterns (fallback)."""
    chunks = []
    lines = content.split("\n")

    func_pattern = re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    class_pattern = re.compile(
        r"^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{",
        re.MULTILINE,
    )
    arrow_pattern = re.compile(
        r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
        re.MULTILINE,
    )

    used_ranges = set()
    for pattern, chunk_type in [
        (func_pattern, "function"),
        (class_pattern, "class"),
        (arrow_pattern, "function"),
    ]:
        for match in pattern.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count("\n") + 1
            end_line = _find_block_end(lines, start_line - 1)
            block = "\n".join(lines[start_line - 1:end_line])
            key = (start_line, end_line)
            if key not in used_ranges:
                used_ranges.add(key)
                chunks.append({
                    "file_path": file_path,
                    "chunk_type": chunk_type,
                    "name": name,
                    "content": block,
                    "start_line": start_line,
                    "end_line": end_line,
                    "language": "javascript",
                    "metadata": {"parser": "regex"},
                })

    if not chunks:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": "javascript",
            "metadata": {"parser": "regex"},
        })
    return chunks


def _parse_java_like_fallback(content, file_path):
    """Parse Java/C#/Kotlin files using regex (fallback)."""
    chunks = []
    lines = content.split("\n")
    ext = os.path.splitext(file_path)[1].lower()
    lang = {".java": "java", ".cs": "csharp", ".kt": "kotlin"}.get(ext, "java")

    class_pattern = re.compile(
        r"(?:public|private|protected)?\s*(?:static\s+)?class\s+(\w+)", re.MULTILINE,
    )
    method_pattern = re.compile(
        r"(?:public|private|protected)\s+(?:static\s+)?(?:\w+)\s+(\w+)\s*\(", re.MULTILINE,
    )

    for pattern, chunk_type in [(class_pattern, "class"), (method_pattern, "function")]:
        for match in pattern.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            start_line = content[:start_pos].count("\n") + 1
            end_line = _find_block_end(lines, start_line - 1)
            block = "\n".join(lines[start_line - 1:end_line])
            chunks.append({
                "file_path": file_path,
                "chunk_type": chunk_type,
                "name": name,
                "content": block,
                "start_line": start_line,
                "end_line": end_line,
                "language": lang,
                "metadata": {"parser": "regex"},
            })

    if not chunks:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": lang,
            "metadata": {"parser": "regex"},
        })
    return chunks


def _parse_generic_file(content, file_path, ext):
    """Fallback parser: chunk by logical blocks."""
    lines = content.split("\n")
    lang = ext.lstrip(".")
    chunks = []

    current_block = []
    block_start = 1
    for i, line in enumerate(lines, 1):
        if line.strip() == "" and current_block:
            block_content = "\n".join(current_block)
            if len(block_content.strip()) > 20:
                chunks.append({
                    "file_path": file_path,
                    "chunk_type": "block",
                    "name": f"block_{block_start}",
                    "content": block_content,
                    "start_line": block_start,
                    "end_line": i - 1,
                    "language": lang,
                    "metadata": {"parser": "generic"},
                })
            current_block = []
            block_start = i + 1
        else:
            current_block.append(line)

    if current_block:
        block_content = "\n".join(current_block)
        if len(block_content.strip()) > 20:
            chunks.append({
                "file_path": file_path,
                "chunk_type": "block",
                "name": f"block_{block_start}",
                "content": block_content,
                "start_line": block_start,
                "end_line": len(lines),
                "language": lang,
                "metadata": {"parser": "generic"},
            })

    if not chunks:
        chunks.append({
            "file_path": file_path,
            "chunk_type": "module",
            "name": os.path.basename(file_path),
            "content": content,
            "start_line": 1,
            "end_line": len(lines),
            "language": lang,
            "metadata": {"parser": "generic"},
        })
    return chunks


def _find_block_end(lines, start_idx):
    """Find the end of a brace-delimited block."""
    depth = 0
    started = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
        if started and depth <= 0:
            return i + 1
    return len(lines)
