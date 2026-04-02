"""Static call graph analysis — scans project files at init to enable intelligent cost attribution."""
from __future__ import annotations

import ast
import json
import logging
import os
from collections import deque
from typing import Any

import httpx

logger = logging.getLogger("costkey")

_SKIP_DIRS = frozenset({
    "venv", ".venv", "env", "__pycache__", "site-packages",
    ".git", "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build",
})

_MAX_FILES = 500

# Attribute chains that indicate AI provider calls
_AI_CALL_PATTERNS: list[tuple[str, ...]] = [
    ("messages", "create"),                  # Anthropic
    ("chat", "completions", "create"),       # OpenAI
    ("completions", "create"),               # OpenAI alt
    ("generate_content",),                   # Google
]

_AI_IMPORT_MODULES = frozenset({
    "openai", "anthropic", "google.generativeai", "cohere", "mistralai", "groq",
})


def discover_files(root: str) -> list[str]:
    """Walk directory tree finding .py files, skipping common non-project dirs."""
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter dirs in-place to prune walk
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for fname in filenames:
            if fname.endswith(".py"):
                result.append(os.path.join(dirpath, fname))
                if len(result) >= _MAX_FILES:
                    return result
    return result


class _CallVisitor(ast.NodeVisitor):
    """AST visitor that collects function definitions and their calls."""

    def __init__(self) -> None:
        self.functions: dict[str, dict[str, Any]] = {}
        self.imports_ai: bool = False
        self._current_func: str | None = None
        self._current_class: str | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in _AI_IMPORT_MODULES:
                self.imports_ai = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and any(
            node.module == m or node.module.startswith(m + ".")
            for m in _AI_IMPORT_MODULES
        ):
            self.imports_ai = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = prev

    def _visit_funcdef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._current_class:
            name = f"{self._current_class}.{node.name}"
        else:
            name = node.name

        prev = self._current_func
        self._current_func = name
        self.functions[name] = {
            "lineno": node.lineno,
            "calls_ai": False,
            "callees": [],
            "call_names": [],
        }
        self.generic_visit(node)
        self._current_func = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_funcdef(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_funcdef(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is None:
            self.generic_visit(node)
            return

        func_info = self.functions[self._current_func]

        # Extract the attribute chain from the call
        chain = _extract_attr_chain(node.func)
        if chain:
            # Check against AI patterns
            for pattern in _AI_CALL_PATTERNS:
                plen = len(pattern)
                if len(chain) >= plen and tuple(chain[-plen:]) == pattern:
                    func_info["calls_ai"] = True
                    break

            # Record the call name (last part) for intra-file resolution
            func_info["call_names"].append(chain[-1])
        elif isinstance(node.func, ast.Name):
            func_info["call_names"].append(node.func.id)

        self.generic_visit(node)


def _extract_attr_chain(node: ast.expr) -> list[str]:
    """Extract attribute chain from a node, e.g. client.chat.completions.create -> ['client','chat','completions','create']."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    return parts


def build_call_graph(files: list[str], project_root: str) -> dict[str, dict[str, Any]]:
    """Parse files and build a call graph keyed by 'relative/path.py:function_name'."""
    graph: dict[str, dict[str, Any]] = {}

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except (OSError, IOError):
            continue

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            continue

        visitor = _CallVisitor()
        visitor.visit(tree)

        rel = os.path.relpath(filepath, project_root)
        # Normalize to forward slashes
        rel = rel.replace(os.sep, "/")

        # Collect all function names in this file for intra-file resolution
        all_names = set(visitor.functions.keys())

        for func_name, info in visitor.functions.items():
            key = f"{rel}:{func_name}"
            # Resolve call_names to callees within the same file
            callees: list[str] = []
            for cname in info["call_names"]:
                # Match plain name or ClassName.method
                if cname in all_names and cname != func_name:
                    callees.append(cname)

            graph[key] = {
                "calls_ai": info["calls_ai"],
                "callees": callees,
                "lineno": info["lineno"],
            }

    return graph


def compute_scores(graph: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute fan_in, ai_distance, and score for each function in the graph."""
    # Count fan_in: how many other functions call each function
    fan_in: dict[str, int] = {}
    for key in graph:
        fan_in[key] = 0

    # Build a mapping from short function names to full keys for callee resolution
    # Within the same file prefix
    for key, info in graph.items():
        file_prefix = key.rsplit(":", 1)[0]
        for callee_name in info["callees"]:
            callee_key = f"{file_prefix}:{callee_name}"
            if callee_key in fan_in:
                fan_in[callee_key] += 1

    # BFS to compute ai_distance for each function
    # ai_distance = 0 means it directly calls AI
    # ai_distance = 1 means it calls a function that calls AI, etc.
    ai_distance: dict[str, int] = {}

    # Build reverse adjacency: for each function, who does it call (as full keys)?
    adjacency: dict[str, list[str]] = {}
    for key, info in graph.items():
        file_prefix = key.rsplit(":", 1)[0]
        adj = []
        for callee_name in info["callees"]:
            callee_key = f"{file_prefix}:{callee_name}"
            if callee_key in graph:
                adj.append(callee_key)
        adjacency[key] = adj

    # Seed: functions that directly call AI
    queue: deque[str] = deque()
    for key, info in graph.items():
        if info["calls_ai"]:
            ai_distance[key] = 0
            queue.append(key)

    # Build reverse graph: who calls this function?
    reverse_adj: dict[str, list[str]] = {k: [] for k in graph}
    for key, callees in adjacency.items():
        for callee_key in callees:
            if callee_key in reverse_adj:
                reverse_adj[callee_key].append(key)

    # BFS from AI-calling functions through callers
    while queue:
        current = queue.popleft()
        current_dist = ai_distance[current]
        # Check callers of current (they are 1 hop further from AI)
        for caller in reverse_adj.get(current, []):
            if caller not in ai_distance:
                ai_distance[caller] = current_dist + 1
                queue.append(caller)
        # Also check callees — if current calls something that calls AI
        for callee in adjacency.get(current, []):
            if callee not in ai_distance:
                ai_distance[callee] = current_dist + 1
                queue.append(callee)

    # Compute scores
    result: dict[str, dict[str, Any]] = {}
    for key, info in graph.items():
        dist = ai_distance.get(key)
        if dist is None or dist > 3:
            continue  # Skip irrelevant functions

        fi = fan_in.get(key, 0)
        score = fi * 10
        if info["calls_ai"]:
            score += 5
        if dist == 1:
            score -= 15
        elif dist == 2:
            score -= 10

        result[key] = {
            "line": info["lineno"],
            "score": float(score),
            "fan_in": fi,
            "calls_ai": info["calls_ai"],
            "ai_distance": dist,
        }

    return result


_MAX_PAYLOAD_BYTES = 50 * 1024  # 50KB


def scan_and_send(
    base_url: str,
    auth_key: str,
    project_id: str,
    project_root: str,
    debug: bool,
) -> None:
    """Discover files, build call graph, compute scores, and POST to server."""
    try:
        files = discover_files(project_root)
        if not files:
            if debug:
                logger.info("[costkey] No Python files found for call graph")
            return

        graph = build_call_graph(files, project_root)
        if not graph:
            if debug:
                logger.info("[costkey] No functions found in call graph")
            return

        scored = compute_scores(graph)
        if not scored:
            if debug:
                logger.info("[costkey] No AI-related functions found in call graph")
            return

        # Build payload
        functions: dict[str, dict[str, Any]] = {}
        for key, info in scored.items():
            functions[key] = {
                "line": info["line"],
                "score": info["score"],
                "fan_in": info["fan_in"],
                "calls_ai": info["calls_ai"],
            }

        payload: dict[str, Any] = {
            "sdkVersion": "python-0.2.3",
            "functions": functions,
        }

        # Trim to fit size limit
        payload_json = json.dumps(payload)
        if len(payload_json.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            # Sort by score descending (most interesting first), keep trimming
            sorted_keys = sorted(
                functions.keys(),
                key=lambda k: functions[k]["score"],
                reverse=True,
            )
            while len(payload_json.encode("utf-8")) > _MAX_PAYLOAD_BYTES and sorted_keys:
                remove_key = sorted_keys.pop()  # Remove least interesting
                del functions[remove_key]
                payload["functions"] = functions
                payload_json = json.dumps(payload)

        if not functions:
            return

        url = f"{base_url}/api/v1/projects/{project_id}/callgraph"
        resp = httpx.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {auth_key}",
                "User-Agent": "costkey-python/0.2.3",
            },
            timeout=10,
        )
        if debug:
            if resp.is_success:
                logger.info(f"[costkey] Call graph sent ({len(functions)} functions)")
            else:
                logger.warning(f"[costkey] Call graph upload returned {resp.status_code}")

    except Exception as e:
        if debug:
            logger.warning(f"[costkey] Call graph scan failed: {e}")
