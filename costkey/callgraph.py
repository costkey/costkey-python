"""Static call graph: raw functions and call expressions, scanned at init.

The SDK sends what the source says: each function's line, the functions it
calls and the expressions it calls through. It does not score functions,
count fan-in, measure distances or decide which calls are AI calls; the
server does all of that (contract detail 6).
"""
from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any

import httpx
from costkey.version import __version__

logger = logging.getLogger("costkey")

_SKIP_DIRS = frozenset({
    "venv", ".venv", "env", "__pycache__", "site-packages",
    ".git", "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build",
})

_MAX_FILES = 500
_MAX_CALLS_PER_FUNCTION = 64
_MAX_EXPRESSION_CHARS = 200
_MAX_PAYLOAD_BYTES = 1024 * 1024


def discover_files(root: str) -> list[str]:
    """Walk directory tree finding .py files, skipping common non-project dirs."""
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        )
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                result.append(os.path.join(dirpath, fname))
                if len(result) >= _MAX_FILES:
                    return result
    return result


def _expression(node: ast.expr) -> str:
    """The callee as written: ``client.chat.completions.create``. Calls in the
    chain render as ``()``; anything else (subscripts, literals) as ``?``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append(_expression(node.func) + "()")
    else:
        parts.append("?")
    return ".".join(reversed(parts))[:_MAX_EXPRESSION_CHARS]


class _CallVisitor(ast.NodeVisitor):
    """Collects function definitions with the calls made in their bodies."""

    def __init__(self) -> None:
        self.functions: dict[str, dict[str, Any]] = {}
        self._current_func: str | None = None
        self._current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = prev

    def _visit_funcdef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = f"{self._current_class}.{node.name}" if self._current_class else node.name
        prev = self._current_func
        self._current_func = name
        self.functions.setdefault(name, {"line": node.lineno, "names": [], "expressions": []})
        self.generic_visit(node)
        self._current_func = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_funcdef(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_funcdef(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is not None:
            info = self.functions[self._current_func]
            if isinstance(node.func, ast.Attribute):
                info["names"].append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                info["names"].append(node.func.id)
            info["expressions"].append(_expression(node.func))
        self.generic_visit(node)


def build_call_graph(files: list[str], project_root: str) -> dict[str, dict[str, Any]]:
    """``{"path:function": {"line", "callees", "callExpressions"}}`` for the files."""
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    by_name: dict[str, list[str]] = {}
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
                tree = ast.parse(handle.read(), filename=filepath)
        except (OSError, SyntaxError, ValueError):
            continue
        visitor = _CallVisitor()
        visitor.visit(tree)
        rel = os.path.relpath(filepath, project_root).replace(os.sep, "/")
        parsed[rel] = visitor.functions
        for func_name in visitor.functions:
            key = f"{rel}:{func_name}"
            by_name.setdefault(func_name.split(".")[-1], []).append(key)

    graph: dict[str, dict[str, Any]] = {}
    for rel, functions in parsed.items():
        for func_name, info in functions.items():
            key = f"{rel}:{func_name}"
            callees: list[str] = []
            for name in info["names"]:
                candidates = by_name.get(name, [])
                # A definition in the same file wins; otherwise the first one.
                same_file = [c for c in candidates if c.startswith(rel + ":")]
                target = (same_file or candidates or [None])[0]
                if target and target != key and target not in callees:
                    callees.append(target)
            graph[key] = {
                "line": info["line"],
                "callees": callees[:_MAX_CALLS_PER_FUNCTION],
                "callExpressions": info["expressions"][:_MAX_CALLS_PER_FUNCTION],
            }
    return graph


def build_payload(graph: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The upload body, capped in size by dropping functions from the end."""
    functions = dict(sorted(graph.items()))
    payload: dict[str, Any] = {"sdkVersion": f"python-{__version__}", "functions": functions}
    encoded = len(json.dumps(payload).encode("utf-8"))
    if encoded <= _MAX_PAYLOAD_BYTES:
        return payload
    payload["truncated"] = True
    for key in reversed(list(functions)):
        encoded -= len(json.dumps({key: functions[key]}).encode("utf-8"))
        del functions[key]
        if encoded <= _MAX_PAYLOAD_BYTES:
            break
    return payload


def scan_and_send(
    base_url: str,
    auth_key: str,
    project_id: str,
    project_root: str,
    debug: bool,
) -> None:
    """Discover files, build the raw call graph and POST it to the server."""
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
        payload = build_payload(graph)
        url = f"{base_url}/api/v1/projects/{project_id}/callgraph"
        from costkey.patch import internal_http
        with internal_http():
            resp = httpx.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {auth_key}",
                    "User-Agent": f"costkey-python/{__version__}",
                },
                timeout=10,
            )
        if debug:
            if resp.is_success:
                logger.info(f"[costkey] Call graph sent ({len(payload['functions'])} functions)")
            else:
                logger.warning(f"[costkey] Call graph upload returned {resp.status_code}")
    except Exception as e:
        if debug:
            logger.warning(f"[costkey] Call graph scan failed: {e}")
