"""Call-site capture: every frame of the stack, raw.

The SDK does not decide which frames are "user code"; the server filters.
Frames are innermost first.

For the first few in-app frames (the caller's own code, not libraries, the
standard library or CostKey itself) the SDK also attaches the source line
and a few lines around it, so the console can show the code that made the
call.
"""
from __future__ import annotations

import linecache
import os
import re
import sys
import sysconfig
import traceback
from contextvars import ContextVar, Token

from costkey.types import CallSite, StackFrame

# How many in-app frames get source context, and how much.
MAX_CONTEXT_FRAMES = 8
CONTEXT_LINES = 3
MAX_LINE_CHARS = 500

_COSTKEY_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _stdlib_dirs() -> tuple[str, ...]:
    dirs: set[str] = set()
    for key in ("stdlib", "platstdlib"):
        try:
            path = sysconfig.get_paths().get(key)
        except Exception:
            path = None
        if path:
            dirs.add(os.path.abspath(path) + os.sep)
    return tuple(dirs)


_STDLIB_DIRS = _stdlib_dirs()


def is_in_app(file_name: str | None) -> bool:
    """True for the caller's own code: not libraries, stdlib or CostKey."""
    if not file_name or file_name.startswith("<"):
        return False
    norm = file_name.replace("\\", "/")
    if "/site-packages/" in norm or "/dist-packages/" in norm or "<frozen" in norm:
        return False
    path = os.path.abspath(file_name)
    if path.startswith(_COSTKEY_DIR):
        return False
    if _STDLIB_DIRS and path.startswith(_STDLIB_DIRS):
        return False
    return True


# Source lines can hold hardcoded secrets. Redact the common shapes before
# they leave the process: provider keys and tokens, URL credentials and query
# strings, and string literals assigned to secret-looking names.
_TOKEN = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}|AIza[A-Za-z0-9_\-]{30,}"
    r"|eyJ[A-Za-z0-9_\-]{20,}(?:\.[A-Za-z0-9_\-]*){0,2}|xox[bpras]-[A-Za-z0-9-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|(?<=Bearer )[A-Za-z0-9._\-]{20,}"
)
_URL_USERINFO = re.compile(r"(://)[^/\s'\"@]+@")
_URL_QUERY = re.compile(r"(://[^\s'\"?#]*\?)[^\s'\"#]*")
_QUERY_SECRET = re.compile(
    r"([?&](?:key|api[_-]?key|token|access[_-]?token|secret|sig|signature|password|auth)=)[^&\s'\"#]+",
    re.IGNORECASE,
)
_SECRET_ASSIGN = re.compile(
    r"((?:api[_-]?key|secret(?:[_-]?key)?|token|access[_-]?token|refresh[_-]?token|password|passwd"
    r"|credential|authorization|private[_-]?key)['\"]?\s*[:=]\s*)(['\"])(?:(?!\2).)+\2",
    re.IGNORECASE,
)
REDACTED = "[REDACTED]"


def scrub_source_line(line: str) -> str:
    line = _TOKEN.sub(REDACTED, line)
    line = _URL_USERINFO.sub(r"\1" + REDACTED + "@", line)
    line = _URL_QUERY.sub(r"\1" + REDACTED, line)
    line = _QUERY_SECRET.sub(r"\1" + REDACTED, line)
    line = _SECRET_ASSIGN.sub(lambda m: m.group(1) + m.group(2) + REDACTED + m.group(2), line)
    return line


def _clip(line: str) -> str:
    line = line.rstrip("\r\n")[:MAX_LINE_CHARS]
    try:
        return scrub_source_line(line)
    except Exception:
        return line


def _attach_context(frame: StackFrame, cache: dict[str, list[str]]) -> bool:
    """Fill the frame's source context. Returns True when it was filled."""
    file_name = frame.file_name
    lineno = frame.line_number
    if not file_name or not lineno or lineno < 1:
        return False
    lines = cache.get(file_name)
    if lines is None:
        try:
            lines = linecache.getlines(file_name)
        except Exception:
            lines = []
        cache[file_name] = lines
    if lineno > len(lines):
        return False
    idx = lineno - 1
    frame.context_line = _clip(lines[idx])
    frame.pre_context = [_clip(l) for l in lines[max(0, idx - CONTEXT_LINES):idx]]
    frame.post_context = [_clip(l) for l in lines[idx + 1:idx + 1 + CONTEXT_LINES]]
    return True


def add_source_context(frames: list[StackFrame]) -> None:
    """Attach source lines to the first in-app frames. Never raises."""
    cache: dict[str, list[str]] = {}
    seen = 0
    for frame in frames:
        if seen >= MAX_CONTEXT_FRAMES:
            break
        try:
            if not is_in_app(frame.file_name):
                continue
            seen += 1
            _attach_context(frame, cache)
        except Exception:
            continue


def _is_entry_point(frame: StackFrame) -> bool:
    """A process entry point (``__main__.py``, a module body) says nothing about which code made a call."""
    base = os.path.basename(frame.file_name or "")
    return base == "__main__.py" or frame.function_name == "<module>"


def has_own_code(frames: list[StackFrame]) -> bool:
    """Whether a stack passes through the caller's own code, beyond the process entry point."""
    return any(is_in_app(frame.file_name) and not _is_entry_point(frame) for frame in frames)


# Where the innermost agent, step or trace was opened in the caller's code.
# Frameworks such as the OpenAI Agents SDK make their model calls on their own
# tasks, whose stacks hold only library code and the server's entry point; those
# calls are attributed to the code that opened the scope they run in.
_scope_site: ContextVar[CallSite | None] = ContextVar("costkey_scope_site", default=None)


def _capture(skip: int) -> CallSite | None:
    try:
        summary = traceback.extract_stack(sys._getframe(skip + 1))
    except Exception:
        return None
    frames = [
        StackFrame(function_name=frame.name, file_name=frame.filename, line_number=frame.lineno)
        for frame in reversed(summary)
    ]
    if not frames:
        return None
    add_source_context(frames)
    return CallSite(raw="".join(traceback.format_list(summary)), frames=frames)


def remember_scope_site() -> Token[CallSite | None] | None:
    """Record where the caller's code opened a scope. Scopes opened by library
    code (integrations) keep the enclosing one's. Returns a token to reset."""
    try:
        site = _capture(2)
        if site is None or not has_own_code(site.frames):
            return None
        return _scope_site.set(site)
    except Exception:
        return None


def forget_scope_site(token: Token[CallSite | None] | None) -> None:
    if token is None:
        return
    try:
        _scope_site.reset(token)
    except ValueError:
        # Closed from another context (framework hooks): nothing to restore here.
        pass


def capture_call_site() -> CallSite | None:
    site = _capture(1)
    if site is not None and has_own_code(site.frames):
        return site
    scope = _scope_site.get()
    return scope if scope is not None else site
