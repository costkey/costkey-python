"""Claude Agent SDK integration (``pip install "costkey[claude-agent-sdk]"``).

The Claude Agent SDK runs the model loop in a separate Claude Code process,
so its model calls never pass through this process's HTTP clients. This
integration reads the SDK's message stream and hooks instead::

    from claude_agent_sdk import ClaudeAgentOptions, query
    from costkey.integrations.claude_agent_sdk import instrument

    claude = instrument()
    options = claude.with_hooks(ClaudeAgentOptions(model="claude-sonnet-4-5"))
    async for message in query(prompt="Fix the failing test", options=options):
        ...

- Each session becomes an ``agent`` span, each subagent (``SubagentStart``/
  ``SubagentStop``) an ``agent`` span and each tool use (``PreToolUse``/
  ``PostToolUse``/``PostToolUseFailure``) a ``tool`` span.
- Every message that carries usage is sent as a call event whose
  ``telemetry`` follows contract detail 8: ``source = "claude-agent-sdk"``,
  ``request`` null and the skeleton of the raw CLI message (usage included)
  in ``response.body``. Nothing is summed, deduplicated or priced here.

``instrument()`` reads the raw CLI messages by wrapping
``claude_agent_sdk._internal.query.Query.receive_messages`` (verified against
claude-agent-sdk 0.2.165); the hooks are the SDK's public hook API.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import threading
import uuid
from typing import Any, Mapping

from costkey import telemetry as _telemetry
from costkey.integrations._tracker import SpanTracker

logger = logging.getLogger("costkey")

FRAMEWORK = "claude_agent_sdk"
SOURCE = "claude-agent-sdk"


def _has_usage(message: Any) -> bool:
    """Whether a raw message carries usage: ``usage`` or ``modelUsage`` on the
    message itself, ``message``, ``event`` or ``event.message``."""
    if not isinstance(message, Mapping):
        return False
    event = message.get("event")
    candidates = (
        message,
        message.get("message"),
        event,
        event.get("message") if isinstance(event, Mapping) else None,
    )
    return any(
        isinstance(candidate, Mapping) and ("usage" in candidate or "modelUsage" in candidate)
        for candidate in candidates
    )


class ClaudeAgentTracker:
    """Turns Claude Agent SDK hooks and raw messages into CostKey telemetry."""

    def __init__(self) -> None:
        self._tracker = SpanTracker(FRAMEWORK)
        self._lock = threading.Lock()

    # ── spans ──

    def _ensure_session(self, session_id: Any, attributes: Mapping[str, Any] | None = None) -> None:
        if not isinstance(session_id, str) or not session_id:
            return
        key = ("session", session_id)
        with self._lock:
            if self._tracker.is_open(key):
                return
            self._tracker.start(
                key, "claude-agent", "agent",
                attributes={"session_id": session_id, **dict(attributes or {})},
                detached=True,
            )

    def _session_key(self, session_id: Any) -> tuple[str, Any]:
        return ("session", session_id)

    async def _pre_tool_use(self, data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        data = data or {}
        session_id = data.get("session_id")
        self._ensure_session(session_id)
        agent_id = data.get("agent_id")
        parent = ("subagent", agent_id) if agent_id else self._session_key(session_id)
        call_id = data.get("tool_use_id") or tool_use_id
        self._tracker.start(
            ("tool", call_id), str(data.get("tool_name") or "tool"), "tool",
            parent=parent, detached=True,
            attributes={"tool_use_id": call_id, "agent_id": agent_id, "agent_type": data.get("agent_type")},
            input=data.get("tool_input"),
        )
        return {}

    async def _post_tool_use(self, data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        data = data or {}
        call_id = data.get("tool_use_id") or tool_use_id
        self._tracker.end(("tool", call_id), output=data.get("tool_response"))
        return {}

    async def _post_tool_use_failure(self, data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        data = data or {}
        call_id = data.get("tool_use_id") or tool_use_id
        # The hook reports an error message, not an exception class; the raw
        # hook event name stands in for it.
        self._tracker.end(
            ("tool", call_id), error="PostToolUseFailure",
            attributes={"is_interrupt": bool(data.get("is_interrupt"))},
        )
        return {}

    async def _subagent_start(self, data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        data = data or {}
        session_id = data.get("session_id")
        self._ensure_session(session_id)
        agent_id = data.get("agent_id")
        self._tracker.start(
            ("subagent", agent_id), str(data.get("agent_type") or "subagent"), "agent",
            parent=self._session_key(session_id), detached=True,
            attributes={"agent_id": agent_id, "agent_type": data.get("agent_type")},
        )
        return {}

    async def _subagent_stop(self, data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        data = data or {}
        self._tracker.end(("subagent", data.get("agent_id")))
        return {}

    def hooks(self) -> dict[str, list[Any]]:
        """``ClaudeAgentOptions.hooks`` entries recording tool and subagent spans."""
        from claude_agent_sdk import HookMatcher

        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[self._pre_tool_use])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[self._post_tool_use])],
            "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[self._post_tool_use_failure])],
            "SubagentStart": [HookMatcher(matcher=None, hooks=[self._subagent_start])],
            "SubagentStop": [HookMatcher(matcher=None, hooks=[self._subagent_stop])],
        }

    def with_hooks(self, options: Any) -> Any:
        """A copy of ``options`` with CostKey's hooks added to any existing ones."""
        merged: dict[str, list[Any]] = {key: list(value) for key, value in (getattr(options, "hooks", None) or {}).items()}
        for event, matchers in self.hooks().items():
            merged.setdefault(event, []).extend(matchers)
        return dataclasses.replace(options, hooks=merged)

    # ── messages ──

    def observe(self, message: Any) -> None:
        """Record one raw CLI message (a dict from the stream-json protocol)."""
        if not isinstance(message, Mapping):
            return
        try:
            session_id = message.get("session_id")
            if message.get("type") == "system" and message.get("subtype") == "init":
                self._ensure_session(session_id, {
                    "model": message.get("model") if isinstance(message.get("model"), str) else None,
                })
            else:
                self._ensure_session(session_id)
            if _has_usage(message):
                self._emit_usage(message)
            if message.get("type") == "result" and isinstance(session_id, str):
                error = message.get("subtype") if message.get("is_error") else None
                self._tracker.end(self._session_key(session_id), error=error)
        except Exception:
            logger.debug("[costkey] claude-agent-sdk message capture failed", exc_info=True)

    def _emit_usage(self, message: Mapping[str, Any]) -> None:
        from costkey.client import _project_id
        from costkey.patch import _dispatch_event, _scrub, _state, get_context
        from costkey.stack import capture_call_site
        from costkey.types import CostKeyEvent

        if not _project_id() or _state.transport is None:
            return
        parent_tool = message.get("parent_tool_use_id")
        span_context = (
            self._tracker.active_context(("tool", parent_tool)) if parent_tool else None
        ) or self._tracker.active_context(self._session_key(message.get("session_id"))) or {}
        ctx = {**_state.default_context, **get_context(), **span_context}
        if ctx.get("costkeyCapture") in {"manual", "explicit", "disabled"}:
            return
        raw_id = message.get("uuid")
        now = _telemetry.iso_now()
        event = CostKeyEvent(
            id=raw_id if isinstance(raw_id, str) and raw_id else uuid.uuid4().hex,
            timestamp=now,
            project_id=_state.project_id,
            provider=SOURCE,
            url="",
            method="",
            call_site=capture_call_site(),
            context=_scrub(ctx),
            telemetry={
                "v": _telemetry.TELEMETRY_VERSION,
                "source": SOURCE,
                "request": None,
                "response": {
                    "status": None,
                    "headers": {},
                    "body": _telemetry.skeleton(dict(message)),
                    "bytes": None,
                },
                "stream": None,
                "error": None,
                "startedAt": now,
                "durationMs": 0,
            },
            event_type="call",
            source=SOURCE,
            trace_id=ctx.get("traceId"),
            trace_name=ctx.get("traceName"),
            execution_id=ctx.get("executionId"),
            span_id=ctx.get("spanId") if isinstance(ctx.get("spanId"), str) else None,
        )
        _dispatch_event(event)

    def close(self) -> None:
        self._tracker.end_all()


_tracker_instance: ClaudeAgentTracker | None = None
_install_lock = threading.Lock()


def instrument() -> ClaudeAgentTracker:
    """Record Claude Agent SDK sessions; returns the tracker (for ``with_hooks``)."""
    global _tracker_instance
    from claude_agent_sdk._internal.query import Query

    with _install_lock:
        if _tracker_instance is not None:
            return _tracker_instance
        tracker = ClaudeAgentTracker()
        original = Query.receive_messages

        @functools.wraps(original)
        async def receive_messages(self: Any, *args: Any, **kwargs: Any) -> Any:
            async for message in original(self, *args, **kwargs):
                tracker.observe(message)
                yield message

        receive_messages.__costkey_original__ = original  # type: ignore[attr-defined]
        Query.receive_messages = receive_messages
        _tracker_instance = tracker
        return tracker


def uninstrument() -> None:
    """Restore ``Query.receive_messages``."""
    global _tracker_instance
    from claude_agent_sdk._internal.query import Query

    with _install_lock:
        original = getattr(Query.receive_messages, "__costkey_original__", None)
        if original is not None:
            Query.receive_messages = original
        if _tracker_instance is not None:
            _tracker_instance.close()
        _tracker_instance = None


__all__ = ["ClaudeAgentTracker", "instrument", "uninstrument"]
