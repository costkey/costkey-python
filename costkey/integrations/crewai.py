"""CrewAI integration (``pip install "costkey[crewai]"``).

Create the listener once, before kicking off crews or flows::

    from costkey.integrations.crewai import CostKeyCrewAIListener
    listener = CostKeyCrewAIListener()  # registers on crewai_event_bus

CrewAI's own paired events become spans: crew kickoff/train/test and flows
are ``workflow`` spans, flow methods ``step`` spans, agent executions
``agent`` spans, tasks ``task`` spans, tool usages and MCP tool executions
``tool`` spans, memory and knowledge queries ``retrieval`` spans and
delegations ``handoff`` spans. Each span keeps CrewAI's event timestamps and
its parent from ``parent_event_id``. LLM call events get no span: the HTTP
capture records the calls and attributes them to the innermost CrewAI scope
(read from CrewAI's event scope stack).

CrewAI runs synchronous handlers on a thread pool, so start and end events
can be handled out of order; span ids are derived from CrewAI's event ids so
the tree is the same either way.
"""
from __future__ import annotations

import datetime
import logging
import threading
import uuid
from typing import Any

from crewai.events import BaseEventListener
from crewai.events.base_events import BaseEvent
from crewai.events.event_context import VALID_EVENT_PAIRS, _event_id_stack

from costkey.patch import (
    _span_closed,
    _span_opened,
    emit_lifecycle,
    emit_span,
    framework_attributes,
    register_span_resolver,
    span_sequence,
)

logger = logging.getLogger("costkey")

FRAMEWORK = "crewai"
_TRACE_NAMESPACE = uuid.UUID("6f1c3a52-5d0e-4c1b-9a51-0c0574c0a1e2")

# Starting event type → span kind. Event types not listed get no span.
_KINDS = {
    "crew_kickoff_started": "workflow",
    "crew_train_started": "workflow",
    "crew_test_started": "workflow",
    "flow_started": "workflow",
    "method_execution_started": "step",
    "agent_execution_started": "agent",
    "lite_agent_execution_started": "agent",
    "agent_evaluation_started": "judge",
    "agent_reasoning_started": "step",
    "task_started": "task",
    "tool_usage_started": "tool",
    "mcp_tool_execution_started": "tool",
    "mcp_connection_started": "custom",
    "llm_guardrail_started": "guardrail",
    "memory_retrieval_started": "retrieval",
    "memory_save_started": "custom",
    "memory_query_started": "retrieval",
    "knowledge_query_started": "retrieval",
    "knowledge_search_query_started": "retrieval",
    "a2a_delegation_started": "handoff",
    "a2a_conversation_started": "handoff",
    "a2a_server_task_started": "task",
    "a2a_parallel_delegation_started": "handoff",
}
_NAME_FIELDS = ("crew_name", "flow_name", "method_name", "tool_name", "agent_role", "task_name", "server_name")
_RAW_FIELDS = ("agent_id", "agent_role", "task_id", "task_name", "tool_name", "crew_name", "flow_name",
               "method_name", "server_name", "emission_sequence")


def _span_id(event_id: str) -> str:
    return f"sp_crewai_{event_id}"


def _iso(value: Any) -> str:
    if isinstance(value, datetime.datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc)
    return stamp.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _event_type(cls: type) -> str | None:
    field = getattr(cls, "model_fields", {}).get("type")
    default = getattr(field, "default", None)
    return default if isinstance(default, str) else None


def _event_classes() -> list[type]:
    found: list[type] = []
    pending = list(BaseEvent.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def _context() -> dict[str, Any]:
    from costkey.client import _current_context

    return _current_context()


def _trace_for(stack: tuple[Any, ...], fallback_event_id: str, ctx: dict[str, Any]) -> tuple[str, bool]:
    """The trace a CrewAI scope belongs to, and whether CrewAI's root owns it."""
    if ctx.get("traceId"):
        return str(ctx["traceId"]), False
    root = stack[0][0] if stack else fallback_event_id
    return uuid.uuid5(_TRACE_NAMESPACE, str(root)).hex, True


class CostKeyCrewAIListener(BaseEventListener):
    """A CrewAI event listener that records CostKey spans."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[str, dict[str, Any]] = {}
        self._early_ends: dict[str, Any] = {}
        super().__init__()
        register_span_resolver(self._current)

    def setup_listeners(self, crewai_event_bus: Any) -> None:
        for cls in _event_classes():
            event_type = _event_type(cls)
            if event_type in _KINDS:
                crewai_event_bus.on(cls)(self._on_start)
            elif event_type is not None and VALID_EVENT_PAIRS.get(event_type) in _KINDS:
                crewai_event_bus.on(cls)(self._on_end)

    # ── handlers (run on CrewAI's handler pool) ──

    def _on_start(self, source: Any, event: Any) -> None:
        try:
            self._start(event)
        except Exception:
            logger.debug("[costkey] crewai start handler failed", exc_info=True)

    def _on_end(self, source: Any, event: Any) -> None:
        try:
            started = getattr(event, "started_event_id", None)
            if not isinstance(started, str):
                return
            with self._lock:
                record = self._open.pop(started, None)
                if record is None:
                    self._early_ends[started] = event
                    return
            self._finish(record, event)
        except Exception:
            logger.debug("[costkey] crewai end handler failed", exc_info=True)

    def _start(self, event: Any) -> None:
        ctx = _context()
        event_id = str(event.event_id)
        trace_id, owns_trace = _trace_for(_event_id_stack.get(), event_id, ctx)
        parent_event = getattr(event, "parent_event_id", None)
        agent_role = getattr(getattr(event, "agent", None), "role", None)
        name = next(
            (str(getattr(event, field)) for field in _NAME_FIELDS if getattr(event, field, None)),
            str(agent_role) if isinstance(agent_role, str) and agent_role else str(event.type),
        )
        attributes = {
            "event_type": event.type,
            "event_id": event_id,
            **{field: getattr(event, field) for field in _RAW_FIELDS
               if isinstance(getattr(event, field, None), (str, int, float, bool))},
        }
        span_id = _span_id(event_id)
        context = {**ctx, "traceId": trace_id, "spanId": span_id}
        if owns_trace:
            context.setdefault("traceName", name)
        record = {
            "span_id": span_id,
            "trace_id": trace_id,
            "trace_name": context.get("traceName"),
            "parent_span_id": _span_id(parent_event) if isinstance(parent_event, str) else ctx.get("spanId"),
            "execution_id": ctx.get("executionId"),
            "kind": _KINDS[event.type],
            "name": name,
            "started_at": _iso(getattr(event, "timestamp", None)),
            "context": context,
            "attributes": attributes,
            "root": owns_trace and not isinstance(parent_event, str),
        }
        if span_sequence(span_id) == 0:
            _span_opened(span_id)
        if record["root"]:
            emit_lifecycle(entity_type="trace", entity_id=trace_id, lifecycle="started", context=context,
                           trace_id=trace_id, trace_name=record["trace_name"], started_at=record["started_at"])
        emit_span(**self._base(record), status="running", attributes=framework_attributes(FRAMEWORK, attributes))
        with self._lock:
            early = self._early_ends.pop(event_id, None)
            if early is None:
                self._open[event_id] = record
                return
        self._finish(record, early)

    @staticmethod
    def _base(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in (
            "span_id", "trace_id", "trace_name", "parent_span_id", "execution_id",
            "kind", "name", "started_at", "context",
        )}

    def _finish(self, record: dict[str, Any], event: Any) -> None:
        event_type = str(getattr(event, "type", ""))
        # CrewAI reports an error message, not an exception; the raw ending
        # event type stands in for the error name.
        failed = event_type.endswith(("_failed", "_error", "_canceled"))
        ended_at = _iso(getattr(event, "timestamp", None))
        started = datetime.datetime.fromisoformat(record["started_at"].replace("Z", "+00:00"))
        ended = datetime.datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        duration = (ended - started).total_seconds() * 1000
        emit_span(
            **self._base(record), status="error" if failed else "ok", ended_at=ended_at,
            duration_ms=duration, error_type=event_type if failed else None,
            attributes=framework_attributes(FRAMEWORK, record["attributes"]),
        )
        _span_closed(record["span_id"])
        if record["root"]:
            emit_lifecycle(entity_type="trace", entity_id=record["trace_id"],
                           lifecycle="failed" if failed else "completed", context=record["context"],
                           trace_id=record["trace_id"], trace_name=record["trace_name"],
                           started_at=record["started_at"], ended_at=ended_at, duration_ms=duration,
                           error_type=event_type if failed else None)

    # ── attribution of in-process calls ──

    def _current(self) -> dict[str, Any] | None:
        stack = _event_id_stack.get()
        for event_id, event_type in reversed(stack):
            if event_type in _KINDS:
                span_id = _span_id(str(event_id))
                if span_sequence(span_id) == 0:
                    _span_opened(span_id)
                trace_id, _ = _trace_for(stack, str(event_id), _context())
                return {"spanId": span_id, "traceId": trace_id}
        return None


__all__ = ["CostKeyCrewAIListener"]
