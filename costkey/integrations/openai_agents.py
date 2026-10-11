"""OpenAI Agents SDK integration (``pip install "costkey[openai-agents]"``).

Two ways in; use one, not both.

A tracing processor, so every run is covered without passing anything::

    from costkey.integrations.openai_agents import instrument
    instrument()  # agents.add_trace_processor(CostKeyTracingProcessor())

Traces become workflow spans; agent, tool (function), handoff, guardrail,
task, turn and MCP list-tools spans become CostKey spans. Model calls
(generation/response spans) get no span of their own: the HTTP capture
already records them, attributed to the agent or tool span they ran in.
Tracing processors only run while the Agents SDK's tracing is enabled.

Or run hooks, which fire even with ``RunConfig(tracing_disabled=True)``::

    from costkey.integrations.openai_agents import CostKeyRunHooks
    result = await Runner.run(agent, input, hooks=CostKeyRunHooks())

The SDK runs hooks in child tasks, so with hooks the span tree is recorded
but model calls are attributed to the span around ``Runner.run`` rather than
to each agent; the tracing processor attributes them per turn.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from costkey.integrations._tracker import SpanTracker

logger = logging.getLogger("costkey")

try:  # pragma: no cover - exercised only when the framework is installed
    from agents import RunHooks as _RunHooksBase
except Exception:  # The framework is optional; hooks are duck-typed anyway.
    _RunHooksBase = object  # type: ignore[assignment,misc]

try:  # pragma: no cover - exercised only when the framework is installed
    from agents.tracing import TracingProcessor as _TracingProcessorBase
except Exception:
    _TracingProcessorBase = object  # type: ignore[assignment,misc]

FRAMEWORK = "openai_agents"


def _name(obj: Any, fallback: str) -> str:
    value = getattr(obj, "name", None)
    return value if isinstance(value, str) and value else fallback


def _model_name(agent: Any) -> str | None:
    model = getattr(agent, "model", None)
    if isinstance(model, str):
        return model
    name = getattr(model, "model", None)
    return name if isinstance(name, str) else None


class CostKeyRunHooks(_RunHooksBase):  # type: ignore[misc,valid-type]
    """RunHooks that record agent, tool and handoff spans.

    The Agents SDK awaits run hooks through ``asyncio.gather``, i.e. in child
    tasks, so a hook cannot change the context the run continues in. These
    spans are therefore detached and linked by explicit parents (tool →
    agent). Model calls are attributed to the span active around
    ``Runner.run``; use :func:`instrument` (the tracing processor) to have
    them attributed to the agent turn that made them.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tracker = SpanTracker(FRAMEWORK)
        self._agents: dict[int, list[tuple[Any, ...]]] = {}
        self._runs = 0

    @staticmethod
    def _tool_key(context: Any, tool: Any) -> tuple[Any, ...]:
        call_id = getattr(context, "tool_call_id", None)
        return ("tool", call_id) if isinstance(call_id, str) and call_id else ("tool", id(context), id(tool))

    def _current_agent(self, agent: Any) -> tuple[Any, ...] | None:
        stack = self._agents.get(id(agent))
        return stack[-1] if stack else None

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        # The start and end hooks receive different context objects; an
        # agent's runs nest, so they pair up per agent object.
        self._runs += 1
        key = ("agent", id(agent), self._runs)
        attributes: dict[str, Any] = {}
        model = _model_name(agent)
        if model:
            attributes["model"] = model
        self._tracker.start(key, _name(agent, "agent"), "agent", attributes=attributes, detached=True)
        self._agents.setdefault(id(agent), []).append(key)

    async def on_agent_end(self, context: Any, agent: Any, output: Any = None) -> None:
        stack = self._agents.get(id(agent))
        if not stack:
            return
        key = stack.pop()
        if not stack:
            self._agents.pop(id(agent), None)
        self._tracker.end(key, output=output)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        call_id = getattr(context, "tool_call_id", None)
        self._tracker.start(
            self._tool_key(context, tool), _name(tool, "tool"), "tool",
            parent=self._current_agent(agent), detached=True,
            attributes={"tool_call_id": call_id} if isinstance(call_id, str) else None,
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any = None) -> None:
        self._tracker.end(self._tool_key(context, tool), output=result)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        key = ("handoff", id(context), id(from_agent), id(to_agent))
        self._tracker.start(
            key, f"{_name(from_agent, 'agent')} → {_name(to_agent, 'agent')}", "handoff",
            parent=self._current_agent(from_agent), detached=True,
            attributes={"from": _name(from_agent, "agent"), "to": _name(to_agent, "agent")},
        )
        self._tracker.end(key)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        # Model calls are captured at the HTTP layer.
        return None

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        return None


# Span data types → CostKey span kinds. Model-call span types are absent on
# purpose: the HTTP capture records those calls.
_KINDS = {
    "agent": "agent",
    "function": "tool",
    "handoff": "handoff",
    "guardrail": "guardrail",
    "custom": "custom",
    "task": "workflow",
    "turn": "step",
    "mcp_tools": "tool",
    "speech_group": "custom",
}
_MODEL_SPAN_TYPES = frozenset({"generation", "response", "transcription", "speech"})


def _span_type(span: Any) -> str:
    data = getattr(span, "span_data", None)
    try:
        return str(data.type)
    except Exception:
        return "custom"


def _span_name(data: Any, span_type: str) -> str:
    if span_type == "handoff":
        return f"{getattr(data, 'from_agent', None) or 'agent'} → {getattr(data, 'to_agent', None) or 'agent'}"
    if span_type == "turn":
        return f"turn {getattr(data, 'turn', '')}".strip()
    if span_type == "mcp_tools":
        return f"list tools ({getattr(data, 'server', None) or 'mcp'})"
    name = getattr(data, "name", None)
    return name if isinstance(name, str) and name else span_type


def _span_attributes(span: Any, data: Any, span_type: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "type": span_type,
        "span_id": getattr(span, "span_id", None),
        "trace_id": getattr(span, "trace_id", None),
    }
    if span_type == "handoff":
        attributes["from"] = getattr(data, "from_agent", None)
        attributes["to"] = getattr(data, "to_agent", None)
    elif span_type == "agent":
        attributes["output_type"] = getattr(data, "output_type", None)
    elif span_type == "turn":
        attributes["turn"] = getattr(data, "turn", None)
        attributes["agent_name"] = getattr(data, "agent_name", None)
    elif span_type == "mcp_tools":
        attributes["server"] = getattr(data, "server", None)
    return {key: value for key, value in attributes.items() if value is None or isinstance(value, (str, int, float, bool))}


class CostKeyTracingProcessor(_TracingProcessorBase):  # type: ignore[misc,valid-type]
    """An ``agents`` TracingProcessor that turns traces and spans into CostKey spans."""

    def __init__(self) -> None:
        self._tracker = SpanTracker(FRAMEWORK)

    def on_trace_start(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", None)
        self._tracker.start(
            ("trace", trace_id), _name(trace, "workflow"), "workflow",
            attributes={"trace_id": trace_id},
        )

    def on_trace_end(self, trace: Any) -> None:
        self._tracker.end(("trace", getattr(trace, "trace_id", None)))

    def on_span_start(self, span: Any) -> None:
        span_type = _span_type(span)
        key = ("span", getattr(span, "span_id", None))
        parent_id = getattr(span, "parent_id", None)
        parent = ("span", parent_id) if parent_id else ("trace", getattr(span, "trace_id", None))
        if span_type in _MODEL_SPAN_TYPES:
            self._tracker.skip(key, parent)
            return
        data = getattr(span, "span_data", None)
        self._tracker.start(
            key, _span_name(data, span_type), _KINDS.get(span_type, "custom"),
            parent=parent,
            attributes=_span_attributes(span, data, span_type),
            input=getattr(data, "input", None) if span_type == "function" else None,
        )

    def on_span_end(self, span: Any) -> None:
        key = ("span", getattr(span, "span_id", None))
        if not self._tracker.is_open(key):
            self._tracker.forget(key)
            return
        data = getattr(span, "span_data", None)
        span_type = _span_type(span)
        attributes: dict[str, Any] = {}
        if span_type == "guardrail":
            attributes["triggered"] = bool(getattr(data, "triggered", False))
        # SpanError carries a message and data, never an exception class.
        error = "SpanError" if getattr(span, "error", None) else None
        self._tracker.end(
            key,
            output=getattr(data, "output", None) if span_type == "function" else None,
            error=error,
            attributes=attributes,
        )

    def shutdown(self) -> None:
        self._tracker.end_all()

    def force_flush(self) -> None:
        try:
            from costkey.client import flush
            flush()
        except Exception:
            logger.debug("[costkey] flush failed", exc_info=True)


_installed: CostKeyTracingProcessor | None = None
_install_lock = threading.Lock()


def instrument() -> CostKeyTracingProcessor:
    """Register one CostKeyTracingProcessor with ``agents.add_trace_processor``."""
    global _installed
    from agents import add_trace_processor

    with _install_lock:
        if _installed is None:
            _installed = CostKeyTracingProcessor()
            add_trace_processor(_installed)
        return _installed


__all__ = ["CostKeyRunHooks", "CostKeyTracingProcessor", "instrument"]
