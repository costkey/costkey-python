"""Strands Agents integration (``pip install "costkey[strands]"``).

Pass the hook provider to an agent (or a Graph/Swarm)::

    from costkey.integrations.strands import CostKeyHooks

    agent = Agent(model=model, tools=[...], hooks=[CostKeyHooks()])

Each invocation becomes an ``agent`` span, each tool call a ``tool`` span,
and each multi-agent node a ``step`` span inside a ``workflow`` span. Model
calls get no span: the HTTP capture (httpx, or boto3 for Bedrock) records
them inside the invocation.
"""
from __future__ import annotations

from typing import Any

from strands.hooks import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
)

from costkey.integrations._tracker import SpanTracker

FRAMEWORK = "strands"

try:  # Multi-agent hooks exist in newer Strands releases.
    from strands.hooks import (
        AfterMultiAgentInvocationEvent,
        AfterNodeCallEvent,
        BeforeMultiAgentInvocationEvent,
        BeforeNodeCallEvent,
    )
except ImportError:  # pragma: no cover - older Strands
    AfterMultiAgentInvocationEvent = AfterNodeCallEvent = None  # type: ignore[assignment,misc]
    BeforeMultiAgentInvocationEvent = BeforeNodeCallEvent = None  # type: ignore[assignment,misc]


def _agent_name(agent: Any) -> str:
    name = getattr(agent, "name", None)
    return name if isinstance(name, str) and name else "agent"


def _model_id(agent: Any) -> Any:
    config = getattr(getattr(agent, "model", None), "config", None)
    if isinstance(config, dict):
        return config.get("model_id")
    return None


class CostKeyHooks(HookProvider):
    """A Strands HookProvider that records CostKey spans."""

    def __init__(self) -> None:
        self._tracker = SpanTracker(FRAMEWORK)

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before_invocation)
        registry.add_callback(AfterInvocationEvent, self._after_invocation)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        if BeforeMultiAgentInvocationEvent is not None:
            registry.add_callback(BeforeMultiAgentInvocationEvent, self._before_multi)
            registry.add_callback(AfterMultiAgentInvocationEvent, self._after_multi)
            registry.add_callback(BeforeNodeCallEvent, self._before_node)
            registry.add_callback(AfterNodeCallEvent, self._after_node)

    # Keys: one open invocation per agent object, one span per tool use.

    def _before_invocation(self, event: Any) -> None:
        agent = event.agent
        self._tracker.start(
            ("agent", id(agent)), _agent_name(agent), "agent",
            attributes={"model": _model_id(agent), "agent_id": getattr(agent, "agent_id", None)},
        )

    def _after_invocation(self, event: Any) -> None:
        result = getattr(event, "result", None)
        stop = getattr(result, "stop_reason", None)
        self._tracker.end(
            ("agent", id(event.agent)),
            attributes={"stop_reason": stop} if isinstance(stop, str) else None,
        )

    def _before_tool(self, event: Any) -> None:
        tool_use = event.tool_use or {}
        tool_id = tool_use.get("toolUseId")
        self._tracker.start(
            ("tool", tool_id), str(tool_use.get("name") or "tool"), "tool",
            parent=("agent", id(event.agent)),
            attributes={"tool_use_id": tool_id},
            input=tool_use.get("input"),
        )

    def _after_tool(self, event: Any) -> None:
        tool_use = event.tool_use or {}
        result = getattr(event, "result", None)
        status = result.get("status") if isinstance(result, dict) else None
        self._tracker.end(
            ("tool", tool_use.get("toolUseId")),
            error=getattr(event, "exception", None),
            attributes={"status": status} if isinstance(status, str) else None,
        )

    def _before_multi(self, event: Any) -> None:
        source = event.source
        self._tracker.start(("multi", id(source)), type(source).__name__, "workflow")

    def _after_multi(self, event: Any) -> None:
        self._tracker.end(("multi", id(event.source)))

    def _before_node(self, event: Any) -> None:
        self._tracker.start(
            ("node", id(event.source), event.node_id), str(event.node_id), "step",
            parent=("multi", id(event.source)), attributes={"node_id": event.node_id},
        )

    def _after_node(self, event: Any) -> None:
        self._tracker.end(("node", id(event.source), event.node_id))


__all__ = ["CostKeyHooks"]
