"""OpenAI Agents SDK integration.

Pass the hooks to a run::

    from costkey.integrations.openai_agents import CostKeyRunHooks

    result = await Runner.run(agent, input, hooks=CostKeyRunHooks())

Agent runs, tool calls and handoffs become spans, and every model call the
run makes is attributed to the agent or tool it ran inside. Hooks fire even
with ``RunConfig(tracing_disabled=True)``, so this does not depend on the
Agents SDK's own tracing.
"""
from __future__ import annotations

import logging
from typing import Any

from costkey.client import _SpanScope

logger = logging.getLogger("costkey")

try:  # pragma: no cover - exercised only when the framework is installed
    from agents import RunHooks as _RunHooksBase
except Exception:  # The framework is optional; hooks are duck-typed anyway.
    _RunHooksBase = object  # type: ignore[assignment,misc]


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
    """RunHooks that record agent, tool and handoff spans."""

    def __init__(self) -> None:
        super().__init__()
        self._open: dict[tuple[Any, ...], list[_SpanScope]] = {}

    def _push(self, key: tuple[Any, ...], scope: _SpanScope) -> None:
        try:
            scope.__enter__()
        except Exception:
            logger.debug("[costkey] could not open span", exc_info=True)
            return
        self._open.setdefault(key, []).append(scope)

    def _pop(self, key: tuple[Any, ...], output: Any = None) -> None:
        stack = self._open.get(key)
        if not stack:
            return
        scope = stack.pop()
        if not stack:
            self._open.pop(key, None)
        try:
            if output is not None and scope._handle is not None:
                scope._handle.set_output(output)
            scope.__exit__(None, None, None)
        except Exception:
            logger.debug("[costkey] could not close span", exc_info=True)

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        attributes: dict[str, Any] = {}
        model = _model_name(agent)
        if model:
            attributes["model"] = model
        self._push(("agent", id(context), id(agent)), _SpanScope(_name(agent, "agent"), "agent", attributes=attributes))

    async def on_agent_end(self, context: Any, agent: Any, output: Any = None) -> None:
        self._pop(("agent", id(context), id(agent)), output)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self._push(("tool", id(context), id(tool)), _SpanScope(_name(tool, "tool"), "tool"))

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any = None) -> None:
        self._pop(("tool", id(context), id(tool)), result)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        scope = _SpanScope(
            f"{_name(from_agent, 'agent')} → {_name(to_agent, 'agent')}",
            "handoff",
            attributes={"from": _name(from_agent, "agent"), "to": _name(to_agent, "agent")},
        )
        try:
            with scope:
                pass
        except Exception:
            logger.debug("[costkey] could not record handoff", exc_info=True)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        # Model calls are captured at the HTTP layer with the active span.
        return None

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        return None
