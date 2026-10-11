"""AG2 integration (``pip install "costkey[ag2]"``), for the ``ag2`` package.

Add the middleware to an agent::

    from costkey.integrations.ag2 import costkey_middleware

    agent = Agent("researcher", config=config, middleware=[costkey_middleware("researcher")])

Each agent turn becomes an ``agent`` span (``on_turn``) and each tool
execution a ``tool`` span (``on_tool_execution``). LLM calls get no span:
the HTTP capture records them inside the turn.
"""
from __future__ import annotations

from typing import Any

from ag2 import Middleware
from ag2.middleware.base import BaseMiddleware

from costkey.client import _SpanScope

FRAMEWORK = "ag2"


class CostKeyMiddleware(BaseMiddleware):
    """AG2 middleware recording turns and tool executions as CostKey spans."""

    def __init__(self, event: Any, context: Any, name: str | None = None) -> None:
        super().__init__(event, context)
        self._name = name or "agent"

    async def on_turn(self, call_next: Any, event: Any, context: Any) -> Any:
        attributes = {"event": type(event).__name__}
        async with _SpanScope(self._name, "agent", attributes=attributes, framework=FRAMEWORK):
            return await call_next(event, context)

    async def on_tool_execution(self, call_next: Any, event: Any, context: Any) -> Any:
        attributes = {"tool_call_id": getattr(event, "id", None)}
        name = getattr(event, "name", None) or "tool"
        async with _SpanScope(str(name), "tool", attributes=attributes,
                              input=getattr(event, "arguments", None), framework=FRAMEWORK) as span:
            result = await call_next(event, context)
            error = type(result).__name__ if type(result).__name__ == "ToolErrorEvent" else None
            if error:
                span.set_attribute("result", error)
            return result


def costkey_middleware(name: str | None = None) -> Middleware:
    """A middleware factory for ``Agent(..., middleware=[...])``."""
    return Middleware(CostKeyMiddleware, name=name)


__all__ = ["CostKeyMiddleware", "costkey_middleware"]
