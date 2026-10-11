"""Pydantic AI integration (``pip install "costkey[pydantic-ai]"``).

Add the capability to an agent::

    from costkey.integrations.pydantic_ai import CostKeyCapability

    agent = Agent("openai:gpt-5", capabilities=[CostKeyCapability()])

Each run becomes an ``agent`` span (``wrap_run``) and each tool execution a
``tool`` span (``wrap_tool_execute``). Model requests get no span: the HTTP
capture records them inside the run or tool span they belong to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

from costkey.client import _SpanScope

FRAMEWORK = "pydantic_ai"


def _model_name(ctx: Any) -> str | None:
    model = getattr(ctx, "model", None)
    for attr in ("model_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class CostKeyCapability(AbstractCapability[Any]):
    """Records agent runs and tool executions as CostKey spans."""

    async def wrap_run(self, ctx: Any, *, handler: Any) -> Any:
        agent = getattr(ctx, "agent", None)
        name = getattr(agent, "name", None) or "agent"
        attributes = {
            "run_id": getattr(ctx, "run_id", None),
            "model": _model_name(ctx),
        }
        async with _SpanScope(str(name), "agent", attributes=attributes, input=getattr(ctx, "prompt", None),
                              framework=FRAMEWORK) as span:
            result = await handler()
            span.set_output(getattr(result, "output", None))
            return result

    async def wrap_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        attributes = {
            "tool_call_id": getattr(call, "tool_call_id", None),
            "run_step": getattr(ctx, "run_step", None),
        }
        name = getattr(call, "tool_name", None) or getattr(tool_def, "name", None) or "tool"
        async with _SpanScope(str(name), "tool", attributes=attributes, input=args, framework=FRAMEWORK) as span:
            result = await handler(args)
            span.set_output(result)
            return result


__all__ = ["CostKeyCapability"]
