"""browser-use integration.

Wrap the run in an agent span and pass the step hooks::

    from costkey.integrations.browser_use import step_hooks

    on_step_start, on_step_end = step_hooks()
    with costkey.agent("browser-agent", user_id=user_id):
        await agent.run(on_step_start=on_step_start, on_step_end=on_step_end)

Every step becomes a span with its step number, and the vision call plus any
reviewer calls made during the step are attributed to it. Label a reviewer
pass with ``costkey.span("action review", kind="judge")``.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from costkey.client import _SpanScope

logger = logging.getLogger("costkey")

StepHook = Callable[[Any], Awaitable[None]]


def _step_number(agent: Any) -> int | None:
    state = getattr(agent, "state", None)
    for holder in (state, agent):
        value = getattr(holder, "n_steps", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def step_hooks(name: str = "step") -> tuple[StepHook, StepHook]:
    """Return ``(on_step_start, on_step_end)`` hooks that record step spans."""
    open_steps: dict[int, list[_SpanScope]] = {}

    async def on_step_start(agent: Any) -> None:
        number = _step_number(agent)
        scope = _SpanScope(name, "step", attributes={"step": number} if number is not None else None)
        try:
            scope.__enter__()
        except Exception:
            logger.debug("[costkey] could not open step span", exc_info=True)
            return
        open_steps.setdefault(id(agent), []).append(scope)

    async def on_step_end(agent: Any) -> None:
        stack = open_steps.get(id(agent))
        if not stack:
            return
        scope = stack.pop()
        if not stack:
            open_steps.pop(id(agent), None)
        try:
            scope.__exit__(None, None, None)
        except Exception:
            logger.debug("[costkey] could not close step span", exc_info=True)

    return on_step_start, on_step_end
