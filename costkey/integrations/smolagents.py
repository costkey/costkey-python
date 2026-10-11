"""smolagents integration (``pip install "costkey[smolagents]"``).

Instrument an agent::

    from costkey.integrations.smolagents import instrument

    agent = instrument(CodeAgent(tools=[...], model=model))
    agent.run("task")

``instrument`` wraps ``agent.run`` in an ``agent`` span and registers a step
callback for ``ActionStep`` and ``PlanningStep``. smolagents only calls step
callbacks once a step has finished, so each step is sent as a completed
``step`` span with the step's own start and end times. Model calls are
captured at the HTTP layer inside the agent span; the server places them in
their step by time.

To register the callback yourself::

    agent = CodeAgent(..., step_callbacks={ActionStep: CostKeyStepCallback(),
                                           PlanningStep: CostKeyStepCallback()})
"""
from __future__ import annotations

import functools
import uuid
from typing import Any

from costkey.client import _SpanScope
from costkey.integrations._tracker import record_completed_span

FRAMEWORK = "smolagents"


def _timing(step: Any) -> tuple[float | None, float | None]:
    timing = getattr(step, "timing", None)
    return getattr(timing, "start_time", None), getattr(timing, "end_time", None)


class CostKeyStepCallback:
    """A smolagents step callback that records each finished step as a span."""

    def __init__(self) -> None:
        # Steps of an agent run outside any CostKey trace share one trace id.
        self._fallback_traces: dict[int, str] = {}

    def __call__(self, memory_step: Any, agent: Any = None) -> None:
        step_type = type(memory_step).__name__
        attributes: dict[str, Any] = {"step_type": step_type}
        number = getattr(memory_step, "step_number", None)
        if isinstance(number, int):
            attributes["step_number"] = number
        calls = getattr(memory_step, "tool_calls", None)
        if isinstance(calls, list):
            attributes["tool_calls"] = [str(getattr(call, "name", "")) for call in calls]
        if getattr(memory_step, "is_final_answer", False):
            attributes["is_final_answer"] = True
        if agent is not None:
            attributes["agent"] = getattr(agent, "name", None) or type(agent).__name__
        if step_type == "PlanningStep":
            name = "planning"
        elif isinstance(number, int):
            name = f"step {number}"
        else:
            name = step_type
        started, ended = _timing(memory_step)
        key = id(agent)
        trace_id = self._fallback_traces.setdefault(key, uuid.uuid4().hex)
        record_completed_span(
            name, "step", framework=FRAMEWORK, started_at=started, ended_at=ended,
            attributes=attributes, error=getattr(memory_step, "error", None), trace_id=trace_id,
        )


def instrument(agent: Any, *, name: str | None = None) -> Any:
    """Record ``agent``'s runs and steps as CostKey spans; returns the agent."""
    from smolagents.memory import ActionStep, PlanningStep

    if getattr(agent, "_costkey_instrumented", False):
        return agent
    callback = CostKeyStepCallback()
    agent.step_callbacks.register(ActionStep, callback)
    agent.step_callbacks.register(PlanningStep, callback)
    original_run = agent.run
    span_name = name or getattr(agent, "name", None) or type(agent).__name__

    def scope() -> _SpanScope:
        return _SpanScope(span_name, "agent", framework=FRAMEWORK)

    @functools.wraps(original_run)
    def run(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _stream(original_run(*args, **kwargs), scope())
        with scope() as span:
            result = original_run(*args, **kwargs)
            span.set_output(getattr(result, "output", result))
            return result

    agent.run = run
    agent._costkey_instrumented = True
    return agent


def _stream(steps: Any, scope: _SpanScope) -> Any:
    with scope:
        yield from steps


__all__ = ["CostKeyStepCallback", "instrument"]
