"""Google ADK integration (``pip install "costkey[google-adk]"``).

Register the plugin on the runner::

    from costkey.integrations.google_adk import CostKeyPlugin

    runner = Runner(agent=root_agent, app_name="app", session_service=sessions,
                    plugins=[CostKeyPlugin()])

Each runner invocation becomes a ``workflow`` span, each agent run an
``agent`` span and each tool call a ``tool`` span. Model calls get no span:
the HTTP capture records them (google-genai over httpx/aiohttp, or any other
provider) inside the agent span.
"""
from __future__ import annotations

from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from costkey.integrations._tracker import SpanTracker

FRAMEWORK = "google_adk"


def _invocation_id(context: Any) -> Any:
    return getattr(context, "invocation_id", None)


class CostKeyPlugin(BasePlugin):
    """An ADK plugin that records invocations, agents and tools as CostKey spans."""

    def __init__(self, name: str = "costkey") -> None:
        super().__init__(name=name)
        self._tracker = SpanTracker(FRAMEWORK)

    # ── runner invocation ──

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        app = getattr(invocation_context, "app_name", None)
        agent = getattr(invocation_context, "agent", None)
        self._tracker.start(
            ("run", _invocation_id(invocation_context)),
            str(app or getattr(agent, "name", None) or "invocation"), "workflow",
            attributes={"invocation_id": _invocation_id(invocation_context), "app_name": app},
        )
        return None

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        self._tracker.end(("run", _invocation_id(invocation_context)))
        return None

    async def on_run_error_callback(self, *, invocation_context: Any, error: Exception) -> None:
        self._tracker.end(("run", _invocation_id(invocation_context)), error=error)
        return None

    # ── agents ──

    def _agent_key(self, agent: Any, callback_context: Any) -> tuple[Any, ...]:
        return ("agent", _invocation_id(callback_context), getattr(agent, "name", None))

    async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        model = getattr(agent, "model", None)
        self._tracker.start(
            self._agent_key(agent, callback_context), str(getattr(agent, "name", None) or "agent"), "agent",
            parent=("run", _invocation_id(callback_context)),
            attributes={
                "agent_class": type(agent).__name__,
                "model": model if isinstance(model, str) else getattr(model, "model", None),
            },
        )
        return None

    async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        self._tracker.end(self._agent_key(agent, callback_context))
        return None

    async def on_agent_error_callback(self, *, agent: Any, callback_context: Any, error: Exception) -> None:
        self._tracker.end(self._agent_key(agent, callback_context), error=error)
        return None

    # ── tools ──

    async def before_tool_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any) -> None:
        call_id = getattr(tool_context, "function_call_id", None)
        self._tracker.start(
            ("tool", call_id or id(tool_context)), str(getattr(tool, "name", None) or "tool"), "tool",
            parent=("agent", _invocation_id(tool_context), getattr(tool_context, "agent_name", None)),
            attributes={"tool_call_id": call_id, "tool_class": type(tool).__name__},
            input=tool_args,
        )
        return None

    async def after_tool_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any,
                                  result: Any) -> None:
        call_id = getattr(tool_context, "function_call_id", None)
        self._tracker.end(("tool", call_id or id(tool_context)), output=result)
        return None

    async def on_tool_error_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any,
                                     error: Exception) -> None:
        call_id = getattr(tool_context, "function_call_id", None)
        self._tracker.end(("tool", call_id or id(tool_context)), error=error)
        return None


__all__ = ["CostKeyPlugin"]
