"""AutoGen AgentChat integration (``pip install "costkey[autogen]"``).

AutoGen AgentChat traces through OpenTelemetry: ``create_agent``,
``invoke_agent`` and ``execute_tool`` spans with GenAI semantic-convention
attributes, on the global tracer provider (or the runtime's
``tracer_provider``). CostKey records them through its OpenTelemetry bridge::

    from costkey.integrations.autogen import instrument
    instrument()  # same as costkey.integrations.opentelemetry.instrument()

Model calls made by AutoGen's model clients are captured at the HTTP layer
and attributed to the ``invoke_agent``/``execute_tool`` span they ran in.
For the ``ag2`` package use :mod:`costkey.integrations.ag2`.
"""
from __future__ import annotations

from typing import Any

from costkey.integrations.opentelemetry import CostKeySpanProcessor
from costkey.integrations.opentelemetry import instrument as _instrument


def instrument(tracer_provider: Any = None) -> CostKeySpanProcessor:
    """Bridge AutoGen's OpenTelemetry spans into CostKey."""
    return _instrument(tracer_provider)


__all__ = ["CostKeySpanProcessor", "instrument"]
