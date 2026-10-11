"""OpenTelemetry bridge (``pip install "costkey[opentelemetry]"``).

Add the span processor to your tracer provider, or let ``instrument`` do it::

    from costkey.integrations.opentelemetry import instrument
    instrument()  # adds CostKeySpanProcessor to the global SDK TracerProvider

Every ended OTel span becomes a CostKey span (contract detail 9) of kind
``custom`` whose ``attributes`` carry ``otel`` (the raw OTel attributes),
``otelKind``, ``otelName`` and ``otelScope``. The server maps GenAI semantic
conventions; the SDK does not interpret them. Frameworks that trace through
OpenTelemetry (AutoGen AgentChat, Strands, Google ADK, Pydantic AI's
``instrument=True``, OpenLLMetry/OpenInference instrumentors) are covered by
this bridge.

When the project does not allow content capture, ``otel`` is an attribute
skeleton: strings become ``{"$s": N}`` unless the key, or its last dotted
segment, is a skeleton keep key (``gen_ai.request.model`` survives,
``gen_ai.prompt`` does not).
Provider calls made inside OTel spans are attributed to the innermost span;
the bridge never emits call events.
"""
from __future__ import annotations

import threading
from typing import Any, Mapping

from opentelemetry import trace as _trace
from opentelemetry.sdk.trace import SpanProcessor

from costkey import telemetry as _telemetry
from costkey.integrations._tracker import SpanTracker
from costkey.patch import _state, register_span_resolver, unregister_span_resolver

FRAMEWORK = "opentelemetry"
def otel_attributes(attributes: Mapping[str, Any] | None, *, capture: bool) -> Any:
    """The span's raw OTel attributes: raw when body capture is on, otherwise
    an attribute skeleton (a dotted key keeps its string only when its last
    segment is a keep key, so ``gen_ai.request.model`` survives and
    ``gen_ai.prompt`` does not)."""
    values = {str(key): value for key, value in (attributes or {}).items()}
    if capture:
        return _telemetry.cap_size(_json_ready(values))
    return _telemetry.attribute_skeleton(values)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _hex(value: int, width: int) -> str:
    return format(value, f"0{width}x")


def _error_name(span: Any) -> str | None:
    status = getattr(span, "status", None)
    code = getattr(getattr(status, "status_code", None), "name", None)
    if code != "ERROR":
        return None
    for event in getattr(span, "events", None) or ():
        if getattr(event, "name", None) == "exception":
            kind = (getattr(event, "attributes", None) or {}).get("exception.type")
            if isinstance(kind, str) and kind:
                return kind
    return "StatusCode.ERROR"


class CostKeySpanProcessor(SpanProcessor):
    """An OpenTelemetry SDK span processor that records CostKey spans."""

    def __init__(self) -> None:
        self._tracker = SpanTracker(FRAMEWORK, nest=False)
        self._shut_down = False
        register_span_resolver(self._current)

    def _attributes(self, span: Any) -> dict[str, Any]:
        context = span.get_span_context()
        scope = getattr(span, "instrumentation_scope", None)
        kind = getattr(span, "kind", None)
        parent = getattr(span, "parent", None)
        return {
            "otel": otel_attributes(getattr(span, "attributes", None), capture=bool(_state.capture_body)),
            "otelKind": getattr(kind, "name", str(kind) if kind is not None else None),
            "otelName": getattr(span, "name", None),
            "otelScope": getattr(scope, "name", None),
            "otelTraceId": _hex(context.trace_id, 32),
            "otelSpanId": _hex(context.span_id, 16),
            "otelParentSpanId": _hex(parent.span_id, 16) if parent is not None else None,
        }

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if self._shut_down:
            return
        context = span.get_span_context()
        parent = getattr(span, "parent", None)
        parent_key = (parent.trace_id, parent.span_id) if parent is not None else None
        self._tracker.start(
            (context.trace_id, context.span_id), span.name or "span", "custom",
            parent=parent_key, attributes=self._attributes(span), detached=True,
        )

    def on_end(self, span: Any) -> None:
        if self._shut_down:
            return
        context = span.get_span_context()
        self._tracker.end(
            (context.trace_id, context.span_id),
            error=_error_name(span),
            attributes=self._attributes(span),
        )

    def _current(self) -> dict[str, Any] | None:
        context = _trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return self._tracker.active_context((context.trace_id, context.span_id))

    def shutdown(self) -> None:
        """Stop recording (OTel providers cannot remove a processor)."""
        global _installed
        self._shut_down = True
        unregister_span_resolver(self._current)
        self._tracker.end_all()
        if _installed is self:
            _installed = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            from costkey.client import flush
            flush()
        except Exception:
            return False
        return True


_installed: CostKeySpanProcessor | None = None
_install_lock = threading.Lock()


def instrument(tracer_provider: Any = None) -> CostKeySpanProcessor:
    """Add one CostKeySpanProcessor to ``tracer_provider`` (default: the
    global provider, which is set to a new SDK TracerProvider if unset)."""
    global _installed
    with _install_lock:
        if _installed is not None and tracer_provider is None:
            return _installed
        provider = tracer_provider or _trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            from opentelemetry.sdk.trace import TracerProvider

            provider = TracerProvider()
            _trace.set_tracer_provider(provider)
        processor = CostKeySpanProcessor()
        provider.add_span_processor(processor)
        if tracer_provider is None:
            _installed = processor
        return processor


__all__ = ["CostKeySpanProcessor", "instrument", "otel_attributes"]
