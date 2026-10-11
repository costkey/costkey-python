"""Shared plumbing for framework integrations.

Frameworks report "run started"/"run ended" from callbacks that do not nest
like ``with`` blocks, may fire on other threads or tasks, and identify runs by
their own ids. ``SpanTracker`` opens a CostKey span per framework run, links
it to its parent by the framework's parent id, and closes it from whichever
context the end callback runs in.

Integrations only create spans with raw attributes. Model calls made in
process are captured by the HTTP patches and attributed to the innermost
active span, so integrations never emit call events for them.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Hashable, Mapping

from costkey.client import SpanHandle, _SpanScope

logger = logging.getLogger("costkey")

_TRACE_KEYS = ("traceId", "traceName", "executionId", "groupId", "groupName", "groupKind")


def scalar_attributes(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the raw scalar attributes (str, int, float, bool) of a mapping."""
    result: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if value is None or isinstance(value, (str, bool, int, float)):
            result[str(key)] = value
    return result


def error_name(error: Any) -> str | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return type(error).__name__
    if isinstance(error, type) and issubclass(error, BaseException):
        return error.__name__
    return str(error)


class _NamedError(Exception):
    """Carries a raw framework error name to the span end event."""


class SpanTracker:
    """Opens and closes CostKey spans keyed by a framework's run ids."""

    def __init__(self, framework: str, *, nest: bool = True) -> None:
        self.framework = framework
        # Attributes go under ``attributes.<framework>`` unless the caller
        # already shapes them (the OpenTelemetry bridge).
        self.nest = nest
        self._lock = threading.RLock()
        self._open: dict[Hashable, _SpanScope] = {}
        # Runs that get no span of their own (hidden runs, model calls)
        # resolve to their nearest visible ancestor.
        self._alias: dict[Hashable, Hashable | None] = {}

    def _resolve(self, key: Hashable | None) -> _SpanScope | None:
        seen = 0
        with self._lock:
            while key is not None and seen < 64:
                scope = self._open.get(key)
                if scope is not None:
                    return scope
                if key not in self._alias:
                    return None
                key = self._alias[key]
                seen += 1
        return None

    def is_open(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._open

    def span_id(self, key: Hashable) -> str | None:
        scope = self._resolve(key)
        return scope.span_id if scope is not None else None

    def handle(self, key: Hashable) -> SpanHandle | None:
        scope = self._resolve(key)
        return scope._handle if scope is not None else None

    def active_context(self, key: Hashable) -> dict[str, Any] | None:
        """The context a call made inside this run should carry."""
        scope = self._resolve(key)
        if scope is None:
            return None
        base = scope._base.get("context") or {}
        values = {k: base[k] for k in _TRACE_KEYS if base.get(k) is not None}
        values["spanId"] = scope.span_id
        return values

    def skip(self, key: Hashable, parent: Hashable | None) -> None:
        """Record a run that has no span; its children attach to ``parent``."""
        with self._lock:
            self._alias[key] = parent

    def forget(self, key: Hashable) -> None:
        """Drop a skipped run once it has ended."""
        with self._lock:
            self._alias.pop(key, None)

    def start(
        self,
        key: Hashable,
        name: str,
        kind: str,
        *,
        parent: Hashable | None = None,
        attributes: Mapping[str, Any] | None = None,
        input: Any = None,
        detached: bool = False,
    ) -> SpanHandle | None:
        """Open a span. A detached span does not become the active span of
        the calling context; pair it with a span resolver for attribution."""
        attrs = dict(attributes or {})
        parent_scope = self._resolve(parent)
        context: dict[str, Any] = {}
        parent_span_id = None
        if parent_scope is not None:
            parent_span_id = parent_scope.span_id
            # Keep the parent's trace even when this callback runs in a fresh
            # thread or task whose context never saw it.
            base = parent_scope._base.get("context") or {}
            context = {key_: base[key_] for key_ in _TRACE_KEYS if base.get(key_) is not None}
        try:
            scope = _SpanScope(
                name if isinstance(name, str) and name.strip() else kind,
                kind, attributes=attrs, input=input,
                parent_span_id=parent_span_id, context=context, detached=detached,
                framework=self.framework if self.nest else None,
            )
            handle = scope.__enter__()
            scope.span_id = handle.id
        except Exception:
            logger.debug("[costkey] %s span start failed", self.framework, exc_info=True)
            return None
        with self._lock:
            self._open[key] = scope
        return handle

    def end(
        self,
        key: Hashable,
        *,
        output: Any = None,
        error: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            scope = self._open.pop(key, None)
            self._alias.pop(key, None)
        if scope is None:
            return
        try:
            handle = scope._handle
            if handle is not None:
                for attr_key, value in (attributes or {}).items():
                    handle.set_attribute(attr_key, value)
                if output is not None:
                    handle.set_output(output)
            if error is None:
                scope.__exit__(None, None, None)
            elif isinstance(error, BaseException):
                scope.__exit__(type(error), error, None)
            else:
                named = type(str(error_name(error)), (_NamedError,), {})
                scope.__exit__(named, named(), None)
        except Exception:
            logger.debug("[costkey] %s span end failed", self.framework, exc_info=True)

    def end_all(self, error: Any = None) -> None:
        with self._lock:
            keys = list(self._open)
        for key in reversed(keys):
            self.end(key, error=error)


def record_completed_span(
    name: str,
    kind: str,
    *,
    framework: str,
    started_at: float | None,
    ended_at: float | None,
    attributes: Mapping[str, Any] | None = None,
    error: Any = None,
    trace_id: str | None = None,
) -> str | None:
    """Emit one already-finished span (for frameworks that only report a step
    after it ran). Its parent is the span active in the caller's context.

    ``started_at``/``ended_at`` are the framework's own epoch seconds.
    """
    import datetime
    import uuid

    from costkey.client import _current_context, _iso_timestamp
    from costkey.patch import emit_span, framework_attributes

    try:
        ctx = _current_context()
        resolved_trace = ctx.get("traceId") or trace_id
        if not resolved_trace:
            return None
        now = datetime.datetime.now(datetime.timezone.utc)

        def stamp(value: float | None) -> datetime.datetime:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return datetime.datetime.fromtimestamp(value, datetime.timezone.utc)
            return now

        started = stamp(started_at)
        ended = stamp(ended_at)
        span_id = f"sp_{uuid.uuid4().hex}"
        emit_span(
            span_id=span_id,
            trace_id=str(resolved_trace),
            trace_name=ctx.get("traceName"),
            kind=kind,
            name=name,
            status="ok" if error is None else "error",
            started_at=_iso_timestamp(started),
            ended_at=_iso_timestamp(ended),
            duration_ms=(ended - started).total_seconds() * 1000,
            error_type=error_name(error),
            parent_span_id=ctx.get("spanId"),
            execution_id=ctx.get("executionId"),
            context={**ctx, "spanId": span_id},
            attributes=framework_attributes(framework, dict(attributes or {})),
        )
        return span_id
    except Exception:
        logger.debug("[costkey] %s span dropped", framework, exc_info=True)
        return None
