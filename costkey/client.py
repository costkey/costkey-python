"""CostKey Python SDK — AI cost observability."""
from __future__ import annotations
import datetime
import json
import math
import os
import uuid
import logging
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Generator
from costkey.types import CostKeyOptions, CostKeyEvent
from costkey.transport import Transport
from costkey.patch import (
    patch,
    unpatch,
    _context,
    get_context,
    emit_lifecycle,
    emit_span,
    _dispatch_event,
    _scrub,
    start_runtime_config_fetch,
)
from costkey.providers import register_extractor as _register_extractor
from costkey.pricing import register_pricing as _register_pricing

logger = logging.getLogger("costkey")

_transport: Transport | None = None
_initialized = False
_usage_lock = threading.Lock()
_usage_updates: dict[tuple[str, str, int], str] = {}
_usage_modes: dict[tuple[str, str], str] = {}
_usage_event_ids: dict[str, str] = {}
_trace_scope_owned: ContextVar[bool] = ContextVar("costkey_trace_scope_owned", default=False)
_execution_scope_owned: ContextVar[bool] = ContextVar("costkey_execution_scope_owned", default=False)

_CONTEXT_ALIASES = {
    "trace_id": "traceId",
    "trace_name": "traceName",
    "execution_id": "executionId",
    "group_name": "groupName",
    "group_kind": "groupKind",
    "group_id": "groupId",
    "dataset_revision": "datasetRevision",
    "case_id": "caseId",
    "conversation_id": "conversationId",
    "customer_id": "customerId",
    "voice_session_id": "voiceSessionId",
    "span_id": "spanId",
    "user_id": "userId",
}

# Only these fields may cross an explicitly supplied service/queue carrier.
# Arbitrary context (including prompts, credentials, and application secrets)
# stays local unless the caller deliberately maps it into a supported field.
_EXPORT_CONTEXT_KEYS = frozenset({
    "traceId", "traceName", "executionId", "groupName", "groupKind",
    "groupId",
    "phase", "datasetRevision", "caseId", "environment", "service",
    "release", "feature", "customer", "customerId", "function", "operation", "conversationId",
    "voiceSessionId", "spanId", "agent", "tool", "userId",
})


def _parse_dsn(dsn: str) -> tuple[str, str, str]:
    """Parse DSN → (endpoint, auth_key, project_id).

    Expected format: https://<key>@app.costkey.dev/<project-id>
    """
    from urllib.parse import urlparse

    _DSN_HELP = 'Invalid DSN format. Expected: https://<key>@app.costkey.dev/<project-id>  Get your DSN at https://app.costkey.dev'

    if not dsn or not isinstance(dsn, str):
        raise ValueError(f"[costkey] {_DSN_HELP}")

    parsed = urlparse(dsn)

    if not parsed.hostname:
        raise ValueError(f"[costkey] {_DSN_HELP}")

    auth_key = parsed.username or ""
    if not auth_key:
        raise ValueError(f"[costkey] DSN missing auth key. {_DSN_HELP}")

    project_id = parsed.path.lstrip("/")
    if not project_id:
        raise ValueError(f"[costkey] DSN missing project ID. {_DSN_HELP}")

    endpoint = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        endpoint += f":{parsed.port}"
    endpoint += "/api/v1/events"
    return endpoint, auth_key, project_id


def _canonical_context(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy context and normalize the documented snake_case aliases."""
    if isinstance(values, str):
        try:
            parsed = json.loads(values)
        except Exception:
            return {}
        values = parsed if isinstance(parsed, Mapping) else None
    result: dict[str, Any] = {}
    for key, value in (values or {}).items():
        canonical = _CONTEXT_ALIASES.get(str(key), str(key))
        # A canonical key supplied alongside an alias wins deterministically.
        if canonical not in result or str(key) == canonical:
            result[canonical] = _json_safe(value)
    return result


def _iso_timestamp(value: Any = None) -> str:
    if value is None:
        return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, datetime.datetime):
        timestamp = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
        return timestamp.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, str) and value:
        return value
    raise TypeError("timestamp must be an ISO string or datetime")


def _json_safe(value: Any) -> Any:
    """Bound carrier/config values to JSON-compatible, non-secret data."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _carrier_scalar(value: Any) -> str | int | float | bool | None:
    """Keep explicit service/queue carriers to finite JSON scalar values."""
    if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _non_empty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _currency_code(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("amount currency must be a non-empty string")
    normalized = value.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_-]{2,9}", normalized):
        raise ValueError("amount currency must be an ISO-like code")
    return normalized


def _amount_payload(amount: Any, currency: str | None) -> tuple[dict[str, Any] | None, float | None, str | None]:
    if amount is None:
        return None, None, _currency_code(currency)
    if isinstance(amount, Mapping):
        value = amount.get("value", amount.get("amount"))
        amount_currency = amount.get("currency", currency or "USD")
    else:
        value = amount
        amount_currency = currency or "USD"
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise TypeError("amount value must be a finite non-negative number")
    normalized_currency = _currency_code(amount_currency)
    assert normalized_currency is not None
    value_float = float(value)
    return {"value": value_float, "currency": normalized_currency}, (
        value_float if normalized_currency == "USD" else None
    ), normalized_currency


def _usage_payload(usage: Any, native_units: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    def usage_value(value: Any) -> Any:
        # Keep non-finite floats intact until validation so NaN/Infinity are
        # rejected rather than silently converted into an unknown null unit.
        if isinstance(value, float) and not math.isfinite(value):
            return value
        return _json_safe(value)

    if isinstance(usage, Mapping):
        result = {str(key): usage_value(value) for key, value in usage.items()}
    elif usage is None:
        result = {}
    elif hasattr(usage, "__dataclass_fields__"):
        result = {}
        for key in (
            "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
            "cache_read_tokens", "cache_creation_tokens", "image_output_tokens",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                result[key] = value
        result.update({str(key): usage_value(value) for key, value in getattr(usage, "native_units", {}).items()})
    else:
        raise TypeError("usage must be a mapping or NormalizedUsage")
    if native_units:
        result.update({str(key): usage_value(value) for key, value in native_units.items()})
    for key, value in result.items():
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise TypeError(f"usage.{key} must be a finite non-negative number or null")
    return result or None


def _fingerprint(values: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(values), sort_keys=True, separators=(",", ":"), default=str)


def _current_context() -> dict[str, Any]:
    from costkey.patch import _state
    return {**_state.default_context, **get_context()}


def _resolve_policy(
    *,
    policy: Any,
    policy_admission: Any,
    policy_endpoint: str | None,
    policy_mode: str,
    policy_timeout: float,
    endpoint: str,
    auth_key: str,
) -> tuple[Any, str]:
    """Resolve explicit admission configuration without enabling blocking by default."""
    mode = policy_mode
    admission = policy_admission
    configured_endpoint = policy_endpoint
    integration: Mapping[str, Any] | None = None
    scope: Mapping[str, Any] | None = None
    usage_bounds: Mapping[str, Any] | None = None
    claimed_cost_usd: float | None = None

    if isinstance(policy, Mapping):
        mode = str(policy.get("mode", policy.get("policyMode", mode)))
        admission = policy.get("admission", admission)
        configured_endpoint = policy.get("endpoint", configured_endpoint)
        policy_timeout = float(policy.get("timeout", policy_timeout))
        integration = policy.get("integration", policy.get("capability"))
        scope = policy.get("scope")
        usage_bounds = policy.get("usageBounds", policy.get("usage_bounds"))
        claimed_cost_usd = policy.get("claimedCostUsd", policy.get("claimed_cost_usd"))
    elif policy is not None and any(hasattr(policy, name) for name in ("mode", "admission", "endpoint")):
        mode = str(getattr(policy, "mode", mode))
        admission = getattr(policy, "admission", admission)
        configured_endpoint = getattr(policy, "endpoint", configured_endpoint)
        policy_timeout = float(getattr(policy, "timeout", policy_timeout))
        integration = getattr(policy, "integration", None)
        scope = getattr(policy, "scope", None)
        usage_bounds = getattr(policy, "usage_bounds", getattr(policy, "usageBounds", None))
        claimed_cost_usd = getattr(policy, "claimed_cost_usd", getattr(policy, "claimedCostUsd", None))
    elif callable(policy) and admission is None:
        admission = policy
    elif isinstance(policy, str) and configured_endpoint is None:
        configured_endpoint = policy
    elif policy is True and configured_endpoint is None:
        mode = "enforce"

    if mode not in {"monitor", "enforce"}:
        raise ValueError("policy_mode must be 'monitor' or 'enforce'")

    if configured_endpoint and admission is None:
        from costkey.policy import PolicyAdmissionOptions
        admission = PolicyAdmissionOptions(
            endpoint=str(configured_endpoint),
            auth_key=auth_key,
            mode=mode,
            timeout=policy_timeout,
            integration=integration if isinstance(integration, Mapping) else None,
            scope=scope if isinstance(scope, Mapping) else None,
            usage_bounds=usage_bounds if isinstance(usage_bounds, Mapping) else None,
            claimed_cost_usd=claimed_cost_usd,
        )
    elif mode == "enforce" and admission is None:
        # Derive only when the caller explicitly opts into enforcement. A
        # normal init never performs a policy request or blocks a call.
        from costkey.policy import PolicyAdmissionOptions
        base_url = endpoint.removesuffix("/api/v1/events")
        admission = PolicyAdmissionOptions(
            endpoint=f"{base_url}/api/v1/sdk/policy/admit",
            auth_key=auth_key,
            mode=mode,
            timeout=policy_timeout,
            integration=integration if isinstance(integration, Mapping) else None,
            scope=scope if isinstance(scope, Mapping) else None,
            usage_bounds=usage_bounds if isinstance(usage_bounds, Mapping) else None,
            claimed_cost_usd=claimed_cost_usd,
        )

    return admission, mode


def init(dsn: str | None = None, *, capture_body: bool | None = None,
         before_send: Callable[[CostKeyEvent], CostKeyEvent | None] | None = None,
         before_send_event: Callable[[CostKeyEvent], CostKeyEvent | None] | None = None,
         max_batch_size: int = 50, flush_interval: float = 5.0,
         debug: bool = False, default_context: dict[str, Any] | None = None,
         release: str | None = None,
         project_root: str | None = None,
         scan_callgraph: bool = True,
         policy: Any = None,
         policy_admission: Any = None,
         policy_endpoint: str | None = None,
         policy_mode: str = "monitor",
         policy_timeout: float = 2.0,
         config_refresh_interval: float | None = 60.0) -> None:
    """
    Initialize CostKey. Call once at app startup.

    >>> import costkey
    >>> costkey.init(dsn="https://ck_abc123@app.costkey.dev/my-project")
    >>> # That's it. Supported provider calls are observed in-process.

    If no DSN is provided, falls back to COSTKEY_DSN env var.
    If neither exists, silently no-ops (test mode).
    """
    global _transport, _initialized

    if _initialized:
        if debug:
            logger.warning("[costkey] Already initialized, skipping")
        return

    # DSN resolution: param → env var → test mode (no-op)
    resolved_dsn = dsn or os.environ.get("COSTKEY_DSN")
    if not resolved_dsn:
        if debug:
            logger.info("[costkey] No DSN provided, running in test mode (no events will be sent)")
        return

    endpoint, auth_key, project_id = _parse_dsn(resolved_dsn)

    admission, resolved_policy_mode = _resolve_policy(
        policy=policy,
        policy_admission=policy_admission,
        policy_endpoint=policy_endpoint,
        policy_mode=policy_mode,
        policy_timeout=policy_timeout,
        endpoint=endpoint,
        auth_key=auth_key,
    )

    _transport = Transport(
        endpoint=endpoint, auth_key=auth_key,
        max_batch_size=max_batch_size, flush_interval=flush_interval,
        debug=debug, release=release,
    )

    initial_context = _canonical_context(default_context)
    if release is not None:
        initial_context.setdefault("release", release)

    patch(
        transport=_transport, project_id=project_id,
        capture_body=capture_body, before_send=before_send,
        default_context=initial_context, debug=debug,
        before_send_event=before_send_event,
        admission=admission,
        admission_mode=resolved_policy_mode,
    )

    _transport.start()
    _initialized = True
    start_runtime_config_fetch(
        endpoint.removesuffix("/api/v1/events"),
        auth_key,
        project_id,
        debug,
        refresh_interval=config_refresh_interval,
    )

    if scan_callgraph:
        try:
            from costkey.callgraph import scan_and_send
            # Derive base URL by stripping /api/v1/events from the endpoint
            base_url = endpoint.replace("/api/v1/events", "")
            root = project_root or os.getcwd()
            t = threading.Thread(
                target=scan_and_send,
                args=(base_url, auth_key, project_id, root, debug),
                daemon=True,
            )
            t.start()
        except Exception:
            if debug:
                logger.warning("[costkey] Failed to start call graph scan")

    if debug:
        logger.info(f"[costkey] Initialized for project {project_id}")


# Alias for MLflow familiarity
autolog = init


def shutdown() -> None:
    """Flush pending events and restore original HTTP clients."""
    global _transport, _initialized
    if _transport:
        _transport.flush()
        _transport.stop()
        _transport = None
    unpatch()
    _initialized = False
    with _usage_lock:
        _usage_updates.clear()
        _usage_modes.clear()
        _usage_event_ids.clear()


def flush() -> None:
    """Flush all pending events without shutting down."""
    if _transport:
        _transport.flush()


@contextmanager
def with_context(
    context: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Generator[None, None, None]:
    """Attach inherited metadata to calls in this synchronous/async scope.

    ContextVars copy values at asyncio task creation, and the token reset in
    ``finally`` prevents nested scopes from leaking into sibling work.

    Both forms are supported for a small migration seam::

        with costkey.with_context(feature="search"):
            ...
        with costkey.with_context({"feature": "search"}):
            ...
    """
    parent = get_context()
    is_header_mapping = isinstance(context, Mapping) and any(
        str(key).lower() == "x-costkey-context" for key in context
    )
    is_nested_carrier = isinstance(context, Mapping) and isinstance(context.get("context"), Mapping)
    supplied = (
        import_context(context)
        if isinstance(context, str) or is_header_mapping or is_nested_carrier
        else _canonical_context(context)
    )
    merged = {**parent, **supplied, **_canonical_context(kwargs)}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def start_trace(
    name: str | None = None,
    trace_id: str | None = None,
    **context: Any,
) -> Generator[None, None, None]:
    """Bound one trace and report started/completed/failed lifecycle events."""
    parent = _current_context()
    # An imported trace identity joins the upstream operation. A fresh scope
    # with no imported identity receives a new trace ID.
    inherited_trace = None if _trace_scope_owned.get() else (
        parent.get("traceId") or parent.get("trace_id")
    )
    tid = trace_id or inherited_trace or uuid.uuid4().hex
    trace_name = name if name is not None else parent.get("traceName", parent.get("trace_name"))
    started_at = _iso_timestamp()
    started = datetime.datetime.now(datetime.timezone.utc)
    child_context = {
        **parent,
        **_canonical_context(context),
        "traceId": tid,
        "traceName": trace_name,
    }
    owned_token = _trace_scope_owned.set(True)
    try:
        with with_context(child_context):
            emit_lifecycle(
                entity_type="trace",
                entity_id=tid,
                lifecycle="started",
                context=child_context,
                trace_id=tid,
                trace_name=trace_name,
                started_at=started_at,
            )
            try:
                yield
            except BaseException as exc:
                ended_at = _iso_timestamp()
                elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() * 1000
                import asyncio
                cancelled = isinstance(exc, asyncio.CancelledError)
                emit_lifecycle(
                    entity_type="trace",
                    entity_id=tid,
                    lifecycle="cancelled" if cancelled else "failed",
                    context=child_context,
                    trace_id=tid,
                    trace_name=trace_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=elapsed,
                    error_type=None if cancelled else type(exc).__name__,
                )
                raise
            else:
                ended_at = _iso_timestamp()
                elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() * 1000
                emit_lifecycle(
                    entity_type="trace",
                    entity_id=tid,
                    lifecycle="completed",
                    context=child_context,
                    trace_id=tid,
                    trace_name=trace_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=elapsed,
                )
    finally:
        _trace_scope_owned.reset(owned_token)


@contextmanager
def start_execution(
    name: str,
    kind: str = "batch",
    execution_id: str | None = None,
    *,
    group_name: str | None = None,
    group_id: str | None = None,
    expected_items: int | None = None,
    completed_items: int | None = None,
    failed_items: int | None = None,
    configuration: Mapping[str, Any] | None = None,
    dataset_revision: str | None = None,
    case_id: str | None = None,
    phase: str | None = None,
    **context: Any,
) -> Generator[None, None, None]:
    """Bound one fresh grouped-work invocation.

    ``name``/``kind`` are stable group identity; ``execution_id`` is fresh by
    default. Member traces inherit this context and remain independently
    identifiable.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("execution name is required")
    if not isinstance(kind, str) or not kind:
        raise ValueError("execution kind is required")
    if expected_items is not None and (
        not isinstance(expected_items, int) or isinstance(expected_items, bool) or expected_items < 0
    ):
        raise ValueError("expected_items must be a non-negative integer")
    for label, value in (("completed_items", completed_items), ("failed_items", failed_items)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{label} must be a non-negative integer")
    parent = _current_context()
    inherited_execution = None if _execution_scope_owned.get() else (
        parent.get("executionId") or parent.get("execution_id")
    )
    eid = execution_id or inherited_execution or uuid.uuid4().hex
    stable_group = group_name or name
    started_at = _iso_timestamp()
    started = datetime.datetime.now(datetime.timezone.utc)
    child_context = {
        **parent,
        **_canonical_context(context),
        "groupName": stable_group,
        "groupKind": kind,
        "executionId": eid,
    }
    if group_id is not None:
        child_context["groupId"] = group_id
    if dataset_revision is not None:
        child_context["datasetRevision"] = dataset_revision
    if case_id is not None:
        child_context["caseId"] = case_id
    if phase is not None:
        child_context["phase"] = phase
    safe_configuration = _json_safe(configuration) if configuration is not None else None
    if isinstance(safe_configuration, dict):
        child_context["configuration"] = safe_configuration
    owned_token = _execution_scope_owned.set(True)
    try:
        with with_context(child_context):
            emit_lifecycle(
                entity_type="execution",
                entity_id=eid,
                lifecycle="started",
                context=child_context,
                execution_id=eid,
                group_name=stable_group,
                group_kind=kind,
                group_id=group_id,
                started_at=started_at,
                expected_items=expected_items,
                completed_items=completed_items,
                failed_items=failed_items,
                configuration=safe_configuration if isinstance(safe_configuration, dict) else None,
            )
            try:
                yield
            except BaseException as exc:
                ended_at = _iso_timestamp()
                elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() * 1000
                import asyncio
                cancelled = isinstance(exc, asyncio.CancelledError)
                emit_lifecycle(
                    entity_type="execution",
                    entity_id=eid,
                    lifecycle="cancelled" if cancelled else "failed",
                    context=child_context,
                    execution_id=eid,
                    group_name=stable_group,
                    group_kind=kind,
                    group_id=group_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=elapsed,
                    error_type=None if cancelled else type(exc).__name__,
                    expected_items=expected_items,
                    completed_items=completed_items,
                    failed_items=failed_items,
                    configuration=safe_configuration if isinstance(safe_configuration, dict) else None,
                )
                raise
            else:
                ended_at = _iso_timestamp()
                elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() * 1000
                emit_lifecycle(
                    entity_type="execution",
                    entity_id=eid,
                    lifecycle="completed",
                    context=child_context,
                    execution_id=eid,
                    group_name=stable_group,
                    group_kind=kind,
                    group_id=group_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=elapsed,
                    expected_items=expected_items,
                    completed_items=completed_items,
                    failed_items=failed_items,
                    configuration=safe_configuration if isinstance(safe_configuration, dict) else None,
                )
    finally:
        _execution_scope_owned.reset(owned_token)


_SPAN_KINDS = frozenset({
    "agent", "step", "tool", "workflow", "task", "judge", "retrieval",
    "handoff", "guardrail", "chain", "custom",
})


class SpanHandle:
    """Returned by a span scope: add attributes or record the span's output."""

    def __init__(self, span_id: str, trace_id: str) -> None:
        self.id = span_id
        self.trace_id = trace_id
        self._attributes: dict[str, Any] = {}
        self._output: Any = None

    def set_attribute(self, key: str, value: str | int | float | bool | None) -> None:
        self._attributes[str(key)] = value

    def set_output(self, value: Any) -> None:
        self._output = value


class _SpanScope:
    """A span usable as ``with``, ``async with`` or a (sync/async) decorator.

    Spans nest through ContextVars: a span opened inside another becomes its
    child, and every provider call made inside is attributed to the innermost
    span. Outside any run, the outermost span starts one.
    """

    def __init__(
        self,
        name: str,
        kind: str = "custom",
        *,
        attributes: Mapping[str, Any] | None = None,
        input: Any = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("span name is required")
        self.name = name.strip()
        self.kind = kind if kind in _SPAN_KINDS else "custom"
        self.attributes = dict(attributes or {})
        self.input = input
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.context = dict(context or {})
        self._trace_cm: Any = None
        self._token: Any = None
        self._previous: dict[str, Any] | None = None
        self._handle: SpanHandle | None = None
        self._base: dict[str, Any] = {}
        self._started: datetime.datetime | None = None

    def _clone(self) -> "_SpanScope":
        return _SpanScope(
            self.name, self.kind, attributes=self.attributes, input=self.input,
            span_id=self.span_id, parent_span_id=self.parent_span_id, context=self.context,
        )

    def __enter__(self) -> SpanHandle:
        if not _current_context().get("traceId"):
            self._trace_cm = start_trace(name=self.name)
            self._trace_cm.__enter__()
        ctx = get_context()
        sid = self.span_id or f"sp_{uuid.uuid4().hex}"
        inherited = ctx.get("spanId") if ctx.get("spanId") != sid else None
        child = {**ctx, **_canonical_context(self.context), "spanId": sid}
        if self.kind == "agent":
            child["agent"] = self.name
        if self.kind == "tool":
            child["tool"] = self.name
        self._previous = _context.get()
        self._token = _context.set(child)
        handle = SpanHandle(sid, str(child.get("traceId")))
        handle._attributes.update(self.attributes)
        self._handle = handle
        self._started = datetime.datetime.now(datetime.timezone.utc)
        self._base = {
            "span_id": sid,
            "trace_id": str(child.get("traceId")),
            "trace_name": child.get("traceName"),
            "parent_span_id": self.parent_span_id or inherited,
            "execution_id": child.get("executionId"),
            "kind": self.kind,
            "name": self.name,
            "started_at": _iso_timestamp(self._started),
            "context": child,
        }
        try:
            emit_span(**self._base, status="running", attributes=dict(handle._attributes), input=self.input)
        except Exception:
            logger.debug("[costkey] span start dropped", exc_info=True)
        return handle

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            import asyncio
            if exc_type is None:
                status, error = "ok", None
            elif issubclass(exc_type, asyncio.CancelledError):
                status, error = "cancelled", None
            else:
                status, error = "error", exc_type.__name__
            ended = datetime.datetime.now(datetime.timezone.utc)
            handle = self._handle
            emit_span(
                **self._base,
                status=status,
                ended_at=_iso_timestamp(ended),
                duration_ms=(ended - (self._started or ended)).total_seconds() * 1000,
                error_type=error,
                attributes=dict(handle._attributes) if handle else None,
                output=handle._output if handle else None,
            )
        except Exception:
            logger.debug("[costkey] span end dropped", exc_info=True)
        finally:
            try:
                _context.reset(self._token)
            except ValueError:
                # Closed from a different context (e.g. framework hooks): restore by value.
                _context.set(self._previous or {})
            if self._trace_cm is not None:
                self._trace_cm.__exit__(exc_type, exc, tb)
        return False

    async def __aenter__(self) -> SpanHandle:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        import functools
        import inspect

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self._clone() as span:
                    result = await fn(*args, **kwargs)
                    span.set_output(result)
                    return result
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self._clone() as span:
                result = fn(*args, **kwargs)
                span.set_output(result)
                return result
        return wrapper


_SPAN_OPTION_KEYS = frozenset({"attributes", "input", "span_id", "parent_span_id", "context"})


def _span_scope(name: str, kind: str, options: dict[str, Any]) -> _SpanScope:
    # Any keyword that isn't a span option is a context label (user_id=...).
    labels = {key: value for key, value in options.items() if key not in _SPAN_OPTION_KEYS}
    known = {key: value for key, value in options.items() if key in _SPAN_OPTION_KEYS}
    if labels:
        known["context"] = {**dict(known.get("context") or {}), **labels}
    return _SpanScope(name, kind, **known)


def span(name: str, kind: str = "custom", **options: Any) -> _SpanScope:
    """Open a span: ``with costkey.span("rerank", kind="retrieval"):``"""
    return _span_scope(name, kind, options)


def agent(name: str, **options: Any) -> _SpanScope:
    """An agent run. Model calls, tools and sub-agents inside are attributed to it."""
    return _span_scope(name, "agent", options)


def step(name: str, **options: Any) -> _SpanScope:
    """One step of an agent loop (a turn, a browser action, a plan stage)."""
    return _span_scope(name, "step", options)


def tool(name: str, **options: Any) -> _SpanScope:
    """A tool call. Its latency and any model calls it makes are attributed to it."""
    return _span_scope(name, "tool", options)


def export_context(*, keys: set[str] | list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Return an explicit, allowlisted carrier for trusted handoff only."""
    current = get_context()
    requested = _EXPORT_CONTEXT_KEYS if keys is None else {
        _CONTEXT_ALIASES.get(str(key), str(key)) for key in keys
    }
    unknown = requested - _EXPORT_CONTEXT_KEYS
    if unknown:
        raise ValueError(f"context keys are not exportable: {', '.join(sorted(unknown))}")
    result: dict[str, Any] = {}
    for key in requested:
        if key in current:
            value = _carrier_scalar(current[key])
            if value is not None:
                result[key] = _scrub(value)
    return result


def export_context_headers(*, keys: set[str] | list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Return an explicit header carrier for a trusted request/queue only."""
    return {"x-costkey-context": json.dumps(export_context(keys=keys), separators=(",", ":"))}


def import_context(carrier: Mapping[str, Any] | str) -> dict[str, Any]:
    """Parse an explicit carrier; arbitrary labels are discarded."""
    if isinstance(carrier, Mapping):
        header_key = next(
            (key for key in carrier if str(key).lower() == "x-costkey-context"),
            None,
        )
        if header_key is not None:
            carrier = str(carrier[header_key])
        elif isinstance(carrier.get("context"), Mapping):
            carrier = carrier["context"]
    values = _canonical_context(carrier)
    nested = values.get("context")
    if isinstance(nested, Mapping):
        values = _canonical_context(nested)
    return {
        key: scalar
        for key, value in values.items()
        if key in _EXPORT_CONTEXT_KEYS
        for scalar in (_carrier_scalar(value),)
        if scalar is not None
    }


def record_usage(
    event: Mapping[str, Any] | None = None,
    *,
    id: str | None = None,
    event_id: str | None = None,
    component_id: str | None = None,
    charge_id: str | None = None,
    provider: Any = "unknown",
    model: str | None = None,
    unit: str | None = None,
    usage: Mapping[str, Any] | Any = None,
    native_units: Mapping[str, Any] | None = None,
    amount: Any = None,
    cost_usd: float | None = None,
    currency: str | None = None,
    amount_kind: str | None = None,
    pricing_id: str | None = None,
    timestamp: Any = None,
    aggregation: str = "delta",
    revision: int | None = None,
    unit_kind: str | None = None,
    unit_value: int | float | None = None,
    trace_id: str | None = None,
    execution_id: str | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    billing_units: Mapping[str, Any] | None = None,
    provider_request_id: str | None = None,
    attempt_id: str | None = None,
    parent_event_id: str | None = None,
) -> str:
    """Report one external usage component with explicit revision semantics.

    The caller supplies a unique envelope ``event_id`` and a revision. Delta
    revisions are additive downstream; cumulative revisions are full snapshots
    and are never converted to deltas in the SDK.
    """
    values = dict(event or {})
    # Mapping input is a migration seam for callers that already construct a
    # wire-shaped usage object. Explicit keyword arguments take precedence.
    event_id = event_id or id or values.pop("id", None) or values.pop("eventId", None) or values.pop("event_id", None)
    component_id = component_id or values.pop("componentId", None) or values.pop("component_id", None) or values.pop("component", None)
    charge_id = charge_id or values.pop("chargeId", None) or values.pop("charge_id", None)
    provider = provider if provider != "unknown" else values.pop("provider", provider)
    model = model or values.pop("model", None)
    unit = unit or values.pop("unit", None) or values.pop("unitKind", None)
    usage = usage if usage is not None else values.pop("usage", values.pop("nativeUsage", None))
    native_units = native_units or values.pop("nativeUnits", None) or values.pop("native_units", None)
    amount = amount if amount is not None else values.pop("amount", None)
    if amount is None:
        amount = values.pop("cost", None)
    if amount is None:
        amount = values.pop("costUsd", values.pop("cost_usd", None))
    cost_usd = cost_usd if cost_usd is not None else values.pop("costUsd", values.pop("cost_usd", None))
    if amount is None and cost_usd is not None:
        amount = cost_usd
    currency = currency or values.pop("currency", None)
    amount_kind = amount_kind or values.pop("amountKind", None) or values.pop("amount_kind", None)
    pricing_id = pricing_id or values.pop("pricingId", None) or values.pop("pricing_id", None)
    timestamp = timestamp if timestamp is not None else values.pop("timestamp", None)
    aggregation = values.pop("aggregation", aggregation)
    revision = revision if revision is not None else values.pop("revision", None)
    unit_kind = unit_kind or values.pop("unitKind", None) or values.pop("unit_kind", None)
    unit_value = unit_value if unit_value is not None else values.pop("unitValue", values.pop("unit_value", None))
    trace_id = trace_id or values.pop("traceId", None) or values.pop("trace_id", None)
    execution_id = execution_id or values.pop("executionId", None) or values.pop("execution_id", None)
    context = context or values.pop("context", None)
    metadata = metadata or values.pop("metadata", None)
    billing_units = billing_units or values.pop("billingUnits", None) or values.pop("billing_units", None)
    provider_request_id = provider_request_id or values.pop("providerRequestId", None) or values.pop("provider_request_id", None)
    attempt_id = attempt_id or values.pop("attemptId", None) or values.pop("attempt_id", None)
    parent_event_id = parent_event_id or values.pop("parentEventId", None) or values.pop("parent_event_id", None)

    unit = _non_empty(unit)
    unit_kind = _non_empty(unit_kind)
    pricing_id = _non_empty(pricing_id)
    provider_request_id = _non_empty(provider_request_id)
    attempt_id = _non_empty(attempt_id)
    parent_event_id = _non_empty(parent_event_id)

    event_id = _non_empty(event_id)
    component_id = _non_empty(component_id)
    if not event_id:
        raise ValueError("record_usage requires a caller-supplied event_id")
    if not component_id:
        raise ValueError("record_usage requires component_id")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise ValueError("record_usage requires a non-negative integer revision")
    if aggregation not in {"delta", "cumulative"}:
        raise ValueError("aggregation must be 'delta' or 'cumulative'")
    if charge_id is None:
        charge_id = component_id
    charge_id = _non_empty(charge_id)
    if not charge_id:
        raise ValueError("charge_id must be a non-empty string")

    usage_data = _usage_payload(usage, native_units)
    amount_data, cost_usd, amount_currency = _amount_payload(amount, currency)
    billing_data = _usage_payload(billing_units) if billing_units is not None else None
    provider_name = _non_empty(getattr(provider, "value", provider)) or "unknown"
    if amount_kind is None and amount_data is not None:
        amount_kind = "reported"
    if amount_kind is not None and amount_kind not in {"reported", "estimated"}:
        raise ValueError("amount_kind must be 'reported' or 'estimated'")
    if unit_kind is not None and (not isinstance(unit_kind, str) or not unit_kind):
        raise ValueError("unit_kind must be a non-empty string")
    if unit_value is not None and (
        not isinstance(unit_value, (int, float))
        or isinstance(unit_value, bool)
        or not math.isfinite(unit_value)
        or unit_value < 0
    ):
        raise TypeError("unit_value must be a finite non-negative number")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    event_context = {
        **_current_context(),
        **_canonical_context(context),
    }
    if trace_id is not None:
        event_context["traceId"] = trace_id
    if execution_id is not None:
        event_context["executionId"] = execution_id

    fingerprint = _fingerprint({
        "componentId": component_id,
        "chargeId": charge_id,
        "provider": provider_name,
        "model": model,
        "unit": unit,
        "usage": usage_data,
        "amount": amount_data,
        "currency": amount_currency,
        "amountKind": amount_kind,
        "pricingId": pricing_id,
        "unitKind": unit_kind or unit,
        "unitValue": unit_value,
        "metadata": _scrub(_json_safe(metadata)) if isinstance(metadata, Mapping) else None,
        "billingUnits": billing_data,
        "providerRequestId": provider_request_id,
        "attemptId": attempt_id,
        "parentEventId": parent_event_id,
        "aggregation": aggregation,
        "revision": revision,
    })
    project_id = _project_id()
    from costkey.patch import _state
    if not project_id or _state.transport is None:
        # Match the SDK lifecycle: pre-init/test-mode usage is a validation
        # no-op and must not poison the revision ledger used after init.
        return event_id
    scope_key = (project_id, charge_id)
    key = (project_id, charge_id, revision)
    with _usage_lock:
        previous_mode = _usage_modes.get(scope_key)
        if previous_mode is not None and previous_mode != aggregation:
            raise ValueError(f"charge_id {charge_id!r} cannot switch aggregation mode")
        _usage_modes[scope_key] = aggregation
        previous_fingerprint = _usage_updates.get(key)
        if previous_fingerprint is not None:
            if previous_fingerprint != fingerprint:
                raise ValueError(f"conflicting usage update for charge_id={charge_id!r}, revision={revision}")
            _usage_event_ids[event_id] = fingerprint
            return event_id
        previous_event = _usage_event_ids.get(event_id)
        if previous_event is not None and previous_event != fingerprint:
            raise ValueError(f"conflicting usage event_id {event_id!r}")
        if aggregation == "cumulative":
            prior_revisions = [
                previous_revision
                for (previous_project, previous_charge, previous_revision) in _usage_updates
                if previous_project == project_id and previous_charge == charge_id
            ]
            if prior_revisions and revision < max(prior_revisions):
                # An older full snapshot cannot safely be applied after a
                # newer one. Return its ID without emitting it.
                _usage_event_ids[event_id] = fingerprint
                return event_id
        _usage_updates[key] = fingerprint
        _usage_event_ids[event_id] = fingerprint

    explicit_event = CostKeyEvent(
        id=event_id,
        timestamp=_iso_timestamp(timestamp),
        project_id=project_id,
        provider=provider_name,
        model=model,
        usage=None,
        cost_usd=cost_usd,
        context=_scrub(event_context),
        trace_id=event_context.get("traceId"),
        trace_name=event_context.get("traceName"),
        event_type="usage",
        source="explicit",
        component_id=component_id,
        charge_id=charge_id,
        revision=revision,
        aggregation=aggregation,
        native_usage=usage_data,
        billing_units=billing_data,
        amount=amount_data,
        currency=amount_currency,
        amount_kind=amount_kind,
        pricing_id=pricing_id,
        unit=unit,
        unit_kind=unit_kind or unit,
        unit_value=unit_value,
        execution_id=event_context.get("executionId"),
        group_id=event_context.get("groupId"),
        group_name=event_context.get("groupName"),
        group_kind=event_context.get("groupKind"),
        phase=event_context.get("phase"),
        dataset_revision=event_context.get("datasetRevision"),
        case_id=event_context.get("caseId"),
        metadata=_scrub(_json_safe(metadata)) if isinstance(metadata, Mapping) else None,
        provider_request_id=provider_request_id,
        attempt_id=attempt_id,
        parent_event_id=parent_event_id,
    )
    _dispatch_event(explicit_event)
    return event_id


def _project_id() -> str:
    # Keep client.py independent of patch internals while preserving the
    # existing no-op test mode. The transport owns the authoritative DSN.
    from costkey.patch import _state
    return _state.project_id


get_current_context = get_context


def get_runtime_config_status() -> str:
    from costkey.patch import runtime_config_status
    return runtime_config_status()


def register_extractor(extractor: Any) -> None:
    """Register a custom provider extractor."""
    _register_extractor(extractor)


def register_pricing(model: str, input_per_1m: float, output_per_1m: float, **kwargs: Any) -> None:
    """Register custom model pricing."""
    _register_pricing(model, input_per_1m, output_per_1m, **kwargs)
