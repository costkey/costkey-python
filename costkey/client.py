"""CostKey Python SDK — AI cost observability."""
from __future__ import annotations
import datetime
import json
import math
import os
import uuid
import logging
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Generator
from costkey.types import CostKeyEvent
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

logger = logging.getLogger("costkey")

_transport: Transport | None = None
_initialized = False
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

    Expected format: http(s)://<key>@<your CostKey host>[/<base-path>]/<project-id>
    """
    from urllib.parse import urlparse

    _DSN_HELP = 'Invalid DSN format. Expected: http(s)://<key>@<your CostKey host>/<project-id>. Find it in your CostKey console under Settings, Projects.'

    if not dsn or not isinstance(dsn, str):
        raise ValueError(f"[costkey] {_DSN_HELP}")

    parsed = urlparse(dsn)

    if not parsed.hostname:
        raise ValueError(f"[costkey] {_DSN_HELP}")

    auth_key = parsed.username or ""
    if not auth_key:
        raise ValueError(f"[costkey] DSN missing auth key. {_DSN_HELP}")

    # The project ID is the last path segment. Anything before it is the path
    # CostKey is served under, e.g. https://<key>@portal.example.com/costkey/<project-id>.
    segments = [part for part in parsed.path.split("/") if part]
    project_id = segments.pop() if segments else ""
    if not project_id:
        raise ValueError(f"[costkey] DSN missing project ID. {_DSN_HELP}")

    endpoint = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        endpoint += f":{parsed.port}"
    if segments:
        endpoint += "/" + "/".join(segments)
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


def _caller_map(value: Any, label: str) -> dict[str, Any] | None:
    """A caller-supplied mapping, made JSON-safe and otherwise untouched."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        try:
            from costkey.telemetry import _plain
            value = _plain(value)
        except Exception:
            raise TypeError(f"{label} must be a mapping") from None
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    return {str(key): _json_safe(item) for key, item in value.items()}


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
         commit: str | None = None,
         repository: str | None = None,
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
    >>> costkey.init(dsn="http://ck_abc123@localhost:4100/my-project")
    >>> # That's it. Supported provider calls are observed in-process.

    If no DSN is provided, falls back to COSTKEY_DSN env var.
    If neither exists, silently no-ops (test mode).

    ``release`` falls back to ``COSTKEY_RELEASE``, then to the commit. The
    commit and repository are detected when not given: ``COSTKEY_COMMIT`` /
    ``COSTKEY_REPOSITORY``, then CI and hosting platform variables, then the
    ``.git`` directory above ``project_root`` (or the working directory).
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

    from costkey.source import detect_commit, detect_repository
    resolved_commit = detect_commit(commit, start=project_root)
    resolved_repository = detect_repository(repository, start=project_root)
    release = release or os.environ.get("COSTKEY_RELEASE") or resolved_commit

    _transport = Transport(
        endpoint=endpoint, auth_key=auth_key,
        max_batch_size=max_batch_size, flush_interval=flush_interval,
        debug=debug, release=release,
        commit=resolved_commit, repository=resolved_repository,
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
        logger.info(f"[costkey] release={release or 'unknown'} commit={resolved_commit or 'unknown'} "
                    f"repository={resolved_repository or 'unknown'}")


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
                # The raw exception class name; the server decides whether it
                # was a failure or a cancellation.
                emit_lifecycle(
                    entity_type="trace",
                    entity_id=tid,
                    lifecycle="failed",
                    context=child_context,
                    trace_id=tid,
                    trace_name=trace_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=elapsed,
                    error_type=type(exc).__name__,
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
                emit_lifecycle(
                    entity_type="execution",
                    entity_id=eid,
                    lifecycle="failed",
                    context=child_context,
                    execution_id=eid,
                    group_name=stable_group,
                    group_kind=kind,
                    group_id=group_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=elapsed,
                    error_type=type(exc).__name__,
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


def _span_call_site() -> Any:
    from costkey.stack import capture_call_site
    try:
        return capture_call_site()
    except Exception:
        return None


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
        detached: bool = False,
        framework: str | None = None,
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
        # A detached span is recorded without becoming the active span of the
        # current context (for spans owned by another tracer, e.g. OTel).
        self.detached = detached
        # Integrations pass their framework's raw attributes; they are sent
        # under ``attributes.<framework>`` (see ``framework_attributes``).
        self.framework = framework
        self._owns_trace = False
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
            detached=self.detached, framework=self.framework,
        )

    def _wire_attributes(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if self.framework:
            from costkey.patch import framework_attributes
            return framework_attributes(self.framework, dict(values))
        return dict(values)

    def __enter__(self) -> SpanHandle:
        has_trace = bool(self.context.get("traceId") or _current_context().get("traceId"))
        if not has_trace and not self.detached:
            self._trace_cm = start_trace(name=self.name)
            self._trace_cm.__enter__()
        ctx = get_context()
        sid = self.span_id or f"sp_{uuid.uuid4().hex}"
        inherited = ctx.get("spanId") if ctx.get("spanId") != sid else None
        child = {**ctx, **_canonical_context(self.context), "spanId": sid}
        if self.detached and not has_trace:
            # No trace to join: this span starts one without touching the
            # context (the lifecycle events are emitted by hand).
            child["traceId"] = uuid.uuid4().hex
            child["traceName"] = self.name
            self._owns_trace = True
        if self.kind == "agent":
            child["agent"] = self.name
        if self.kind == "tool":
            child["tool"] = self.name
        if not self.detached:
            self._previous = _context.get()
            self._token = _context.set(child)
        from costkey.patch import _span_opened
        _span_opened(sid)
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
            # Where the span was opened, raw; the server attributes calls made
            # inside it with no code of their own (a framework's task) to it.
            "call_site": _span_call_site(),
        }
        if self._owns_trace:
            self._emit_trace("started")
        try:
            emit_span(**self._base, status="running", attributes=self._wire_attributes(handle._attributes),
                      input=self.input)
        except Exception:
            logger.debug("[costkey] span start dropped", exc_info=True)
        return handle

    def _emit_trace(self, lifecycle: str, error: str | None = None) -> None:
        trace_id = self._base.get("trace_id")
        try:
            ended = None if lifecycle == "started" else _iso_timestamp()
            emit_lifecycle(
                entity_type="trace", entity_id=str(trace_id), lifecycle=lifecycle,
                context=dict(self._base.get("context") or {}), trace_id=str(trace_id),
                trace_name=self._base.get("trace_name"), started_at=self._base.get("started_at"),
                ended_at=ended,
                duration_ms=(
                    (datetime.datetime.now(datetime.timezone.utc) - self._started).total_seconds() * 1000
                    if ended and self._started else None
                ),
                error_type=error,
            )
        except Exception:
            logger.debug("[costkey] trace lifecycle dropped", exc_info=True)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            # The raw exception class name (CancelledError included); the
            # server maps it to failed or cancelled.
            if exc_type is None:
                status, error = "ok", None
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
                attributes=self._wire_attributes(handle._attributes) if handle else None,
                output=handle._output if handle else None,
            )
        except Exception:
            logger.debug("[costkey] span end dropped", exc_info=True)
        finally:
            from costkey.patch import _span_closed
            _span_closed(self._base.get("span_id"))
            if self._owns_trace:
                self._owns_trace = False
                self._emit_trace("completed" if exc_type is None else "failed",
                                 None if exc_type is None else exc_type.__name__)
            if not self.detached:
                self._restore_context(exc_type, exc, tb)
        return False

    def _restore_context(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            _context.reset(self._token)
        except ValueError:
            # Closed from a different context (e.g. framework hooks): restore by value.
            _context.set(self._previous or {})
        if self._trace_cm is not None:
            trace_cm, self._trace_cm = self._trace_cm, None
            try:
                trace_cm.__exit__(exc_type, exc, tb)
            except ValueError:
                # The trace scope was opened in another context.
                _context.set(self._previous or {})

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
    provider: Any = None,
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
    """Report one external usage component exactly as the caller describes it.

    The SDK sends what you pass: no defaults are filled in and nothing is
    derived. Dedup, revision fences, charge ids, currency and pricing are the
    server's job. ``event_id``, ``component_id`` and a non-negative integer
    ``revision`` are required because the server rejects usage without them.
    """
    values = dict(event or {})
    # Mapping input is a migration seam for callers that already construct a
    # wire-shaped usage object. Explicit keyword arguments take precedence.
    event_id = event_id or id or values.pop("id", None) or values.pop("eventId", None) or values.pop("event_id", None)
    component_id = component_id or values.pop("componentId", None) or values.pop("component_id", None) or values.pop("component", None)
    charge_id = charge_id or values.pop("chargeId", None) or values.pop("charge_id", None)
    provider = provider if provider is not None else values.pop("provider", None)
    model = model or values.pop("model", None)
    unit = unit or values.pop("unit", None)
    usage = usage if usage is not None else values.pop("usage", values.pop("nativeUsage", None))
    native_units = native_units or values.pop("nativeUnits", None) or values.pop("native_units", None)
    amount = amount if amount is not None else values.pop("amount", None)
    cost_usd = cost_usd if cost_usd is not None else values.pop("costUsd", values.pop("cost_usd", None))
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

    usage_data = _caller_map(usage, "usage")
    native_data = _caller_map(native_units, "native_units")
    if native_data:
        usage_data = {**(usage_data or {}), **native_data}
    event_context = {
        **_current_context(),
        **_canonical_context(context),
    }
    if trace_id is not None:
        event_context["traceId"] = trace_id
    if execution_id is not None:
        event_context["executionId"] = execution_id

    project_id = _project_id()
    from costkey.patch import _state
    if not project_id or _state.transport is None:
        # Before init (or in test mode) usage is a validation-only no-op.
        return event_id

    provider_value = getattr(provider, "value", provider)
    explicit_event = CostKeyEvent(
        id=event_id,
        timestamp=_iso_timestamp(timestamp),
        project_id=project_id,
        provider=str(provider_value) if provider_value is not None else "unknown",
        model=model,
        cost_usd=_json_safe(cost_usd),
        context=_scrub(event_context),
        trace_id=event_context.get("traceId"),
        trace_name=event_context.get("traceName"),
        event_type="usage",
        source="explicit",
        component_id=component_id,
        charge_id=_non_empty(charge_id),
        revision=revision,
        aggregation=aggregation,
        native_usage=usage_data,
        billing_units=_caller_map(billing_units, "billing_units"),
        amount=_json_safe(amount),
        currency=currency,
        amount_kind=amount_kind,
        pricing_id=pricing_id,
        unit=unit,
        unit_kind=unit_kind,
        unit_value=_json_safe(unit_value),
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
