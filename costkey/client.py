"""CostKey Python SDK — AI cost observability."""
from __future__ import annotations
import os
import uuid
import logging
import threading
from contextlib import contextmanager
from contextvars import copy_context
from typing import Any, Callable, Generator
from costkey.types import CostKeyOptions, CostKeyEvent
from costkey.transport import Transport
from costkey.patch import patch, unpatch, _context, set_context, get_context
from costkey.providers import register_extractor as _register_extractor
from costkey.pricing import register_pricing as _register_pricing

logger = logging.getLogger("costkey")

_transport: Transport | None = None
_initialized = False


def _parse_dsn(dsn: str) -> tuple[str, str, str]:
    """Parse DSN → (endpoint, auth_key, project_id).

    Expected format: https://<key>@app.costkey.dev/<project-id>
    """
    from urllib.parse import urlparse

    _DSN_HELP = 'Invalid DSN format. Expected: https://<key>@app.costkey.dev/<project-id>'

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


def init(dsn: str, *, capture_body: bool = True,
         before_send: Callable[[CostKeyEvent], CostKeyEvent | None] | None = None,
         max_batch_size: int = 50, flush_interval: float = 5.0,
         debug: bool = False, default_context: dict[str, Any] | None = None,
         release: str | None = None,
         project_root: str | None = None,
         scan_callgraph: bool = True) -> None:
    """
    Initialize CostKey. Call once at app startup.

    >>> import costkey
    >>> costkey.init(dsn="https://ck_abc123@app.costkey.dev/my-project")
    >>> # That's it. Every AI call is now tracked.
    """
    global _transport, _initialized

    if _initialized:
        if debug:
            logger.warning("[costkey] Already initialized, skipping")
        return

    endpoint, auth_key, project_id = _parse_dsn(dsn)

    _transport = Transport(
        endpoint=endpoint, auth_key=auth_key,
        max_batch_size=max_batch_size, flush_interval=flush_interval,
        debug=debug, release=release,
    )

    patch(
        transport=_transport, project_id=project_id,
        capture_body=capture_body, before_send=before_send,
        default_context=default_context or {}, debug=debug,
    )

    _transport.start()
    _initialized = True

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


def flush() -> None:
    """Flush all pending events without shutting down."""
    if _transport:
        _transport.flush()


@contextmanager
def with_context(**kwargs: Any) -> Generator[None, None, None]:
    """
    Tag AI calls with custom context.

    >>> with costkey.with_context(task="summarize", team="search"):
    ...     openai.chat.completions.create(...)
    """
    parent = get_context()
    merged = {**parent, **kwargs}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def start_trace(name: str | None = None, trace_id: str | None = None) -> Generator[None, None, None]:
    """
    Start a trace. All AI calls inside are grouped under one trace ID.

    >>> with costkey.start_trace(name="POST /api/search"):
    ...     classify_intent(query)
    ...     results = search(query)
    ...     summary = summarize(results)
    """
    tid = trace_id or uuid.uuid4().hex
    with with_context(traceId=tid, traceName=name):
        yield


def register_extractor(extractor: Any) -> None:
    """Register a custom provider extractor."""
    _register_extractor(extractor)


def register_pricing(model: str, input_per_1m: float, output_per_1m: float, **kwargs: Any) -> None:
    """Register custom model pricing."""
    _register_pricing(model, input_per_1m, output_per_1m, **kwargs)
