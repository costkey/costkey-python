"""Monkey-patch HTTP clients to intercept AI provider calls."""
from __future__ import annotations
import json
import time
import uuid
import re
import logging
from contextvars import ContextVar
from typing import Any, Callable
from costkey.types import CostKeyEvent, NormalizedUsage, Provider
from costkey.providers import find_extractor
from costkey.stack import capture_call_site
from costkey.pricing import compute_cost
from costkey.transport import Transport

logger = logging.getLogger("costkey")

# Context var for tracing / manual context
_context: ContextVar[dict[str, Any]] = ContextVar("costkey_context", default={})

# Secret patterns to scrub from bodies
_SECRET_PATTERNS = [
    re.compile(r"^sk-[a-zA-Z0-9]{20,}$"),
    re.compile(r"^sk-ant-[a-zA-Z0-9\-]{20,}$"),
    re.compile(r"^AIza[a-zA-Z0-9_\-]{30,}$"),
    re.compile(r"^Bearer\s+.{20,}$"),
    re.compile(r"^eyJ[a-zA-Z0-9_\-]{20,}"),
]
_SECRET_KEYS = frozenset({
    "api_key", "apikey", "api-key", "secret", "secret_key",
    "token", "access_token", "refresh_token", "password",
    "authorization", "auth", "private_key",
})


def _scrub(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        for pat in _SECRET_PATTERNS:
            if pat.match(obj):
                return "[REDACTED]"
        return obj
    if isinstance(obj, list):
        return [_scrub(item) for item in obj]
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k.lower() in _SECRET_KEYS else _scrub(v)
            for k, v in obj.items()
        }
    return obj


class _PatchState:
    def __init__(self) -> None:
        self.transport: Transport | None = None
        self.project_id: str = ""
        self.capture_body: bool = True
        self.before_send: Callable | None = None
        self.default_context: dict[str, Any] = {}
        self.debug: bool = False
        self._original_httpx_send: Any = None
        self._original_httpx_async_send: Any = None
        self._original_requests_send: Any = None
        self.patched = False


_state = _PatchState()


def patch(transport: Transport, project_id: str, capture_body: bool,
          before_send: Callable | None, default_context: dict[str, Any], debug: bool) -> None:
    if _state.patched:
        return

    _state.transport = transport
    _state.project_id = project_id
    _state.capture_body = capture_body
    _state.before_send = before_send
    _state.default_context = default_context
    _state.debug = debug

    _patch_httpx()
    _patch_requests()
    _state.patched = True


def unpatch() -> None:
    _unpatch_httpx()
    _unpatch_requests()
    _state.patched = False


def _patch_httpx() -> None:
    try:
        import httpx

        _state._original_httpx_send = httpx.Client.send

        def patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
            url = str(request.url)
            extractor = find_extractor(url)

            if not extractor:
                return _state._original_httpx_send(self, request, **kwargs)

            call_site = capture_call_site()
            ctx = {**_state.default_context, **_context.get()}
            start = time.perf_counter()

            request_body = None
            if request.content:
                try:
                    request_body = json.loads(request.content)
                except Exception:
                    pass

            response = _state._original_httpx_send(self, request, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            try:
                response_body = response.json()
            except Exception:
                response_body = None

            _process(extractor, url, request.method, response.status_code,
                     request_body, response_body, duration_ms, call_site, ctx)

            return response

        httpx.Client.send = patched_send

        # Also patch async client
        _state._original_httpx_async_send = httpx.AsyncClient.send

        async def patched_async_send(self: Any, request: Any, **kwargs: Any) -> Any:
            url = str(request.url)
            extractor = find_extractor(url)

            if not extractor:
                return await _state._original_httpx_async_send(self, request, **kwargs)

            call_site = capture_call_site()
            ctx = {**_state.default_context, **_context.get()}
            start = time.perf_counter()

            request_body = None
            if request.content:
                try:
                    request_body = json.loads(request.content)
                except Exception:
                    pass

            response = await _state._original_httpx_async_send(self, request, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            try:
                response_body = response.json()
            except Exception:
                response_body = None

            _process(extractor, url, request.method, response.status_code,
                     request_body, response_body, duration_ms, call_site, ctx)

            return response

        httpx.AsyncClient.send = patched_async_send

    except ImportError:
        if _state.debug:
            logger.debug("[costkey] httpx not installed, skipping patch")


def _unpatch_httpx() -> None:
    try:
        import httpx
        if _state._original_httpx_send:
            httpx.Client.send = _state._original_httpx_send
        if _state._original_httpx_async_send:
            httpx.AsyncClient.send = _state._original_httpx_async_send
    except ImportError:
        pass


def _patch_requests() -> None:
    try:
        import requests

        _state._original_requests_send = requests.Session.send

        def patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
            url = str(request.url)
            extractor = find_extractor(url)

            if not extractor:
                return _state._original_requests_send(self, request, **kwargs)

            call_site = capture_call_site()
            ctx = {**_state.default_context, **_context.get()}
            start = time.perf_counter()

            request_body = None
            if request.body:
                try:
                    request_body = json.loads(request.body)
                except Exception:
                    pass

            response = _state._original_requests_send(self, request, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            try:
                response_body = response.json()
            except Exception:
                response_body = None

            _process(extractor, url, request.method, response.status_code,
                     request_body, response_body, duration_ms, call_site, ctx)

            return response

        requests.Session.send = patched_send

    except ImportError:
        if _state.debug:
            logger.debug("[costkey] requests not installed, skipping patch")


def _unpatch_requests() -> None:
    try:
        import requests
        if _state._original_requests_send:
            requests.Session.send = _state._original_requests_send
    except ImportError:
        pass


def _process(extractor: Any, url: str, method: str, status_code: int | None,
             request_body: Any, response_body: Any,
             duration_ms: float, call_site: Any, ctx: dict[str, Any]) -> None:
    try:
        usage = extractor.extract_usage(response_body) if response_body else None
        model = extractor.extract_model(request_body, response_body)
        cost_usd = compute_cost(model, usage) if model and usage else None

        event = CostKeyEvent(
            id=uuid.uuid4().hex,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            project_id=_state.project_id,
            provider=extractor.provider,
            model=model,
            url=url,
            method=method,
            status_code=status_code,
            usage=usage,
            cost_usd=cost_usd,
            duration_ms=round(duration_ms, 2),
            streaming=False,
            call_site=call_site,
            context=ctx,
            request_body=_scrub(request_body) if _state.capture_body else None,
            response_body=_scrub(response_body) if _state.capture_body else None,
        )

        if _state.before_send:
            try:
                event = _state.before_send(event)
            except Exception:
                if _state.debug:
                    logger.warning("[costkey] before_send threw, dropping event")
                return

        if event and _state.transport:
            _state.transport.enqueue(event)
    except Exception:
        if _state.debug:
            logger.warning("[costkey] Error processing event", exc_info=True)


def get_context() -> dict[str, Any]:
    return _context.get()


def set_context(ctx: dict[str, Any]) -> None:
    _context.set(ctx)
