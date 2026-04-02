"""Monkey-patch HTTP clients to intercept AI provider calls."""
from __future__ import annotations
import json
import time
import uuid
import re
import logging
import functools
from contextvars import ContextVar
from typing import Any, Callable, Iterator, AsyncIterator
from costkey.types import CostKeyEvent, NormalizedUsage, StreamTiming, Provider
from costkey.providers import find_extractor
from costkey.stack import capture_call_site
from costkey.transport import Transport

logger = logging.getLogger("costkey")

_context: ContextVar[dict[str, Any]] = ContextVar("costkey_context", default={})

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


def _is_streaming_request(request_body: Any) -> bool:
    if isinstance(request_body, dict):
        return request_body.get("stream") is True
    return False


def _extract_sse_usage(text: str, extractor: Any) -> NormalizedUsage | None:
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
            usage = extractor.extract_usage(parsed)
            if usage:
                return usage
        except Exception:
            continue
    return None


def _extract_sse_model(text: str, request_body: Any, extractor: Any) -> str | None:
    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
            model = extractor.extract_model(request_body, parsed)
            if model:
                return model
        except Exception:
            continue
    return None


# ── httpx patching ──

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

            is_streaming = _is_streaming_request(request_body) or kwargs.get("stream", False)

            response = _state._original_httpx_send(self, request, **kwargs)

            if is_streaming:
                # Wrap the response's iteration methods to capture streaming data
                _wrap_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                )
            else:
                # Non-streaming — read body normally
                duration_ms = (time.perf_counter() - start) * 1000
                response_body = None
                try:
                    response_body = response.json()
                except Exception:
                    pass
                _process(extractor, url, request.method, response.status_code,
                         request_body, response_body, duration_ms,
                         False, None, call_site, ctx)

            return response

        httpx.Client.send = patched_send

        # Async client
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

            is_streaming = _is_streaming_request(request_body) or kwargs.get("stream", False)

            response = await _state._original_httpx_async_send(self, request, **kwargs)

            if is_streaming:
                _wrap_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                response_body = None
                try:
                    response_body = response.json()
                except Exception:
                    pass
                _process(extractor, url, request.method, response.status_code,
                         request_body, response_body, duration_ms,
                         False, None, call_site, ctx)

            return response

        httpx.AsyncClient.send = patched_async_send

    except ImportError:
        if _state.debug:
            logger.debug("[costkey] httpx not installed, skipping patch")


def _wrap_streaming_response(
    response: Any, extractor: Any, url: str, method: str,
    request_body: Any, start: float,
    call_site: Any, ctx: dict[str, Any],
) -> None:
    """
    Wrap httpx Response iteration methods to capture streaming data.

    The Anthropic/OpenAI SDKs use response.iter_lines() or response.iter_bytes()
    to consume SSE streams. We wrap these methods to accumulate the text,
    then extract usage + timing when iteration finishes.
    """
    accumulated_text = ""
    first_chunk_time: float | None = None
    chunk_count = 0
    event_sent = False

    def _on_chunk(chunk_bytes: bytes) -> None:
        nonlocal accumulated_text, first_chunk_time, chunk_count
        chunk_count += 1
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        try:
            accumulated_text += chunk_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass

    def _on_done() -> None:
        nonlocal event_sent
        if event_sent:
            return
        event_sent = True

        end = time.perf_counter()
        duration_ms = (end - start) * 1000

        usage = _extract_sse_usage(accumulated_text, extractor)
        model = _extract_sse_model(accumulated_text, request_body, extractor)

        ttft = ((first_chunk_time - start) * 1000) if first_chunk_time else None
        tps = None
        if usage and usage.output_tokens and first_chunk_time:
            stream_secs = end - first_chunk_time
            if stream_secs > 0:
                tps = usage.output_tokens / stream_secs

        stream_timing = StreamTiming(
            ttft=round(ttft, 2) if ttft else None,
            tps=round(tps, 2) if tps else None,
            stream_duration=round(duration_ms, 2),
            chunk_count=chunk_count,
        )

        _process(extractor, url, method, response.status_code,
                 request_body, None, duration_ms,
                 True, stream_timing, call_site, ctx)

    # Wrap iter_bytes
    original_iter_bytes = response.iter_bytes

    @functools.wraps(original_iter_bytes)
    def wrapped_iter_bytes(*a: Any, **kw: Any) -> Iterator[bytes]:
        try:
            for chunk in original_iter_bytes(*a, **kw):
                _on_chunk(chunk)
                yield chunk
        finally:
            _on_done()

    response.iter_bytes = wrapped_iter_bytes

    # Wrap iter_lines
    original_iter_lines = response.iter_lines

    @functools.wraps(original_iter_lines)
    def wrapped_iter_lines(*a: Any, **kw: Any) -> Iterator[str]:
        try:
            for line in original_iter_lines(*a, **kw):
                _on_chunk(line.encode("utf-8") if isinstance(line, str) else line)
                yield line
        finally:
            _on_done()

    response.iter_lines = wrapped_iter_lines

    # Wrap iter_text
    if hasattr(response, "iter_text"):
        original_iter_text = response.iter_text

        @functools.wraps(original_iter_text)
        def wrapped_iter_text(*a: Any, **kw: Any) -> Iterator[str]:
            try:
                for text in original_iter_text(*a, **kw):
                    _on_chunk(text.encode("utf-8") if isinstance(text, str) else text)
                    yield text
            finally:
                _on_done()

        response.iter_text = wrapped_iter_text

    # Wrap iter_raw
    if hasattr(response, "iter_raw"):
        original_iter_raw = response.iter_raw

        @functools.wraps(original_iter_raw)
        def wrapped_iter_raw(*a: Any, **kw: Any) -> Iterator[bytes]:
            try:
                for chunk in original_iter_raw(*a, **kw):
                    _on_chunk(chunk)
                    yield chunk
            finally:
                _on_done()

        response.iter_raw = wrapped_iter_raw

    # Wrap read() for cases where the body is read all at once
    original_read = response.read

    @functools.wraps(original_read)
    def wrapped_read(*a: Any, **kw: Any) -> bytes:
        data = original_read(*a, **kw)
        _on_chunk(data)
        # Only fire _on_done for successful responses (200-299).
        # Error responses (4xx/5xx) use read() to get the error body,
        # not as stream completion.
        status = getattr(response, "status_code", 200)
        if 200 <= status < 300:
            _on_done()
        return data

    response.read = wrapped_read

    # Wrap async variants if they exist
    if hasattr(response, "aiter_bytes"):
        original_aiter_bytes = response.aiter_bytes

        @functools.wraps(original_aiter_bytes)
        async def wrapped_aiter_bytes(*a: Any, **kw: Any) -> AsyncIterator[bytes]:
            try:
                async for chunk in original_aiter_bytes(*a, **kw):
                    _on_chunk(chunk)
                    yield chunk
            finally:
                _on_done()

        response.aiter_bytes = wrapped_aiter_bytes

    if hasattr(response, "aiter_lines"):
        original_aiter_lines = response.aiter_lines

        @functools.wraps(original_aiter_lines)
        async def wrapped_aiter_lines(*a: Any, **kw: Any) -> AsyncIterator[str]:
            try:
                async for line in original_aiter_lines(*a, **kw):
                    _on_chunk(line.encode("utf-8") if isinstance(line, str) else line)
                    yield line
            finally:
                _on_done()

        response.aiter_lines = wrapped_aiter_lines

    if hasattr(response, "aread"):
        original_aread = response.aread

        @functools.wraps(original_aread)
        async def wrapped_aread(*a: Any, **kw: Any) -> bytes:
            data = await original_aread(*a, **kw)
            _on_chunk(data)
            status = getattr(response, "status_code", 200)
            if 200 <= status < 300:
                _on_done()
            return data

        response.aread = wrapped_aread


# ── requests patching ──

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
                     request_body, response_body, duration_ms,
                     False, None, call_site, ctx)

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


# ── Event processing ──

def _process(extractor: Any, url: str, method: str, status_code: int | None,
             request_body: Any, response_body: Any,
             duration_ms: float, streaming: bool, stream_timing: StreamTiming | None,
             call_site: Any, ctx: dict[str, Any]) -> None:
    try:
        usage = extractor.extract_usage(response_body) if response_body else None
        model = extractor.extract_model(request_body, response_body)

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
            cost_usd=None,  # Server calculates cost
            duration_ms=round(duration_ms, 2),
            streaming=streaming,
            stream_timing=stream_timing,
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
