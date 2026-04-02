"""Monkey-patch HTTP clients to intercept AI provider calls."""
from __future__ import annotations
import json
import time
import uuid
import re
import logging
from contextvars import ContextVar
from typing import Any, Callable
from costkey.types import CostKeyEvent, NormalizedUsage, StreamTiming, Provider
from costkey.providers import find_extractor
from costkey.stack import capture_call_site
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


def _is_streaming_request(request_body: Any) -> bool:
    """Check if the request body has stream: true."""
    if isinstance(request_body, dict):
        return request_body.get("stream") is True
    return False


def _extract_sse_usage(text: str, extractor: Any) -> NormalizedUsage | None:
    """Extract usage from accumulated SSE text — scan from the end for the final data chunk."""
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
    """Extract model name from SSE text."""
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

            is_streaming = _is_streaming_request(request_body)

            # Force stream=True in kwargs so we can intercept the response stream
            if is_streaming:
                kwargs.setdefault("stream", True)

            response = _state._original_httpx_send(self, request, **kwargs)

            if is_streaming and hasattr(response, "stream"):
                # Streaming response — wrap the stream to capture timing + usage
                _handle_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                )
            else:
                # Non-streaming — read response body
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

            is_streaming = _is_streaming_request(request_body)

            if is_streaming:
                kwargs.setdefault("stream", True)

            response = await _state._original_httpx_async_send(self, request, **kwargs)

            if is_streaming and hasattr(response, "stream"):
                _handle_streaming_response(
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


def _handle_streaming_response(
    response: Any, extractor: Any, url: str, method: str,
    request_body: Any, start: float,
    call_site: Any, ctx: dict[str, Any],
) -> None:
    """Hook into httpx streaming response to capture TTFT, TPS, and usage."""
    original_iter = response.stream.__class__.__iter__
    original_aiter = getattr(response.stream.__class__, "__aiter__", None)

    accumulated_text = ""
    first_chunk_time: float | None = None
    chunk_count = 0
    event_sent = False

    class StreamWrapper:
        """Wraps the httpx stream to intercept chunks without buffering."""
        def __init__(self, original_stream: Any) -> None:
            self._stream = original_stream

        def __iter__(self) -> Any:
            nonlocal accumulated_text, first_chunk_time, chunk_count, event_sent
            try:
                for chunk in self._stream:
                    chunk_count += 1
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    if isinstance(chunk, bytes):
                        accumulated_text += chunk.decode("utf-8", errors="replace")
                    else:
                        accumulated_text += str(chunk)
                    yield chunk
            finally:
                if not event_sent:
                    event_sent = True
                    _finalize_stream(
                        extractor, url, method, response.status_code,
                        request_body, accumulated_text,
                        start, first_chunk_time, chunk_count,
                        call_site, ctx,
                    )

        async def __aiter__(self) -> Any:
            nonlocal accumulated_text, first_chunk_time, chunk_count, event_sent
            try:
                async for chunk in self._stream:
                    chunk_count += 1
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    if isinstance(chunk, bytes):
                        accumulated_text += chunk.decode("utf-8", errors="replace")
                    else:
                        accumulated_text += str(chunk)
                    yield chunk
            finally:
                if not event_sent:
                    event_sent = True
                    _finalize_stream(
                        extractor, url, method, response.status_code,
                        request_body, accumulated_text,
                        start, first_chunk_time, chunk_count,
                        call_site, ctx,
                    )

        def close(self) -> None:
            if hasattr(self._stream, "close"):
                self._stream.close()

        async def aclose(self) -> None:
            if hasattr(self._stream, "aclose"):
                await self._stream.aclose()

    # Replace the stream object
    response.stream = StreamWrapper(response.stream)


def _finalize_stream(
    extractor: Any, url: str, method: str, status_code: int | None,
    request_body: Any, accumulated_text: str,
    start: float, first_chunk_time: float | None, chunk_count: int,
    call_site: Any, ctx: dict[str, Any],
) -> None:
    """Called when the stream ends — extract usage, compute timing, send event."""
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

    _process(extractor, url, method, status_code,
             request_body, None, duration_ms,
             True, stream_timing, call_site, ctx)


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


def _process(extractor: Any, url: str, method: str, status_code: int | None,
             request_body: Any, response_body: Any,
             duration_ms: float, streaming: bool, stream_timing: StreamTiming | None,
             call_site: Any, ctx: dict[str, Any]) -> None:
    try:
        usage = extractor.extract_usage(response_body) if response_body else None
        model = extractor.extract_model(request_body, response_body)

        # For streaming, usage comes from the stream finalizer, not response_body
        if streaming and stream_timing and not usage:
            usage = None  # Already handled in _finalize_stream

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
