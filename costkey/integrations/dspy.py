"""DSPy integration (``pip install "costkey[dspy]"``).

Register the callback globally (or per module with ``callbacks=[...]``)::

    import dspy
    from costkey.integrations.dspy import CostKeyCallback

    dspy.configure(lm=lm, callbacks=[CostKeyCallback()])

Module calls (``forward``) become ``chain`` spans (``ReAct`` and other agent
modules ``agent`` spans), tools ``tool`` spans, and ``Evaluate`` runs and
optimizer ``compile`` calls ``workflow`` spans. LM calls and adapter
format/parse calls get no span: the transport capture records each model call
inside the module that made it.

DSPy 3.4 sends model requests through its vendored ``lm15`` HTTP transport
(raw sockets), which the httpx/requests/aiohttp patches never see. Creating a
``CostKeyCallback`` (or calling ``instrument_transport()``) therefore also
patches ``lm15``'s ``StdlibTransport.stream`` and
``StdlibAsyncTransport._do_stream`` so those requests produce the same raw
call telemetry as any other transport. LiteLLM-backed LMs keep going through
httpx/aiohttp and are captured there.
"""
from __future__ import annotations

from typing import Any

from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback

import functools
import threading

from costkey.integrations._tracker import SpanTracker

FRAMEWORK = "dspy"
_AGENT_MODULES = frozenset({"ReAct", "CodeAct", "ProgramOfThought", "RLM"})


def _name(instance: Any, fallback: str) -> str:
    name = getattr(instance, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(instance).__name__ if instance is not None else fallback


class CostKeyCallback(BaseCallback):
    """A DSPy callback that records modules, tools, evaluations and compiles as spans."""

    def __init__(self) -> None:
        self._tracker = SpanTracker(FRAMEWORK)
        instrument_transport()

    def _start(self, call_id: str, name: str, kind: str, inputs: Any = None,
               attributes: dict[str, Any] | None = None) -> None:
        self._tracker.start(call_id, name, kind, parent=ACTIVE_CALL_ID.get(),
                            attributes={"call_id": call_id, **(attributes or {})}, input=inputs)

    def _skip(self, call_id: str) -> None:
        self._tracker.skip(call_id, ACTIVE_CALL_ID.get())

    def _end(self, call_id: str, outputs: Any = None, exception: BaseException | None = None) -> None:
        if self._tracker.is_open(call_id):
            self._tracker.end(call_id, error=exception)
        else:
            self._tracker.forget(call_id)

    # modules
    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        kind = "agent" if type(instance).__name__ in _AGENT_MODULES else "chain"
        self._start(call_id, type(instance).__name__, kind, inputs)

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._end(call_id, outputs, exception)

    # tools
    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._start(call_id, _name(instance, "tool"), "tool", inputs)

    def on_tool_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._end(call_id, outputs, exception)

    # evaluation and optimization
    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._start(call_id, "Evaluate", "workflow")

    def on_evaluate_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._end(call_id, None, exception)

    def on_compile_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._start(call_id, f"{type(instance).__name__}.compile", "workflow")

    def on_compile_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._end(call_id, None, exception)

    # model calls and adapters: no span (the HTTP capture records the calls)
    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._skip(call_id)

    def on_lm_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._tracker.forget(call_id)

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._skip(call_id)

    def on_adapter_format_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._tracker.forget(call_id)

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._skip(call_id)

    def on_adapter_parse_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        self._tracker.forget(call_id)


# ── DSPy's vendored lm15 transport ──

_transport_lock = threading.Lock()
_transport_patched = False


def _lm15_call(request: Any) -> Any:
    from costkey.patch import _Call, _captured_provider
    from costkey import telemetry as _telemetry

    url = str(getattr(request, "url", ""))
    provider = _captured_provider(url)
    if provider is None:
        return None
    headers = list(getattr(request, "headers", None) or [])
    body = getattr(request, "body", b"") or b""
    kind = _telemetry.content_type(headers)
    return _Call(provider, url, str(getattr(request, "method", "POST")), headers=headers,
                 body=_telemetry.body_skeleton(body, kind) if body else None,
                 body_bytes=len(body), capture=lambda: _json_or_none(body))


def _json_or_none(body: bytes) -> Any:
    import json
    try:
        return json.loads(body)
    except Exception:
        return None


def instrument_transport() -> bool:
    """Capture requests sent through DSPy's vendored lm15 transport."""
    global _transport_patched
    try:
        from dspy._vendor.lm15.transports import StdlibAsyncTransport, StdlibTransport
    except Exception:
        return False
    from costkey.patch import (
        _emit_when_collected, _run_admission_async, _run_admission_sync, _settle_async,
        _settle_sync, _StreamRecorder,
    )

    with _transport_lock:
        if _transport_patched:
            return True
        original_stream = StdlibTransport.stream
        original_do_stream = StdlibAsyncTransport._do_stream

        @functools.wraps(original_stream)
        def stream(self: Any, request: Any) -> Any:
            call = _lm15_call(request)
            if call is None:
                return original_stream(self, request)
            call.decision = _run_admission_sync(call.admission_payload())
            try:
                response = original_stream(self, request)
            except BaseException as exc:
                call.finish(error=exc, responded=False)
                _settle_sync(call.decision, call.event_id)
                raise
            recorder = _StreamRecorder(call, status_code=response.status, headers=response.headers)
            chunks = response._chunks

            def recorded() -> Any:
                try:
                    for chunk in chunks:
                        recorder.chunk(chunk)
                        yield chunk
                except BaseException as exc:
                    recorder.fail(exc)
                    raise
                finally:
                    recorder.done()

            response._chunks = recorded()
            _emit_when_collected(response, recorder)
            return response

        @functools.wraps(original_do_stream)
        async def do_stream(self: Any, request: Any) -> Any:
            call = _lm15_call(request)
            if call is None:
                return await original_do_stream(self, request)
            call.decision = await _run_admission_async(call.admission_payload())
            try:
                response = await original_do_stream(self, request)
            except BaseException as exc:
                call.finish(error=exc, responded=False)
                await _settle_async(call.decision, call.event_id)
                raise
            recorder = _StreamRecorder(call, status_code=response.status, headers=response.headers,
                                       settlement_async=True)
            chunks = response._chunks

            async def recorded() -> Any:
                try:
                    async for chunk in chunks:
                        recorder.chunk(chunk)
                        yield chunk
                except BaseException as exc:
                    recorder.fail(exc)
                    raise
                finally:
                    await recorder.finish_async()

            response._chunks = recorded()
            _emit_when_collected(response, recorder)
            return response

        StdlibTransport.stream = stream
        StdlibAsyncTransport._do_stream = do_stream
        _transport_patched = True
        return True


__all__ = ["CostKeyCallback", "instrument_transport"]
