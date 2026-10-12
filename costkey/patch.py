"""Patch HTTP clients to capture AI provider calls as raw telemetry.

The SDK only sends telemetry: each captured call carries the request and
response skeletons, headers, stream chunks, timing and the raw error name.
Provider, model, usage and cost are derived on the server.
"""
from __future__ import annotations
import asyncio
import json
import datetime
import io
import math
import time
import uuid
import weakref
import re
import logging
import functools
import inspect
import itertools
import threading
from contextvars import ContextVar
from typing import Any, Callable, Iterator, AsyncIterator
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit
from costkey import telemetry as _telemetry
from costkey.types import CostKeyEvent, Provider
from costkey import providers as _providers
from costkey.providers import match_provider
from costkey.stack import capture_call_site
from costkey.transport import Transport

logger = logging.getLogger("costkey")

_context: ContextVar[dict[str, Any]] = ContextVar("costkey_context", default={})
# Set while the SDK makes its own requests (ingest, config, admission,
# settlement, call graph) so they are never captured as provider calls.
_internal: ContextVar[bool] = ContextVar("costkey_internal_request", default=False)


class internal_http:
    """``with internal_http():`` marks the SDK's own HTTP requests."""

    def __enter__(self) -> None:
        self._token = _internal.set(True)

    def __exit__(self, *exc: Any) -> None:
        _internal.reset(self._token)


def _captured_provider(url: str) -> str | None:
    if _internal.get():
        return None
    return match_provider(url)

_SECRET_PATTERNS = [
    re.compile(r"^sk-[a-zA-Z0-9]{20,}$"),
    re.compile(r"^sk-ant-[a-zA-Z0-9\-]{20,}$"),
    re.compile(r"^AIza[a-zA-Z0-9_\-]{30,}$"),
    re.compile(r"^Bearer\s+.{20,}$"),
    re.compile(r"^eyJ[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"^xox[bpras]-[a-zA-Z0-9-]{20,}$"),
    re.compile(r"^ghp_[a-zA-Z0-9]{30,}$"),
]
_SECRET_KEYS = frozenset({
    "api_key", "apikey", "api-key", "secret", "secret_key",
    "token", "access_token", "refresh_token", "password",
    "passwd", "credential", "authorization", "auth", "private_key",
    "privatekey", "secretkey",
})
_ADMISSION_CONTEXT_KEYS = frozenset({
    "traceId", "traceName", "executionId", "groupId", "groupName", "groupKind",
    "phase", "caseId", "datasetRevision", "environment", "service", "release",
    "feature", "customer", "customerId", "function", "operation", "conversationId",
})
_CONTEXT_HEADER = "x-costkey-context"
_TRACE_ID_HEADER = "x-costkey-trace-id"
_TRACE_NAME_HEADER = "x-costkey-trace-name"
_CARRIER_HEADERS = frozenset({_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER})
_BLOB_MIN_CHARS = 16_384
_BASE64_BLOB = re.compile(r"(?:data:[\w/+.-]+;base64,)?[A-Za-z0-9+/=_-]+")
_MULTIPART_FIELD = re.compile(
    rb'content-disposition: form-data; name="([^"]+)"\r\n(?:[^\r\n]+\r\n)*\r\n(.*?)\r\n--',
    re.IGNORECASE | re.DOTALL,
)
_STREAM_TYPES = ("text/event-stream", "application/x-ndjson", "application/vnd.amazon.eventstream")


def _is_stream_response(headers: Any) -> bool:
    return _telemetry.content_type(headers).startswith(_STREAM_TYPES)


def _is_binary_response(headers: Any) -> bool:
    return _telemetry.is_binary_type(_telemetry.content_type(headers))


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Keep captured bodies serializable and small: bytes, non-finite floats
    and inline base64 media (audio, images) never go on the wire."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if len(value) >= _BLOB_MIN_CHARS and _BASE64_BLOB.fullmatch(value[:4096]):
            return f"[omitted {len(value)} chars]"
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[{len(value)} bytes]"
    if depth > 64:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return f"[{type(value).__name__}]"


def _form_fields(raw: Any, content_type: str | None = None) -> dict[str, Any] | None:
    """Plain (non-file) form fields, e.g. ``model_id`` on an upload."""
    fields: dict[str, Any] = {}
    if isinstance(raw, dict):
        fields = {str(key): value for key, value in raw.items() if isinstance(value, (str, int, float, bool))}
    elif isinstance(getattr(raw, "_fields", None), list):  # aiohttp.FormData
        for item in raw._fields:
            if isinstance(item, tuple) and len(item) == 3:
                options, _headers, value = item
                if isinstance(value, str) and not options.get("filename") and options.get("name"):
                    fields[str(options["name"])] = value
    elif isinstance(getattr(raw, "fields", None), list):  # httpx MultipartStream
        for field in raw.fields:
            value = getattr(field, "value", None)
            if not hasattr(field, "filename") and isinstance(value, (str, bytes)):
                fields[field.name] = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    elif isinstance(raw, (bytes, bytearray, str)):
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        kind = (content_type or "").lower()
        if "multipart/form-data" in kind:
            for match in _MULTIPART_FIELD.finditer(data):
                fields[match.group(1).decode("utf-8", "replace")] = match.group(2).decode("utf-8", "replace")
        elif "application/x-www-form-urlencoded" in kind:
            fields = dict(parse_qsl(data.decode("utf-8", "replace")))
    return fields or None


def _decode_request_body(raw: Any, content_type: str | None = None) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, str)):
        try:
            return json.loads(raw)
        except Exception:
            pass
    try:
        return _form_fields(raw, content_type)
    except Exception:
        return None


def _httpx_request_bytes(request: Any) -> bytes | None:
    # A multipart upload is a stream that has not been read yet.
    try:
        return request.content
    except Exception:
        return None


def _without_carrier_headers(headers: Any) -> Any:
    try:
        items = list(headers.items())
    except Exception:
        return headers
    if not any(str(key).lower() in _CARRIER_HEADERS for key, _value in items):
        return headers
    kept = [(key, value) for key, value in items if str(key).lower() not in _CARRIER_HEADERS]
    try:
        from multidict import CIMultiDict
        return CIMultiDict(kept)
    except ImportError:
        return dict(kept)


def _lower_headers(headers: Any) -> Any:
    """A case-insensitive view for plain-dict request headers."""
    if headers is None or hasattr(headers, "getall"):
        return headers
    try:
        items = headers.items() if hasattr(headers, "items") else headers
        return {str(key).lower(): value for key, value in items}
    except Exception:
        return None


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



def _safe_url(url: str) -> str:
    """Keep provider path while removing URL credentials, queries, and fragments."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if not hostname:
            return ""
        netloc = hostname
        if ":" in hostname and not hostname.startswith("["):
            netloc = f"[{hostname}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port:
            netloc += f":{port}"
        # Provider model/endpoint identity is in the path/body. Query values
        # are not needed for accounting and may contain credentials, so clear
        # the entire query rather than maintaining a fragile allowlist.
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except Exception:
        # Even malformed URLs must not put credentials back into telemetry.
        return ""


def _carrier_scalar(value: Any) -> str | int | float | bool | None:
    """Return the scalar values safe to copy across a service boundary.

    A carrier is an identity handoff, not a general-purpose context dump.
    Restricting it to JSON scalars keeps an allowlisted key (for example
    ``feature``) from becoming a covert body/prompt channel when a caller
    accidentally supplies a nested object.
    """
    if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _incoming_context(headers: Any) -> dict[str, Any]:
    """Read only an explicitly supplied carrier from a provider request."""
    if headers is None:
        return {}
    try:
        carrier = headers.get(_CONTEXT_HEADER)
        if carrier:
            parsed = json.loads(carrier)
            if isinstance(parsed, dict) and isinstance(parsed.get("context"), dict):
                parsed = parsed["context"]
            if isinstance(parsed, dict):
                return _scrub({
                    key: scalar
                    for key, value in parsed.items()
                    if key in _ADMISSION_CONTEXT_KEYS
                    for scalar in (_carrier_scalar(value),)
                    if scalar is not None
                })
    except Exception:
        pass
    try:
        trace_id = headers.get(_TRACE_ID_HEADER)
        trace_name = headers.get(_TRACE_NAME_HEADER)
        return {
            key: scalar
            for key, value in (("traceId", trace_id), ("traceName", trace_name))
            if value
            for scalar in (_carrier_scalar(value),)
            if scalar is not None
        }
    except Exception:
        return {}


def _strip_httpx_context_headers(request: Any) -> Any:
    try:
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in {_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER}
        }
        content = request.content
        injected = _providers.inject_stream_usage_bytes(str(request.url), content)
        if injected is not content:
            content = injected
            headers = {key: value for key, value in headers.items() if key.lower() != "content-length"}
        return type(request)(
            request.method,
            request.url,
            headers=headers,
            content=content,
            extensions=request.extensions,
        )
    except Exception:
        try:
            request.headers = {
                key: value for key, value in request.headers.items()
                if key.lower() not in {_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER}
            }
        except Exception:
            pass
        return request


def _strip_requests_context_headers(request: Any) -> Any:
    try:
        import copy
        cleaned = copy.copy(request)
        cleaned.headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in {_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER}
        }
        body = getattr(request, "body", None)
        injected = _providers.inject_stream_usage_bytes(str(getattr(request, "url", "")), body)
        if injected is not body:
            cleaned.body = injected
            cleaned.headers["Content-Length"] = str(len(injected.encode("utf-8") if isinstance(injected, str) else injected))
        return cleaned
    except Exception:
        try:
            for key in (_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER):
                request.headers.pop(key, None)
        except Exception:
            pass
        return request




class _PatchState:
    def __init__(self) -> None:
        self.transport: Transport | None = None
        self.project_id: str = ""
        self.capture_body: bool = False
        self.local_capture_body: bool | None = None
        self.config_known: bool = False
        self.config_status: str = "unknown"
        self.redaction_enabled: bool = True
        self.config_refresh_timer: threading.Timer | None = None
        self.config_generation: int = 0
        self.before_send: Callable | None = None
        self.before_send_event: Callable | None = None
        self.default_context: dict[str, Any] = {}
        self.debug: bool = False
        self.admission: Any = None
        self.admission_mode: str = "monitor"
        # httpx and httpx2 (openai 3.x, anthropic 1.x and mistralai 3.x use httpx2):
        # module name -> (Client.send, AsyncClient.send) before patching.
        self._original_httpx: dict[str, tuple[Any, Any]] = {}
        self._original_requests_send: Any = None
        self._original_aiohttp_request: Any = None
        self._original_botocore_call: Any = None
        self.patched = False


_state = _PatchState()
_lifecycle_lock = threading.Lock()
_lifecycle_sequences: dict[tuple[str, str, str], int] = {}


def patch(transport: Transport, project_id: str, capture_body: bool | None,
          before_send: Callable | None, default_context: dict[str, Any], debug: bool,
          before_send_event: Callable | None = None, admission: Any = None,
          admission_mode: str = "monitor") -> None:
    if _state.patched:
        return
    _state.transport = transport
    _state.project_id = project_id
    _state.local_capture_body = capture_body
    # Metadata-safe until the project setting is fetched. Explicit local false
    # remains false even when a remote project setting later allows text.
    _state.capture_body = False
    _state.config_known = False
    _state.config_status = "unknown"
    _state.redaction_enabled = True
    _state.config_generation += 1
    _state.before_send = before_send
    _state.before_send_event = before_send_event
    _state.default_context = dict(default_context)
    _state.debug = debug
    _state.admission = admission
    _state.admission_mode = admission_mode
    _patch_httpx()
    _patch_requests()
    _patch_aiohttp()
    _patch_botocore()
    _state.patched = True


def unpatch() -> None:
    if _state.config_refresh_timer:
        _state.config_refresh_timer.cancel()
        _state.config_refresh_timer = None
    _state.config_generation += 1
    _unpatch_httpx()
    _unpatch_requests()
    _unpatch_aiohttp()
    _unpatch_botocore()
    _state.patched = False
    _state.transport = None
    _state.before_send = None
    _state.before_send_event = None
    _state.admission = None
    _state.default_context = {}
    _state.project_id = ""
    _state.admission_mode = "monitor"
    _state.capture_body = False
    _state.local_capture_body = None
    _state.config_known = False
    _state.config_status = "unknown"
    _state.redaction_enabled = True
    try:
        from costkey.providers import clear_runtime_hosts
        clear_runtime_hosts()
    except Exception:
        pass


def apply_runtime_config(config: Any, *, project_id: str | None = None) -> bool:
    """Apply DSN-authenticated capture settings; fail closed for bad data."""
    if project_id is not None and project_id != _state.project_id:
        return False
    if not isinstance(config, dict):
        return False
    if project_id is not None and config.get("projectId") != project_id:
        return False
    capture = config.get("capture")
    capture = capture if isinstance(capture, dict) else config
    mode = capture.get("mode")
    body_enabled = capture.get("bodyCaptureEnabled")
    redaction_enabled = capture.get("redactionEnabled")
    if mode not in {"metadata", "text"} or not isinstance(body_enabled, bool) or not isinstance(redaction_enabled, bool):
        return False
    if "providers" in config and not isinstance(config["providers"], list):
        return False
    remote_allows_body = mode == "text" and body_enabled is True
    _state.capture_body = _state.local_capture_body is not False and remote_allows_body
    _state.redaction_enabled = redaction_enabled
    try:
        from costkey.providers import apply_runtime_hosts
        apply_runtime_hosts(config.get("providers", []))
    except Exception:
        # Configured provider enrichment must never affect provider traffic.
        if _state.debug:
            logger.warning("[costkey] Runtime provider config could not be applied")
    _state.config_known = True
    _state.config_status = "available"
    return True


def _mark_runtime_config(status: str, *, project_id: str | None = None) -> None:
    if project_id is not None and project_id != _state.project_id:
        return
    _state.config_status = status
    # A revoked/unavailable/invalid config never unlocks body capture.
    _state.capture_body = False
    _state.config_known = status == "revoked"
    try:
        from costkey.providers import clear_runtime_hosts
        clear_runtime_hosts()
    except Exception:
        pass


def runtime_config_status() -> str:
    return _state.config_status


def start_runtime_config_fetch(
    base_url: str,
    auth_key: str,
    project_id: str,
    debug: bool = False,
    refresh_interval: float | None = None,
) -> None:
    """Fetch project capture policy off the provider-call path."""
    if _state.config_refresh_timer:
        _state.config_refresh_timer.cancel()
        _state.config_refresh_timer = None
    _state.config_generation += 1
    generation = _state.config_generation
    interval = (
        refresh_interval
        if isinstance(refresh_interval, (int, float))
        and not isinstance(refresh_interval, bool)
        and math.isfinite(refresh_interval)
        and refresh_interval > 0
        else None
    )

    def is_current() -> bool:
        return (
            _state.config_generation == generation
            and _state.project_id == project_id
            and (_state.patched or _state.transport is not None)
        )

    def schedule_next() -> None:
        if interval is None or not is_current():
            return
        timer = threading.Timer(interval, fetch)
        timer.daemon = True
        _state.config_refresh_timer = timer
        timer.start()

    def fetch() -> None:
        try:
            import httpx
            from urllib.parse import quote
            if not is_current() and _state.transport is not None:
                return
            url = f"{base_url.rstrip('/')}/api/v1/projects/{quote(project_id, safe='')}/config"
            with internal_http():
                response = httpx.get(
                    url,
                    headers={"Authorization": f"Bearer {auth_key}"},
                    timeout=5,
                )
            if not is_current() and _state.transport is not None:
                return
            if response.status_code in {401, 403}:
                _mark_runtime_config("revoked", project_id=project_id)
                if debug:
                    logger.warning("[costkey] Runtime capture config rejected; body capture remains disabled")
                return
            if not (200 <= response.status_code < 300):
                _mark_runtime_config("unavailable", project_id=project_id)
                if debug:
                    logger.warning("[costkey] Runtime capture config unavailable (%s)", response.status_code)
                return
            try:
                body = response.json()
            except Exception:
                _mark_runtime_config("invalid", project_id=project_id)
                return
            if not apply_runtime_config(body, project_id=project_id):
                _mark_runtime_config("invalid", project_id=project_id)
        except Exception as exc:
            _mark_runtime_config("unavailable", project_id=project_id)
            if debug:
                logger.warning("[costkey] Runtime capture config unavailable: %s", exc)
        finally:
            schedule_next()

    worker = threading.Thread(target=fetch, name="costkey-runtime-config", daemon=True)
    worker.start()



_STREAM_PATH = re.compile(r"(:streamGenerateContent$|:streamRawPredict$|/invoke-with-response-stream$|/converse-stream$)")


def _is_streaming_request(request_body: Any, url: str | None = None) -> bool:
    """Streaming is declared in the body (OpenAI-style) or the URL (Gemini, Bedrock)."""
    if isinstance(request_body, dict) and request_body.get("stream") is True:
        return True
    if url:
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            if _STREAM_PATH.search(parsed.path):
                return True
            if parse_qs(parsed.query).get("alt") == ["sse"]:
                return True
        except Exception:
            return False
    return False


def _policy_refusal(reason: str, details: dict[str, Any] | None = None) -> Exception:
    # Imported lazily so the legacy SDK can still be used without the policy
    # helper's optional HTTP implementation.
    from costkey.policy import CostKeyPolicyRefused

    values = details or {}
    return CostKeyPolicyRefused(
        reason=reason,
        policy_id=values.get("policyId") or values.get("policy_id"),
        scope=values.get("scope"),
        allowance=values.get("allowance"),
        window=values.get("window"),
        retry_after=values.get("retryAfter") or values.get("retry_after"),
        reset_at=values.get("resetAt") or values.get("reset_at"),
        policy_version=values.get("policyVersion") or values.get("policy_version"),
        replay=values.get("replay") if isinstance(values.get("replay"), bool) else None,
        policies=values.get("policies"),
        message=values.get("message") if isinstance(values.get("message"), str) else None,
    )


def _decision_allowed(result: Any) -> tuple[bool | None, dict[str, Any]]:
    """Normalize callback/HTTP admission results without guessing unknowns."""
    if isinstance(result, bool):
        return result, {}
    if result is None:
        return None, {}
    if isinstance(result, str):
        value = result.lower().strip()
        if value in {"allow", "allowed", "ok", "pass"}:
            return True, {}
        if value in {"deny", "denied", "refuse", "refused", "block", "blocked"}:
            return False, {}
        return None, {}
    if not isinstance(result, dict):
        allowed = getattr(result, "allowed", None)
        if isinstance(allowed, bool):
            return allowed, vars(result) if hasattr(result, "__dict__") else {}
        return None, {}

    details = dict(result)
    if isinstance(result.get("allowed"), bool):
        return result["allowed"], details
    decision = result.get("decision") or result.get("status")
    if isinstance(decision, str):
        value = decision.lower().strip()
        if value in {"allow", "allowed", "ok", "pass", "approved"}:
            return True, details
        if value in {"deny", "denied", "refuse", "refused", "block", "blocked"}:
            return False, details
    if isinstance(result.get("ok"), bool):
        return result["ok"], details
    return None, details


def _decision_details(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    values = getattr(result, "__dict__", None)
    return dict(values) if isinstance(values, dict) else None


def _handle_admission_result(result: Any) -> None:
    allowed, details = _decision_allowed(result)
    if allowed is True:
        if details.get("providerCallAllowed") is False or details.get("provider_call_allowed") is False:
            raise _policy_refusal("idempotency_conflict", details)
        return
    if allowed is False:
        reason_value = details.get("reason") or details.get("code")
        if not isinstance(reason_value, str) or not reason_value:
            raise _policy_refusal("decision_unavailable", details)
        reason = reason_value
        raise _policy_refusal(reason, details)
    raise _policy_refusal("decision_unavailable", details)


def _run_admission_sync(payload: dict[str, Any]) -> dict[str, Any] | None:
    admission = _state.admission
    if admission is None:
        if _state.admission_mode == "enforce":
            raise _policy_refusal("decision_unavailable")
        return None
    try:
        if callable(admission):
            result = admission(payload)
        elif hasattr(admission, "check"):
            result = admission.check(payload)
        else:
            result = None
        if inspect.isawaitable(result):
            # An async callback must be used from the async patch path. A
            # synchronous provider call cannot safely drive an active loop.
            close = getattr(result, "close", None)
            if close:
                close()
            raise RuntimeError("async admission callback used by synchronous request")
        if isinstance(result, dict) and (
            result.get("providerCallAllowed") is False or result.get("provider_call_allowed") is False
        ):
            raise _policy_refusal("idempotency_conflict", result)
        if _state.admission_mode == "enforce":
            _handle_admission_result(result)
        return _decision_details(result)
    except Exception as exc:
        from costkey.policy import CostKeyPolicyRefused
        if isinstance(exc, CostKeyPolicyRefused):
            if _state.admission_mode == "enforce" or exc.reason == "idempotency_conflict":
                raise
            if _state.debug:
                logger.warning("[costkey] monitor-mode admission refusal ignored")
            return None
        if _state.admission_mode == "enforce":
            raise _policy_refusal("decision_unavailable") from exc
        if _state.debug:
            logger.warning("[costkey] admission check unavailable; allowing monitor-mode call")
        return None


async def _run_admission_async(payload: dict[str, Any]) -> dict[str, Any] | None:
    admission = _state.admission
    if admission is None:
        if _state.admission_mode == "enforce":
            raise _policy_refusal("decision_unavailable")
        return None
    try:
        if callable(admission):
            result = admission(payload)
        elif hasattr(admission, "check"):
            result = admission.check(payload)
        else:
            result = None
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict) and (
            result.get("providerCallAllowed") is False or result.get("provider_call_allowed") is False
        ):
            raise _policy_refusal("idempotency_conflict", result)
        if _state.admission_mode == "enforce":
            _handle_admission_result(result)
        return _decision_details(result)
    except Exception as exc:
        from costkey.policy import CostKeyPolicyRefused
        if isinstance(exc, CostKeyPolicyRefused):
            if _state.admission_mode == "enforce" or exc.reason == "idempotency_conflict":
                raise
            if _state.debug:
                logger.warning("[costkey] monitor-mode admission refusal ignored")
            return None
        if _state.admission_mode == "enforce":
            raise _policy_refusal("decision_unavailable") from exc
        if _state.debug:
            logger.warning("[costkey] admission check unavailable; allowing monitor-mode call")
        return None


def _safe_admission_context(ctx: dict[str, Any]) -> dict[str, Any]:
    return _scrub({
        key: scalar
        for key, value in ctx.items()
        if key in _ADMISSION_CONTEXT_KEYS
        for scalar in (_carrier_scalar(value),)
        if scalar is not None
    })


def _reservation_id(decision: dict[str, Any] | None) -> str | None:
    if not isinstance(decision, dict):
        return None
    value = decision.get("reservationId") or decision.get("reservation_id")
    if isinstance(value, str) and value:
        return value
    reservation = decision.get("reservation")
    if isinstance(reservation, dict):
        value = reservation.get("reservationId") or reservation.get("reservation_id")
        if isinstance(value, str) and value:
            return value
    return None



class _ReplayProtocol:
    """Flow control for a reader that only replays an already-read body."""
    _reading_paused = True
    connected = False

    def pause_reading(self, *_a: Any, **_kw: Any) -> None:
        pass

    def resume_reading(self, *_a: Any, **_kw: Any) -> None:
        pass


def _replay_content(response: Any, body: bytes) -> None:
    """Give a pre-read response a fresh ``content`` stream holding its body,
    so callers that read ``response.content`` directly still see it."""
    try:
        from aiohttp import StreamReader
        reader = StreamReader(_ReplayProtocol(), 2 ** 16, loop=asyncio.get_running_loop())
        if body:
            reader.feed_data(body)
        reader.feed_eof()
        response.content = reader
    except Exception:
        logger.debug("[costkey] aiohttp body replay unavailable", exc_info=True)


class _TeeStreamReader:
    """Forwards an aiohttp StreamReader, recording what the caller reads."""

    def __init__(self, reader: Any, recorder: _StreamRecorder) -> None:
        self._reader = reader
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    def _seen(self, data: Any) -> Any:
        try:
            self._recorder.chunk(data)
            if self._reader.at_eof():
                self._recorder.done()
        except Exception:
            logger.debug("[costkey] aiohttp stream capture failed", exc_info=True)
        return data

    async def read(self, n: int = -1) -> bytes:
        return self._seen(await self._reader.read(n))

    async def readline(self, *a: Any, **kw: Any) -> bytes:
        return self._seen(await self._reader.readline(*a, **kw))

    async def readuntil(self, *a: Any, **kw: Any) -> bytes:
        return self._seen(await self._reader.readuntil(*a, **kw))

    async def readany(self) -> bytes:
        return self._seen(await self._reader.readany())

    async def readexactly(self, n: int) -> bytes:
        return self._seen(await self._reader.readexactly(n))

    async def readchunk(self) -> tuple[bytes, bool]:
        data, end_of_chunk = await self._reader.readchunk()
        self._seen(data)
        return data, end_of_chunk

    def read_nowait(self, n: int = -1) -> bytes:
        return self._seen(self._reader.read_nowait(n))

    def __aiter__(self) -> Any:
        from aiohttp.streams import AsyncStreamIterator
        return AsyncStreamIterator(self.readline)

    def iter_chunked(self, n: int) -> Any:
        from aiohttp.streams import AsyncStreamIterator
        return AsyncStreamIterator(lambda: self.read(n))

    def iter_any(self) -> Any:
        from aiohttp.streams import AsyncStreamIterator
        return AsyncStreamIterator(self.readany)

    def iter_chunks(self) -> Any:
        from aiohttp.streams import ChunkTupleAsyncStreamIterator
        return ChunkTupleAsyncStreamIterator(self)


def _record_aiohttp_stream(response: Any, recorder: _StreamRecorder) -> None:
    response.content = _TeeStreamReader(response.content, recorder)
    # A caller that stops early releases or closes the response.
    for name in ("release", "close"):
        original = getattr(response, name)

        def wrapped(*a: Any, _original: Any = original, **kw: Any) -> Any:
            try:
                return _original(*a, **kw)
            finally:
                recorder.done()

        setattr(response, name, wrapped)
    _emit_when_collected(response, recorder)


def _params_request_sse(params: Any) -> bool:
    try:
        if isinstance(params, str):
            return "alt=sse" in params
        items = params.items() if hasattr(params, "items") else params or ()
        return any(str(key) == "alt" and str(value) == "sse" for key, value in items)
    except Exception:
        return False


def _url_with_params(url: str, params: Any) -> str:
    """aiohttp sends ``params=`` separately; some providers (Deepgram) name
    the model in the query."""
    if not params:
        return url
    try:
        from yarl import URL
        built = URL(url)
        extend = getattr(built, "extend_query", None) or built.update_query
        return str(extend(params))
    except Exception:
        return url



def _reservation_id_of(decision: Any) -> str | None:
    return _reservation_id(decision) if isinstance(decision, dict) else None


def _settlement_payload(decision: dict[str, Any] | None, event_id: str) -> dict[str, Any] | None:
    """The provider was attempted; the server reconciles the outcome from the
    call event (same ``eventId``) rather than the SDK classifying it."""
    reservation_id = _reservation_id_of(decision)
    if not reservation_id:
        return None
    return {
        "projectId": _state.project_id,
        "reservationId": reservation_id,
        "settlementId": f"{event_id}:settlement",
        "eventId": event_id,
        "outcome": "unknown",
        "providerAttempted": True,
    }


def _settle_sync(decision: dict[str, Any] | None, event_id: str) -> None:
    payload = _settlement_payload(decision, event_id)
    settler = getattr(_state.admission, "settle", None)
    if not payload or not callable(settler):
        return
    for attempt in range(2):
        try:
            settler(payload)
            return
        except Exception:
            if attempt == 1 and _state.debug:
                logger.warning("[costkey] policy settlement unavailable; reservation remains pending")


async def _settle_async(decision: dict[str, Any] | None, event_id: str) -> None:
    payload = _settlement_payload(decision, event_id)
    settler = getattr(_state.admission, "settle", None)
    if not payload or not callable(settler):
        return
    for attempt in range(2):
        try:
            if inspect.iscoroutinefunction(settler):
                await settler(payload)
            else:
                await asyncio.to_thread(settler, payload)
            return
        except Exception:
            if attempt == 1 and _state.debug:
                logger.warning("[costkey] policy settlement unavailable; reservation remains pending")


# ── Active spans owned by other tracers ──
#
# Spans opened through ``costkey.span`` become the active span of the current
# context. Spans bridged from another tracer (OpenTelemetry, CrewAI's event
# bus) are detached: the other tracer owns "current", so a resolver reports
# which of its spans is current, and a call is attributed to whichever active
# span started last (the inner one).

_span_seq_lock = threading.Lock()
_span_seq: dict[str, int] = {}
_span_counter = itertools.count(1)
_span_resolvers: list[Callable[[], dict[str, Any] | None]] = []


def _span_opened(span_id: Any) -> None:
    if isinstance(span_id, str):
        with _span_seq_lock:
            _span_seq[span_id] = next(_span_counter)


def _span_closed(span_id: Any) -> None:
    if isinstance(span_id, str):
        with _span_seq_lock:
            _span_seq.pop(span_id, None)


def span_sequence(span_id: Any) -> int:
    """When an open span started, relative to other open spans (0 if closed)."""
    with _span_seq_lock:
        return _span_seq.get(span_id, 0) if isinstance(span_id, str) else 0


def register_span_resolver(resolver: Callable[[], dict[str, Any] | None]) -> None:
    """Add a callable returning the context of another tracer's current span."""
    if resolver not in _span_resolvers:
        _span_resolvers.append(resolver)


def unregister_span_resolver(resolver: Callable[[], dict[str, Any] | None]) -> None:
    if resolver in _span_resolvers:
        _span_resolvers.remove(resolver)


def _with_active_span(ctx: dict[str, Any]) -> dict[str, Any]:
    if not _span_resolvers:
        return ctx
    best_seq = span_sequence(ctx.get("spanId"))
    best: dict[str, Any] | None = None
    for resolver in list(_span_resolvers):
        try:
            candidate = resolver()
        except Exception:
            continue
        if not isinstance(candidate, dict):
            continue
        seq = span_sequence(candidate.get("spanId"))
        if seq > best_seq:
            best_seq, best = seq, candidate
    return {**ctx, **best} if best else ctx


# ── One captured call ──

class _Call:
    """A provider call in flight: its request telemetry, context and timing."""

    def __init__(self, provider: str, url: str, method: str, *, headers: Any,
                 body: Any, body_bytes: int | None, capture: Callable[[], Any] | None = None,
                 extra_context_headers: Any = None) -> None:
        self.provider = provider
        self.url = url
        self.method = (method or "GET").upper()
        self.request_headers = headers
        self.request_body = body
        self.request_bytes = body_bytes
        self.call_site = capture_call_site()
        ctx = _with_active_span({**_state.default_context, **get_context()})
        for key, value in _incoming_context(extra_context_headers if extra_context_headers is not None else headers).items():
            ctx.setdefault(key, value)
        self.ctx = ctx
        self.capture_body = _state.capture_body
        self.captured_request = None
        if self.capture_body and capture is not None:
            try:
                self.captured_request = capture()
            except Exception:
                self.captured_request = None
        self.event_id = uuid.uuid4().hex
        self.started_at = _telemetry.iso_now()
        self.start = time.perf_counter()
        self.decision: dict[str, Any] | None = None
        self.streaming = False
        self.sent = False

    def request_block(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": _telemetry.sanitize_url(self.url),
            "headers": _telemetry.request_headers(self.request_headers),
            "body": self.request_body,
            "bytes": self.request_bytes,
        }

    def admission_payload(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "admissionId": self.event_id,
            "requestId": self.event_id,
            "projectId": _state.project_id,
            "provider": self.provider,
            "context": _safe_admission_context(self.ctx),
            # Metadata only: the request skeleton, never prompt text or
            # credentials. The server estimates usage bounds from it.
            "request": self.request_block(),
        }

    def finish(self, *, status: int | None = None, headers: Any = None, body: Any = None,
               body_bytes: int | None = None, stream: dict[str, Any] | None = None,
               error: BaseException | dict[str, Any] | None = None, responded: bool = True,
               captured_response: Any = None) -> None:
        if self.sent:
            return
        self.sent = True
        try:
            duration_ms = (time.perf_counter() - self.start) * 1000
            data = _telemetry.build(
                method=self.method,
                url=self.url,
                request_header_values=self.request_headers,
                request_body=self.request_body,
                request_bytes=self.request_bytes,
                status=status,
                response_header_values=headers,
                response_body=body,
                response_bytes=body_bytes,
                stream=stream,
                error=error,
                started_at=self.started_at,
                duration_ms=duration_ms,
                responded=responded,
            )
            _emit_call(self, data, status, duration_ms, stream is not None, captured_response)
        except Exception:
            if _state.debug:
                logger.warning("[costkey] Error processing event", exc_info=True)


def _emit_call(call: _Call, data: dict[str, Any], status: int | None, duration_ms: float,
               streamed: bool, captured_response: Any) -> None:
    ctx = call.ctx
    # LiveKit/session adapters can own a component's charge; callers scope
    # this opt-out around the overlapping provider call.
    if ctx.get("costkeyCapture") in {"manual", "explicit", "disabled"}:
        return
    attempt_label = ctx.get("attemptId", ctx.get("attempt_id"))
    parent_event = ctx.get("parentEventId", ctx.get("parent_event_id"))
    capture = call.capture_body
    event = CostKeyEvent(
        id=call.event_id,
        timestamp=call.started_at,
        project_id=_state.project_id,
        provider=call.provider,
        url=_safe_url(call.url),
        method=call.method,
        status_code=status,
        duration_ms=round(duration_ms, 2),
        streaming=streamed,
        call_site=call.call_site,
        context=_scrub(ctx),
        request_body=_json_safe(_scrub(call.captured_request)) if capture else None,
        response_body=_json_safe(_scrub(captured_response)) if capture else None,
        telemetry=data,
        event_type="call",
        source="auto",
        trace_id=ctx.get("traceId", ctx.get("trace_id")),
        trace_name=ctx.get("traceName", ctx.get("trace_name")),
        group_id=ctx.get("groupId", ctx.get("group_id")),
        group_name=ctx.get("groupName", ctx.get("group_name")),
        group_kind=ctx.get("groupKind", ctx.get("group_kind")),
        execution_id=ctx.get("executionId", ctx.get("execution_id")),
        phase=ctx.get("phase"),
        dataset_revision=ctx.get("datasetRevision", ctx.get("dataset_revision")),
        case_id=ctx.get("caseId", ctx.get("case_id")),
        configuration=_scrub(ctx.get("configuration")) if isinstance(ctx.get("configuration"), dict) else None,
        attempt_id=attempt_label if isinstance(attempt_label, str) and attempt_label.strip() else call.event_id,
        parent_event_id=parent_event if isinstance(parent_event, str) and parent_event.strip() else None,
        span_id=ctx.get("spanId") if isinstance(ctx.get("spanId"), str) else None,
    )
    _dispatch_event(event, legacy_hook=True)


def _json_body(data: Any) -> Any:
    """Parse a body for opt-in content capture only."""
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


# ── Streamed (or deferred) responses ──

def _content_length(headers: Any) -> int | None:
    for key, value in _telemetry._header_items(headers):
        if key == "content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _is_multipart(body: Any) -> bool:
    return isinstance(body, dict) and "$multipart" in body

class _StreamRecorder:
    """Records one streamed response and emits its call event once.

    Clients feed it the bytes (or typed events) the caller consumes; it emits
    when the stream ends, or when the caller stops early and closes it.
    """

    def __init__(self, call: _Call, *, status_code: int | None, headers: Any,
                 settlement_async: bool = False) -> None:
        self.call = call
        self.status_code = status_code
        self.headers = headers
        self.settlement_async = settlement_async
        self.collector = _telemetry.StreamCollector(
            _telemetry.content_type(headers), start=call.start, content_length=_content_length(headers),
        )
        # Client read methods call each other (iter_lines -> iter_text ->
        # iter_bytes); only the outermost one records.
        self.depth = 0
        self.sent = False
        self.error: BaseException | None = None
        self.settlement_task: Any = None

    def chunk(self, data: Any) -> None:
        if self.sent or not data:
            return
        try:
            self.collector.feed(data)
        except Exception:
            logger.debug("[costkey] stream chunk capture failed", exc_info=True)

    def line(self, data: Any) -> None:
        # Line iterators strip the separator the stream framing needs.
        if isinstance(data, str):
            self.chunk(data + "\n")
        elif isinstance(data, (bytes, bytearray)):
            self.chunk(bytes(data) + b"\n")

    def event(self, payload: Any) -> None:
        """Typed stream events (botocore) arrive already decoded."""
        if self.sent:
            return
        try:
            self.collector.event(payload)
        except Exception:
            logger.debug("[costkey] stream event capture failed", exc_info=True)

    def fail(self, exc: BaseException) -> None:
        if not isinstance(exc, GeneratorExit):
            self.error = exc

    def done(self) -> None:
        if self.sent:
            return
        self.sent = True
        try:
            stream, body = self.collector.finish()
            self.call.finish(
                status=self.status_code, headers=self.headers, body=body,
                body_bytes=self.collector.bytes, stream=stream, error=self.error,
            )
        except Exception:
            logger.debug("[costkey] stream capture failed", exc_info=True)
        if _reservation_id_of(self.call.decision) is None:
            return
        if self.settlement_async:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self.settlement_task = loop.create_task(_settle_async(self.call.decision, self.call.event_id))
                return
        _settle_sync(self.call.decision, self.call.event_id)

    async def finish_async(self) -> None:
        self.done()
        if self.settlement_task is not None:
            await self.settlement_task


def _record_iter(recorder: _StreamRecorder, original: Any, record: Callable[[Any], None]) -> Any:
    @functools.wraps(original)
    def wrapped(*a: Any, **kw: Any) -> Iterator[Any]:
        recorder.depth += 1
        outer = recorder.depth == 1
        try:
            for item in original(*a, **kw):
                if outer:
                    record(item)
                yield item
        except BaseException as exc:
            if outer:
                recorder.fail(exc)
            raise
        finally:
            recorder.depth -= 1
            if outer:
                recorder.done()
    return wrapped


def _record_aiter(recorder: _StreamRecorder, original: Any, record: Callable[[Any], None]) -> Any:
    @functools.wraps(original)
    async def wrapped(*a: Any, **kw: Any) -> AsyncIterator[Any]:
        recorder.depth += 1
        outer = recorder.depth == 1
        try:
            async for item in original(*a, **kw):
                if outer:
                    record(item)
                yield item
        except BaseException as exc:
            if outer:
                recorder.fail(exc)
            raise
        finally:
            recorder.depth -= 1
            if outer:
                await recorder.finish_async()
    return wrapped


def _emit_when_collected(owner: Any, recorder: _StreamRecorder) -> None:
    """A stream the caller drops without finishing or closing still reports."""
    try:
        weakref.finalize(owner, recorder.done).atexit = False
    except TypeError:
        pass


# ── httpx ──

def _httpx_request(request: Any) -> tuple[Any, int | None, Callable[[], Any]]:
    """Request body skeleton, byte length and a capture-only decoder."""
    kind = _telemetry.content_type(request.headers)
    try:
        raw = request.content
    except Exception:
        raw = None
    if raw is not None:
        return (
            _telemetry.body_skeleton(raw, kind),
            len(raw),
            lambda: _decode_request_body(raw, kind),
        )
    stream = getattr(request, "stream", None)
    size = None
    try:
        size = stream.get_content_length() if stream is not None else None
    except Exception:
        size = None
    body = _telemetry.multipart_from_httpx(stream)
    return (
        body,
        None if _is_multipart(body) else size,
        lambda: _decode_request_body(stream, kind),
    )


def _httpx_call(request: Any) -> _Call | None:
    url = str(request.url)
    provider = _captured_provider(url)
    if provider is None:
        return None
    body, size, capture = _httpx_request(request)
    return _Call(provider, url, request.method, headers=request.headers,
                 body=body, body_bytes=size, capture=capture)


def _finish_httpx(call: _Call, response: Any, settlement_async: bool) -> _StreamRecorder | None:
    """Emit a fully-read response now; return a recorder for an unread one."""
    headers = response.headers
    if _is_stream_response(headers) or _is_streaming_request_body(call):
        recorder = _StreamRecorder(call, status_code=response.status_code, headers=headers,
                                   settlement_async=settlement_async)
        recorder.chunk(response.content)
        return recorder
    content = response.content
    call.finish(
        status=response.status_code, headers=headers,
        body=_telemetry.body_skeleton(content, _telemetry.content_type(headers)),
        body_bytes=len(content),
        captured_response=_json_body(content) if call.capture_body and not _is_binary_response(headers) else None,
    )
    return None


def _is_streaming_request_body(call: _Call) -> bool:
    body = call.request_body
    return isinstance(body, dict) and body.get("stream") is True or _is_streaming_request(None, call.url)


_HTTPX_MODULES = ("httpx", "httpx2")


def _patch_httpx() -> None:
    import importlib
    for name in _HTTPX_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            if _state.debug:
                logger.debug("[costkey] %s not installed, skipping patch", name)
            continue
        if name in _state._original_httpx:
            continue
        _patch_httpx_module(name, module)


def _patch_httpx_module(name: str, httpx: Any) -> None:
    original_send = httpx.Client.send
    original_async_send = httpx.AsyncClient.send
    _state._original_httpx[name] = (original_send, original_async_send)

    def patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
        try:
            call = _httpx_call(request)
        except Exception:
            logger.debug("[costkey] httpx request inspection failed", exc_info=True)
            call = None
        if call is None:
            return original_send(self, request, **kwargs)
        request_to_send = _strip_httpx_context_headers(request)
        call.decision = _run_admission_sync(call.admission_payload())
        try:
            response = original_send(self, request_to_send, **kwargs)
        except BaseException as exc:
            call.finish(error=exc, responded=False)
            _settle_sync(call.decision, call.event_id)
            raise
        try:
            if kwargs.get("stream", False):
                _wrap_streaming_response(response, call)
            else:
                recorder = _finish_httpx(call, response, settlement_async=False)
                if recorder is not None:
                    recorder.done()
                else:
                    _settle_sync(call.decision, call.event_id)
        except Exception:
            logger.debug("[costkey] httpx response capture failed", exc_info=True)
        return response

    async def patched_async_send(self: Any, request: Any, **kwargs: Any) -> Any:
        try:
            call = _httpx_call(request)
        except Exception:
            logger.debug("[costkey] httpx request inspection failed", exc_info=True)
            call = None
        if call is None:
            return await original_async_send(self, request, **kwargs)
        request_to_send = _strip_httpx_context_headers(request)
        call.decision = await _run_admission_async(call.admission_payload())
        try:
            response = await original_async_send(self, request_to_send, **kwargs)
        except BaseException as exc:
            call.finish(error=exc, responded=False)
            await _settle_async(call.decision, call.event_id)
            raise
        try:
            if kwargs.get("stream", False):
                _wrap_streaming_response(response, call, settlement_async=True)
            else:
                recorder = _finish_httpx(call, response, settlement_async=True)
                if recorder is not None:
                    await recorder.finish_async()
                else:
                    await _settle_async(call.decision, call.event_id)
        except Exception:
            logger.debug("[costkey] httpx response capture failed", exc_info=True)
        return response

    httpx.Client.send = patched_send
    httpx.AsyncClient.send = patched_async_send


def _wrap_streaming_response(response: Any, call: _Call, settlement_async: bool = False) -> _StreamRecorder:
    """Record what the caller reads from an httpx streaming response.

    Provider SDKs consume SSE through ``iter_lines``/``iter_bytes`` (or the
    async variants); every read path is wrapped and the outermost records.
    """
    recorder = _StreamRecorder(
        call, status_code=getattr(response, "status_code", None),
        headers=getattr(response, "headers", None), settlement_async=settlement_async,
    )

    for name, record in (
        ("iter_bytes", recorder.chunk), ("iter_lines", recorder.line),
        ("iter_text", recorder.chunk), ("iter_raw", recorder.chunk),
    ):
        original = getattr(response, name, None)
        if original is not None:
            setattr(response, name, _record_iter(recorder, original, record))

    for name, record in (
        ("aiter_bytes", recorder.chunk), ("aiter_lines", recorder.line),
        ("aiter_text", recorder.chunk), ("aiter_raw", recorder.chunk),
    ):
        original = getattr(response, name, None)
        if original is not None:
            setattr(response, name, _record_aiter(recorder, original, record))

    original_read = getattr(response, "read", None)
    if original_read is not None:
        @functools.wraps(original_read)
        def wrapped_read(*a: Any, **kw: Any) -> bytes:
            recorder.depth += 1
            try:
                data = original_read(*a, **kw)
            except BaseException as exc:
                recorder.fail(exc)
                raise
            finally:
                recorder.depth -= 1
            if recorder.depth == 0:
                recorder.chunk(data)
                recorder.done()
            return data

        response.read = wrapped_read

    original_aread = getattr(response, "aread", None)
    if original_aread is not None:
        @functools.wraps(original_aread)
        async def wrapped_aread(*a: Any, **kw: Any) -> bytes:
            recorder.depth += 1
            try:
                data = await original_aread(*a, **kw)
            except BaseException as exc:
                recorder.fail(exc)
                raise
            finally:
                recorder.depth -= 1
            if recorder.depth == 0:
                recorder.chunk(data)
                await recorder.finish_async()
            return data

        response.aread = wrapped_aread

    # A stream closed without being iterated still reports. httpx closes the
    # response itself at the end of iteration, before decoders flush, so an
    # in-progress iteration reports from its own finally instead.
    original_close = getattr(response, "close", None)
    if original_close is not None:
        @functools.wraps(original_close)
        def wrapped_close(*a: Any, **kw: Any) -> Any:
            try:
                return original_close(*a, **kw)
            finally:
                if recorder.depth == 0:
                    recorder.done()

        response.close = wrapped_close

    original_aclose = getattr(response, "aclose", None)
    if original_aclose is not None:
        @functools.wraps(original_aclose)
        async def wrapped_aclose(*a: Any, **kw: Any) -> Any:
            try:
                return await original_aclose(*a, **kw)
            finally:
                if recorder.depth == 0:
                    await recorder.finish_async()

        response.aclose = wrapped_aclose

    _emit_when_collected(response, recorder)
    return recorder


def _unpatch_httpx() -> None:
    import importlib
    for name, (send, async_send) in list(_state._original_httpx.items()):
        try:
            module = importlib.import_module(name)
            module.Client.send = send
            module.AsyncClient.send = async_send
        except ImportError:
            pass
    _state._original_httpx.clear()


# ── requests ──

def _requests_call(request: Any) -> _Call | None:
    url = str(request.url)
    provider = _captured_provider(url)
    if provider is None:
        return None
    kind = _telemetry.content_type(request.headers)
    raw = request.body
    if isinstance(raw, (bytes, bytearray, str)):
        body = _telemetry.body_skeleton(raw, kind)
        size: int | None = len(raw.encode("utf-8") if isinstance(raw, str) else raw)
        if _is_multipart(body):
            size = None  # contract: a multipart request has ``bytes`` null
    else:
        body = None
        try:
            size = int(request.headers.get("content-length"))
        except Exception:
            size = None
    return _Call(provider, url, request.method, headers=request.headers, body=body, body_bytes=size,
                 capture=lambda: _decode_request_body(raw, kind))


def _patch_requests() -> None:
    try:
        import requests
    except ImportError:
        if _state.debug:
            logger.debug("[costkey] requests not installed, skipping patch")
        return

    _state._original_requests_send = requests.Session.send

    def patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
        try:
            call = _requests_call(request)
        except Exception:
            logger.debug("[costkey] requests inspection failed", exc_info=True)
            call = None
        if call is None:
            return _state._original_requests_send(self, request, **kwargs)
        request_to_send = _strip_requests_context_headers(request)
        call.decision = _run_admission_sync(call.admission_payload())
        try:
            response = _state._original_requests_send(self, request_to_send, **kwargs)
        except BaseException as exc:
            call.finish(error=exc, responded=False)
            _settle_sync(call.decision, call.event_id)
            raise
        try:
            headers = response.headers
            if kwargs.get("stream", False):
                # Reading here would buffer the stream before the caller sees
                # it; record what the caller reads instead.
                recorder = _StreamRecorder(call, status_code=response.status_code, headers=headers)
                _wrap_requests_stream(response, recorder)
            elif _is_stream_response(headers) or _is_streaming_request_body(call):
                recorder = _StreamRecorder(call, status_code=response.status_code, headers=headers)
                recorder.chunk(response.content)
                recorder.done()
            else:
                content = response.content or b""
                call.finish(
                    status=response.status_code, headers=headers,
                    body=_telemetry.body_skeleton(content, _telemetry.content_type(headers)),
                    body_bytes=len(content),
                    captured_response=_json_body(content) if call.capture_body and not _is_binary_response(headers) else None,
                )
                _settle_sync(call.decision, call.event_id)
        except Exception:
            logger.debug("[costkey] requests response capture failed", exc_info=True)
        return response

    requests.Session.send = patched_send


def _wrap_requests_stream(response: Any, recorder: _StreamRecorder) -> None:
    # iter_lines(), .content, .text and .json() all read through iter_content.
    response.iter_content = _record_iter(recorder, response.iter_content, recorder.chunk)
    original_close = response.close

    @functools.wraps(original_close)
    def wrapped_close(*a: Any, **kw: Any) -> Any:
        try:
            return original_close(*a, **kw)
        finally:
            recorder.done()

    response.close = wrapped_close
    _emit_when_collected(response, recorder)


def _unpatch_requests() -> None:
    try:
        import requests
        if _state._original_requests_send:
            requests.Session.send = _state._original_requests_send
    except ImportError:
        pass
    _state._original_requests_send = None


# ── aiohttp ──

def _patch_aiohttp() -> None:
    try:
        import aiohttp
        original = aiohttp.ClientSession._request
    except Exception:
        if _state.debug:
            logger.debug("[costkey] aiohttp not available, skipping patch")
        return
    if _state._original_aiohttp_request is not None:
        return
    _state._original_aiohttp_request = original

    async def patched_request(self: Any, method: str, str_or_url: Any, **kwargs: Any) -> Any:
        try:
            prepared = _aiohttp_call(self, method, str_or_url, kwargs)
        except Exception:
            logger.debug("[costkey] aiohttp request inspection failed", exc_info=True)
            prepared = None
        if prepared is None:
            return await original(self, method, str_or_url, **kwargs)
        call, send_kwargs = prepared
        call.decision = await _run_admission_async(call.admission_payload())
        try:
            response = await original(self, method, str_or_url, **send_kwargs)
        except BaseException as exc:
            call.finish(error=exc, responded=False)
            await _settle_async(call.decision, call.event_id)
            raise
        try:
            await _observe_aiohttp_response(response, call)
        except Exception:
            logger.debug("[costkey] aiohttp response capture failed", exc_info=True)
        return response

    aiohttp.ClientSession._request = patched_request


def _aiohttp_body(kwargs: dict[str, Any], kind: str) -> tuple[Any, int | None, Callable[[], Any]]:
    if kwargs.get("json") is not None:
        payload = kwargs["json"]
        try:
            size: int | None = len(json.dumps(payload).encode("utf-8"))
        except Exception:
            size = None
        return _telemetry.skeleton(payload), size, lambda: payload
    data = kwargs.get("data")
    if data is None:
        return None, None, lambda: None
    if isinstance(data, (bytes, bytearray, str)):
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        body = _telemetry.body_skeleton(raw, kind)
        return body, None if _is_multipart(body) else len(raw), lambda: _decode_request_body(raw, kind)
    if isinstance(data, dict):
        return _telemetry.skeleton(data), None, lambda: _decode_request_body(data, kind)
    form = _telemetry.multipart_from_aiohttp(data)
    return form, None, lambda: _decode_request_body(data, kind)


def _aiohttp_call(session: Any, method: str, str_or_url: Any,
                  kwargs: dict[str, Any]) -> tuple[_Call, dict[str, Any]] | None:
    try:
        url = str(session._build_url(str_or_url))
    except Exception:
        url = str(str_or_url)
    url = _url_with_params(url, kwargs.get("params"))
    provider = _captured_provider(url)
    if provider is None or url.startswith(("ws://", "wss://")):
        return None
    headers = _lower_headers(kwargs.get("headers"))
    if headers is not None and str(headers.get("upgrade") or "").lower() == "websocket":
        return None
    send_kwargs = kwargs
    stripped = _without_carrier_headers(kwargs.get("headers"))
    if stripped is not kwargs.get("headers"):
        send_kwargs = {**kwargs, "headers": stripped}
    if kwargs.get("json") is not None:
        injected = _providers.inject_stream_usage(url, kwargs["json"])
        if injected is not kwargs["json"]:
            send_kwargs = {**send_kwargs, "json": injected}
    kind = _telemetry.content_type(headers)
    body, size, capture = _aiohttp_body(kwargs, kind)
    call = _Call(provider, url, method, headers=headers, body=body, body_bytes=size, capture=capture)
    call.streaming = _is_streaming_request(kwargs.get("json"), url) or _params_request_sse(kwargs.get("params"))
    return call, send_kwargs


async def _observe_aiohttp_response(response: Any, call: _Call) -> None:
    status = getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    if status == 101:
        return
    streaming = call.streaming or _is_streaming_request_body(call)
    if streaming or _is_stream_response(headers) or _is_binary_response(headers):
        # Streams and media are recorded as the caller reads them; reading
        # here would buffer the whole body before the caller sees it.
        recorder = _StreamRecorder(call, status_code=status, headers=headers, settlement_async=True)
        _record_aiohttp_stream(response, recorder)
        return
    try:
        # aiohttp caches the body for read()/text()/json(); direct
        # ``content`` readers get a replay of it.
        content = await response.read()
    except Exception:
        logger.debug("[costkey] aiohttp body read failed", exc_info=True)
        content = None
    if content is not None:
        _replay_content(response, content)
    call.finish(
        status=status, headers=headers,
        body=_telemetry.body_skeleton(content, _telemetry.content_type(headers)) if content is not None else None,
        body_bytes=len(content) if content is not None else None,
        captured_response=_json_body(content) if call.capture_body else None,
    )
    await _settle_async(call.decision, call.event_id)


def _unpatch_aiohttp() -> None:
    if _state._original_aiohttp_request is None:
        return
    try:
        import aiohttp
        aiohttp.ClientSession._request = _state._original_aiohttp_request
    except Exception:
        pass
    _state._original_aiohttp_request = None


# ── botocore (boto3) Bedrock ──

_BEDROCK_OPERATIONS = {
    "Converse": "converse",
    "ConverseStream": "converse-stream",
    "InvokeModel": "invoke",
    "InvokeModelWithResponseStream": "invoke-with-response-stream",
}


class _BedrockEventStream:
    """Iterates a botocore EventStream, recording each event for one call."""

    def __init__(self, stream: Any, recorder: _StreamRecorder) -> None:
        self._stream = stream
        self._recorder = recorder
        _emit_when_collected(self, recorder)

    def __iter__(self) -> Iterator[Any]:
        try:
            for event in self._stream:
                self._recorder.event(event)
                yield event
        except BaseException as exc:
            self._recorder.fail(exc)
            raise
        finally:
            self._recorder.done()

    def close(self) -> Any:
        try:
            return self._stream.close()
        finally:
            self._recorder.done()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _bedrock_response_meta(response: Any) -> tuple[int | None, Any]:
    meta = response.get("ResponseMetadata") if isinstance(response, dict) else None
    if not isinstance(meta, dict):
        return None, None
    status = meta.get("HTTPStatusCode")
    return (status if isinstance(status, int) else None), meta.get("HTTPHeaders")


def _patch_botocore() -> None:
    try:
        from botocore.client import BaseClient
        original = BaseClient._make_api_call
    except Exception:
        if _state.debug:
            logger.debug("[costkey] botocore not available, skipping patch")
        return
    if _state._original_botocore_call is not None:
        return
    _state._original_botocore_call = original

    def patched_make_api_call(self: Any, operation_name: str, api_params: Any) -> Any:
        path = _BEDROCK_OPERATIONS.get(operation_name)
        try:
            bedrock = path is not None and self.meta.service_model.service_name == "bedrock-runtime"
        except Exception:
            bedrock = False
        if not bedrock:
            return original(self, operation_name, api_params)
        return _bedrock_api_call(original, self, operation_name, path, api_params)

    BaseClient._make_api_call = patched_make_api_call


def _bedrock_call(client: Any, operation_name: str, path: str, api_params: Any) -> _Call:
    params = api_params if isinstance(api_params, dict) else {}
    model_id = params.get("modelId")
    endpoint = str(getattr(client.meta, "endpoint_url", "") or "https://bedrock-runtime.amazonaws.com")
    url = f"{endpoint.rstrip('/')}/model/{quote(str(model_id or ''), safe='')}/{path}"
    if operation_name.startswith("InvokeModel"):
        raw = params.get("body")
        kind = str(params.get("contentType") or "application/json")
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
        body = _telemetry.body_skeleton(raw_bytes, kind) if isinstance(raw_bytes, (bytes, bytearray)) else None
        size = len(raw_bytes) if isinstance(raw_bytes, (bytes, bytearray)) else None
        headers = {"content-type": kind}
        capture = lambda: _json_body(raw_bytes)  # noqa: E731
    else:
        # Converse sends its parameters (minus the model id, which is in the
        # path) as the JSON body.
        payload = {key: value for key, value in params.items() if key != "modelId"}
        body = _telemetry.skeleton(payload)
        size = None
        headers = {"content-type": "application/json"}
        capture = lambda: _json_safe(payload)  # noqa: E731
    return _Call(Provider.BEDROCK.value, url, "POST", headers=headers, body=body,
                 body_bytes=size, capture=capture)


def _bedrock_api_call(original: Any, client: Any, operation_name: str, path: str, api_params: Any) -> Any:
    try:
        call = _bedrock_call(client, operation_name, path, api_params)
    except Exception:
        logger.debug("[costkey] bedrock request inspection failed", exc_info=True)
        return original(client, operation_name, api_params)

    call.decision = _run_admission_sync(call.admission_payload())
    try:
        response = original(client, operation_name, api_params)
    except BaseException as exc:
        status, headers = _bedrock_response_meta(getattr(exc, "response", None))
        error_body = getattr(exc, "response", None)
        call.finish(
            status=status, headers=headers,
            body=_telemetry.skeleton({k: v for k, v in error_body.items() if k != "ResponseMetadata"})
            if isinstance(error_body, dict) else None,
            error=exc, responded=status is not None,
        )
        _settle_sync(call.decision, call.event_id)
        raise

    try:
        status, headers = _bedrock_response_meta(response)
        if operation_name.endswith("Stream"):
            key = "stream" if operation_name == "ConverseStream" else "body"
            recorder = _StreamRecorder(call, status_code=status, headers=headers)
            recorder.collector.format = "aws-eventstream"
            if response.get(key) is not None:
                response[key] = _BedrockEventStream(response[key], recorder)
            else:
                recorder.done()
            return response

        if operation_name == "InvokeModel":
            data = None
            raw = response.get("body")
            if raw is not None and hasattr(raw, "read"):
                data = raw.read()
                from botocore.response import StreamingBody
                response["body"] = StreamingBody(io.BytesIO(data), len(data))
            kind = str(response.get("contentType") or "application/json")
            call.finish(
                status=status, headers=headers,
                body=_telemetry.body_skeleton(data, kind) if data is not None else None,
                body_bytes=len(data) if data is not None else None,
                captured_response=_json_body(data) if call.capture_body else None,
            )
        else:
            payload = {k: v for k, v in response.items() if k != "ResponseMetadata"}
            call.finish(
                status=status, headers=headers, body=_telemetry.skeleton(payload),
                captured_response=_json_safe(payload) if call.capture_body else None,
            )
        _settle_sync(call.decision, call.event_id)
    except Exception:
        logger.debug("[costkey] bedrock response capture failed", exc_info=True)
    return response


def _unpatch_botocore() -> None:
    if _state._original_botocore_call is None:
        return
    try:
        from botocore.client import BaseClient
        BaseClient._make_api_call = _state._original_botocore_call
    except Exception:
        pass
    _state._original_botocore_call = None


# ── Event processing ──

def _dispatch_event(event: CostKeyEvent, *, legacy_hook: bool = False) -> bool:
    """Apply the appropriate hook and enqueue one event, without leaking errors."""
    try:
        hook = _state.before_send if legacy_hook else _state.before_send_event
        if hook:
            event = hook(event)
            if inspect.isawaitable(event):
                close = getattr(event, "close", None)
                if close:
                    close()
                return False
        if event and _state.transport:
            _state.transport.enqueue(event)
            return True
    except Exception:
        if _state.debug:
            logger.warning("[costkey] event hook/transport failed; dropping event", exc_info=True)
    return False


def emit_lifecycle(
    *,
    entity_type: str,
    entity_id: str,
    lifecycle: str,
    context: dict[str, Any],
    sequence: int | None = None,
    trace_id: str | None = None,
    trace_name: str | None = None,
    group_name: str | None = None,
    group_kind: str | None = None,
    group_id: str | None = None,
    execution_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_ms: float | None = None,
    error_type: str | None = None,
    expected_items: int | None = None,
    completed_items: int | None = None,
    failed_items: int | None = None,
    configuration: dict[str, Any] | None = None,
) -> str:
    """Enqueue a non-billable trace/execution lifecycle event."""
    if entity_type not in {"trace", "execution"}:
        raise ValueError("entity_type must be 'trace' or 'execution'")
    if lifecycle not in {"started", "completed", "failed", "cancelled"}:
        raise ValueError("unsupported lifecycle state")
    if sequence is not None and (
        not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0
    ):
        raise ValueError("sequence must be a non-negative integer")
    for label, value in (
        ("expected_items", expected_items),
        ("completed_items", completed_items),
        ("failed_items", failed_items),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{label} must be a non-negative integer")
    event_id = uuid.uuid4().hex
    event_context = dict(context)
    if trace_id is not None:
        event_context.setdefault("traceId", trace_id)
    if trace_name is not None:
        event_context.setdefault("traceName", trace_name)
    if execution_id is not None:
        event_context.setdefault("executionId", execution_id)
    if group_name is not None:
        event_context.setdefault("groupName", group_name)
    if group_kind is not None:
        event_context.setdefault("groupKind", group_kind)
    if group_id is not None:
        event_context.setdefault("groupId", group_id)
    # Sequence fences are project-scoped just like trace/execution identity on
    # the server; the same caller ID in another project starts independently.
    key = (_state.project_id, entity_type, entity_id)
    with _lifecycle_lock:
        if sequence is None:
            sequence = _lifecycle_sequences.get(key, 0)
            _lifecycle_sequences[key] = sequence + 1
        else:
            _lifecycle_sequences[key] = max(
                _lifecycle_sequences.get(key, 0), sequence + 1,
            )
    event = CostKeyEvent(
        id=event_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        project_id=_state.project_id,
        provider=Provider.UNKNOWN,
        event_type="lifecycle",
        source="explicit",
        entity_type=entity_type,
        entity_id=entity_id,
        sequence=sequence,
        lifecycle=lifecycle,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=round(duration_ms or 0, 2),
        error_type=error_type,
        expected_items=expected_items,
        completed_items=completed_items,
        failed_items=failed_items,
        context=_scrub(event_context),
        trace_id=trace_id or event_context.get("traceId", event_context.get("trace_id")),
        trace_name=trace_name or event_context.get("traceName", event_context.get("trace_name")),
        group_name=group_name,
        group_kind=group_kind,
        group_id=group_id,
        execution_id=execution_id,
        phase=event_context.get("phase"),
        dataset_revision=event_context.get("datasetRevision", event_context.get("dataset_revision")),
        case_id=event_context.get("caseId", event_context.get("case_id")),
        configuration=configuration,
    )
    # trace_id/trace_name are carried in context for compatibility; the
    # transport also promotes standard identity fields from that context.
    _dispatch_event(event)
    return event_id


def framework_attributes(framework: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Span attributes for a framework integration: the framework's raw
    attributes under ``attributes.<framework>``, whole (scrubbed) when body
    capture is on and as an attribute skeleton when it is off."""
    values = dict(raw or {})
    nested = (
        _json_safe(_scrub(values))
        if _state.capture_body
        else _telemetry.attribute_skeleton(values)
    )
    return {"framework": framework, framework: nested}


def _span_payload(value: Any) -> str | None:
    """Span input/output are content: whole, scrubbed, capture-only."""
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        return None
    text = _scrub(text)
    if not isinstance(text, str):
        return None
    return text


def emit_span(
    *,
    span_id: str,
    trace_id: str,
    kind: str,
    name: str,
    status: str,
    started_at: str,
    context: dict[str, Any],
    trace_name: str | None = None,
    parent_span_id: str | None = None,
    execution_id: str | None = None,
    ended_at: str | None = None,
    duration_ms: float | None = None,
    error_type: str | None = None,
    attributes: dict[str, Any] | None = None,
    input: Any = None,
    output: Any = None,
) -> None:
    """Enqueue one delivery of an agent/step/tool span (start or end)."""
    attrs = dict(attributes or {})
    if _state.capture_body:
        for key, value in (("input", input), ("output", output)):
            payload = _span_payload(value)
            if payload is not None:
                attrs[key] = payload
    event = CostKeyEvent(
        id=span_id,
        timestamp=ended_at or started_at,
        project_id=_state.project_id,
        provider=Provider.UNKNOWN,
        event_type="span",
        source="explicit",
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=round(duration_ms or 0, 2),
        error_type=error_type,
        context=_scrub(dict(context)),
        trace_id=trace_id,
        trace_name=trace_name,
        execution_id=execution_id,
        parent_span_id=parent_span_id,
        span_kind=kind,
        span_name=name,
        span_status=status,
        attributes=_scrub(attrs) if attrs else None,
    )
    _dispatch_event(event)



def get_context() -> dict[str, Any]:
    # Return a copy so callers cannot mutate the ContextVar value shared by
    # sibling tasks created from the same context.
    return dict(_context.get())


def set_context(ctx: dict[str, Any]) -> None:
    _context.set(dict(ctx))
