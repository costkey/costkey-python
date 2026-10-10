"""Monkey-patch HTTP clients to intercept AI provider calls."""
from __future__ import annotations
import asyncio
import base64
import json
import datetime
import io
import math
import struct
import time
import uuid
import weakref
import re
import logging
import functools
import inspect
import threading
from contextvars import ContextVar
from typing import Any, Callable, Iterator, AsyncIterator
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit
from costkey.types import CostKeyEvent, NormalizedUsage, StreamTiming, Provider
from costkey.providers import (
    BedrockExtractor, call_model, call_usage, find_extractor, is_audio_response, merge_usage,
)
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
_EVENT_STREAM_TYPE = "application/vnd.amazon.eventstream"
_BINARY_TYPES = ("audio/", "video/", "image/", "application/octet-stream")
_BLOB_MIN_CHARS = 16_384
_BASE64_BLOB = re.compile(r"(?:data:[\w/+.-]+;base64,)?[A-Za-z0-9+/=_-]+")
_MULTIPART_FIELD = re.compile(
    rb'content-disposition: form-data; name="([^"]+)"\r\n(?:[^\r\n]+\r\n)*\r\n(.{0,4096}?)\r\n--',
    re.IGNORECASE | re.DOTALL,
)


def _content_type(headers: Any) -> str:
    try:
        return str(headers.get("content-type") or "").lower() if headers is not None else ""
    except Exception:
        return ""


def _is_binary_response(headers: Any) -> bool:
    """Audio/image bodies carry no usage; never buffer or JSON-parse them."""
    return _content_type(headers).startswith(_BINARY_TYPES)


def _reads_body(extractor: Any, url: str, headers: Any) -> bool:
    """Whether a response may be read for usage: never audio, whether sent as
    audio bytes or wrapped in JSON (base64 or hex)."""
    return not _is_binary_response(headers) and not is_audio_response(extractor, url)


def _is_event_stream(headers: Any, url: str) -> bool:
    content_type = _content_type(headers)
    if content_type:
        return content_type.startswith(_EVENT_STREAM_TYPE)
    path = urlsplit(url).path
    return path.endswith("/invoke-with-response-stream") or path.endswith("/converse-stream")


def _is_stream_response(headers: Any) -> bool:
    content_type = _content_type(headers)
    return content_type.startswith(("text/event-stream", "application/x-ndjson", _EVENT_STREAM_TYPE))


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
        import httpx
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in {_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER}
        }
        return httpx.Request(
            request.method,
            request.url,
            headers=headers,
            content=request.content,
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
        return cleaned
    except Exception:
        try:
            for key in (_CONTEXT_HEADER, _TRACE_ID_HEADER, _TRACE_NAME_HEADER):
                request.headers.pop(key, None)
        except Exception:
            pass
        return request


def _text_from_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts) if parts else None


def _extract_system_prompt(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None

    direct = _text_from_content(
        body.get("system") or body.get("system_prompt") or body.get("systemPrompt")
    )
    if direct:
        return direct

    system_instruction = body.get("system_instruction") or body.get("systemInstruction")
    if isinstance(system_instruction, dict):
        from_parts = _text_from_content(system_instruction.get("parts"))
        if from_parts:
            return from_parts

    messages = body.get("messages")
    if isinstance(messages, list):
        prompts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") not in ("system", "developer"):
                continue
            content = _text_from_content(message.get("content"))
            if content:
                prompts.append(content)
        if prompts:
            return "\n\n".join(prompts)

    return None


def _collect_tool_results(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []
    results: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") in ("tool", "function"):
            results.append(message)
        content = message.get("content")
        if isinstance(content, list):
            results.extend(
                block for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            )
    return results


def _collect_response_items(response_body: Any) -> tuple[list[Any], list[Any], list[Any]]:
    tool_calls: list[Any] = []
    web_searches: list[Any] = []
    citations: list[Any] = []
    if not isinstance(response_body, dict):
        return tool_calls, web_searches, citations

    choices = response_body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                continue
            message = choice["message"]
            if isinstance(message.get("tool_calls"), list):
                tool_calls.extend(message["tool_calls"])
            if message.get("function_call"):
                tool_calls.append(message["function_call"])
            if isinstance(message.get("annotations"), list):
                citations.extend(message["annotations"])

    output = response_body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", ""))
            if "function_call" in item_type or "tool_call" in item_type:
                tool_calls.append(item)
            if "web_search" in item_type:
                web_searches.append(item)
            if isinstance(item.get("content"), list):
                for content in item["content"]:
                    if isinstance(content, dict) and isinstance(content.get("annotations"), list):
                        citations.extend(content["annotations"])

    content = response_body.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(block)
            if isinstance(block.get("citations"), list):
                citations.extend(block["citations"])

    function_calls = response_body.get("functionCalls") or response_body.get("function_calls")
    if isinstance(function_calls, list):
        tool_calls.extend(function_calls)

    candidates = response_body.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            grounding = candidate.get("groundingMetadata") or candidate.get("grounding_metadata")
            if grounding:
                web_searches.append(grounding)

    usage = response_body.get("usage")
    if isinstance(usage, dict) and usage.get("server_tool_use"):
        web_searches.append({"usage": usage["server_tool_use"]})

    return tool_calls, web_searches, citations


def _extract_metadata(request_body: Any, response_body: Any) -> dict[str, Any] | None:
    if not isinstance(request_body, dict) and not isinstance(response_body, dict):
        return None
    req = request_body if isinstance(request_body, dict) else {}
    metadata: dict[str, Any] = {}

    messages = req.get("messages")
    if isinstance(messages, list):
        metadata["prompts"] = [
            {
                "role": str(message.get("role", "unknown")) if isinstance(message, dict) else "unknown",
                "content": message.get("content") if isinstance(message, dict) else message,
            }
            for message in messages
        ]
    elif isinstance(req.get("contents"), list):
        metadata["prompts"] = [
            {
                "role": str(content.get("role", "user")) if isinstance(content, dict) else "user",
                "content": content,
            }
            for content in req["contents"]
        ]
    elif "input" in req:
        metadata["prompts"] = [{"role": "input", "content": req["input"]}]

    if isinstance(req.get("tools"), list) and req["tools"]:
        metadata["toolDefinitions"] = req["tools"]
    if "tool_choice" in req or "toolChoice" in req:
        metadata["toolChoice"] = req.get("tool_choice", req.get("toolChoice"))

    tool_calls, web_searches, citations = _collect_response_items(response_body)
    tool_results = _collect_tool_results(messages)
    if tool_calls:
        metadata["toolCalls"] = tool_calls
    if tool_results:
        metadata["toolResults"] = tool_results
    if web_searches:
        metadata["webSearches"] = web_searches
    if citations:
        metadata["citations"] = citations

    return metadata or None


def _extract_safe_metadata(request_body: Any, response_body: Any) -> dict[str, Any] | None:
    """Return structural metadata without copying provider content.

    Metadata-only projects still benefit from coverage/debugging facts such as
    message counts and response shape.  Deliberately avoid arbitrary role/type
    values and all message/tool contents: malformed provider payloads can put
    user data in those fields too.
    """
    request = request_body if isinstance(request_body, dict) else {}
    response = response_body if isinstance(response_body, dict) else {}
    summary: dict[str, Any] = {}

    messages = request.get("messages")
    if isinstance(messages, list):
        summary["promptCount"] = len(messages)
        summary["promptRoles"] = [
            message.get("role")
            if isinstance(message, dict)
            and message.get("role") in {"system", "developer", "user", "assistant", "tool", "function"}
            else "unknown"
            for message in messages
        ]
    elif isinstance(request.get("contents"), list):
        summary["promptCount"] = len(request["contents"])
    elif "input" in request and request.get("input") is not None:
        summary["promptCount"] = 1

    tools = request.get("tools")
    if isinstance(tools, list):
        summary["toolDefinitionCount"] = len(tools)
    if "tool_choice" in request or "toolChoice" in request:
        summary["hasToolChoice"] = True

    for key, result_key in (
        ("choices", "choiceCount"),
        ("output", "outputItemCount"),
        ("content", "responseContentCount"),
        ("candidates", "candidateCount"),
    ):
        value = response.get(key)
        if isinstance(value, list):
            summary[result_key] = len(value)

    return {"raw": summary} if summary else None


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
        self._original_httpx_send: Any = None
        self._original_httpx_async_send: Any = None
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
        from costkey.providers import clear_runtime_providers
        clear_runtime_providers()
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
        from costkey.providers import apply_runtime_providers
        apply_runtime_providers(config.get("providers", []))
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
        from costkey.providers import clear_runtime_providers
        clear_runtime_providers()
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


def _sse_bodies(text: str) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for raw in text.split("\n"):
        line = raw.lstrip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except Exception:
            continue
        if isinstance(parsed, dict):
            bodies.append(parsed)
    return bodies


def _ndjson_bodies(text: str) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            bodies.append(parsed)
    return bodies


def _stream_bodies_usage(bodies: list[dict[str, Any]], extractor: Any) -> NormalizedUsage | None:
    merged: NormalizedUsage | None = None
    for parsed in bodies:
        for candidate in (parsed, parsed.get("response"), parsed.get("message")):
            if not isinstance(candidate, dict):
                continue
            try:
                usage = extractor.extract_usage(candidate)
            except Exception:
                usage = None
            if usage:
                merged = merge_usage(merged, usage)
    stream_units = getattr(extractor, "extract_stream_units", None)
    if callable(stream_units):
        try:
            units = stream_units(bodies)
        except Exception:
            units = None
        if units:
            merged = merged or NormalizedUsage()
            merged.native_units = {**(merged.native_units or {}), **units}
    if merged and merged.total_tokens is None and merged.input_tokens is not None and merged.output_tokens is not None:
        merged.total_tokens = merged.input_tokens + merged.output_tokens
    return merged


def _extract_sse_usage(text: str, extractor: Any) -> NormalizedUsage | None:
    """Merge usage across a whole SSE stream.

    OpenAI chat reports usage in the final chunk, the Responses API under
    ``response.completed``'s ``response``, Anthropic input in ``message_start``
    and output in ``message_delta``, Gemini cumulatively on every chunk. Each
    field keeps its largest reported value, which is correct for all of them.
    """
    return _stream_bodies_usage(_sse_bodies(text), extractor)



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


_EVENT_HEADER_SIZES = {2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}


def _event_stream_headers(raw: bytes) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    i = 0
    while i < len(raw):
        name_length = raw[i]
        name = raw[i + 1:i + 1 + name_length].decode("utf-8", "replace")
        i += 1 + name_length
        kind = raw[i]
        i += 1
        if kind in (0, 1):
            value: Any = kind == 0
        elif kind in (6, 7):
            (length,) = struct.unpack_from(">H", raw, i)
            value = raw[i + 2:i + 2 + length]
            i += 2 + length
            if kind == 7:
                value = value.decode("utf-8", "replace")
        elif kind in _EVENT_HEADER_SIZES:
            value = raw[i:i + _EVENT_HEADER_SIZES[kind]]
            i += _EVENT_HEADER_SIZES[kind]
        else:
            break
        headers[name] = value
    return headers


def _decode_event_stream(data: bytes) -> list[tuple[dict[str, Any], bytes]]:
    """Split an AWS event stream into (headers, payload) messages.

    Each message is [total length u32][headers length u32][prelude crc u32]
    [headers][payload][message crc u32], big-endian.
    """
    messages: list[tuple[dict[str, Any], bytes]] = []
    offset = 0
    while offset + 16 <= len(data):
        total, headers_length = struct.unpack_from(">II", data, offset)
        end = offset + total
        if total < 16 or end > len(data) or 16 + headers_length > total:
            break
        try:
            headers = _event_stream_headers(data[offset + 12:offset + 12 + headers_length])
        except Exception:
            headers = {}
        messages.append((headers, data[offset + 12 + headers_length:end - 4]))
        offset = end
    return messages


def _bedrock_payload_usage(payload: Any, extractor: Any) -> NormalizedUsage | None:
    """Usage in one Bedrock stream event: Converse ``metadata`` events, or
    InvokeModel ``chunk`` events whose bytes hold the model's own JSON."""
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("chunk"), dict):
        payload = payload["chunk"]
    blob = payload.get("bytes")
    if isinstance(blob, (str, bytes, bytearray)):
        try:
            payload = json.loads(base64.b64decode(blob) if isinstance(blob, str) else bytes(blob))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
    merged: NormalizedUsage | None = None
    for candidate in (payload, payload.get("message"), payload.get("metadata")):
        if not isinstance(candidate, dict):
            continue
        try:
            usage = extractor.extract_usage(candidate)
        except Exception:
            usage = None
        if usage:
            merged = merge_usage(merged, usage)
    return merged


def _extract_event_stream_usage(data: bytes, extractor: Any) -> NormalizedUsage | None:
    merged: NormalizedUsage | None = None
    for headers, payload in _decode_event_stream(data):
        if headers.get(":message-type", "event") != "event":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        usage = _bedrock_payload_usage(parsed, extractor)
        if usage:
            merged = merge_usage(merged, usage)
    return merged


def _stream_summary(data: bytes, extractor: Any, url: str, request_body: Any, headers: Any
                    ) -> tuple[NormalizedUsage | None, str | None, str | None, dict[str, Any] | None]:
    """Usage, model, provider request id and (for a plain JSON body read
    through a streaming API) the parsed body, from a whole streamed response."""
    if _is_event_stream(headers, url):
        return _extract_event_stream_usage(data, extractor), None, None, None
    text = data.decode("utf-8", errors="replace")
    bodies = _sse_bodies(text)
    body = None
    if not bodies:
        # Gemini without alt=sse streams a single JSON array of chunks.
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        body = parsed if isinstance(parsed, dict) else None
        bodies = [item for item in (parsed if isinstance(parsed, list) else [parsed]) if isinstance(item, dict)]
        if parsed is None:
            # Newline-delimited JSON (Ollama's native streams).
            bodies = _ndjson_bodies(text)
    model = None
    for item in bodies:
        try:
            model = extractor.extract_model(request_body, item)
        except Exception:
            model = None
        if model:
            break
    request_id = next((
        item[key] for item in bodies
        for key in ("id", "request_id", "requestId", "response_id", "responseId")
        if isinstance(item.get(key), str) and item[key]
    ), None)
    return _stream_bodies_usage(bodies, extractor), model, request_id, body


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


def _provider_value(provider: Any) -> str:
    value = getattr(provider, "value", provider)
    return str(value)


def _estimate_request_usage(request_body: Any, raw_body: Any = None) -> dict[str, int]:
    """Per-call bounds for enforced budgets.

    Input is bounded by the request's byte size (a token is never smaller than
    one byte of request text); output by the request's own max-token setting.
    """
    usage: dict[str, int] = {}
    raw: bytes | None = None
    if isinstance(raw_body, (bytes, bytearray)):
        raw = bytes(raw_body)
    elif isinstance(raw_body, str):
        raw = raw_body.encode("utf-8")
    elif request_body is not None:
        try:
            raw = json.dumps(request_body).encode("utf-8")
        except Exception:
            raw = None
    if raw:
        usage["maxInputTokens"] = len(raw)
    if isinstance(request_body, dict):
        generation = request_body.get("generationConfig") if isinstance(request_body.get("generationConfig"), dict) else {}
        inference = request_body.get("inferenceConfig") if isinstance(request_body.get("inferenceConfig"), dict) else {}
        for value in (
            request_body.get("max_tokens"), request_body.get("max_completion_tokens"),
            request_body.get("max_output_tokens"), generation.get("maxOutputTokens"),
            inference.get("maxTokens"),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage["maxOutputTokens"] = value
                break
    return usage


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


def _actual_usage(usage: NormalizedUsage | None) -> dict[str, int | float]:
    if usage is None:
        return {}
    result: dict[str, int | float] = {}
    for source, target in (
        ("input_tokens", "inputTokens"),
        ("output_tokens", "outputTokens"),
        ("total_tokens", "totalTokens"),
    ):
        value = getattr(usage, source, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[target] = value
    native = usage.native_units
    for names, target in (
        (("audioSeconds", "audio_seconds", "audio_duration"), "audioSeconds"),
        (("characters", "characterCount", "characters_count"), "characters"),
        (("images", "imageCount", "image_count"), "images"),
    ):
        for name in names:
            value = native.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[target] = value
                break
    return result


def _settlement_outcome(
    decision: dict[str, Any] | None,
    usage: NormalizedUsage | None,
    status_code: int | None,
) -> str:
    if not (decision and _reservation_id(decision)):
        return "unknown"
    units = {
        str(policy.get("unit"))
        for policy in decision.get("policies", [])
        if isinstance(policy, dict) and policy.get("result") == "allow"
    }
    if status_code is not None and 200 <= status_code < 300:
        if "money" in units and (usage is None or usage.cost_usd is None):
            return "unknown"
        if "tokens" in units and _actual_usage(usage).get("totalTokens") is None:
            return "unknown"
        return "success"
    if usage is not None and (usage.cost_usd is not None or _actual_usage(usage)):
        return "failed"
    return "unknown"


def _settlement_payload(
    decision: dict[str, Any] | None,
    usage: NormalizedUsage | None,
    status_code: int | None,
    event_id: str,
) -> dict[str, Any] | None:
    reservation_id = _reservation_id(decision)
    if not reservation_id:
        return None
    payload: dict[str, Any] = {
        "projectId": _state.project_id,
        "reservationId": reservation_id,
        "settlementId": f"{event_id}:settlement",
        "outcome": _settlement_outcome(decision, usage, status_code),
        "providerAttempted": True,
    }
    if usage is not None:
        actual = _actual_usage(usage)
        if actual:
            payload["actualUsage"] = actual
        if usage.cost_usd is not None:
            payload["actualCostUsd"] = usage.cost_usd
    return payload


def _settle_sync(
    decision: dict[str, Any] | None,
    usage: NormalizedUsage | None,
    status_code: int | None,
    event_id: str,
) -> None:
    payload = _settlement_payload(decision, usage, status_code, event_id)
    if not payload:
        return
    settler = getattr(_state.admission, "settle", None)
    if not callable(settler):
        return
    for attempt in range(2):
        try:
            settler(payload)
            return
        except Exception:
            if attempt == 1 and _state.debug:
                logger.warning("[costkey] policy settlement unavailable; reservation remains pending")


async def _settle_async(
    decision: dict[str, Any] | None,
    usage: NormalizedUsage | None,
    status_code: int | None,
    event_id: str,
) -> None:
    payload = _settlement_payload(decision, usage, status_code, event_id)
    if not payload:
        return
    settler = getattr(_state.admission, "settle", None)
    if not callable(settler):
        return
    for attempt in range(2):
        try:
            if inspect.iscoroutinefunction(settler):
                await settler(payload)
            else:
                import asyncio
                await asyncio.to_thread(settler, payload)
            return
        except Exception:
            if attempt == 1 and _state.debug:
                logger.warning("[costkey] policy settlement unavailable; reservation remains pending")


def _admission_payload(event_id: str, extractor: Any, url: str, request_body: Any,
                       raw_body: Any, ctx: dict[str, Any], streaming: bool) -> dict[str, Any]:
    return {
        "id": event_id,
        "admissionId": event_id,
        "requestId": event_id,
        "projectId": _state.project_id,
        "provider": _provider_value(extractor.provider),
        "model": call_model(extractor, url, request_body, None),
        "context": _safe_admission_context(ctx),
        # Admission deliberately receives metadata only; request
        # bodies and credentials never leave the application here.
        "usageBounds": _estimate_request_usage(request_body, raw_body),
        "streaming": streaming,
    }


# ── Streamed responses ──

class _StreamRecorder:
    """Records one streamed provider response and emits its call event once.

    Clients feed it the bytes (or typed events) the caller consumes; it emits
    when the stream ends, or when the caller stops early and closes it.
    """

    def __init__(self, extractor: Any, url: str, method: str, request_body: Any,
                 start: float, call_site: Any, ctx: dict[str, Any], *,
                 status_code: int | None = None, headers: Any = None,
                 event_id: str | None = None, admission_decision: dict[str, Any] | None = None,
                 settlement_async: bool = False, capture_body: bool | None = None) -> None:
        self.extractor = extractor
        self.url = url
        self.method = method
        self.request_body = request_body
        self.start = start
        self.call_site = call_site
        self.ctx = ctx
        self.status_code = status_code
        self.headers = headers
        self.event_id = event_id
        self.admission_decision = admission_decision
        self.settlement_async = settlement_async
        self.capture_body = capture_body
        # Audio and other media carry no usage; count them, don't keep them.
        self.buffering = (
            _is_event_stream(headers, url) or not _is_binary_response(headers)
        ) and not is_audio_response(extractor, url)
        self.buffer = bytearray()
        self.usage: NormalizedUsage | None = None
        self.first_chunk_time: float | None = None
        self.chunk_count = 0
        # Client read methods call each other (iter_lines -> iter_text ->
        # iter_bytes); only the outermost one records.
        self.depth = 0
        self.sent = False
        self.settlement_task: Any = None

    def chunk(self, data: Any) -> None:
        if self.sent or not data:
            return
        self.chunk_count += 1
        if self.first_chunk_time is None:
            self.first_chunk_time = time.perf_counter()
        if self.buffering:
            try:
                self.buffer += data.encode("utf-8") if isinstance(data, str) else data
            except Exception:
                pass

    def line(self, data: Any) -> None:
        # Line iterators strip the separator SSE parsing needs.
        if isinstance(data, str):
            self.chunk(data + "\n")
        elif isinstance(data, (bytes, bytearray)):
            self.chunk(bytes(data) + b"\n")

    def observe(self, usage: NormalizedUsage | None) -> None:
        """Typed stream events (botocore) report usage directly."""
        if self.sent:
            return
        self.chunk_count += 1
        if self.first_chunk_time is None:
            self.first_chunk_time = time.perf_counter()
        if usage:
            self.usage = merge_usage(self.usage, usage)

    def done(self) -> None:
        if self.sent:
            return
        self.sent = True
        try:
            self._emit()
        except Exception:
            logger.debug("[costkey] stream capture failed", exc_info=True)

    async def finish_async(self) -> None:
        self.done()
        if self.settlement_task is not None:
            await self.settlement_task

    def _emit(self) -> None:
        end = time.perf_counter()
        duration_ms = (end - self.start) * 1000
        usage, model, provider_request_id, body = self.usage, None, None, None
        if self.buffer:
            streamed, model, provider_request_id, body = _stream_summary(
                bytes(self.buffer), self.extractor, self.url, self.request_body, self.headers,
            )
            if streamed:
                usage = merge_usage(usage, streamed)
        called = call_usage(self.extractor, self.url, self.request_body, body,
                            self.headers, self.status_code, self.method)
        if called:
            usage = merge_usage(usage, called)

        ttft = ((self.first_chunk_time - self.start) * 1000) if self.first_chunk_time else None
        tps = None
        if usage and usage.output_tokens and self.first_chunk_time:
            stream_secs = end - self.first_chunk_time
            if stream_secs > 0:
                tps = usage.output_tokens / stream_secs
        stream_timing = StreamTiming(
            ttft=round(ttft, 2) if ttft else None,
            tps=round(tps, 2) if tps else None,
            stream_duration=round(duration_ms, 2),
            chunk_count=self.chunk_count,
        )

        _process(self.extractor, self.url, self.method, self.status_code,
                 self.request_body, body, duration_ms,
                 True, stream_timing, self.call_site, self.ctx, event_id=self.event_id,
                 provider_request_id=provider_request_id,
                 capture_body=self.capture_body, usage=usage, model=model,
                 response_headers=self.headers)
        event_id = self.event_id or uuid.uuid4().hex
        if _reservation_id(self.admission_decision) is None:
            return
        if self.settlement_async:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self.settlement_task = loop.create_task(_settle_async(
                    self.admission_decision, usage, self.status_code, event_id,
                ))
                return
        _settle_sync(self.admission_decision, usage, self.status_code, event_id)


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
            ctx = {**_state.default_context, **get_context()}
            for key, value in _incoming_context(request.headers).items():
                ctx.setdefault(key, value)
            request_to_send = _strip_httpx_context_headers(request)
            capture_body = _state.capture_body
            start = time.perf_counter()

            raw_body = _httpx_request_bytes(request)
            request_body = _decode_request_body(
                raw_body if raw_body is not None else getattr(request, "stream", None),
                request.headers.get("content-type"),
            )
            stream_requested = bool(kwargs.get("stream", False))
            is_streaming = _is_streaming_request(request_body, url) or stream_requested

            event_id = uuid.uuid4().hex
            admission_decision = _run_admission_sync(_admission_payload(
                event_id, extractor, url, request_body, raw_body, ctx, is_streaming,
            ))

            try:
                response = _state._original_httpx_send(self, request_to_send, **kwargs)
            except Exception:
                _settle_sync(admission_decision, None, None, event_id)
                raise

            if stream_requested:
                # Wrap the response's iteration methods to capture streaming data
                _wrap_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                    event_id=event_id,
                    admission_decision=admission_decision,
                    capture_body=capture_body,
                )
            elif is_streaming or _is_stream_response(response.headers):
                # send() already read the whole stream; nothing will iterate it.
                recorder = _StreamRecorder(
                    extractor, url, request.method, request_body, start, call_site, ctx,
                    status_code=response.status_code, headers=response.headers,
                    event_id=event_id, admission_decision=admission_decision,
                    capture_body=capture_body,
                )
                recorder.chunk(response.content)
                recorder.done()
            else:
                # Non-streaming — read body normally
                duration_ms = (time.perf_counter() - start) * 1000
                response_body = None
                if _reads_body(extractor, url, response.headers):
                    try:
                        response_body = response.json()
                    except Exception:
                        pass
                usage = call_usage(extractor, url, request_body, response_body,
                                   response.headers, response.status_code, request.method)
                _process(extractor, url, request.method, response.status_code,
                         request_body, response_body, duration_ms,
                         False, None, call_site, ctx, event_id=event_id,
                         capture_body=capture_body, usage=usage,
                         response_headers=response.headers)
                _settle_sync(admission_decision, usage, response.status_code, event_id)

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
            ctx = {**_state.default_context, **get_context()}
            for key, value in _incoming_context(request.headers).items():
                ctx.setdefault(key, value)
            request_to_send = _strip_httpx_context_headers(request)
            capture_body = _state.capture_body
            start = time.perf_counter()

            raw_body = _httpx_request_bytes(request)
            request_body = _decode_request_body(
                raw_body if raw_body is not None else getattr(request, "stream", None),
                request.headers.get("content-type"),
            )
            stream_requested = bool(kwargs.get("stream", False))
            is_streaming = _is_streaming_request(request_body, url) or stream_requested

            event_id = uuid.uuid4().hex
            admission_decision = await _run_admission_async(_admission_payload(
                event_id, extractor, url, request_body, raw_body, ctx, is_streaming,
            ))

            try:
                response = await _state._original_httpx_async_send(self, request_to_send, **kwargs)
            except Exception:
                await _settle_async(admission_decision, None, None, event_id)
                raise

            if stream_requested:
                _wrap_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                    event_id=event_id,
                    admission_decision=admission_decision,
                    settlement_async=True,
                    capture_body=capture_body,
                )
            elif is_streaming or _is_stream_response(response.headers):
                recorder = _StreamRecorder(
                    extractor, url, request.method, request_body, start, call_site, ctx,
                    status_code=response.status_code, headers=response.headers,
                    event_id=event_id, admission_decision=admission_decision,
                    settlement_async=True, capture_body=capture_body,
                )
                recorder.chunk(response.content)
                await recorder.finish_async()
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                response_body = None
                if _reads_body(extractor, url, response.headers):
                    try:
                        response_body = response.json()
                    except Exception:
                        pass
                usage = call_usage(extractor, url, request_body, response_body,
                                   response.headers, response.status_code, request.method)
                _process(extractor, url, request.method, response.status_code,
                         request_body, response_body, duration_ms,
                         False, None, call_site, ctx, event_id=event_id,
                         capture_body=capture_body, usage=usage,
                         response_headers=response.headers)
                await _settle_async(admission_decision, usage, response.status_code, event_id)

            return response

        httpx.AsyncClient.send = patched_async_send

    except ImportError:
        if _state.debug:
            logger.debug("[costkey] httpx not installed, skipping patch")


def _wrap_streaming_response(
    response: Any, extractor: Any, url: str, method: str,
    request_body: Any, start: float,
    call_site: Any, ctx: dict[str, Any], event_id: str | None = None,
    admission_decision: dict[str, Any] | None = None,
    settlement_async: bool = False,
    capture_body: bool | None = None,
) -> _StreamRecorder:
    """
    Wrap httpx Response iteration methods to capture streaming data.

    The Anthropic/OpenAI SDKs use response.iter_lines() or response.iter_bytes()
    to consume SSE streams. We wrap these methods to accumulate the bytes,
    then extract usage + timing when iteration finishes.
    """
    recorder = _StreamRecorder(
        extractor, url, method, request_body, start, call_site, ctx,
        status_code=getattr(response, "status_code", None),
        headers=getattr(response, "headers", None),
        event_id=event_id, admission_decision=admission_decision,
        settlement_async=settlement_async, capture_body=capture_body,
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

    # Wrap read() for cases where the body is read all at once
    original_read = getattr(response, "read", None)
    if original_read is not None:
        @functools.wraps(original_read)
        def wrapped_read(*a: Any, **kw: Any) -> bytes:
            recorder.depth += 1
            try:
                data = original_read(*a, **kw)
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
    try:
        import httpx
        if _state._original_httpx_send:
            httpx.Client.send = _state._original_httpx_send
        if _state._original_httpx_async_send:
            httpx.AsyncClient.send = _state._original_httpx_async_send
    except ImportError:
        pass


# ── requests patching ──

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
            ctx = {**_state.default_context, **get_context()}
            for key, value in _incoming_context(request.headers).items():
                ctx.setdefault(key, value)
            request_to_send = _strip_requests_context_headers(request)
            capture_body = _state.capture_body
            start = time.perf_counter()

            request_body = _decode_request_body(request.body, request.headers.get("content-type"))
            stream_requested = bool(kwargs.get("stream", False))
            is_streaming = _is_streaming_request(request_body, url) or stream_requested
            event_id = uuid.uuid4().hex
            admission_decision = _run_admission_sync(_admission_payload(
                event_id, extractor, url, request_body, request.body, ctx, is_streaming,
            ))

            try:
                response = _state._original_requests_send(self, request_to_send, **kwargs)
            except Exception:
                _settle_sync(admission_decision, None, None, event_id)
                raise

            if stream_requested or is_streaming or _is_stream_response(response.headers):
                recorder = _StreamRecorder(
                    extractor, url, request.method, request_body, start, call_site, ctx,
                    status_code=response.status_code, headers=response.headers,
                    event_id=event_id, admission_decision=admission_decision,
                    capture_body=capture_body,
                )
                if stream_requested:
                    # Reading here would buffer the stream before the caller
                    # sees it; record what the caller reads instead.
                    _wrap_requests_stream(response, recorder)
                else:
                    recorder.chunk(response.content)
                    recorder.done()
                return response

            duration_ms = (time.perf_counter() - start) * 1000
            response_body = None
            if _reads_body(extractor, url, response.headers):
                try:
                    response_body = response.json()
                except Exception:
                    response_body = None
            usage = call_usage(extractor, url, request_body, response_body,
                               response.headers, response.status_code, request.method)

            _process(extractor, url, request.method, response.status_code,
                     request_body, response_body, duration_ms,
                     False, None, call_site, ctx, event_id=event_id,
                     capture_body=capture_body, usage=usage,
                     response_headers=response.headers)
            _settle_sync(admission_decision, usage, response.status_code, event_id)

            return response

        requests.Session.send = patched_send

    except ImportError:
        if _state.debug:
            logger.debug("[costkey] requests not installed, skipping patch")


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


# ── aiohttp patching ──

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
            call = _aiohttp_call(self, method, str_or_url, kwargs)
        except Exception:
            logger.debug("[costkey] aiohttp request inspection failed", exc_info=True)
            call = None
        if call is None:
            return await original(self, method, str_or_url, **kwargs)

        admission_decision = await _run_admission_async(call["admission"])
        try:
            response = await original(self, method, str_or_url, **call["kwargs"])
        except Exception:
            await _settle_async(admission_decision, None, None, call["event_id"])
            raise
        try:
            await _observe_aiohttp_response(response, call, admission_decision)
        except Exception:
            logger.debug("[costkey] aiohttp response capture failed", exc_info=True)
        return response

    aiohttp.ClientSession._request = patched_request


def _aiohttp_call(session: Any, method: str, str_or_url: Any, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    try:
        url = str(session._build_url(str_or_url))
    except Exception:
        url = str(str_or_url)
    url = _url_with_params(url, kwargs.get("params"))
    extractor = find_extractor(url)
    if not extractor or url.startswith(("ws://", "wss://")):
        return None
    headers = _lower_headers(kwargs.get("headers"))
    if headers is not None and str(headers.get("upgrade") or "").lower() == "websocket":
        return None

    call_site = capture_call_site()
    ctx = {**_state.default_context, **get_context()}
    for key, value in _incoming_context(headers).items():
        ctx.setdefault(key, value)
    send_kwargs = kwargs
    stripped = _without_carrier_headers(kwargs.get("headers"))
    if stripped is not kwargs.get("headers"):
        send_kwargs = {**kwargs, "headers": stripped}

    raw_body = kwargs.get("data")
    request_body = kwargs.get("json")
    if request_body is None:
        request_body = _decode_request_body(
            raw_body, headers.get("content-type") if headers is not None else None,
        )
    streaming = _is_streaming_request(request_body, url) or _params_request_sse(kwargs.get("params"))
    event_id = uuid.uuid4().hex
    return {
        "extractor": extractor,
        "url": url,
        "method": str(method).upper(),
        "request_body": request_body,
        "streaming": streaming,
        "call_site": call_site,
        "ctx": ctx,
        "capture_body": _state.capture_body,
        "event_id": event_id,
        "kwargs": send_kwargs,
        "admission": _admission_payload(
            event_id, extractor, url, request_body,
            raw_body if isinstance(raw_body, (bytes, bytearray, str)) else None, ctx, streaming,
        ),
        "start": time.perf_counter(),
    }


async def _observe_aiohttp_response(response: Any, call: dict[str, Any],
                                    admission_decision: dict[str, Any] | None) -> None:
    status = getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    if status == 101:
        return
    if call["streaming"] or _is_stream_response(headers):
        recorder = _StreamRecorder(
            call["extractor"], call["url"], call["method"], call["request_body"],
            call["start"], call["call_site"], call["ctx"],
            status_code=status, headers=headers, event_id=call["event_id"],
            admission_decision=admission_decision, settlement_async=True,
            capture_body=call["capture_body"],
        )
        _record_aiohttp_stream(response, recorder)
        return

    response_body = None
    if _reads_body(call["extractor"], call["url"], headers):
        try:
            # aiohttp caches the body for read()/text()/json(); direct
            # ``content`` readers get a replay of it.
            body = await response.read()
        except Exception:
            logger.debug("[costkey] aiohttp body read failed", exc_info=True)
            body = None
        if body is not None:
            _replay_content(response, body)
            try:
                response_body = json.loads(body)
            except Exception:
                response_body = None
    duration_ms = (time.perf_counter() - call["start"]) * 1000
    usage = call_usage(call["extractor"], call["url"], call["request_body"], response_body,
                       headers, status, call["method"])
    _process(call["extractor"], call["url"], call["method"], status,
             call["request_body"], response_body, duration_ms,
             False, None, call["call_site"], call["ctx"], event_id=call["event_id"],
             capture_body=call["capture_body"], usage=usage, response_headers=headers)
    await _settle_async(admission_decision, usage, status, call["event_id"])


def _unpatch_aiohttp() -> None:
    if _state._original_aiohttp_request is None:
        return
    try:
        import aiohttp
        aiohttp.ClientSession._request = _state._original_aiohttp_request
    except Exception:
        pass
    _state._original_aiohttp_request = None


# ── botocore (boto3) Bedrock patching ──

_BEDROCK_OPERATIONS = {
    "Converse": "converse",
    "ConverseStream": "converse-stream",
    "InvokeModel": "invoke",
    "InvokeModelWithResponseStream": "invoke-with-response-stream",
}
_BEDROCK = BedrockExtractor()


class _BedrockEventStream:
    """Iterates a botocore EventStream, recording usage events for one call."""

    def __init__(self, stream: Any, recorder: _StreamRecorder) -> None:
        self._stream = stream
        self._recorder = recorder
        _emit_when_collected(self, recorder)

    def __iter__(self) -> Iterator[Any]:
        try:
            for event in self._stream:
                try:
                    self._recorder.observe(_bedrock_payload_usage(event, _BEDROCK))
                except Exception:
                    logger.debug("[costkey] bedrock stream capture failed", exc_info=True)
                yield event
        finally:
            self._recorder.done()

    def close(self) -> Any:
        try:
            return self._stream.close()
        finally:
            self._recorder.done()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _bedrock_request_body(operation_name: str, api_params: Any) -> Any:
    if not isinstance(api_params, dict):
        return None
    if operation_name.startswith("InvokeModel"):
        body = api_params.get("body")
        if isinstance(body, (bytes, bytearray, str)):
            try:
                return _json_safe(json.loads(body))
            except Exception:
                return None
        return None
    return _json_safe(api_params)


def _bedrock_response_meta(response: Any) -> tuple[int | None, Any, str | None]:
    meta = response.get("ResponseMetadata") if isinstance(response, dict) else None
    if not isinstance(meta, dict):
        return None, None, None
    status = meta.get("HTTPStatusCode")
    request_id = meta.get("RequestId")
    return (
        status if isinstance(status, int) else None,
        meta.get("HTTPHeaders"),
        request_id if isinstance(request_id, str) and request_id else None,
    )


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


def _bedrock_api_call(original: Any, client: Any, operation_name: str, path: str, api_params: Any) -> Any:
    try:
        model_id = api_params.get("modelId") if isinstance(api_params, dict) else None
        endpoint = str(getattr(client.meta, "endpoint_url", "") or "https://bedrock-runtime.amazonaws.com")
        url = f"{endpoint.rstrip('/')}/model/{quote(str(model_id or ''), safe='')}/{path}"
        request_body = _bedrock_request_body(operation_name, api_params)
        call_site = capture_call_site()
        ctx = {**_state.default_context, **get_context()}
        capture_body = _state.capture_body
        streaming = operation_name.endswith("Stream")
        event_id = uuid.uuid4().hex
        payload = _admission_payload(event_id, _BEDROCK, url, request_body,
                                     None, ctx, streaming)
        payload["model"] = model_id
        start = time.perf_counter()
    except Exception:
        logger.debug("[costkey] bedrock request inspection failed", exc_info=True)
        return original(client, operation_name, api_params)

    admission_decision = _run_admission_sync(payload)
    try:
        response = original(client, operation_name, api_params)
    except Exception as exc:
        status, headers, request_id = _bedrock_response_meta(getattr(exc, "response", None))
        if status is not None:
            _process(_BEDROCK, url, "POST", status, request_body, None,
                     (time.perf_counter() - start) * 1000, streaming, None, call_site, ctx,
                     event_id=event_id, provider_request_id=request_id,
                     capture_body=capture_body, usage=None, model=model_id,
                     response_headers=headers)
        _settle_sync(admission_decision, None, status, event_id)
        raise

    try:
        status, headers, request_id = _bedrock_response_meta(response)
        if streaming:
            key = "stream" if operation_name == "ConverseStream" else "body"
            recorder = _StreamRecorder(
                _BEDROCK, url, "POST", request_body, start, call_site, ctx,
                status_code=status, headers=headers, event_id=event_id,
                admission_decision=admission_decision, capture_body=capture_body,
            )
            if response.get(key) is not None:
                response[key] = _BedrockEventStream(response[key], recorder)
            else:
                recorder.done()
            return response

        response_body: Any = None
        if operation_name == "InvokeModel":
            raw = response.get("body")
            if raw is not None and hasattr(raw, "read"):
                data = raw.read()
                from botocore.response import StreamingBody
                response["body"] = StreamingBody(io.BytesIO(data), len(data))
                if "json" in str(response.get("contentType") or "application/json").lower():
                    try:
                        response_body = _json_safe(json.loads(data))
                    except Exception:
                        response_body = None
        else:
            response_body = _json_safe({k: v for k, v in response.items() if k != "ResponseMetadata"})
        usage = call_usage(_BEDROCK, url, request_body, response_body, headers, status, "POST")
        _process(_BEDROCK, url, "POST", status, request_body, response_body,
                 (time.perf_counter() - start) * 1000, False, None, call_site, ctx,
                 event_id=event_id, provider_request_id=request_id,
                 capture_body=capture_body, usage=usage, model=model_id,
                 response_headers=headers)
        _settle_sync(admission_decision, usage, status, event_id)
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

_UNSET: Any = object()


def _process(extractor: Any, url: str, method: str, status_code: int | None,
             request_body: Any, response_body: Any,
             duration_ms: float, streaming: bool, stream_timing: StreamTiming | None,
             call_site: Any, ctx: dict[str, Any], event_id: str | None = None,
             provider_request_id: str | None = None,
             capture_body: bool | None = None, *, usage: Any = _UNSET,
             model: str | None = None, response_headers: Any = None) -> None:
    try:
        # LiveKit/session adapters can report a cumulative LLM component at a
        # different granularity than the underlying HTTP request. Callers may
        # scope this opt-out around the overlapping provider call and keep the
        # explicit session usage as the source of charge.
        if ctx.get("costkeyCapture") in {"manual", "explicit", "disabled"}:
            return
        if usage is _UNSET:
            usage = call_usage(extractor, url, request_body, response_body, response_headers, status_code, method)
        model = model or call_model(extractor, url, request_body, response_body)

        response_charge_id = provider_request_id
        if isinstance(response_body, dict):
            for key in ("id", "request_id", "requestId", "response_id", "responseId"):
                if isinstance(response_body.get(key), str) and response_body[key]:
                    response_charge_id = response_body[key]
                    break
        request_model = call_model(extractor, url, request_body, None)

        should_capture = _state.capture_body if capture_body is None else capture_body
        resolved_id = event_id or uuid.uuid4().hex
        # Calls are stamped when they started: now minus the measured duration.
        started_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(milliseconds=max(duration_ms, 0))
        attempt_label = ctx.get("attemptId", ctx.get("attempt_id"))
        parent_event = ctx.get("parentEventId", ctx.get("parent_event_id"))
        event = CostKeyEvent(
            id=resolved_id,
            timestamp=started_at.isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            project_id=_state.project_id,
            provider=extractor.provider,
            model=model,
            url=_safe_url(url),
            method=method,
            status_code=status_code,
            usage=usage,
            cost_usd=usage.cost_usd if usage else None,
            duration_ms=round(duration_ms, 2),
            streaming=streaming,
            stream_timing=stream_timing,
            call_site=call_site,
            context=_scrub(ctx),
            metadata=(
                _json_safe(_scrub(_extract_metadata(request_body, response_body)))
                if should_capture
                else _extract_safe_metadata(request_body, response_body)
            ),
            system_prompt=_scrub(_extract_system_prompt(request_body)) if should_capture else None,
            request_body=_json_safe(_scrub(request_body)) if should_capture else None,
            response_body=_json_safe(_scrub(response_body)) if should_capture else None,
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
            request_model=request_model if request_model != model else None,
            charge_id=response_charge_id,
            provider_request_id=response_charge_id,
            attempt_id=attempt_label if isinstance(attempt_label, str) and attempt_label.strip() else resolved_id,
            parent_event_id=parent_event if isinstance(parent_event, str) and parent_event.strip() else None,
            span_id=ctx.get("spanId") if isinstance(ctx.get("spanId"), str) else None,
        )

        _dispatch_event(event, legacy_hook=True)
    except Exception:
        if _state.debug:
            logger.warning("[costkey] Error processing event", exc_info=True)


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


_MAX_SPAN_PAYLOAD_CHARS = 2000


def _span_payload(value: Any) -> str | None:
    """Span input/output are content: bounded, scrubbed, capture-only."""
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        return None
    text = _scrub(text)
    if not isinstance(text, str):
        return None
    return text if len(text) <= _MAX_SPAN_PAYLOAD_CHARS else text[:_MAX_SPAN_PAYLOAD_CHARS] + "…"


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


def emit_score(
    *,
    score_id: str,
    name: str,
    value: float,
    timestamp: str,
    trace_id: str | None = None,
    span_id: str | None = None,
    label: str | None = None,
    comment: str | None = None,
    source: str | None = None,
) -> bool:
    """Enqueue one judge/human/code score. The comment is content: capture only."""
    event = CostKeyEvent(
        id=score_id,
        timestamp=timestamp,
        project_id=_state.project_id,
        provider=Provider.UNKNOWN,
        event_type="score",
        # The envelope source carries the score's origin for score events.
        source=source or "explicit",
        trace_id=trace_id,
        span_id=span_id,
        score_name=name,
        score_value=value,
        score_label=label,
        score_comment=comment if _state.capture_body else None,
    )
    return _dispatch_event(event)


def get_context() -> dict[str, Any]:
    # Return a copy so callers cannot mutate the ContextVar value shared by
    # sibling tasks created from the same context.
    return dict(_context.get())


def set_context(ctx: dict[str, Any]) -> None:
    _context.set(dict(ctx))
