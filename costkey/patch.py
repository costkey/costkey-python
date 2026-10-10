"""Monkey-patch HTTP clients to intercept AI provider calls."""
from __future__ import annotations
import json
import datetime
import math
import time
import uuid
import re
import logging
import functools
import inspect
import threading
from contextvars import ContextVar
from typing import Any, Callable, Iterator, AsyncIterator
from urllib.parse import urlsplit, urlunsplit
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
    _state.patched = True


def unpatch() -> None:
    if _state.config_refresh_timer:
        _state.config_refresh_timer.cancel()
        _state.config_refresh_timer = None
    _state.config_generation += 1
    _unpatch_httpx()
    _unpatch_requests()
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


def _extract_sse_provider_request_id(text: str) -> str | None:
    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        for key in ("id", "request_id", "requestId", "response_id", "responseId"):
            value = parsed.get(key)
            if isinstance(value, str) and value:
                return value
    return None


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


def _estimate_request_usage(request_body: Any) -> dict[str, int]:
    """Return a conservative, body-only estimate for policy admission."""
    if not isinstance(request_body, dict):
        return {}
    usage: dict[str, int] = {}
    max_tokens = request_body.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens >= 0:
        usage["outputTokens"] = max_tokens
    max_completion = request_body.get("max_completion_tokens")
    if isinstance(max_completion, int) and not isinstance(max_completion, bool) and max_completion >= 0:
        usage["outputTokens"] = max_completion
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

            request_body = None
            if request.content:
                try:
                    request_body = json.loads(request.content)
                except Exception:
                    pass

            is_streaming = _is_streaming_request(request_body) or kwargs.get("stream", False)

            event_id = uuid.uuid4().hex
            request_model = extractor.extract_model(request_body, None)
            admission_decision = _run_admission_sync({
                "id": event_id,
                "admissionId": event_id,
                "requestId": event_id,
                "projectId": _state.project_id,
                "provider": _provider_value(extractor.provider),
                "model": request_model,
                "context": _safe_admission_context(ctx),
                # Admission deliberately receives metadata only; request
                # bodies and credentials never leave the application here.
                "usageBounds": _estimate_request_usage(request_body),
                "streaming": is_streaming,
            })

            try:
                response = _state._original_httpx_send(self, request_to_send, **kwargs)
            except Exception:
                _settle_sync(admission_decision, None, None, event_id)
                raise

            if is_streaming:
                # Wrap the response's iteration methods to capture streaming data
                _wrap_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                    event_id=event_id,
                    admission_decision=admission_decision,
                    capture_body=capture_body,
                )
            else:
                # Non-streaming — read body normally
                duration_ms = (time.perf_counter() - start) * 1000
                response_body = None
                try:
                    response_body = response.json()
                except Exception:
                    pass
                usage = extractor.extract_usage(response_body) if response_body else None
                _process(extractor, url, request.method, response.status_code,
                         request_body, response_body, duration_ms,
                         False, None, call_site, ctx, event_id=event_id,
                         capture_body=capture_body)
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

            request_body = None
            if request.content:
                try:
                    request_body = json.loads(request.content)
                except Exception:
                    pass

            is_streaming = _is_streaming_request(request_body) or kwargs.get("stream", False)

            event_id = uuid.uuid4().hex
            request_model = extractor.extract_model(request_body, None)
            admission_decision = await _run_admission_async({
                "id": event_id,
                "admissionId": event_id,
                "requestId": event_id,
                "projectId": _state.project_id,
                "provider": _provider_value(extractor.provider),
                "model": request_model,
                "context": _safe_admission_context(ctx),
                "usageBounds": _estimate_request_usage(request_body),
                "streaming": is_streaming,
            })

            try:
                response = await _state._original_httpx_async_send(self, request_to_send, **kwargs)
            except Exception:
                await _settle_async(admission_decision, None, None, event_id)
                raise

            if is_streaming:
                _wrap_streaming_response(
                    response, extractor, url, request.method,
                    request_body, start, call_site, ctx,
                    event_id=event_id,
                    admission_decision=admission_decision,
                    settlement_async=True,
                    capture_body=capture_body,
                )
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                response_body = None
                try:
                    response_body = response.json()
                except Exception:
                    pass
                usage = extractor.extract_usage(response_body) if response_body else None
                _process(extractor, url, request.method, response.status_code,
                         request_body, response_body, duration_ms,
                         False, None, call_site, ctx, event_id=event_id,
                         capture_body=capture_body)
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
    settlement_task: Any = None

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
        nonlocal event_sent, settlement_task
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
                 True, stream_timing, call_site, ctx, event_id=event_id,
                 provider_request_id=_extract_sse_provider_request_id(accumulated_text),
                 capture_body=capture_body)
        if settlement_async:
            try:
                import asyncio
                settlement_task = asyncio.create_task(_settle_async(
                    admission_decision, usage, response.status_code, event_id or uuid.uuid4().hex,
                ))
            except RuntimeError:
                _settle_sync(admission_decision, usage, response.status_code, event_id or uuid.uuid4().hex)
        else:
            _settle_sync(admission_decision, usage, response.status_code, event_id or uuid.uuid4().hex)

    async def _finish_async() -> None:
        _on_done()
        if settlement_task is not None:
            await settlement_task

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
                await _finish_async()

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
                await _finish_async()

        response.aiter_lines = wrapped_aiter_lines

    if hasattr(response, "aread"):
        original_aread = response.aread

        @functools.wraps(original_aread)
        async def wrapped_aread(*a: Any, **kw: Any) -> bytes:
            data = await original_aread(*a, **kw)
            _on_chunk(data)
            await _finish_async()
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
            ctx = {**_state.default_context, **get_context()}
            for key, value in _incoming_context(request.headers).items():
                ctx.setdefault(key, value)
            request_to_send = _strip_requests_context_headers(request)
            capture_body = _state.capture_body
            start = time.perf_counter()

            request_body = None
            if request.body:
                try:
                    request_body = json.loads(request.body)
                except Exception:
                    pass

            event_id = uuid.uuid4().hex
            request_model = extractor.extract_model(request_body, None)
            admission_decision = _run_admission_sync({
                "id": event_id,
                "admissionId": event_id,
                "requestId": event_id,
                "projectId": _state.project_id,
                "provider": _provider_value(extractor.provider),
                "model": request_model,
                "context": _safe_admission_context(ctx),
                "usageBounds": _estimate_request_usage(request_body),
                "streaming": _is_streaming_request(request_body),
            })

            try:
                response = _state._original_requests_send(self, request_to_send, **kwargs)
            except Exception:
                _settle_sync(admission_decision, None, None, event_id)
                raise
            duration_ms = (time.perf_counter() - start) * 1000

            try:
                response_body = response.json()
            except Exception:
                response_body = None
            usage = extractor.extract_usage(response_body) if response_body else None

            _process(extractor, url, request.method, response.status_code,
                     request_body, response_body, duration_ms,
                     False, None, call_site, ctx, event_id=event_id,
                     capture_body=capture_body)
            _settle_sync(admission_decision, usage, response.status_code, event_id)

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
             call_site: Any, ctx: dict[str, Any], event_id: str | None = None,
             provider_request_id: str | None = None,
             capture_body: bool | None = None) -> None:
    try:
        # LiveKit/session adapters can report a cumulative LLM component at a
        # different granularity than the underlying HTTP request. Callers may
        # scope this opt-out around the overlapping provider call and keep the
        # explicit session usage as the source of charge.
        if ctx.get("costkeyCapture") in {"manual", "explicit", "disabled"}:
            return
        usage = extractor.extract_usage(response_body) if response_body else None
        model = extractor.extract_model(request_body, response_body)

        response_charge_id = provider_request_id
        if isinstance(response_body, dict):
            for key in ("id", "request_id", "requestId", "response_id", "responseId"):
                if isinstance(response_body.get(key), str) and response_body[key]:
                    response_charge_id = response_body[key]
                    break
        request_model = extractor.extract_model(request_body, None)

        should_capture = _state.capture_body if capture_body is None else capture_body
        event = CostKeyEvent(
            id=event_id or uuid.uuid4().hex,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
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
                _scrub(_extract_metadata(request_body, response_body))
                if should_capture
                else _extract_safe_metadata(request_body, response_body)
            ),
            system_prompt=_scrub(_extract_system_prompt(request_body)) if should_capture else None,
            request_body=_scrub(request_body) if should_capture else None,
            response_body=_scrub(response_body) if should_capture else None,
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
            attempt_id=event_id,
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


def get_context() -> dict[str, Any]:
    # Return a copy so callers cannot mutate the ContextVar value shared by
    # sibling tasks created from the same context.
    return dict(_context.get())


def set_context(ctx: dict[str, Any]) -> None:
    _context.set(dict(ctx))
