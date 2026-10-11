"""Raw telemetry (contract v1): skeletons, header and URL rules, stream capture.

The SDK only sends telemetry; the server does all processing. The only
transformations applied here are the three the contract allows:

1. content removal (the skeleton): strings become ``{"$s": N}`` unless their
   key is in ``KEEP_STRING_KEYS``;
2. secret removal from URLs and headers;
3. size caps.

Nothing in this module prices, normalizes, counts tokens, infers models or
classifies anything.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import datetime
import enum
import errno as _errno
import json
import math
import re
import struct
import time
from collections import deque
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlsplit, urlunsplit

TELEMETRY_VERSION = 1

KEEP_STRING_KEYS = frozenset({
    "model", "model_id", "modelId", "modelVersion", "id", "object", "type", "status",
    "finish_reason", "stop_reason", "finishReason", "service_tier", "system_fingerprint",
    "responseId", "request_id", "requestId", "search_depth", "topic", "output_format",
    "response_format", "encoding", "language", "language_code", "voice_id", "size",
    "quality", "style", "mode", "operation", "event", "provider", "modality", "tier",
    "currency", "unit", "job_id", "jobId", "trafficType", "created", "created_at",
    "speech_model", "response_id",
})

MAX_DEPTH = 12
MAX_KEPT_STRING = 200
MAX_ARRAY_ITEMS = 32
ARRAY_HEAD_ITEMS = 8
MAX_SKELETON_BYTES = 64 * 1024
MAX_STREAM_CHUNKS = 200
STREAM_HEAD_CHUNKS = 20
STREAM_TAIL_CHUNKS = 50

REQUEST_HEADER_ALLOWLIST = frozenset({
    "content-type", "anthropic-version", "anthropic-beta", "openai-beta",
    "x-goog-api-client", "x-stainless-lang",
})
RESPONSE_HEADER_DENYLIST = frozenset({
    "set-cookie", "cookie", "authorization", "proxy-authorization", "x-api-key", "api-key",
})
_DENIED_QUERY_PARAMS = frozenset({
    "key", "api_key", "apikey", "api-key", "token", "access_token", "sig", "signature", "code",
})
_MARKER_KEYS = frozenset({"$s", "$bytes", "$more", "$trunc"})
_BINARY_TYPES = ("audio/", "video/", "image/", "application/octet-stream")
_EVENT_STREAM_TYPE = "application/vnd.amazon.eventstream"


def iso_now() -> str:
    return iso_timestamp(datetime.datetime.now(datetime.timezone.utc))


def iso_timestamp(value: datetime.datetime) -> str:
    stamp = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    return stamp.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ── Skeleton ──

def _string(key: str | None, value: str) -> Any:
    if key is not None and key in KEEP_STRING_KEYS and len(value) <= MAX_KEPT_STRING:
        return value
    return {"$s": len(value)}


def _plain(value: Any) -> Any:
    """Turn a framework object into JSON-shaped data without interpreting it."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name, None) for field in dataclasses.fields(value)}
    if isinstance(value, (set, frozenset)):
        return list(value)
    values = getattr(value, "__dict__", None)
    if isinstance(values, dict):
        return {key: item for key, item in values.items() if not str(key).startswith("_")}
    raise TypeError


def _skel(value: Any, key: str | None, depth: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _string(key, value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$bytes": len(value)}
    if isinstance(value, Mapping):
        if depth >= MAX_DEPTH:
            return {"$trunc": True}
        return {str(k): _skel(v, str(k), depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if depth >= MAX_DEPTH:
            return {"$trunc": True}
        if len(value) <= MAX_ARRAY_ITEMS:
            return [_skel(item, None, depth + 1) for item in value]
        head = [_skel(item, None, depth + 1) for item in value[:ARRAY_HEAD_ITEMS]]
        head.append({"$more": len(value) - ARRAY_HEAD_ITEMS})
        return head
    if isinstance(value, enum.Enum):
        return _skel(value.value, key, depth)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return _string(key, value.isoformat())
    try:
        plain = _plain(value)
    except Exception:
        return _string(key, str(value))
    return _skel(plain, key, depth)


def _is_container(node: Any) -> bool:
    if isinstance(node, list):
        return True
    return isinstance(node, dict) and not (len(node) == 1 and next(iter(node)) in _MARKER_KEYS)


def _max_depth(node: Any, depth: int = 0) -> int:
    if not _is_container(node):
        return -1
    children = node.values() if isinstance(node, dict) else node
    deepest = depth
    for child in children:
        deepest = max(deepest, _max_depth(child, depth + 1))
    return deepest


def _truncate(node: Any, cap: int, depth: int = 0) -> Any:
    if not _is_container(node):
        return node
    if depth >= cap:
        return {"$trunc": True}
    if isinstance(node, dict):
        return {key: _truncate(child, cap, depth + 1) for key, child in node.items()}
    return [_truncate(child, cap, depth + 1) for child in node]


def serialized_size(node: Any) -> int:
    return len(json.dumps(node, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def cap_size(node: Any, limit: int = MAX_SKELETON_BYTES) -> Any:
    """Truncate the deepest subtrees first until the serialized form fits."""
    if serialized_size(node) <= limit:
        return node
    cap = _max_depth(node)
    while cap >= 0:
        node = _truncate(node, cap)
        if serialized_size(node) <= limit:
            return node
        cap -= 1
    return {"$trunc": True}


def _attribute_string(key: str | None, value: str) -> Any:
    if key is not None and len(value) <= MAX_KEPT_STRING and (
        key in KEEP_STRING_KEYS or key.rsplit(".", 1)[-1] in KEEP_STRING_KEYS
    ):
        return value
    return {"$s": len(value)}


def _attr_skel(value: Any, key: str | None, depth: int) -> Any:
    if isinstance(value, str):
        return _attribute_string(key, value)
    if isinstance(value, Mapping):
        if depth >= MAX_DEPTH:
            return {"$trunc": True}
        return {str(k): _attr_skel(v, str(k), depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if depth >= MAX_DEPTH:
            return {"$trunc": True}
        items = value if len(value) <= MAX_ARRAY_ITEMS else value[:ARRAY_HEAD_ITEMS]
        result = [_attr_skel(item, None, depth + 1) for item in items]
        if len(value) > MAX_ARRAY_ITEMS:
            result.append({"$more": len(value) - ARRAY_HEAD_ITEMS})
        return result
    if value is None or isinstance(value, (bool, int, float, bytes, bytearray, memoryview)):
        return _skel(value, key, depth)
    try:
        return _attr_skel(_plain(value), key, depth)
    except Exception:
        return _attribute_string(key, str(value))


def attribute_skeleton(value: Any) -> Any:
    """Skeleton for framework span attributes: like :func:`skeleton`, but a
    dotted key also keeps its string when its last segment is a keep key
    (``gen_ai.request.model`` survives, ``gen_ai.prompt`` does not)."""
    return cap_size(_attr_skel(value, None, 0))


def skeleton(value: Any, key: str | None = None) -> Any:
    """The content-free shape of ``value`` (see the contract's skeleton rules)."""
    return cap_size(_skel(value, key, 0))


# ── URLs and headers ──

def _denied_param(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in _DENIED_QUERY_PARAMS
        or lowered.startswith("auth")
        or lowered.startswith("x-amz-")
    )


def sanitize_url(url: Any) -> str:
    """Drop userinfo, the fragment and credential query parameters."""
    try:
        parsed = urlsplit(str(url))
        hostname = parsed.hostname
        if not hostname:
            return ""
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port:
            netloc += f":{port}"
        kept = []
        for part in parsed.query.split("&") if parsed.query else ():
            if not part:
                continue
            name = unquote_plus(part.split("=", 1)[0])
            if not _denied_param(name):
                kept.append(part)
        return urlunsplit((parsed.scheme, netloc, parsed.path, "&".join(kept), ""))
    except Exception:
        return ""


def _header_items(headers: Any) -> list[tuple[str, str]]:
    if headers is None:
        return []
    try:
        if hasattr(headers, "multi_items"):
            items = headers.multi_items()
        elif hasattr(headers, "items"):
            items = headers.items()
        else:
            items = headers
        result = []
        for key, value in items:
            if isinstance(key, bytes):
                key = key.decode("latin-1")
            if isinstance(value, bytes):
                value = value.decode("latin-1")
            result.append((str(key).lower(), str(value)))
        return result
    except Exception:
        return []


def _merge(items: list[tuple[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key, value in items:
        merged[key] = f"{merged[key]}, {value}" if key in merged else value
    return merged


def request_headers(headers: Any) -> dict[str, str]:
    return _merge([(k, v) for k, v in _header_items(headers) if k in REQUEST_HEADER_ALLOWLIST])


def _denied_response_header(name: str) -> bool:
    if name in RESPONSE_HEADER_DENYLIST:
        return True
    return "token" in name and "count" not in name


def response_headers(headers: Any) -> dict[str, str]:
    return _merge([(k, v) for k, v in _header_items(headers) if not _denied_response_header(k)])


def content_type(headers: Any) -> str:
    for key, value in _header_items(headers):
        if key == "content-type":
            return value.lower()
    return ""


def is_binary_type(kind: str) -> bool:
    return kind.lower().startswith(_BINARY_TYPES)


# ── Bodies ──

def _multipart_value(name: str, value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return {"$bytes": len(value)}
    if isinstance(value, str):
        return _string(name, value)
    return _skel(value, name, 1)


def _file_part(name: str, filename: Any, kind: Any, size: Any) -> dict[str, Any]:
    return {
        "name": name,
        "filename": _string("filename", str(filename)) if filename is not None else None,
        "contentType": str(kind) if kind else None,
        "bytes": size if isinstance(size, int) else None,
    }


def _boundary(kind: str) -> bytes | None:
    match = re.search(r'boundary="?([^";]+)"?', kind, re.IGNORECASE)
    return match.group(1).encode("latin-1") if match else None


def _disposition(header: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)\*?="((?:[^"\\]|\\.)*)"|(\w+)=([^;\s]+)', header):
        if match.group(1):
            params[match.group(1).lower()] = match.group(2)
        else:
            params[match.group(3).lower()] = match.group(4)
    return params


def multipart_from_bytes(data: bytes, kind: str) -> dict[str, Any] | None:
    boundary = _boundary(kind)
    if not boundary:
        return None
    parts: list[dict[str, Any]] = []
    for chunk in data.split(b"--" + boundary)[1:]:
        if chunk.startswith(b"--"):
            break
        chunk = chunk[2:] if chunk.startswith(b"\r\n") else chunk
        head, sep, body = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]
        headers: dict[str, str] = {}
        for line in head.decode("latin-1").split("\r\n"):
            key, colon, value = line.partition(":")
            if colon:
                headers[key.strip().lower()] = value.strip()
        params = _disposition(headers.get("content-disposition", ""))
        name = params.get("name", "")
        if "filename" in params:
            parts.append(_file_part(name, params["filename"], headers.get("content-type"), len(body)))
        else:
            parts.append({"name": name, "value": _multipart_value(name, body)})
    return {"$multipart": parts}


def _file_size(value: Any) -> int | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    try:
        position = value.tell()
        value.seek(0, 2)
        end = value.tell()
        value.seek(position)
        return end - position
    except Exception:
        return None


def multipart_from_httpx(stream: Any) -> dict[str, Any] | None:
    fields = getattr(stream, "fields", None)
    if not isinstance(fields, list):
        return None
    parts: list[dict[str, Any]] = []
    for field in fields:
        name = str(getattr(field, "name", ""))
        if hasattr(field, "filename"):
            headers = getattr(field, "headers", {}) or {}
            kind = next((v for k, v in headers.items() if str(k).lower() == "content-type"), None)
            parts.append(_file_part(name, field.filename, kind, _file_size(getattr(field, "file", None))))
        else:
            parts.append({"name": name, "value": _multipart_value(name, getattr(field, "value", None))})
    return {"$multipart": parts}


def multipart_from_aiohttp(form: Any) -> Any:
    fields = getattr(form, "_fields", None)
    if not isinstance(fields, list):
        return None
    parts: list[dict[str, Any]] = []
    plain: dict[str, Any] = {}
    for item in fields:
        if not (isinstance(item, tuple) and len(item) == 3):
            continue
        options, headers, value = item
        name = str(options.get("name", ""))
        filename = options.get("filename")
        if filename is not None or hasattr(value, "read"):
            kind = (headers or {}).get("Content-Type") or (headers or {}).get("content-type")
            parts.append(_file_part(name, filename, kind, _file_size(value)))
        else:
            parts.append({"name": name, "value": _multipart_value(name, value)})
            plain[name] = value
    if not getattr(form, "_is_multipart", True):
        # A FormData without files is sent urlencoded.
        return skeleton(plain)
    return {"$multipart": parts}


def body_skeleton(raw: Any, kind: str = "") -> Any:
    """Skeleton of a raw HTTP body given its content type."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return skeleton(raw)
    data = bytes(raw)
    if not data:
        return None
    original_kind = kind or ""
    kind = original_kind.lower()
    if is_binary_type(kind):
        return {"$bytes": len(data)}
    if "multipart/form-data" in kind:
        # The boundary is case-sensitive; parse it from the original header.
        parsed = multipart_from_bytes(data, original_kind)
        if parsed is not None:
            return parsed
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"$bytes": len(data)}
    if "application/x-www-form-urlencoded" in kind:
        return skeleton(dict(parse_qsl(text, keep_blank_values=True)))
    try:
        return skeleton(json.loads(text))
    except ValueError:
        return {"$s": len(text)}


def error_block(exc: BaseException | None) -> dict[str, Any] | None:
    """The raw exception class name and, when one is attached, its errno name."""
    if exc is None:
        return None
    code = None
    current: BaseException | None = exc
    for _ in range(6):
        if current is None:
            break
        value = getattr(current, "errno", None)
        if isinstance(value, int) and value in _errno.errorcode:
            code = _errno.errorcode[value]
            break
        current = current.__cause__ or current.__context__
    return {"name": type(exc).__name__, "code": code}


# ── Streams ──

def _decode_bedrock_bytes(payload: Any) -> Any:
    """Framing only: a ``bytes`` field holds the model's JSON, base64 on the wire."""
    if not isinstance(payload, dict):
        return payload
    blob = payload.get("bytes")
    if blob is None:
        return payload
    try:
        raw = base64.b64decode(blob, validate=False) if isinstance(blob, str) else bytes(blob)
        decoded = json.loads(raw)
    except (ValueError, TypeError, binascii.Error):
        return payload
    return {**payload, "bytes": decoded}


def bedrock_event(event: Any) -> Any:
    """A decoded Bedrock stream event, ``{"<event-type>": payload}``."""
    if isinstance(event, dict):
        return {key: _decode_bedrock_bytes(value) for key, value in event.items()}
    return event


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


class StreamCollector:
    """Turns a streamed response body into skeleton chunks, incrementally.

    Keeps the first 200 chunks and the last 50; at the end, over 200 chunks
    become the first 20 plus the last 50, and ``chunkCount`` keeps the total.
    """

    def __init__(self, kind: str = "", start: float | None = None,
                 content_length: int | None = None) -> None:
        self.start = start if start is not None else time.perf_counter()
        kind = (kind or "").lower()
        if kind.startswith("text/event-stream"):
            self.format: str | None = "sse"
        elif kind.startswith(("application/x-ndjson", "application/jsonl", "application/json-seq")):
            self.format = "ndjson"
        elif kind.startswith(_EVENT_STREAM_TYPE):
            self.format = "aws-eventstream"
        else:
            self.format = None  # decided from the whole body at the end
        self.binary = is_binary_type(kind) and self.format is None
        self.content_length = content_length
        self.head: list[Any] = []
        self.tail: deque[Any] = deque(maxlen=STREAM_TAIL_CHUNKS)
        self.count = 0
        self.bytes = 0
        self.first_ms: float | None = None
        self.last_ms: float | None = None
        self._buffer = bytearray()
        self._data_lines: list[str] = []

    def _mark(self) -> None:
        now = (time.perf_counter() - self.start) * 1000
        if self.first_ms is None:
            self.first_ms = round(now, 2)
        self.last_ms = round(now, 2)

    def _add(self, payload: Any) -> None:
        chunk = skeleton(payload)
        self.count += 1
        if len(self.head) < MAX_STREAM_CHUNKS:
            self.head.append(chunk)
        self.tail.append(chunk)

    def _add_text(self, text: str) -> None:
        try:
            self._add(json.loads(text))
        except ValueError:
            self._add(text)

    def feed(self, data: Any) -> None:
        if not data:
            return
        if isinstance(data, str):
            data = data.encode("utf-8")
        data = bytes(data)
        self.bytes += len(data)
        self._mark()
        if self.binary:
            return
        self._buffer += data
        if self.format == "sse":
            self._drain_lines(self._sse_line)
        elif self.format == "ndjson":
            self._drain_lines(self._ndjson_line)
        elif self.format == "aws-eventstream":
            self._drain_frames()

    def event(self, payload: Any) -> None:
        """An already-decoded stream event (botocore event streams)."""
        self.format = self.format or "aws-eventstream"
        self._mark()
        self._add(bedrock_event(payload))

    def _drain_lines(self, handle: Any) -> None:
        while True:
            index = self._buffer.find(b"\n")
            if index < 0:
                return
            line = bytes(self._buffer[:index]).rstrip(b"\r")
            del self._buffer[:index + 1]
            handle(line.decode("utf-8", "replace"))

    def _sse_line(self, line: str) -> None:
        if not line:
            self._dispatch_sse()
        elif line.startswith("data:"):
            value = line[5:]
            self._data_lines.append(value[1:] if value.startswith(" ") else value)

    def _dispatch_sse(self) -> None:
        if self._data_lines:
            data = "\n".join(self._data_lines)
            self._data_lines = []
            # ``[DONE]`` is framing (the end-of-stream sentinel), not a record.
            if data.strip() != "[DONE]":
                self._add_text(data)

    def _ndjson_line(self, line: str) -> None:
        if line.strip():
            self._add_text(line)

    def _drain_frames(self) -> None:
        while len(self._buffer) >= 12:
            total, headers_length = struct.unpack_from(">II", self._buffer, 0)
            if total < 16 or 16 + headers_length > total:
                self._buffer.clear()
                return
            if len(self._buffer) < total:
                return
            frame = bytes(self._buffer[:total])
            del self._buffer[:total]
            try:
                headers = _event_stream_headers(frame[12:12 + headers_length])
            except Exception:
                headers = {}
            payload_raw = frame[12 + headers_length:total - 4]
            name = headers.get(":event-type") or headers.get(":exception-type") or "event"
            try:
                payload: Any = json.loads(payload_raw)
            except ValueError:
                payload = {"$bytes": len(payload_raw)}
            self._add(bedrock_event({str(name): payload}))

    def finish(self) -> tuple[dict[str, Any] | None, Any]:
        """Return ``(stream, response_body)``; exactly one describes the body."""
        if self.binary:
            size = self.bytes or self.content_length
            return None, ({"$bytes": size} if size else None)
        if self.format == "sse":
            if self._buffer:
                self._sse_line(bytes(self._buffer).decode("utf-8", "replace").rstrip("\r"))
                self._buffer.clear()
            self._dispatch_sse()
        elif self.format == "ndjson":
            if self._buffer:
                self._ndjson_line(bytes(self._buffer).decode("utf-8", "replace"))
                self._buffer.clear()
        elif self.format is None:
            data = bytes(self._buffer)
            self._buffer.clear()
            if not data.strip():
                return None, None
            text = data.decode("utf-8", "replace")
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                self.format = "json-array"
                for item in parsed:
                    self._add(item)
            elif parsed is not None:
                return None, skeleton(parsed)
            elif re.search(r"^data:", text, re.MULTILINE):
                self.format = "sse"
                self._buffer += data
                self._drain_lines(self._sse_line)
                self._dispatch_sse()
            else:
                self.format = "ndjson"
                for line in text.split("\n"):
                    self._ndjson_line(line)
        if self.format is None:
            return None, None
        chunks = (
            self.head if self.count <= MAX_STREAM_CHUNKS
            else self.head[:STREAM_HEAD_CHUNKS] + list(self.tail)
        )
        return {
            "format": self.format,
            "chunks": chunks,
            "chunkCount": self.count,
            "firstChunkMs": self.first_ms,
            "lastChunkMs": self.last_ms,
        }, None


def build(
    *,
    method: str | None,
    url: str,
    request_header_values: Any = None,
    request_body: Any = None,
    request_bytes: int | None = None,
    status: int | None = None,
    response_header_values: Any = None,
    response_body: Any = None,
    response_bytes: int | None = None,
    stream: dict[str, Any] | None = None,
    error: BaseException | dict[str, Any] | None = None,
    started_at: str,
    duration_ms: float,
    responded: bool = True,
) -> dict[str, Any]:
    """Assemble one call's ``telemetry`` object."""
    return {
        "v": TELEMETRY_VERSION,
        "request": {
            "method": method,
            "url": sanitize_url(url),
            "headers": request_headers(request_header_values),
            "body": request_body,
            "bytes": request_bytes,
        },
        "response": {
            "status": status,
            "headers": response_headers(response_header_values),
            "body": response_body,
            "bytes": response_bytes,
        } if responded else None,
        "stream": stream,
        "error": error if isinstance(error, dict) or error is None else error_block(error),
        "startedAt": started_at,
        "durationMs": round(duration_ms, 2),
    }
