"""Screen capture: images an agent sends to the model (its screenshots).

Each image is uploaded once, keyed by the SHA-256 of its bytes, and the call
event lists every image in request order (newest last). Which image is the
frame for a step is decided on the server. Off unless enabled both locally
(``capture_screens=True``) and for the project. Mirrors the TypeScript SDK.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

#: Images larger than this are listed but not uploaded (size cap, no resizing).
MAX_FRAME_BYTES = 2 * 1024 * 1024

_MAX_DEPTH = 32
_DATA_URI = re.compile(r"^data:(image/[a-z0-9.+-]+)(?:;[a-z0-9=._-]+)*;base64,", re.IGNORECASE)
_MEDIA_TYPE_KEYS = ("media_type", "mimeType", "mime_type")
_IMAGE_PART_TYPES = {"image", "input_image", "base64"}


@dataclass
class ExtractedFrames:
    #: Every image in the request, in order: {"sha256", "mediaType", "bytes"[, "omitted"]}.
    refs: list[dict[str, Any]] = field(default_factory=list)
    #: Uploadable bytes by hash: (media type, bytes). Images over the cap are not here.
    uploads: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    #: Hash by the request's own string value, so captured bodies can name the frame.
    by_value: dict[str, str] = field(default_factory=dict)


def extract_frames(body: Any) -> ExtractedFrames:
    """Every base64 image in a provider request (data URIs and ``{media_type, data}`` parts)."""
    found = ExtractedFrames()
    _walk(body, 0, found)
    return found


def _walk(value: Any, depth: int, found: ExtractedFrames) -> None:
    if depth > _MAX_DEPTH or value is None:
        return
    if isinstance(value, str):
        if len(value) > 64:
            match = _DATA_URI.match(value)
            if match:
                _add(found, value, value[match.end():], match.group(1).lower())
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, depth + 1, found)
        return
    if not isinstance(value, dict):
        return
    data = value.get("data")
    if isinstance(data, str) and len(data) > 64:
        media_type = _declared_type(value)
        if media_type is None and value.get("type") in _IMAGE_PART_TYPES:
            media_type = _sniff_type(data)
        if media_type:
            _add(found, data, data, media_type)
    for key, entry in value.items():
        if key == "data" and isinstance(entry, str):
            continue
        _walk(entry, depth + 1, found)


def _declared_type(part: dict[str, Any]) -> str | None:
    for key in _MEDIA_TYPE_KEYS:
        value = part.get(key)
        if isinstance(value, str) and value.lower().startswith("image/"):
            return value.lower()
    return None


def _sniff_type(data: str) -> str | None:
    """The image type from the first base64 characters, for parts that don't declare one."""
    if data.startswith("iVBORw0KGgo"):
        return "image/png"
    if data.startswith("/9j/"):
        return "image/jpeg"
    if data.startswith("UklGR"):
        return "image/webp"
    if data.startswith("R0lGOD"):
        return "image/gif"
    return None


def _add(found: ExtractedFrames, original: str, encoded: str, media_type: str) -> None:
    known = found.by_value.get(original)
    if known is not None:
        for ref in found.refs:
            if ref["sha256"] == known:
                found.refs.append(ref)
                break
        return
    try:
        raw = base64.b64decode(encoded, validate=False)
    except (binascii.Error, ValueError):
        return
    if not raw:
        return
    sha256 = hashlib.sha256(raw).hexdigest()
    found.by_value[original] = sha256
    if len(raw) > MAX_FRAME_BYTES:
        found.refs.append({"sha256": sha256, "mediaType": media_type, "bytes": len(raw), "omitted": "too_large"})
        return
    found.refs.append({"sha256": sha256, "mediaType": media_type, "bytes": len(raw)})
    found.uploads.setdefault(sha256, (media_type, raw))


def name_frames(body: Any, frames: ExtractedFrames | None, depth: int = 0) -> Any:
    """A captured body with each uploaded image replaced by a placeholder naming its frame."""
    if not frames or not frames.by_value or depth > 64:
        return body
    if isinstance(body, str):
        sha256 = frames.by_value.get(body)
        if sha256 is None:
            return body
        ref = next((item for item in frames.refs if item["sha256"] == sha256), None)
        if ref is None:
            return body
        return f"[image data omitted: {ref['mediaType']}; ~{ref['bytes']} bytes; frame {sha256}]"
    if isinstance(body, dict):
        return {key: name_frames(item, frames, depth + 1) for key, item in body.items()}
    if isinstance(body, (list, tuple)):
        return [name_frames(item, frames, depth + 1) for item in body]
    return body
