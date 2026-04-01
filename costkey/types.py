from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    UNKNOWN = "unknown"


@dataclass
class NormalizedUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


@dataclass
class StackFrame:
    function_name: str | None = None
    file_name: str | None = None
    line_number: int | None = None


@dataclass
class CallSite:
    raw: str = ""
    frames: list[StackFrame] = field(default_factory=list)


@dataclass
class StreamTiming:
    ttft: float | None = None
    tps: float | None = None
    stream_duration: float | None = None
    chunk_count: int = 0


@dataclass
class CostKeyEvent:
    id: str = ""
    timestamp: str = ""
    project_id: str = ""
    provider: Provider = Provider.UNKNOWN
    model: str | None = None
    url: str = ""
    method: str = "POST"
    status_code: int | None = None
    usage: NormalizedUsage | None = None
    cost_usd: float | None = None
    duration_ms: float = 0
    streaming: bool = False
    stream_timing: StreamTiming | None = None
    call_site: CallSite | None = None
    context: dict[str, Any] = field(default_factory=dict)
    request_body: Any = None
    response_body: Any = None


@dataclass
class CostKeyOptions:
    dsn: str = ""
    capture_body: bool = True
    before_send: Callable[[CostKeyEvent], CostKeyEvent | None] | None = None
    max_batch_size: int = 50
    flush_interval: float = 5.0
    debug: bool = False
    default_context: dict[str, Any] = field(default_factory=dict)
