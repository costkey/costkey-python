from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    GROQ = "groq"
    XAI = "xai"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"
    UNKNOWN = "unknown"


@dataclass
class NormalizedUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    # Provider-native units which cannot be reduced to tokens (for example
    # audio seconds, characters, images, or requests).  Keep these separate
    # instead of inventing a cross-modality total.
    native_units: dict[str, int | float | None] = field(default_factory=dict)
    # Some providers (notably routers) return their own amount.  Preserve the
    # amount as reported; the server remains authoritative for estimates.
    cost_usd: float | None = None
    image_output_tokens: int | None = None

    @property
    def units(self) -> dict[str, int | float | None]:
        """Compatibility alias for callers that call native units ``units``."""
        return self.native_units


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
    provider: Provider | str = Provider.UNKNOWN
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
    metadata: dict[str, Any] | None = None
    system_prompt: str | None = None
    request_body: Any = None
    response_body: Any = None
    # Additive DevX event envelope fields.  Defaults preserve the legacy
    # provider-call shape and wire compatibility.
    event_type: str = "call"
    source: str = "auto"
    entity_type: str | None = None
    entity_id: str | None = None
    sequence: int | None = None
    lifecycle: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    error_type: str | None = None
    expected_items: int | None = None
    completed_items: int | None = None
    failed_items: int | None = None
    component_id: str | None = None
    charge_id: str | None = None
    revision: int | None = None
    aggregation: str | None = None
    native_usage: dict[str, Any] | None = None
    billing_units: dict[str, Any] | None = None
    amount: dict[str, Any] | None = None
    currency: str | None = None
    amount_kind: str | None = None
    pricing_id: str | None = None
    unit: str | None = None
    unit_kind: str | None = None
    unit_value: int | float | None = None
    request_model: str | None = None
    attempt_id: str | None = None
    parent_event_id: str | None = None
    group_name: str | None = None
    group_kind: str | None = None
    group_id: str | None = None
    execution_id: str | None = None
    phase: str | None = None
    configuration: dict[str, Any] | None = None
    dataset_revision: str | None = None
    case_id: str | None = None
    trace_id: str | None = None
    trace_name: str | None = None
    provider_request_id: str | None = None
    # Agent/step/tool spans: calls carry span_id; span events carry the rest.
    span_id: str | None = None
    parent_span_id: str | None = None
    span_kind: str | None = None
    span_name: str | None = None
    span_status: str | None = None
    attributes: dict[str, Any] | None = None


@dataclass
class CostKeyOptions:
    dsn: str = ""
    capture_body: bool | None = None
    before_send: Callable[[CostKeyEvent], CostKeyEvent | None] | None = None
    before_send_event: Callable[[CostKeyEvent], CostKeyEvent | None] | None = None
    max_batch_size: int = 50
    flush_interval: float = 5.0
    debug: bool = False
    default_context: dict[str, Any] = field(default_factory=dict)
    # Optional explicit pre-call admission.  Observation remains enabled and
    # non-blocking unless a caller opts into enforcement.
    policy_mode: str = "monitor"
    policy: Any = None
    policy_admission: Any = None
    policy_endpoint: str | None = None
    policy_timeout: float = 2.0
    config_refresh_interval: float | None = 60.0
