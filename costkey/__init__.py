"""CostKey: AI cost observability. The SDK only sends raw telemetry."""
from costkey.client import (
    init,
    shutdown,
    flush,
    with_context,
    start_trace,
    start_execution,
    record_usage,
    export_context,
    export_context_headers,
    import_context,
    get_runtime_config_status,
    get_current_context,
    span,
    agent,
    step,
    tool,
    SpanHandle,
)
from costkey.policy import CostKeyPolicyRefused, PolicyAdmissionOptions, PolicyOptions
from costkey.livekit import record_livekit_usage, record_livekit_session_usage

from costkey.version import __version__
__all__ = [
    "init",
    "shutdown",
    "flush",
    "with_context",
    "start_trace",
    "start_execution",
    "record_usage",
    "export_context",
    "export_context_headers",
    "import_context",
    "get_runtime_config_status",
    "get_current_context",
    "span",
    "agent",
    "step",
    "tool",
    "SpanHandle",
    "CostKeyPolicyRefused",
    "PolicyAdmissionOptions",
    "PolicyOptions",
    "record_livekit_usage",
    "record_livekit_session_usage",
    "__version__",
]
