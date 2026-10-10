"""Bounded background transport with retry/backoff.

Enqueue is non-blocking; explicit flush/shutdown may wait for best-effort
delivery at an application-owned lifecycle boundary.
"""
from __future__ import annotations
import json
import time
import threading
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import httpx
from costkey.types import CostKeyEvent
from costkey.version import __version__

logger = logging.getLogger("costkey")


def _sanitize_url(value: Any) -> str:
    """Defense-in-depth URL sanitization at the final telemetry boundary."""
    if not isinstance(value, str) or not value:
        return ""
    try:
        parsed = urlsplit(value)
        if not parsed.hostname:
            return ""
        hostname = parsed.hostname
        netloc = hostname
        if ":" in hostname and not hostname.startswith("["):
            netloc = f"[{hostname}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port:
            netloc += f":{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except Exception:
        return ""

MAX_QUEUE_SIZE = 500
MAX_RETRIES = 10
BASE_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0


class Transport:
    def __init__(self, endpoint: str, auth_key: str, max_batch_size: int,
                 flush_interval: float, debug: bool, release: str | None = None):
        self._endpoint = endpoint
        self._auth_key = auth_key
        self._max_batch_size = max(1, int(max_batch_size))
        self._flush_interval = max(0.01, float(flush_interval))
        self._debug = debug
        self._release = release
        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        # One long-lived worker handles all automatic delivery.  Enqueue only
        # mutates the bounded in-memory queue and signals this worker; it never
        # performs a network request on a provider-call thread.
        self._flush_lock = threading.Lock()
        self._flush_signal = threading.Event()
        self._stop_signal = threading.Event()
        self._worker: threading.Thread | None = None
        self._started = False
        # Kept as a compatibility attribute for code that inspected the old
        # timer; automatic scheduling now lives in the single worker.
        self._timer: threading.Timer | None = None
        self._consecutive_failures = 0
        self._backoff_until = 0.0
        self._connected = False

    def start(self) -> None:
        with self._lock:
            if self._started and self._worker and self._worker.is_alive():
                return
            self._started = True
            self._stop_signal.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="costkey-transport",
                daemon=True,
            )
            self._worker.start()

    def stop(self) -> None:
        # ``client.shutdown`` flushes first, but making stop itself drain is a
        # safer lifecycle boundary for direct Transport users as well.
        self.flush()
        worker = self._worker
        self._stop_signal.set()
        self._flush_signal.set()
        if worker and worker is not threading.current_thread():
            worker.join(timeout=10)
        with self._lock:
            self._started = False
            self._worker = None
        self._timer = None

    def enqueue(self, event: CostKeyEvent) -> None:
        with self._lock:
            if len(self._queue) >= MAX_QUEUE_SIZE:
                self._queue.pop(0)
                if self._debug:
                    logger.warning("[costkey] Queue full, dropping oldest event")

            self._queue.append(self._serialize(event))

            if len(self._queue) >= self._max_batch_size:
                self._flush_signal.set()

    def flush(self) -> None:
        self._flush_signal.set()
        # Explicit flush is allowed to block at a caller-owned lifecycle
        # boundary.  Serialize it with the worker, but keep network I/O out of
        # the queue lock so concurrent provider calls remain independent.
        with self._flush_lock:
            self._flush_available(force=True)

    def _worker_loop(self) -> None:
        while not self._stop_signal.is_set():
            self._flush_signal.wait(timeout=self._flush_interval)
            self._flush_signal.clear()
            if self._stop_signal.is_set():
                break
            with self._flush_lock:
                self._flush_available(force=False)

    def _flush_available(self, *, force: bool) -> None:
        while True:
            batch = self._take_batch(force=force)
            if batch is None:
                return
            if not self._send_batch(batch):
                return

    # Internal compatibility seam for older callers/tests.  It is explicit
    # caller-owned flushing; enqueue never invokes it.
    def _do_flush(self) -> None:
        with self._flush_lock:
            self._flush_available(force=True)

    def _schedule_flush(self) -> None:
        """Compatibility helper; automatic scheduling uses one worker."""
        if not self._started:
            self.start()
        else:
            self._flush_signal.set()

    def _tick(self) -> None:
        self._flush_signal.set()

    def _take_batch(self, *, force: bool) -> list[dict[str, Any]] | None:
        with self._lock:
            if not self._queue:
                return None
            if not force:
                now = time.monotonic()
                if now < self._backoff_until:
                    if self._debug:
                        wait = round(self._backoff_until - now)
                        logger.warning(f"[costkey] Backing off, retry in {wait}s")
                    return None
            batch = self._queue[:self._max_batch_size]
            self._queue = self._queue[self._max_batch_size:]
            return batch

    def _send_batch(self, batch: list[dict[str, Any]]) -> bool:
        payload: dict[str, Any] = {"sdkVersion": f"python-{__version__}", "events": batch}
        if self._release:
            payload["release"] = self._release

        try:
            resp = httpx.post(
                self._endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._auth_key}",
                    "User-Agent": f"costkey-python/{__version__}",
                },
                timeout=10,
            )
            if resp.status_code in (401, 403):
                # Auth failure — always log (not just debug)
                logger.warning(f"[costkey] Authentication failed ({resp.status_code}). Check your DSN at https://app.costkey.dev/settings")
            elif resp.status_code == 429:
                # Rate limited — put events back
                self._requeue(batch)
                self._apply_backoff()
                if self._debug:
                    logger.warning("[costkey] Rate limited, will retry")
            elif resp.status_code >= 500:
                # Server error — put events back and retry
                self._requeue(batch)
                self._apply_backoff()
                if self._debug:
                    logger.warning(f"[costkey] Server error {resp.status_code}, will retry ({self._consecutive_failures} failures)")
            elif not resp.is_success:
                # Client error — drop, won't succeed on retry
                if self._debug:
                    logger.warning(f"[costkey] Ingest returned {resp.status_code}")
            else:
                # Success — reset backoff
                self._consecutive_failures = 0
                self._backoff_until = 0.0
                if not self._connected:
                    self._connected = True
                    print("[costkey] Connected. Tracking AI calls.")
            return resp.is_success
        except Exception as e:
            # Network error — put events back and retry
            self._requeue(batch)
            self._apply_backoff()
            if self._debug:
                logger.warning(f"[costkey] Network error, {len(self._queue)} events queued, retry in {self._get_backoff_s():.0f}s: {e}")
        return False

    def _requeue(self, batch: list[dict[str, Any]]) -> None:
        with self._lock:
            # Keep the same bounded queue guarantee on failed delivery as on
            # normal enqueue. Oldest events are the first to be discarded.
            combined = batch + self._queue
            self._queue = combined[:MAX_QUEUE_SIZE]

    def _apply_backoff(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            backoff = self._get_backoff_s()
            self._backoff_until = time.monotonic() + backoff

            # After too many failures, drop old events to prevent unbounded growth
            if self._consecutive_failures > MAX_RETRIES and len(self._queue) > MAX_QUEUE_SIZE // 2:
                drop_count = len(self._queue) // 4
                self._queue = self._queue[drop_count:]
            else:
                drop_count = 0
            if self._debug:
                if drop_count:
                    logger.warning(f"[costkey] Too many failures, dropped {drop_count} oldest events")

    def _get_backoff_s(self) -> float:
        # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s...
        return min(BASE_BACKOFF_S * (2 ** (self._consecutive_failures - 1)), MAX_BACKOFF_S)

    def _serialize(self, event: CostKeyEvent) -> dict[str, Any]:
        provider = getattr(event.provider, "value", event.provider)
        d: dict[str, Any] = {
            "id": event.id,
            "timestamp": event.timestamp,
            "projectId": event.project_id,
            "eventType": event.event_type,
            "source": event.source,
            "provider": str(provider),
            "model": event.model,
            "url": _sanitize_url(event.url),
            "method": event.method,
            "statusCode": event.status_code,
            "usage": None,
            "costUsd": event.cost_usd,
            "durationMs": event.duration_ms,
            "streaming": event.streaming,
            "streamTiming": {
                "ttft": event.stream_timing.ttft,
                "tps": event.stream_timing.tps,
                "streamDuration": event.stream_timing.stream_duration,
                "chunkCount": event.stream_timing.chunk_count,
            } if event.stream_timing else None,
            "callSite": {
                "raw": event.call_site.raw,
                "frames": [
                    {"functionName": f.function_name, "fileName": f.file_name,
                     "lineNumber": f.line_number, "columnNumber": None}
                    for f in event.call_site.frames
                ],
            } if event.call_site else None,
            "context": event.context,
            "metadata": event.metadata,
            "systemPrompt": event.system_prompt,
            "requestBody": event.request_body,
            "responseBody": event.response_body,
        }
        if event.usage:
            d["usage"] = {
                "inputTokens": event.usage.input_tokens,
                "outputTokens": event.usage.output_tokens,
                "totalTokens": event.usage.total_tokens,
                "reasoningTokens": event.usage.reasoning_tokens,
                "cacheReadTokens": event.usage.cache_read_tokens,
                "cacheCreationTokens": event.usage.cache_creation_tokens,
                "nativeUnits": event.usage.native_units,
                "costUsd": event.usage.cost_usd,
                "imageOutputTokens": event.usage.image_output_tokens,
            }

        # Promote standard identity fields so new servers can index them while
        # retaining the legacy context object used by older servers.
        ctx = event.context or {}
        identity = {
            "traceId": event.trace_id or ctx.get("traceId", ctx.get("trace_id")),
            "traceName": event.trace_name or ctx.get("traceName", ctx.get("trace_name")),
            "executionId": event.execution_id or ctx.get("executionId", ctx.get("execution_id")),
            "groupName": event.group_name or ctx.get("groupName", ctx.get("group_name")),
            "groupKind": event.group_kind or ctx.get("groupKind", ctx.get("group_kind")),
            "groupId": event.group_id or ctx.get("groupId", ctx.get("group_id")),
            "phase": event.phase or ctx.get("phase"),
            "datasetRevision": event.dataset_revision or ctx.get("datasetRevision", ctx.get("dataset_revision")),
            "caseId": event.case_id or ctx.get("caseId", ctx.get("case_id")),
            "configuration": event.configuration or ctx.get("configuration"),
        }
        for key, value in identity.items():
            if value is not None:
                d[key] = value

        if event.request_model is not None:
            d["requestModel"] = event.request_model
        if event.provider_request_id is not None:
            d["providerRequestId"] = event.provider_request_id
        if event.attempt_id is not None:
            d["attemptId"] = event.attempt_id
        if event.parent_event_id is not None:
            d["parentEventId"] = event.parent_event_id
        if event.span_id is not None:
            d["spanId"] = event.span_id
        if event.event_type == "span":
            d["kind"] = event.span_kind or "custom"
            d["name"] = event.span_name
            d["status"] = event.span_status or "running"
            if event.parent_span_id is not None:
                d["parentSpanId"] = event.parent_span_id
            if event.attributes:
                d["attributes"] = event.attributes
        if event.entity_type is not None:
            d["entityType"] = event.entity_type
        if event.entity_id is not None:
            d["entityId"] = event.entity_id
        if event.sequence is not None:
            d["sequence"] = event.sequence
        if event.lifecycle is not None:
            d["lifecycle"] = event.lifecycle
        if event.started_at is not None:
            d["startedAt"] = event.started_at
        if event.ended_at is not None:
            d["endedAt"] = event.ended_at
        if event.error_type is not None:
            d["errorType"] = event.error_type
        if event.expected_items is not None:
            d["expectedItems"] = event.expected_items
        if event.completed_items is not None:
            d["completedItems"] = event.completed_items
        if event.failed_items is not None:
            d["failedItems"] = event.failed_items
        if event.configuration is not None:
            d["configuration"] = event.configuration
        if event.component_id is not None:
            d["componentId"] = event.component_id
        if event.charge_id is not None:
            d["chargeId"] = event.charge_id
        if event.revision is not None:
            d["revision"] = event.revision
        if event.aggregation is not None:
            d["aggregation"] = event.aggregation
        if event.native_usage is not None:
            d["usage"] = event.native_usage
            # Keep the established adapter alias while ``usage`` remains the
            # canonical field used by the cross-language envelope.
            d["nativeUsage"] = event.native_usage
        if event.billing_units is not None:
            d["billingUnits"] = event.billing_units
        if event.amount is not None:
            d["amount"] = event.amount
        if event.currency is not None:
            d["currency"] = event.currency
        if event.amount_kind is not None:
            d["amountKind"] = event.amount_kind
        if event.pricing_id is not None:
            d["pricingId"] = event.pricing_id
        if event.unit is not None:
            d["unit"] = event.unit
        if event.unit_kind is not None:
            d["unitKind"] = event.unit_kind
        if event.unit_value is not None:
            d["unitValue"] = event.unit_value
        return d
