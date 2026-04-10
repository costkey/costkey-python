"""Batched async transport with retry + backoff. Never blocks. Never throws."""
from __future__ import annotations
import json
import time
import threading
import logging
from typing import Any
import httpx
from costkey.types import CostKeyEvent

logger = logging.getLogger("costkey")

MAX_QUEUE_SIZE = 500
MAX_RETRIES = 10
BASE_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0


class Transport:
    def __init__(self, endpoint: str, auth_key: str, max_batch_size: int,
                 flush_interval: float, debug: bool, release: str | None = None):
        self._endpoint = endpoint
        self._auth_key = auth_key
        self._max_batch_size = max_batch_size
        self._flush_interval = flush_interval
        self._debug = debug
        self._release = release
        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._consecutive_failures = 0
        self._backoff_until = 0.0

    def start(self) -> None:
        self._schedule_flush()

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def enqueue(self, event: CostKeyEvent) -> None:
        with self._lock:
            if len(self._queue) >= MAX_QUEUE_SIZE:
                self._queue.pop(0)
                if self._debug:
                    logger.warning("[costkey] Queue full, dropping oldest event")

            self._queue.append(self._serialize(event))

            if len(self._queue) >= self._max_batch_size:
                self._do_flush()

    def flush(self) -> None:
        with self._lock:
            self._do_flush()

    def _schedule_flush(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        with self._lock:
            self._do_flush()
        self._schedule_flush()

    def _do_flush(self) -> None:
        if not self._queue:
            return

        # Respect backoff
        now = time.monotonic()
        if now < self._backoff_until:
            if self._debug:
                wait = round(self._backoff_until - now)
                logger.warning(f"[costkey] Backing off, retry in {wait}s")
            return

        batch = self._queue[:self._max_batch_size]
        self._queue = self._queue[self._max_batch_size:]

        payload: dict[str, Any] = {"sdkVersion": "python-0.3.0", "events": batch}
        if self._release:
            payload["release"] = self._release

        try:
            resp = httpx.post(
                self._endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._auth_key}",
                    "User-Agent": "costkey-python/0.3.0",
                },
                timeout=10,
            )
            if resp.status_code == 429:
                # Rate limited — put events back
                self._queue = batch + self._queue
                self._apply_backoff()
                if self._debug:
                    logger.warning("[costkey] Rate limited, will retry")
            elif resp.status_code >= 500:
                # Server error — put events back and retry
                self._queue = batch + self._queue
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
        except Exception as e:
            # Network error — put events back and retry
            self._queue = batch + self._queue
            self._apply_backoff()
            if self._debug:
                logger.warning(f"[costkey] Network error, {len(self._queue)} events queued, retry in {self._get_backoff_s():.0f}s: {e}")

    def _apply_backoff(self) -> None:
        self._consecutive_failures += 1
        backoff = self._get_backoff_s()
        self._backoff_until = time.monotonic() + backoff

        # After too many failures, drop old events to prevent unbounded growth
        if self._consecutive_failures > MAX_RETRIES and len(self._queue) > MAX_QUEUE_SIZE // 2:
            drop_count = len(self._queue) // 4
            self._queue = self._queue[drop_count:]
            if self._debug:
                logger.warning(f"[costkey] Too many failures, dropped {drop_count} oldest events")

    def _get_backoff_s(self) -> float:
        # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s...
        return min(BASE_BACKOFF_S * (2 ** (self._consecutive_failures - 1)), MAX_BACKOFF_S)

    def _serialize(self, event: CostKeyEvent) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": event.id,
            "timestamp": event.timestamp,
            "projectId": event.project_id,
            "provider": event.provider.value,
            "model": event.model,
            "url": event.url,
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
            "callSite": None,
            "context": event.context,
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
            }
        if event.call_site:
            d["callSite"] = {
                "raw": event.call_site.raw,
                "frames": [
                    {"functionName": f.function_name, "fileName": f.file_name,
                     "lineNumber": f.line_number, "columnNumber": None}
                    for f in event.call_site.frames
                ],
            }
        return d
