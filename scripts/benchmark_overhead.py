#!/usr/bin/env python3
"""Measure Python SDK request overhead against a loopback HTTP provider.

The benchmark is intentionally local-only. It measures the provider request
round trip plus client-side work in the calling process; it does not contact
CostKey or a real provider. Results are environment-specific evidence, not a
performance guarantee.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import socket
import sys
import threading
import time
from collections import Counter
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Callable

import httpx

# Make a direct checkout invocation use this repository's package rather than
# an unrelated globally installed ``costkey`` distribution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import costkey
from costkey.types import NormalizedUsage


class BenchmarkExtractor:
    provider = "benchmark"

    def match(self, url: str) -> bool:
        return urlsplit(url).path == "/provider-overhead"

    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        return NormalizedUsage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        )

    def extract_model(self, request_body: Any, response_body: Any) -> str | None:
        for body in (response_body, request_body):
            if isinstance(body, dict) and isinstance(body.get("model"), str):
                return body["model"]
        return None


class LoopbackState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: Counter[str] = Counter()

    def count(self, path: str) -> None:
        with self.lock:
            self.counts[path] += 1

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return dict(self.counts)


class LoopbackHandler(BaseHTTPRequestHandler):
    def _state(self) -> LoopbackState:
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        self._state().count(path)
        if path.endswith("/config"):
            project_id = path.split("/projects/", 1)[-1].rsplit("/config", 1)[0]
            self._send_json(200, {
                "projectId": project_id,
                "capture": {
                    "mode": "metadata",
                    "bodyCaptureEnabled": False,
                    "redactionEnabled": True,
                },
                "providers": [],
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        self._state().count(path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}

        if path == "/provider-overhead":
            self._send_json(200, {
                "id": "benchmark-provider-request",
                "model": "benchmark-model",
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            })
        elif path.endswith("/policy/admit"):
            self._send_json(200, {
                "decision": "allow",
                "providerCallAllowed": True,
                "reservationId": f"reservation-{body.get('admissionId', 'benchmark')}",
                "policies": [{"unit": "calls", "result": "allow"}],
            })
        elif path.endswith("/policy/settle"):
            self._send_json(200, {"state": "settled"})
        elif path == "/api/v1/events":
            self._send_json(202, {"accepted": len(body.get("events", [])) if isinstance(body, dict) else 0})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, *_args: Any) -> None:
        return


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _stats(samples_us: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(samples_us),
        "minUs": round(min(samples_us), 3) if samples_us else 0.0,
        "p50Us": round(_percentile(samples_us, 0.50), 3),
        "p95Us": round(_percentile(samples_us, 0.95), 3),
        "maxUs": round(max(samples_us), 3) if samples_us else 0.0,
    }


def _request(client: httpx.Client, url: str) -> None:
    response = client.post(url, json={"model": "benchmark-model"})
    response.raise_for_status()


def _measure(
    url: str,
    iterations: int,
    warmup: int,
    request_fn: Callable[[httpx.Client, str], None] = _request,
) -> dict[str, float | int]:
    samples: list[float] = []
    with httpx.Client(timeout=10) as client:
        for _ in range(warmup):
            request_fn(client, url)
        for _ in range(iterations):
            started = time.perf_counter_ns()
            request_fn(client, url)
            samples.append((time.perf_counter_ns() - started) / 1_000)
    return _stats(samples)


def _wait_for_config(timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while costkey.get_runtime_config_status() == "unknown" and time.monotonic() < deadline:
        time.sleep(0.002)


def _measure_sdk(
    dsn: str,
    url: str,
    iterations: int,
    warmup: int,
    *,
    enforce: bool = False,
) -> dict[str, float | int]:
    options: dict[str, Any] = {
        "dsn": dsn,
        "capture_body": False,
        "max_batch_size": 500,
        "flush_interval": 3600,
        "scan_callgraph": False,
        "config_refresh_interval": None,
    }
    if enforce:
        options["policy"] = {
            "mode": "enforce",
            "integration": {
                "supported": True,
                "admissionSupported": True,
                "verified": True,
            },
        }
    costkey.init(**options)
    try:
        _wait_for_config()
        return _measure(url, iterations, warmup)
    finally:
        costkey.shutdown()


def _loopback_guard() -> tuple[Callable[..., Any], Callable[..., Any]]:
    original_connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(f"benchmark network guard blocked non-loopback host: {host}")
        return original_connect(sock, address)

    socket.socket.connect = guarded_connect  # type: ignore[assignment]
    return original_connect, guarded_connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="also measure an explicitly enforced local admission + settlement path",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")

    original_connect, _guarded_connect = _loopback_guard()
    server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
    server.state = LoopbackState()  # type: ignore[attr-defined]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/provider-overhead"
    dsn = f"http://benchmark-key@127.0.0.1:{port}/benchmark-project"
    costkey.register_extractor(BenchmarkExtractor())

    try:
        baseline = _measure(url, args.iterations, args.warmup)
        # Transport emits its first-connect notice on stdout; keep the
        # benchmark artifact machine-readable JSON.
        with redirect_stdout(io.StringIO()):
            metadata_only = _measure_sdk(dsn, url, args.iterations, args.warmup)
        results: dict[str, Any] = {
            "baseline": baseline,
            "metadataOnly": metadata_only,
        }
        if args.enforce:
            with redirect_stdout(io.StringIO()):
                results["enforcedAdmission"] = _measure_sdk(
                    dsn, url, args.iterations, args.warmup, enforce=True,
                )
        results["metadataOnlyDelta"] = {
            "p50Us": round(float(metadata_only["p50Us"]) - float(baseline["p50Us"]), 3),
            "p95Us": round(float(metadata_only["p95Us"]) - float(baseline["p95Us"]), 3),
        }
        if "enforcedAdmission" in results:
            enforced = results["enforcedAdmission"]
            results["enforcedAdmissionDelta"] = {
                "p50Us": round(float(enforced["p50Us"]) - float(baseline["p50Us"]), 3),
                "p95Us": round(float(enforced["p95Us"]) - float(baseline["p95Us"]), 3),
            }
        report = {
            "schemaVersion": 1,
            "scope": "loopback HTTP provider; no external network",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "runtime": {
                "python": sys.version,
                "pythonExecutable": sys.executable,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "costkey": costkey.__version__,
                "httpx": httpx.__version__,
            },
            "results": results,
            "serverRequests": server.state.snapshot(),  # type: ignore[attr-defined]
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        costkey.shutdown()
        server.shutdown()
        server.server_close()
        socket.socket.connect = original_connect  # type: ignore[assignment]


if __name__ == "__main__":
    main()
