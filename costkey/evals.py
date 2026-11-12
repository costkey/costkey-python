"""Evals that run in your environment.

``run_eval`` reads a labelled dataset from CostKey, runs your task (your model
call, with your keys) on every item as one tracked execution, scores each
answer and reports the results. The calls are captured like any others, so the
console shows what the candidate cost next to the original. A port of
``CostKey.runEval`` in the TypeScript SDK.
"""
from __future__ import annotations

import asyncio
import base64
import contextvars
import datetime
import inspect
import json
import math
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Union
from urllib.parse import quote

__all__ = [
    "EvalItem",
    "EvalVerdict",
    "EvalResult",
    "EvalSummary",
    "run_eval",
    "run_eval_async",
    "default_eval_score",
    "computer_action",
]

TOKEN_HELP = (
    "costkey.run_eval() reads the dataset with a personal access token: pass api_token or set "
    "COSTKEY_API_TOKEN (create one in the console under Settings, MCP & coding agents)"
)


@dataclass
class EvalItem:
    id: str
    label: str
    note: Optional[str]
    expected: Any
    #: The original call's request body, with its screenshots put back.
    input: Any
    #: The original call's response body.
    output: Any
    model: Optional[str]
    provider: Optional[str]
    trace_id: Optional[str]
    source_event_id: str
    #: Some screenshots in the input couldn't be restored (not uploaded, or expired).
    incomplete: bool = False


@dataclass
class EvalVerdict:
    #: None when there's nothing to compare against: the item is reported unscored.
    passed: Optional[bool]
    reason: Optional[str] = None


@dataclass
class EvalResult:
    item_id: str
    trace_id: str
    passed: Optional[bool]
    reason: Optional[str]
    error: Optional[str]


@dataclass
class EvalSummary:
    execution_id: str
    dataset: Dict[str, Any]
    total: int
    scored: int
    passed: int
    failed: int
    errors: int
    results: List[EvalResult] = field(default_factory=list)


Task = Callable[[EvalItem], Any]
Scorer = Callable[[EvalItem, Any], Union[EvalVerdict, Awaitable[EvalVerdict]]]

_PLACEHOLDER = re.compile(r"^\[image data omitted: ([^;\]]+); ~\d+ bytes; frame ([0-9a-f]{64})\]$")


# Loading the dataset


class _Api:
    def __init__(self, api_base: str, project_id: str, token: str) -> None:
        import httpx

        self._base = f"{api_base}/projects/{quote(project_id, safe='')}"
        self._client = httpx.Client(timeout=30.0, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True)
        self._frames: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> Any:
        from costkey.patch import internal_http

        with internal_http():
            response = self._client.get(f"{self._base}{path}")
        if response.status_code >= 400 and response.status_code != 404:
            raise RuntimeError(f"CostKey dataset request failed ({response.status_code}): {response.text[:300]}")
        return response

    def frame(self, sha256: str) -> Optional[str]:
        with self._lock:
            if sha256 in self._frames:
                return self._frames[sha256]
        try:
            response = self.get(f"/frames/{sha256}")
            data = base64.b64encode(response.content).decode("ascii") if response.status_code == 200 else None
        except Exception:  # noqa: BLE001 - an expired frame leaves the placeholder in place
            data = None
        with self._lock:
            self._frames[sha256] = data
        return data

    def close(self) -> None:
        self._client.close()


def _load(dataset: str, api_base: str, project_id: str, token: str) -> tuple:
    api = _Api(api_base, project_id, token)
    try:
        listed = api.get("/datasets").json()
        rows = listed.get("datasets") if isinstance(listed, dict) else None
        rows = rows if isinstance(rows, list) else []
        found = (
            next((row for row in rows if row.get("id") == dataset), None)
            or next((row for row in rows if row.get("name") == dataset), None)
            or next((row for row in rows if str(row.get("name", "")).lower() == dataset.lower()), None)
        )
        if not found:
            raise ValueError(f'No dataset named "{dataset}" in this project')
        exported = api.get(f"/datasets/{quote(str(found['id']), safe='')}/export").text
        items: List[EvalItem] = []
        for line in exported.split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            state = {"incomplete": False}
            restored = _restore_images(row.get("input"), None, api.frame, state)
            items.append(EvalItem(
                id=str(row.get("id")),
                label=str(row.get("label") or ""),
                note=row.get("note") if isinstance(row.get("note"), str) else None,
                expected=row.get("expected"),
                input=restored,
                output=row.get("output"),
                model=row.get("model") if isinstance(row.get("model"), str) else None,
                provider=row.get("provider") if isinstance(row.get("provider"), str) else None,
                trace_id=row.get("traceId") if isinstance(row.get("traceId"), str) else None,
                source_event_id=str(row.get("sourceEventId") or ""),
                incomplete=state["incomplete"],
            ))
        return {"id": str(found["id"]), "name": str(found.get("name", "")), "revision": found.get("revision")}, items
    finally:
        api.close()


def _restore_images(value: Any, key: Optional[str], frame: Callable[[str], Optional[str]], state: Dict[str, bool]) -> Any:
    """Put uploaded screenshots back where the captured body has their placeholder."""
    if isinstance(value, str):
        if not value.startswith("[image data omitted:"):
            return value
        match = _PLACEHOLDER.match(value)
        if not match:
            state["incomplete"] = True
            return value
        data = frame(match.group(2))
        if not data:
            state["incomplete"] = True
            return value
        return data if key == "data" else f"data:{match.group(1)};base64,{data}"
    if isinstance(value, list):
        return [_restore_images(item, key, frame, state) for item in value]
    if isinstance(value, dict):
        return {entry_key: _restore_images(entry, entry_key, frame, state) for entry_key, entry in value.items()}
    return value


# Running it


class _Run:
    """What one run shares between items: the options, the results and the score events."""

    def __init__(self, dataset: Dict[str, Any], items: List[EvalItem], name: str, task: Task, score: Optional[Scorer], tolerance: float) -> None:
        self.dataset = dataset
        self.items = items
        self.name = name
        self.task = task
        self.score = score
        self.tolerance = tolerance
        self.results: List[Optional[EvalResult]] = [None] * len(items)
        self.execution_id = ""

    def verdict_for(self, item: EvalItem, output: Any) -> Any:
        return self.score(item, output) if self.score else default_eval_score(item, output, self.tolerance)

    def record(self, index: int, item: EvalItem, trace_id: str, verdict: EvalVerdict, error: Optional[str]) -> None:
        self.results[index] = EvalResult(item_id=item.id, trace_id=trace_id, passed=verdict.passed, reason=verdict.reason, error=error)
        if verdict.passed is None or not trace_id:
            return
        _emit_score(
            event_id=f"eval_{self.execution_id}_{item.id}",
            trace_id=trace_id,
            name=self.name,
            passed=bool(verdict.passed),
            reason=verdict.reason,
            context={"caseId": item.id, "phase": "candidate", "executionId": self.execution_id},
        )

    def summary(self) -> EvalSummary:
        results = [result for result in self.results if result is not None]
        scored = [result for result in results if result.passed is not None]
        return EvalSummary(
            execution_id=self.execution_id,
            dataset=self.dataset,
            total=len(results),
            scored=len(scored),
            passed=sum(1 for result in scored if result.passed),
            failed=sum(1 for result in scored if not result.passed),
            errors=sum(1 for result in results if result.error),
            results=results,
        )


def _verdict(value: Any) -> EvalVerdict:
    if isinstance(value, EvalVerdict):
        return value
    if isinstance(value, bool) or value is None:
        return EvalVerdict(passed=value)
    if isinstance(value, Mapping):
        passed = value.get("passed", value.get("pass"))
        return EvalVerdict(passed=None if passed is None else bool(passed), reason=value.get("reason"))
    raise TypeError("a scorer returns EvalVerdict(passed=..., reason=...), a bool, or None")


def _failure(error: BaseException) -> tuple:
    return f"{type(error).__name__}: {error}", EvalVerdict(passed=False, reason=f"The task failed ({type(error).__name__})")


def _prepare(dataset: Any, task: Any, api_token: Optional[str]) -> tuple:
    from costkey import client

    if not client._initialized or not client._api_base:
        raise RuntimeError("costkey.run_eval() needs costkey.init() with a DSN first")
    if not dataset or not callable(task):
        raise TypeError("costkey.run_eval() needs dataset= and task=")
    token = api_token or os.environ.get("COSTKEY_API_TOKEN")
    if not token:
        raise RuntimeError(TOKEN_HELP)
    from costkey.patch import _state

    return client._api_base, _state.project_id, token


def _clamp_concurrency(concurrency: int) -> int:
    return max(1, min(16, int(concurrency or 1)))


def run_eval(
    dataset: str,
    task: Task,
    *,
    score: Optional[Scorer] = None,
    name: Optional[str] = None,
    configuration: Optional[Mapping[str, Any]] = None,
    api_token: Optional[str] = None,
    concurrency: int = 2,
    tolerance: float = 40,
) -> EvalSummary:
    """Run ``task`` on every item of a dataset, score each answer and report it.

    ``dataset`` is a dataset name or id. ``task(item)`` is your model call for
    one item; return the response. ``score(item, output)`` is your own scorer
    (default: :func:`default_eval_score`). ``configuration`` says what you
    changed, such as ``{"model": "claude-haiku-4-5"}``, and shows on the run.
    The dataset is read with a personal access token (``api_token`` or
    ``COSTKEY_API_TOKEN``). Items run ``concurrency`` at a time, on threads.
    For an async task, use :func:`run_eval_async`.
    """
    from costkey.client import flush, get_current_context, start_execution, start_trace

    api_base, project_id, token = _prepare(dataset, task, api_token)
    info, items = _load(str(dataset), api_base, project_id, token)
    run = _Run(info, items, (name or f"eval: {info['name']}")[:120], task, score, tolerance)

    def one(index: int, item: EvalItem) -> None:
        with start_trace(run.name, trace_id=uuid.uuid4().hex, caseId=item.id, phase="candidate"):
            trace_id = str(get_current_context().get("traceId") or "")
            error = None
            try:
                output = task(item)
                if inspect.isawaitable(output):
                    _close(output)
                    raise TypeError("task returned a coroutine: use await costkey.run_eval_async(...)")
                verdict = run.verdict_for(item, output)
                if inspect.isawaitable(verdict):
                    _close(verdict)
                    raise TypeError("score returned a coroutine: use await costkey.run_eval_async(...)")
                verdict = _verdict(verdict)
            except Exception as failure:  # noqa: BLE001 - a failing item is a result, not a crash
                error, verdict = _failure(failure)
            run.record(index, item, trace_id, verdict, error)

    with start_execution(
        run.name,
        kind="evaluation",
        dataset_revision=f"{info['id']}@{info['revision']}",
        configuration=configuration,
        expected_items=len(items),
    ):
        run.execution_id = str(get_current_context().get("executionId") or "")
        workers = min(_clamp_concurrency(concurrency), max(1, len(items)))
        if workers == 1:
            for index, item in enumerate(items):
                one(index, item)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="costkey-eval") as pool:
                futures = [pool.submit(contextvars.copy_context().run, one, index, item) for index, item in enumerate(items)]
                for future in futures:
                    future.result()
    flush()
    return run.summary()


async def run_eval_async(
    dataset: str,
    task: Task,
    *,
    score: Optional[Scorer] = None,
    name: Optional[str] = None,
    configuration: Optional[Mapping[str, Any]] = None,
    api_token: Optional[str] = None,
    concurrency: int = 2,
    tolerance: float = 40,
) -> EvalSummary:
    """:func:`run_eval` for async code: ``task`` and ``score`` may be coroutines.
    Items run ``concurrency`` at a time on the running event loop."""
    from costkey.client import flush, get_current_context, start_execution, start_trace

    api_base, project_id, token = _prepare(dataset, task, api_token)
    info, items = await asyncio.get_running_loop().run_in_executor(None, _load, str(dataset), api_base, project_id, token)
    run = _Run(info, items, (name or f"eval: {info['name']}")[:120], task, score, tolerance)
    limit = asyncio.Semaphore(_clamp_concurrency(concurrency))

    async def one(index: int, item: EvalItem) -> None:
        async with limit:
            with start_trace(run.name, trace_id=uuid.uuid4().hex, caseId=item.id, phase="candidate"):
                trace_id = str(get_current_context().get("traceId") or "")
                error = None
                try:
                    output = task(item)
                    if inspect.isawaitable(output):
                        output = await output
                    verdict = run.verdict_for(item, output)
                    if inspect.isawaitable(verdict):
                        verdict = await verdict
                    verdict = _verdict(verdict)
                except Exception as failure:  # noqa: BLE001
                    error, verdict = _failure(failure)
                run.record(index, item, trace_id, verdict, error)

    with start_execution(
        run.name,
        kind="evaluation",
        dataset_revision=f"{info['id']}@{info['revision']}",
        configuration=configuration,
        expected_items=len(items),
    ):
        run.execution_id = str(get_current_context().get("executionId") or "")
        await asyncio.gather(*(one(index, item) for index, item in enumerate(items)))
    await asyncio.get_running_loop().run_in_executor(None, flush)
    return run.summary()


def _close(awaitable: Any) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _emit_score(*, event_id: str, trace_id: str, name: str, passed: bool, reason: Optional[str], context: Dict[str, Any]) -> None:
    from costkey.patch import _dispatch_event, _state
    from costkey.types import CostKeyEvent, Provider

    _dispatch_event(CostKeyEvent(
        id=event_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        project_id=_state.project_id,
        provider=Provider.UNKNOWN,
        event_type="score",
        source="code",
        context=context,
        trace_id=trace_id,
        execution_id=context.get("executionId"),
        phase="candidate",
        case_id=context.get("caseId"),
        score_name=name,
        score_value=1 if passed else 0,
        score_label="pass" if passed else "fail",
        score_comment=reason,
    ))


# The default scorer


def _parse_object(body: Any) -> Optional[Dict[str, Any]]:
    value = body
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or isinstance(value, (list, tuple, int, float, bool)):
        return None
    try:
        from costkey.telemetry import _plain

        plain = _plain(value)
    except Exception:  # noqa: BLE001
        return None
    return dict(plain) if isinstance(plain, Mapping) else None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def computer_action(body: Any) -> Optional[Dict[str, Any]]:
    """The computer action in a response: Anthropic's computer tool or OpenAI's ``computer_call``."""
    response = _parse_object(body)
    if not response:
        return None
    for block in response.get("content") if isinstance(response.get("content"), list) else []:
        block = _parse_object(block) or {}
        tool_input = _parse_object(block.get("input"))
        if block.get("type") != "tool_use" or not isinstance((tool_input or {}).get("action"), str):
            continue
        coordinate = tool_input.get("coordinate") if isinstance(tool_input.get("coordinate"), list) else []
        action: Dict[str, Any] = {"type": tool_input["action"], "x": _number(coordinate[0] if coordinate else None), "y": _number(coordinate[1] if len(coordinate) > 1 else None)}
        if isinstance(tool_input.get("text"), str):
            action["text"] = tool_input["text"]
        return action
    for entry in response.get("output") if isinstance(response.get("output"), list) else []:
        entry = _parse_object(entry) or {}
        call = _parse_object(entry.get("action"))
        if entry.get("type") != "computer_call" or not isinstance((call or {}).get("type"), str):
            continue
        keys = "+".join(str(key) for key in call["keys"]) if isinstance(call.get("keys"), list) else None
        text = call["text"] if isinstance(call.get("text"), str) else keys
        action = {"type": call["type"], "x": _number(call.get("x")), "y": _number(call.get("y"))}
        if text:
            action["text"] = text
        return action
    return None


def _expected_action(expected: Any) -> Optional[Dict[str, Any]]:
    value = _parse_object(expected)
    if not value:
        return None
    kind = value.get("action") if isinstance(value.get("action"), str) else value.get("type") if isinstance(value.get("type"), str) else None
    if not kind:
        return None
    coordinate = value.get("coordinate") if isinstance(value.get("coordinate"), list) else []
    action: Dict[str, Any] = {
        "type": kind,
        "x": _number(value.get("x")) if _number(value.get("x")) is not None else _number(coordinate[0] if coordinate else None),
        "y": _number(value.get("y")) if _number(value.get("y")) is not None else _number(coordinate[1] if len(coordinate) > 1 else None),
    }
    if isinstance(value.get("text"), str):
        action["text"] = value["text"]
    return action


_ALIASES = {"left_click": "click", "click": "click", "keypress": "key", "key": "key"}


def _same_action(a: Dict[str, Any], b: Dict[str, Any], tolerance: float) -> bool:
    if _ALIASES.get(a["type"], a["type"]) != _ALIASES.get(b["type"], b["type"]):
        return False
    if None not in (a.get("x"), a.get("y"), b.get("x"), b.get("y")):
        if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > tolerance:
            return False
    if a.get("text") is not None and b.get("text") is not None and a["text"].strip().lower() != b["text"].strip().lower():
        return False
    return True


def _describe(action: Dict[str, Any]) -> str:
    where = f" at {round(action['x'])}, {round(action['y'])}" if action.get("x") is not None and action.get("y") is not None else ""
    return f"{action['type']}{where}"


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    value = _parse_object(output) or {}
    parts: List[str] = []
    for block in value.get("content") if isinstance(value.get("content"), list) else []:
        text = (_parse_object(block) or {}).get("text")
        if isinstance(text, str):
            parts.append(text)
    for choice in value.get("choices") if isinstance(value.get("choices"), list) else []:
        content = (_parse_object((_parse_object(choice) or {}).get("message")) or {}).get("content")
        if isinstance(content, str):
            parts.append(content)
    if isinstance(value.get("output_text"), str):
        parts.append(value["output_text"])
    return "\n".join(parts)


def default_eval_score(item: EvalItem, output: Any, tolerance: float = 40) -> EvalVerdict:
    """The default scorer:

    - with an expected action, the candidate must take it (same kind; clicks within ``tolerance`` px);
    - for an item labelled Good, it must take the original action;
    - for an item labelled Bad, it must not repeat the original action;
    - with expected text, the answer must contain it;
    - otherwise the item is unscored.
    """
    candidate = computer_action(output)
    expected = _expected_action(item.expected)
    if expected:
        if not candidate:
            return EvalVerdict(False, f"Expected {_describe(expected)}, got no action")
        if _same_action(candidate, expected, tolerance):
            return EvalVerdict(True, f"Took the expected action ({_describe(candidate)})")
        return EvalVerdict(False, f"Expected {_describe(expected)}, got {_describe(candidate)}")
    original = computer_action(item.output)
    label = item.label.strip().lower()
    if original and label == "good":
        if not candidate:
            return EvalVerdict(False, f"Expected {_describe(original)}, got no action")
        if _same_action(candidate, original, tolerance):
            return EvalVerdict(True, f"Matched the good action ({_describe(candidate)})")
        return EvalVerdict(False, f"Expected {_describe(original)}, got {_describe(candidate)}")
    if original and label == "bad":
        if not candidate:
            return EvalVerdict(True, "Didn't repeat the bad action")
        if _same_action(candidate, original, tolerance):
            return EvalVerdict(False, f"Repeated the bad action ({_describe(candidate)})")
        return EvalVerdict(True, f"Avoided the bad action: {_describe(candidate)} instead of {_describe(original)}")
    if isinstance(item.expected, str) and item.expected.strip():
        if item.expected.strip().lower() in _output_text(output).lower():
            return EvalVerdict(True, "The answer contains the expected text")
        return EvalVerdict(False, "The answer doesn't contain the expected text")
    return EvalVerdict(None, "Nothing to compare with: label it Good or Bad, or give it an expected action")
