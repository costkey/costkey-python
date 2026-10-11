"""LiveKit Agents usage: the raw ``session_usage_updated`` snapshot.

LiveKit reports cumulative per-model usage rows. This adapter sends the update
as a skeleton (numbers verbatim, ``provider``/``model``/``type`` kept, other
text replaced by its length). It does not classify rows into STT/LLM/TTS,
infer units, interpret timestamps, sum, pick billing units or compute
coverage: the server does all of that.

Wire it to the session event::

    @session.on("session_usage_updated")
    def _on_usage(ev):
        costkey.record_livekit_usage(ev, session_id=room.name, revision=next(rev))
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from costkey import telemetry as _telemetry


def _field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _names(values: Any) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = (values,)
    return [str(value) for value in values]


def record_livekit_usage(
    update: Any,
    *,
    session_id: str | None = None,
    revision: int | None = None,
    conversation_id: str | None = None,
    trace_id: str | None = None,
    execution_id: str | None = None,
    event_prefix: str | None = None,
    expected_components: list[str] | tuple[str, ...] | None = None,
    exclude_components: list[str] | tuple[str, ...] | set[str] | None = None,
    group_name: str | None = None,
    group_kind: str | None = None,
    phase: str | None = None,
    dataset_revision: str | None = None,
    case_id: str | None = None,
    declared_amounts: Mapping[str, Any] | None = None,
    declared_pricing_ids: Mapping[str, str] | None = None,
    declared_billing_units: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Send one raw LiveKit usage snapshot as a cumulative usage event.

    ``revision`` must increase with each snapshot of the session. The caller
    options (expected/excluded components, declared amounts, pricing ids and
    billing units) are forwarded verbatim for the server to apply.
    Returns the event id in a list (kept for 0.6 call sites).
    """
    from costkey.client import _current_context, _iso_timestamp, _json_safe, _project_id
    from costkey.patch import _dispatch_event, _scrub, _state
    from costkey.types import CostKeyEvent

    session = session_id or _field(update, "sessionId", "session_id", "roomName", "room_name")
    if not isinstance(session, str) or not session:
        raise ValueError("record_livekit_usage requires session_id")
    current_revision = revision if revision is not None else _field(update, "revision", "sequence", "sequenceNumber")
    if not isinstance(current_revision, int) or isinstance(current_revision, bool) or current_revision < 0:
        raise ValueError("record_livekit_usage requires a non-negative revision")
    for label, value in (
        ("declared_amounts", declared_amounts),
        ("declared_pricing_ids", declared_pricing_ids),
        ("declared_billing_units", declared_billing_units),
    ):
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")

    component_id = event_prefix or f"livekit:{session}"
    event_id = f"{component_id}:r{current_revision}"
    project_id = _project_id()
    if not project_id or _state.transport is None:
        return [event_id]

    context: dict[str, Any] = {**_current_context(), "voiceSessionId": session}
    for key, value in (
        ("conversationId", conversation_id), ("traceId", trace_id), ("executionId", execution_id),
        ("groupName", group_name), ("groupKind", group_kind), ("phase", phase),
        ("datasetRevision", dataset_revision), ("caseId", case_id),
    ):
        if value is not None:
            context[key] = value

    options = {
        key: value for key, value in (
            ("expectedComponents", _names(expected_components)),
            ("excludeComponents", _names(exclude_components)),
            ("declaredAmounts", _json_safe(declared_amounts)),
            ("declaredPricingIds", _json_safe(declared_pricing_ids)),
            ("declaredBillingUnits", _json_safe(declared_billing_units)),
        ) if value is not None
    }
    event = CostKeyEvent(
        id=event_id,
        timestamp=_iso_timestamp(),
        project_id=project_id,
        provider="livekit",
        event_type="usage",
        source="livekit",
        component_id=component_id,
        revision=current_revision,
        aggregation="cumulative",
        context=_scrub(context),
        trace_id=context.get("traceId"),
        trace_name=context.get("traceName"),
        execution_id=context.get("executionId"),
        group_name=context.get("groupName"),
        group_kind=context.get("groupKind"),
        phase=context.get("phase"),
        dataset_revision=context.get("datasetRevision"),
        case_id=context.get("caseId"),
        metadata=_scrub({"livekit": {"sessionId": session, "revision": current_revision, **options}}),
        telemetry={
            "v": _telemetry.TELEMETRY_VERSION,
            "source": "livekit",
            "livekit": {
                "sessionId": session,
                "revision": current_revision,
                "update": _telemetry.skeleton(update),
                **_scrub(options),
            },
        },
    )
    _dispatch_event(event)
    return [event_id]


record_livekit_session_usage = record_livekit_usage


__all__ = ["record_livekit_usage", "record_livekit_session_usage"]
