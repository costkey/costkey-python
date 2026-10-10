"""Reference LiveKit usage adapter.

LiveKit exposes session usage updates as cumulative component snapshots. This
adapter intentionally consumes an application callback/event payload; it does
not intercept arbitrary audio or claim universal voice coverage. Providers,
models, native units, and any bundled amount must be supplied by the verified
LiveKit integration.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from costkey.client import record_usage

_COMPONENT_KEYS = {
    "stt": ("stt", "speech_to_text", "speechToText", "stt_usage", "sttUsage"),
    "llm": ("llm", "llm_usage", "llmUsage", "language_model", "languageModel"),
    "tts": ("tts", "text_to_speech", "textToSpeech", "tts_usage", "ttsUsage"),
}


def _field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        try:
            result = asdict(value)
            return result if isinstance(result, Mapping) else None
        except Exception:
            return None
    values = getattr(value, "__dict__", None)
    return values if isinstance(values, Mapping) else None


def _component_entries(update: Any) -> list[tuple[str, Mapping[str, Any] | Any]]:
    components = _field(update, "components", "usage", "componentUsage", "component_usage")
    if isinstance(components, Mapping):
        entries: list[tuple[str, Mapping[str, Any] | Any]] = []
        for name in ("stt", "llm", "tts"):
            value = components.get(name)
            if value is None:
                for alias in _COMPONENT_KEYS[name]:
                    if alias in components:
                        value = components[alias]
                        break
            if value is not None:
                entries.append((name, value))
        if entries:
            return entries

    # LiveKit's verified session-level hook is
    # ``ev.usage.model_usage``. Each row is a cumulative summary for one
    # provider/model pair; classify it from its concrete usage type/fields.
    session_usage = _field(update, "usage")
    model_usage = _field(session_usage, "model_usage", "modelUsage")
    if model_usage is None:
        model_usage = _field(update, "model_usage", "modelUsage")
    if isinstance(model_usage, (list, tuple)):
        entries = []
        for item in model_usage:
            mapping = _mapping(item) or {}
            class_name = type(item).__name__.lower()
            usage_type = str(mapping.get("type", "")).lower()
            if "tts" in class_name or "text_to_speech" in class_name or "tts" in usage_type:
                component = "tts"
            elif "stt" in class_name or "speech_to_text" in class_name or "stt" in usage_type:
                component = "stt"
            elif "llm" in class_name or "language_model" in class_name or "llm" in usage_type:
                component = "llm"
            elif "interruption" in class_name or "eot" in class_name or "interruption" in usage_type or "eot" in usage_type:
                component = "interruption"
            elif (
                "characters_count" in mapping
                or "audio_duration" in mapping
                and "output_audio_tokens" in mapping
            ):
                component = "tts"
            elif "input_audio_tokens" in mapping or "output_text_tokens" in mapping:
                component = "stt"
            else:
                component = "llm"
            entries.append((component, item))
        if entries:
            return entries

    entries = []
    for name, aliases in _COMPONENT_KEYS.items():
        value = _field(update, *aliases)
        if value is None:
            continue
        entries.append((name, value))
    return entries


def _records(value: Any) -> list[Mapping[str, Any] | Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _record_values(value: Any) -> tuple[str, str | None, str | None, str | None, Mapping[str, Any] | Any, Any]:
    mapping = _mapping(value) or {}
    provider = _field(value, "provider", "providerName", "provider_name") or "livekit"
    model = _field(value, "model", "modelName", "model_name")
    unit = _field(value, "unit", "unitKind", "unit_kind")
    amount = _field(value, "amount", "cost", "costUsd", "cost_usd")
    currency = _field(value, "currency")
    usage = _field(value, "usage", "units", "nativeUnits", "native_units")
    if usage is None and mapping:
        reserved = {
            "type",
            "id", "usage_id", "usageId", "component_id", "componentId",
            "request_id", "requestId", "provider_request_id", "providerRequestId",
            "provider", "providerName", "provider_name", "model", "modelName", "model_name",
            "unit", "unitKind", "unit_kind", "amount", "cost", "costUsd", "cost_usd", "currency",
        }
        usage = {key: item for key, item in mapping.items() if key not in reserved}
    if usage is None:
        # Some SDK versions expose slots/properties rather than ``__dict__``.
        native_names = (
            "input_tokens", "output_tokens", "total_tokens", "input_cached_tokens",
            "input_audio_tokens", "input_text_tokens", "input_image_tokens",
            "output_audio_tokens", "output_text_tokens", "characters_count",
            "audio_duration", "session_duration", "total_requests",
        )
        usage = {
            name: _field(value, name)
            for name in native_names
            if _field(value, name) is not None
        }
    if unit is None:
        if mapping.get("audio_duration") is not None:
            unit = "seconds"
        elif mapping.get("characters_count") is not None:
            unit = "characters"
    return str(provider), str(model) if model is not None else None, str(unit) if unit is not None else None, str(currency) if currency is not None else None, usage, amount


def _timestamp(update: Any) -> str | None:
    """Normalize LiveKit's Python seconds timestamp (or JS milliseconds)."""
    value = _field(update, "created_at", "createdAt")
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    # Python LiveKit uses epoch seconds; the TypeScript adapter uses epoch ms.
    seconds = value / 1_000 if value > 100_000_000_000 else value
    try:
        return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).isoformat(
            timespec="milliseconds",
        ).replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _row_key(value: Any) -> tuple[str, str | None]:
    """Return a deterministic identity hint for duplicate model rows.

    LiveKit normally emits one row per provider/model, so that pair is the
    stable component identity. If an integration supplies multiple rows for
    the same pair, a stable row ID is required for identity across revisions;
    otherwise a canonical snapshot hash keeps a reordered snapshot stable.
    """
    stable = _field(
        value,
        "id", "usage_id", "usageId", "component_id", "componentId",
        "request_id", "requestId", "provider_request_id", "providerRequestId",
    )
    if isinstance(stable, str) and stable.strip():
        return stable.strip(), None
    mapping = _mapping(value)
    try:
        encoded = json.dumps(mapping or {}, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        encoded = repr(mapping)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16], encoded


def _id_suffix(value: str) -> str:
    """Keep row IDs URL-safe when they become part of an explicit component ID."""
    cleaned = re.sub(r"[^A-Za-z0-9_.~-]+", "-", value).strip("-")
    return cleaned[:80] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _declared_value(values: Mapping[str, Any] | None, *, component_id: str, component: str,
                    provider: str, model: str | None) -> Any:
    if not isinstance(values, Mapping):
        return None
    for key in (
        component_id,
        f"{component}:{provider}:{model or 'unknown'}",
        component,
    ):
        if key in values:
            return values[key]
    return None


_IDENTITY_FIELDS = {
    "type", "provider", "providerName", "provider_name", "model", "modelName", "model_name",
    "amount", "cost", "costUsd", "cost_usd", "currency", "id", "usage_id", "usageId",
    "component_id", "componentId", "request_id", "requestId", "provider_request_id", "providerRequestId",
}


def _billing_units(value: Any) -> dict[str, int | float] | None:
    """Extract chargeable LiveKit fields in the backend's billing vocabulary.

    The usage payload keeps native provider names for observability. This
    separate map is authoritative for rate lookup and excludes derived
    ``total_tokens`` when input/output parts are present. Duration fields from
    Python LiveKit are already seconds; millisecond aliases are normalized.
    """
    mapping = _mapping(value) or {}
    result: dict[str, int | float] = {}
    token_parts = False
    has_total = False
    for raw_key, raw_value in mapping.items():
        if raw_key in _IDENTITY_FIELDS:
            continue
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool) \
                or not math.isfinite(raw_value) or raw_value < 0:
            continue
        key = str(raw_key)
        lowered = key.lower()
        if lowered in {"input_tokens", "inputtokens", "output_tokens", "outputtokens"}:
            token_parts = True
        if lowered in {"total_tokens", "totaltokens"}:
            has_total = True
        if lowered in {"audio_duration", "audioduration"}:
            result["audio_seconds"] = raw_value
        elif lowered == "audiodurationms":
            result["audio_seconds"] = raw_value / 1_000
        elif lowered in {"session_duration", "sessionduration"}:
            result["session_seconds"] = raw_value
        elif lowered == "sessiondurationms":
            result["session_seconds"] = raw_value / 1_000
        elif lowered in {"characters_count", "characterscount"}:
            result["characters"] = raw_value
        else:
            # The server normalizes camelCase to snake_case for price lookup;
            # retain snake_case fields as-is for Python-native rows.
            result[key] = raw_value
    if token_parts and has_total:
        result.pop("total_tokens", None)
        result.pop("totalTokens", None)
    return result or None


def _billing_family(unit: str) -> str:
    lowered = unit.lower()
    if "token" in lowered:
        return "tokens"
    if any(term in lowered for term in ("second", "minute", "hour", "duration")):
        return "time"
    return "other"


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
    """Record a verified LiveKit session usage snapshot.

    ``revision`` is the LiveKit snapshot revision (or another monotonic value
    supplied by the integration). Every component gets its own stable charge,
    so STT/LLM/TTS updates remain separately attributable under one trace or
    conversation. Missing components are simply not reported and therefore
    remain visibly partial in downstream accounting.

    If LiveKit also reports an aggregate LLM total while CostKey captures the
    underlying LLM HTTP calls, pass ``exclude_components={"llm"}`` and retain
    those HTTP events as the LLM source of charge. Session and per-request
    identities have different granularities and must not be guessed equal.

    ``declared_amounts`` and ``declared_pricing_ids`` are optional explicit
    application inputs for providers whose LiveKit usage rows contain no
    amount. Keys may be a generated component ID, ``component:provider:model``
    or a component name. Declared amounts are marked ``estimated``; the adapter
    never invents provider spend from duration or token counts. When one row
    mixes native billing families, ``declared_billing_units`` can select the
    exact billable map; otherwise the adapter marks that component's billing
    selection as required and leaves its amount unknown.
    """
    session = session_id or _field(update, "sessionId", "session_id", "roomName", "room_name")
    if not isinstance(session, str) or not session:
        raise ValueError("record_livekit_usage requires session_id")
    current_revision = revision if revision is not None else _field(update, "revision", "sequence", "sequenceNumber")
    if not isinstance(current_revision, int) or isinstance(current_revision, bool) or current_revision < 0:
        raise ValueError("record_livekit_usage requires a non-negative revision")

    prefix = event_prefix or f"livekit:{session}"
    recorded: list[str] = []

    def component_names(values: Any) -> set[str]:
        if isinstance(values, str):
            values = (values,)
        return {
            str(component).strip().lower()
            for component in (values or ())
            if str(component).strip()
        }

    excluded = component_names(exclude_components)
    expected = component_names(expected_components) - excluded
    if declared_amounts is not None and not isinstance(declared_amounts, Mapping):
        raise TypeError("declared_amounts must be a mapping")
    if declared_pricing_ids is not None and not isinstance(declared_pricing_ids, Mapping):
        raise TypeError("declared_pricing_ids must be a mapping")
    if declared_billing_units is not None and not isinstance(declared_billing_units, Mapping):
        raise TypeError("declared_billing_units must be a mapping")

    prepared: list[dict[str, Any]] = []
    for component, component_value in _component_entries(update):
        if component.lower() in excluded:
            continue
        for value in _records(component_value):
            provider, model, unit, currency, usage, amount = _record_values(value)
            row_hint, row_snapshot = _row_key(value)
            base_id = f"{prefix}:{component}:{provider}:{model or 'unknown'}"
            prepared.append({
                "component": component,
                "value": value,
                "provider": provider,
                "model": model,
                "unit": unit,
                "currency": currency,
                "usage": usage,
                "amount": amount,
                "base_id": base_id,
                "row_hint": row_hint,
                "row_snapshot": row_snapshot,
            })

    # Assign IDs after seeing the complete snapshot. A single provider/model
    # keeps the compact stable ID; duplicate rows use a stable row ID or a
    # canonical hash instead of an arrival-order index.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        grouped.setdefault(item["base_id"], []).append(item)
    for base_id, items in grouped.items():
        if len(items) == 1:
            item = items[0]
            item["component_id"] = (
                f"{base_id}:{_id_suffix(item['row_hint'])}"
                if item["row_snapshot"] is None
                else base_id
            )
            continue
        ordered = sorted(items, key=lambda item: (item["row_hint"], item["row_snapshot"] or ""))
        suffix_counts: dict[str, int] = {}
        for item in ordered:
            suffix = _id_suffix(item["row_hint"])
            # A canonical hash is used for rows without a caller-owned stable
            # ID. Ties remain deterministic for identical rows.
            if item["row_snapshot"] is not None:
                suffix = f"snapshot-{suffix}"
            duplicate_rank = suffix_counts.get(suffix, 0)
            suffix_counts[suffix] = duplicate_rank + 1
            item["component_id"] = f"{base_id}:{suffix}"
            if duplicate_rank:
                item["component_id"] += f":duplicate-{duplicate_rank}"

    observed_components = {item["component"] for item in prepared}
    coverage = {
        "expectedComponents": sorted(expected),
        "observedComponents": sorted(observed_components),
        "missingComponents": sorted(expected - observed_components),
        "excludedComponents": sorted(excluded),
    } if expected else None

    for item in prepared:
        component = item["component"]
        value = item["value"]
        provider = item["provider"]
        model = item["model"]
        unit = item["unit"]
        currency = item["currency"]
        usage = item["usage"]
        amount = item["amount"]
        component_id = item["component_id"]
        declared_amount = _declared_value(
            declared_amounts,
            component_id=component_id,
            component=component,
            provider=provider,
            model=model,
        )
        amount_was_declared = amount is None and declared_amount is not None
        if amount is None:
            amount = declared_amount
        explicit_billing_units = _declared_value(
            declared_billing_units,
            component_id=component_id,
            component=component,
            provider=provider,
            model=model,
        )
        if explicit_billing_units is not None and not isinstance(explicit_billing_units, Mapping):
            raise TypeError("declared_billing_units values must be mappings")
        billing_units = (
            dict(explicit_billing_units)
            if isinstance(explicit_billing_units, Mapping)
            else _billing_units(value)
        )
        billing_units_required = False
        if amount is None and billing_units:
            billing_families = {_billing_family(unit_name) for unit_name in billing_units}
            billing_units_required = len(billing_families) > 1 and explicit_billing_units is None
            if billing_units_required:
                # Multiple native families (for example LLM tokens plus
                # session seconds) are not a billable selection. Keep usage
                # visible but require an explicit map or amount rather than
                # silently choosing one family.
                billing_units = None
        declared_pricing_id = _declared_value(
            declared_pricing_ids,
            component_id=component_id,
            component=component,
            provider=provider,
            model=model,
        )
        event_id = f"{component_id}:r{current_revision}"
        context: dict[str, Any] = {"voiceSessionId": session}
        if conversation_id is not None:
            context["conversationId"] = conversation_id
        if trace_id is not None:
            context["traceId"] = trace_id
        if execution_id is not None:
            context["executionId"] = execution_id
        if group_name is not None:
            context["groupName"] = group_name
        if group_kind is not None:
            context["groupKind"] = group_kind
        if phase is not None:
            context["phase"] = phase
        if dataset_revision is not None:
            context["datasetRevision"] = dataset_revision
        if case_id is not None:
            context["caseId"] = case_id
        if coverage is not None:
            context["voiceCoverage"] = coverage
        if billing_units_required:
            context["voiceBilling"] = {
                "status": "selection_required",
                "componentId": component_id,
                "reason": "mixed_native_unit_families",
            }
        row_type = _field(value, "type")
        metadata = {
            "livekit": {
                "type": row_type if isinstance(row_type, str) else None,
                "sessionId": session,
                "revision": current_revision,
                **({"voiceCoverage": coverage} if coverage is not None else {}),
            },
        }
        recorded.append(record_usage(
            event_id=event_id,
            component_id=component_id,
            charge_id=component_id,
            provider=provider,
            model=model,
            unit=unit,
            usage=usage,
            amount=amount,
            amount_kind="estimated" if amount_was_declared else None,
            pricing_id=declared_pricing_id if isinstance(declared_pricing_id, str) else None,
            currency=currency,
            timestamp=_timestamp(update),
            aggregation="cumulative",
            revision=current_revision,
            context=context,
            metadata=metadata,
            billing_units=billing_units,
        ))
    return recorded


record_livekit_session_usage = record_livekit_usage


__all__ = ["record_livekit_usage", "record_livekit_session_usage"]
