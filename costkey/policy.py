"""Optional, explicit pre-call policy admission for CostKey.

Observation never depends on this module.  Applications opt into ``enforce``
mode only when they provide a real admission integration (a callable or the
CostKey admission endpoint).  An unavailable required decision fails closed
with a typed error; monitor mode remains fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx


class CostKeyPolicyRefused(RuntimeError):
    """A supported call was refused before provider execution."""

    code = "COSTKEY_POLICY_REFUSED"

    def __init__(
        self,
        *,
        reason: str,
        policy_id: str | None = None,
        scope: Any = None,
        allowance: Any = None,
        window: Any = None,
        retry_after: Any = None,
        reset_at: Any = None,
        policy_version: Any = None,
        replay: bool | None = None,
        policies: Any = None,
        message: str | None = None,
    ) -> None:
        self.reason = reason
        self.policy_id = policy_id
        self.scope = scope
        self.allowance = allowance
        self.window = window
        self.retry_after = retry_after
        self.reset_at = reset_at
        self.policy_version = policy_version
        self.replay = replay
        self.policies = policies
        detail = f" [{policy_id}]" if policy_id else ""
        super().__init__(message or f"CostKey policy refused this call{detail}: {reason}")

    @property
    def retry_at(self) -> Any:
        return self.reset_at if self.reset_at is not None else self.retry_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "reason": self.reason,
            "message": str(self),
            **({"policyId": self.policy_id} if self.policy_id else {}),
            **({"policyVersion": self.policy_version} if self.policy_version is not None else {}),
            **({"scope": self.scope} if self.scope is not None else {}),
            **({"allowance": self.allowance} if self.allowance is not None else {}),
            **({"window": self.window} if self.window is not None else {}),
            **({"retryAt": self.reset_at if self.reset_at is not None else self.retry_after}
               if self.reset_at is not None or self.retry_after is not None else {}),
            **({"replay": self.replay} if self.replay is not None else {}),
            **({"policies": self.policies} if self.policies is not None else {}),
        }


@dataclass(frozen=True)
class PolicyOptions:
    """Explicit policy client configuration accepted by ``costkey.init``."""

    mode: str = "monitor"
    admission: Any = None
    endpoint: str | None = None
    timeout: float = 2.0
    integration: Mapping[str, Any] | None = None
    scope: Mapping[str, Any] | None = None
    usage_bounds: Mapping[str, Any] | None = None
    claimed_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"monitor", "enforce"}:
            raise ValueError("policy mode must be 'monitor' or 'enforce'")
        if self.timeout <= 0:
            raise ValueError("policy timeout must be positive")


@dataclass(frozen=True)
class PolicyAdmissionOptions:
    """Configuration for the optional CostKey admission endpoint."""

    endpoint: str
    auth_key: str
    mode: str = "monitor"
    timeout: float = 2.0
    # These fields are supplied only when the application has verified the
    # provider/transport capability and can honestly bound the request.
    integration: Mapping[str, Any] | None = None
    scope: Mapping[str, Any] | None = None
    usage_bounds: Mapping[str, Any] | None = None
    claimed_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"monitor", "enforce"}:
            raise ValueError("policy mode must be 'monitor' or 'enforce'")
        if not self.endpoint:
            raise ValueError("policy admission endpoint is required")
        if self.timeout <= 0:
            raise ValueError("policy admission timeout must be positive")

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request.setdefault("admissionId", payload.get("id"))
        request.setdefault("requestId", payload.get("requestId") or payload.get("id"))
        if self.integration is not None:
            request["integration"] = dict(self.integration)
        if self.scope is not None:
            request["scope"] = dict(self.scope)
        if self.usage_bounds is not None:
            request["usageBounds"] = dict(self.usage_bounds)
        if self.claimed_cost_usd is not None:
            request["claimedCostUsd"] = self.claimed_cost_usd
        from costkey.patch import internal_http
        with internal_http():
            response = httpx.post(
                self.endpoint,
                    json=request,
                headers={
                    "Authorization": f"Bearer {self.auth_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        try:
            body = response.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            result = dict(body)
        else:
            result = {}

        # A refusal can be returned as a normal decision body or as a 4xx
        # response.  Preserve server-provided reason and reset metadata.
        # Status alone is not a policy decision: a proxy/auth/route error
        # must fail closed as ``decision_unavailable`` rather than inventing a
        # budget refusal. Only an explicit typed body can refuse.
        if not (200 <= response.status_code < 300):
            has_typed_decision = isinstance(result.get("allowed"), bool) or isinstance(result.get("decision"), str)
            if not has_typed_decision:
                result["allowed"] = None
                result["reason"] = "decision_unavailable"
        return result

    def settle(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Release/reconcile a reservation created by ``__call__``."""
        endpoint = self.endpoint.rstrip("/")
        if endpoint.endswith("/admit"):
            endpoint = endpoint[:-len("/admit")] + "/settle"
        from costkey.patch import internal_http
        with internal_http():
            response = httpx.post(
                endpoint,
                json=dict(payload),
                headers={
                    "Authorization": f"Bearer {self.auth_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        try:
            body = response.json()
        except Exception:
            body = {}
        if not (200 <= response.status_code < 300):
            raise RuntimeError(f"policy settlement returned HTTP {response.status_code}")
        return dict(body) if isinstance(body, dict) else {}


__all__ = ["CostKeyPolicyRefused", "PolicyAdmissionOptions", "PolicyOptions"]
