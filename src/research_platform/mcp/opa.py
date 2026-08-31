"""Ask Open Policy Agent for the authorization decision.

The engine fails closed: a timeout, a transport error, an HTTP error status or a response
the platform cannot read all refuse the call. An authorization service that is unreachable
must never be equivalent to one that said yes.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from research_platform.identity import Principal
from research_platform.mcp.policy import (
    AuthorizationRequest,
    PolicyDecision,
    PolicyEngine,
)
from research_platform.mcp.registry import Capability

DEFAULT_DECISION_PATH = "/v1/data/research/authz/decision"
UNAVAILABLE_VERSION = "opa-unavailable"


def _principal_document(principal: Principal) -> dict[str, Any]:
    return {
        "tenant_id": principal.tenant_id,
        "subject_id": principal.subject_id,
        "roles": sorted(role.value for role in principal.roles),
        "agent_role": principal.agent_role.value if principal.agent_role else None,
        "clearance": principal.clearance.value,
    }


def _capability_document(capability: Capability) -> dict[str, Any]:
    return {
        "server": capability.server,
        "name": capability.name,
        "kind": capability.kind.value,
        "required_roles": sorted(role.value for role in capability.required_roles),
        "allowed_agents": sorted(agent.value for agent in capability.allowed_agents),
        "max_access_class": capability.max_access_class.value,
        "side_effecting": capability.side_effecting,
        "requires_approval": capability.requires_approval,
        "tenant_scope": sorted(capability.tenant_scope),
    }


def build_policy_input(request: AuthorizationRequest) -> dict[str, Any]:
    """Describe the call for the policy.

    Argument names are sent but their values are not. The policy decides on the
    capability and the identity, and forwarding argument values would copy potentially
    sensitive data into a second service and its decision logs.
    """
    return {
        "input": {
            "principal": _principal_document(request.principal),
            "capability": _capability_document(request.capability),
            "job_id": str(request.job_id),
            "task_id": str(request.task_id),
            "argument_names": sorted(request.arguments),
        }
    }


class OpaPolicyEngine:
    """A policy engine backed by an Open Policy Agent deployment."""

    def __init__(
        self,
        base_url: str,
        *,
        decision_path: str = DEFAULT_DECISION_PATH,
        timeout_seconds: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("an OPA base URL is required")
        self._decision_path = decision_path
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self._policy_version = UNAVAILABLE_VERSION

    def __enter__(self) -> OpaPolicyEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def policy_version(self) -> str:
        """The version of the last decision, or a marker until one has been made."""
        return self._policy_version

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision:
        try:
            response = self._client.post(
                self._decision_path,
                json=build_policy_input(request),
            )
            response.raise_for_status()
            document = response.json()
        except httpx.TimeoutException:
            return self._unavailable("the authorization service did not respond in time")
        except httpx.HTTPStatusError as error:
            return self._unavailable(
                f"the authorization service returned {error.response.status_code}"
            )
        except (httpx.HTTPError, ValueError):
            return self._unavailable("the authorization service could not be reached")

        return self._read_decision(document)

    def _read_decision(self, document: object) -> PolicyDecision:
        if not isinstance(document, dict):
            return self._unavailable("the authorization service returned an unreadable decision")
        result = document.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("allow"), bool):
            return self._unavailable("the authorization service returned no decision")

        version = result.get("policy_version")
        policy_version = version if isinstance(version, str) and version else UNAVAILABLE_VERSION
        self._policy_version = policy_version

        reason = result.get("reason")
        described = reason if isinstance(reason, str) and reason else "no reason was given"
        if result["allow"]:
            return PolicyDecision.allow(policy_version)
        return PolicyDecision.deny(described, policy_version)

    def _unavailable(self, reason: str) -> PolicyDecision:
        """Refuse the call, keeping the reason distinguishable from a policy refusal."""
        self._policy_version = UNAVAILABLE_VERSION
        return PolicyDecision.deny(reason, UNAVAILABLE_VERSION)


class AllOfPolicyEngine:
    """Require every configured engine to permit the call.

    The registry engine is the platform's own invariant and the Open Policy Agent bundle
    is the organization's policy. Both must agree, so a misconfigured bundle cannot widen
    access beyond the registered capability boundary, and the boundary cannot override a
    policy that has narrowed it.
    """

    def __init__(self, engines: list[PolicyEngine]) -> None:
        if not engines:
            raise ValueError("at least one policy engine is required")
        self._engines = engines

    @property
    def policy_version(self) -> str:
        return "+".join(engine.policy_version for engine in self._engines)

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision:
        refusals: list[str] = []
        for engine in self._engines:
            decision = engine.evaluate(request)
            if not decision.allowed:
                refusals.append(decision.reason)
        if refusals:
            return PolicyDecision.deny("; ".join(refusals), self.policy_version)
        return PolicyDecision.allow(self.policy_version)
