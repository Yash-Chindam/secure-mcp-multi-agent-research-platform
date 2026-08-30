"""Authorization decisions taken immediately before a capability executes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from research_platform.domain.models import clearance_covers
from research_platform.identity import Principal
from research_platform.mcp.registry import Capability

LOCAL_POLICY_VERSION = "local-2026-08-01"


@dataclass(frozen=True)
class AuthorizationRequest:
    """Everything a policy engine may consider, with no ambient state."""

    principal: Principal
    capability: Capability
    job_id: UUID
    task_id: UUID
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    policy_version: str

    @classmethod
    def allow(cls, policy_version: str = LOCAL_POLICY_VERSION) -> PolicyDecision:
        return cls(allowed=True, reason="permitted by policy", policy_version=policy_version)

    @classmethod
    def deny(cls, reason: str, policy_version: str = LOCAL_POLICY_VERSION) -> PolicyDecision:
        return cls(allowed=False, reason=reason, policy_version=policy_version)


class PolicyEngine(Protocol):
    """The seam an external Open Policy Agent deployment will implement."""

    @property
    def policy_version(self) -> str: ...

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision: ...


class RegistryPolicyEngine:
    """Decide from the registered capability boundary alone.

    This re-checks tenant, clearance and role at execution time rather than trusting the
    discovery result the agent was given, because the plan and the call are separated by
    agent reasoning that must not be treated as an authorization decision.
    """

    def __init__(self, policy_version: str = LOCAL_POLICY_VERSION) -> None:
        self._policy_version = policy_version

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision:
        capability = request.capability
        principal = request.principal

        if not capability.is_available_to_tenant(principal.tenant_id):
            return self._deny(f"capability is not offered to tenant {principal.tenant_id}")
        if not clearance_covers(principal.clearance, capability.max_access_class):
            return self._deny(
                f"clearance {principal.clearance} cannot reach {capability.max_access_class} data"
            )
        if not capability.is_executable_by(principal):
            return self._deny(f"{self._describe(principal)} may not execute {capability.name}")
        return PolicyDecision.allow(self._policy_version)

    def _deny(self, reason: str) -> PolicyDecision:
        return PolicyDecision.deny(reason, self._policy_version)

    @staticmethod
    def _describe(principal: Principal) -> str:
        if principal.agent_role is not None:
            return f"agent role {principal.agent_role}"
        return f"roles {sorted(role.value for role in principal.roles)}"
