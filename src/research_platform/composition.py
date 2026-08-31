"""Assemble the governed platform from configuration.

Wiring lives here rather than in the API module so a worker can build the same governed
gateway the HTTP boundary uses, and so the policy layering is decided in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityExecutor, CapabilityGateway
from research_platform.mcp.opa import AllOfPolicyEngine, OpaPolicyEngine
from research_platform.mcp.policy import PolicyEngine, RegistryPolicyEngine
from research_platform.mcp.registry import CapabilityRegistry
from research_platform.settings import Settings


@dataclass(frozen=True)
class PolicyStack:
    """The engine the gateway will use, and how it was assembled."""

    engine: PolicyEngine
    externally_enforced: bool

    @property
    def description(self) -> str:
        if self.externally_enforced:
            return "registry boundary and Open Policy Agent bundle"
        return "registry boundary only (no authorization service configured)"


def build_policy_stack(settings: Settings) -> PolicyStack:
    """Layer the registry boundary with the policy bundle when one is configured.

    Both must permit a call, so a misconfigured bundle cannot widen access beyond the
    registered capability boundary and the boundary cannot override a narrowed policy.
    """
    registry_engine = RegistryPolicyEngine()
    if not settings.opa_url:
        return PolicyStack(engine=registry_engine, externally_enforced=False)

    opa = OpaPolicyEngine(settings.opa_url, timeout_seconds=settings.opa_timeout_seconds)
    return PolicyStack(
        engine=AllOfPolicyEngine([registry_engine, opa]),
        externally_enforced=True,
    )


def build_gateway(
    *,
    executor: CapabilityExecutor,
    settings: Settings,
    registry: CapabilityRegistry | None = None,
) -> CapabilityGateway:
    """Build the single governed path to the MCP capabilities."""
    return CapabilityGateway(
        registry=registry or default_registry(),
        executor=executor,
        policy=build_policy_stack(settings).engine,
    )
