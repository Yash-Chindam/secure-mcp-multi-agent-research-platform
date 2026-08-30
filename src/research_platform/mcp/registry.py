"""Capability registry and identity-scoped MCP discovery."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from research_platform.domain.models import AccessClass, clearance_covers
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role

ALL_TENANTS = "*"

METADATA_ONLY_AGENTS = frozenset({AgentRole.PLANNER})
"""Agents that may read capability metadata but never execute a capability (section 7)."""

Name = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class CapabilityKind(StrEnum):
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"


class CapabilityNotFound(LookupError):
    def __init__(self, server: str, name: str) -> None:
        super().__init__(f"capability {server}.{name} is not registered")
        self.server = server
        self.name = name


class DuplicateCapability(ValueError):
    pass


class Capability(BaseModel):
    """One capability an MCP server exposes, with the boundary it must be used within."""

    model_config = {"frozen": True}

    server: Name
    name: Name
    kind: CapabilityKind = CapabilityKind.TOOL
    description: str = Field(min_length=1, max_length=500)
    required_roles: frozenset[Role] = Field(default_factory=lambda: frozenset({Role.REQUESTER}))
    allowed_agents: frozenset[AgentRole] = Field(default_factory=frozenset)
    max_access_class: AccessClass = AccessClass.PUBLIC
    side_effecting: bool = False
    requires_approval: bool = False
    tenant_scope: frozenset[str] = Field(default_factory=lambda: frozenset({ALL_TENANTS}))
    max_result_bytes: Annotated[int, Field(ge=1_024, le=8_388_608)] = 262_144
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0

    @model_validator(mode="after")
    def side_effecting_capabilities_need_approval(self) -> Capability:
        if self.side_effecting and not self.requires_approval:
            raise ValueError(
                f"side-effecting capability {self.qualified_name} must require approval"
            )
        if not self.tenant_scope:
            raise ValueError("a capability must be scoped to at least one tenant")
        return self

    @property
    def qualified_name(self) -> str:
        return f"{self.server}.{self.name}"

    def is_available_to_tenant(self, tenant_id: str) -> bool:
        return ALL_TENANTS in self.tenant_scope or tenant_id in self.tenant_scope

    def is_visible_to(self, principal: Principal) -> bool:
        """Report whether discovery may reveal this capability to the caller.

        Visibility is deliberately narrower than execution: a capability the caller could
        never be authorized to run is never named to them in the first place.
        """
        if not self.is_available_to_tenant(principal.tenant_id):
            return False
        if not clearance_covers(principal.clearance, self.max_access_class):
            return False
        if principal.agent_role is not None:
            return (
                principal.agent_role in self.allowed_agents
                or principal.agent_role in METADATA_ONLY_AGENTS
            )
        return principal.has_any_role(self.required_roles)

    def is_executable_by(self, principal: Principal) -> bool:
        """Report whether the caller may actually run this capability.

        The planner reads capability metadata to build a plan but never executes, so
        execution is strictly narrower than visibility.
        """
        if not self.is_visible_to(principal):
            return False
        if principal.agent_role is not None:
            return principal.agent_role in self.allowed_agents
        return principal.has_any_role(self.required_roles)


class CapabilityRegistry:
    """The administrator-managed catalogue of registered MCP capabilities."""

    def __init__(self, capabilities: list[Capability] | None = None) -> None:
        self._capabilities: dict[tuple[str, str], Capability] = {}
        for capability in capabilities or []:
            self.register(capability)

    def register(self, capability: Capability) -> Capability:
        key = (capability.server, capability.name)
        if key in self._capabilities:
            raise DuplicateCapability(f"{capability.qualified_name} is already registered")
        self._capabilities[key] = capability
        return capability

    def get(self, server: str, name: str) -> Capability:
        try:
            return self._capabilities[(server, name)]
        except KeyError as error:
            raise CapabilityNotFound(server, name) from error

    def resolve_for(self, principal: Principal, server: str, name: str) -> Capability:
        """Return a capability only when discovery would have revealed it to the caller.

        An invisible capability is reported as missing rather than forbidden, so probing
        cannot be used to enumerate another tenant's tools.
        """
        capability = self.get(server, name)
        if not capability.is_visible_to(principal):
            raise CapabilityNotFound(server, name)
        return capability

    def discover(self, principal: Principal) -> list[Capability]:
        visible = [
            capability
            for capability in self._capabilities.values()
            if capability.is_visible_to(principal)
        ]
        return sorted(visible, key=lambda capability: capability.qualified_name)

    def servers(self) -> list[str]:
        return sorted({capability.server for capability in self._capabilities.values()})

    def __len__(self) -> int:
        return len(self._capabilities)
