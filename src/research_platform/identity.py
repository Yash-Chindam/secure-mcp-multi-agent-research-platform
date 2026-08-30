"""The identity every authorization and discovery decision is made against."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from research_platform.domain.models import AccessClass
from research_platform.domain.tasks import AgentRole


class Role(StrEnum):
    """Human roles from section 3 of the design specification."""

    REQUESTER = "requester"
    REVIEWER = "reviewer"
    ADMINISTRATOR = "administrator"


@dataclass(frozen=True)
class Principal:
    """A tenant-scoped caller, which may be a person or an agent acting for one."""

    tenant_id: str
    subject_id: str
    roles: frozenset[Role] = field(default_factory=frozenset)
    agent_role: AgentRole | None = None
    clearance: AccessClass = AccessClass.PUBLIC

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("a principal must belong to a tenant")
        if not self.subject_id:
            raise ValueError("a principal must have a subject identifier")

    @property
    def is_agent(self) -> bool:
        return self.agent_role is not None

    def has_any_role(self, roles: frozenset[Role]) -> bool:
        return bool(self.roles & roles)

    def for_agent(self, agent_role: AgentRole) -> Principal:
        """Derive the agent identity that acts on this principal's behalf."""
        return Principal(
            tenant_id=self.tenant_id,
            subject_id=self.subject_id,
            roles=self.roles,
            agent_role=agent_role,
            clearance=self.clearance,
        )
