"""The default capability catalogue for the five MCP servers in section 8."""

from __future__ import annotations

from research_platform.domain.models import AccessClass
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Role
from research_platform.mcp.registry import Capability, CapabilityKind, CapabilityRegistry

RESEARCHERS = frozenset({AgentRole.RESEARCHER})
ANALYSTS = frozenset({AgentRole.ANALYST})
EVIDENCE_READERS = frozenset({AgentRole.CRITIC, AgentRole.REPORTER})
OPERATORS = frozenset({Role.REQUESTER, Role.REVIEWER, Role.ADMINISTRATOR})
ADMINISTRATORS = frozenset({Role.ADMINISTRATOR})


WEB_RESEARCH: tuple[Capability, ...] = (
    Capability(
        server="web-research",
        name="search",
        description="Search approved public sources for candidate documents.",
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS,
        max_result_bytes=131_072,
        timeout_seconds=20,
    ),
    Capability(
        server="web-research",
        name="fetch",
        description="Fetch and extract readable text from an approved public URL.",
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS,
        max_result_bytes=524_288,
        timeout_seconds=30,
    ),
)

FILESYSTEM: tuple[Capability, ...] = (
    Capability(
        server="filesystem",
        name="list_workspace",
        description="List readable files inside the tenant workspace root.",
        kind=CapabilityKind.RESOURCE,
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS,
        max_access_class=AccessClass.INTERNAL,
        max_result_bytes=65_536,
    ),
    Capability(
        server="filesystem",
        name="read_document",
        description="Read one approved document from the tenant workspace root.",
        kind=CapabilityKind.RESOURCE,
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS | EVIDENCE_READERS,
        max_access_class=AccessClass.INTERNAL,
        max_result_bytes=524_288,
    ),
)

POSTGRES: tuple[Capability, ...] = (
    Capability(
        server="postgres",
        name="describe_schema",
        description="Inspect tables and columns exposed to the tenant read-only role.",
        kind=CapabilityKind.RESOURCE,
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS | ANALYSTS,
        max_access_class=AccessClass.INTERNAL,
        max_result_bytes=65_536,
    ),
    Capability(
        server="postgres",
        name="run_analytical_query",
        description="Run one parsed read-only query against the tenant analytical schema.",
        required_roles=OPERATORS,
        allowed_agents=ANALYSTS,
        max_access_class=AccessClass.RESTRICTED,
        requires_approval=True,
        max_result_bytes=1_048_576,
        timeout_seconds=60,
    ),
)

GITHUB: tuple[Capability, ...] = (
    Capability(
        server="github",
        name="read_repository",
        description="Read files and metadata from an allowlisted repository.",
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS,
        max_access_class=AccessClass.INTERNAL,
        max_result_bytes=524_288,
    ),
    Capability(
        server="github",
        name="read_pull_requests",
        description="Read pull request and issue metadata from an allowlisted repository.",
        required_roles=OPERATORS,
        allowed_agents=RESEARCHERS,
        max_access_class=AccessClass.INTERNAL,
        max_result_bytes=262_144,
    ),
)

PYTHON_ANALYSIS: tuple[Capability, ...] = (
    Capability(
        server="python-analysis",
        name="run_calculation",
        description="Run a reproducible calculation in a network-isolated sandbox.",
        required_roles=OPERATORS,
        allowed_agents=ANALYSTS,
        max_result_bytes=262_144,
        timeout_seconds=60,
    ),
)

EVIDENCE: tuple[Capability, ...] = (
    Capability(
        server="evidence",
        name="retrieve",
        description="Retrieve recorded evidence for a job by identifier.",
        kind=CapabilityKind.RESOURCE,
        required_roles=OPERATORS,
        allowed_agents=EVIDENCE_READERS | ANALYSTS,
        max_access_class=AccessClass.INTERNAL,
        max_result_bytes=524_288,
    ),
)

ADMINISTRATION: tuple[Capability, ...] = (
    Capability(
        server="registry",
        name="reload_policy",
        description="Reload the authorization policy bundle across MCP services.",
        required_roles=ADMINISTRATORS,
        side_effecting=True,
        requires_approval=True,
        max_access_class=AccessClass.RESTRICTED,
    ),
)

DEFAULT_CAPABILITIES: tuple[Capability, ...] = (
    *WEB_RESEARCH,
    *FILESYSTEM,
    *POSTGRES,
    *GITHUB,
    *PYTHON_ANALYSIS,
    *EVIDENCE,
    *ADMINISTRATION,
)


def default_registry() -> CapabilityRegistry:
    """Build the registry the platform starts with."""
    return CapabilityRegistry(list(DEFAULT_CAPABILITIES))
