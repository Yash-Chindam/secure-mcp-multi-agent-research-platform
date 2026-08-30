import pytest

from research_platform.domain.models import AccessClass
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.registry import (
    ALL_TENANTS,
    Capability,
    CapabilityNotFound,
    CapabilityRegistry,
    DuplicateCapability,
)


def make_capability(**overrides: object) -> Capability:
    defaults: dict[str, object] = {
        "server": "web-research",
        "name": "fetch",
        "description": "Fetch an approved public URL.",
        "required_roles": frozenset({Role.REQUESTER}),
        "allowed_agents": frozenset({AgentRole.RESEARCHER}),
    }
    return Capability(**(defaults | overrides))  # type: ignore[arg-type]


def person(**overrides: object) -> Principal:
    defaults: dict[str, object] = {
        "tenant_id": "acme",
        "subject_id": "user-1",
        "roles": frozenset({Role.REQUESTER}),
    }
    return Principal(**(defaults | overrides))  # type: ignore[arg-type]


def test_a_principal_must_carry_a_tenant_and_subject() -> None:
    with pytest.raises(ValueError, match="must belong to a tenant"):
        Principal(tenant_id="", subject_id="user-1")
    with pytest.raises(ValueError, match="subject identifier"):
        Principal(tenant_id="acme", subject_id="")


def test_for_agent_derives_an_agent_identity_on_the_same_tenant() -> None:
    agent = person(clearance=AccessClass.INTERNAL).for_agent(AgentRole.RESEARCHER)

    assert agent.is_agent is True
    assert agent.agent_role is AgentRole.RESEARCHER
    assert agent.tenant_id == "acme"
    assert agent.clearance is AccessClass.INTERNAL


def test_a_side_effecting_capability_must_require_approval() -> None:
    with pytest.raises(ValueError, match="must require approval"):
        make_capability(side_effecting=True)


def test_a_capability_must_be_scoped_to_a_tenant() -> None:
    with pytest.raises(ValueError, match="at least one tenant"):
        make_capability(tenant_scope=frozenset())


def test_registering_the_same_capability_twice_is_rejected() -> None:
    registry = CapabilityRegistry([make_capability()])

    with pytest.raises(DuplicateCapability):
        registry.register(make_capability())


def test_discovery_hides_capabilities_scoped_to_another_tenant() -> None:
    registry = CapabilityRegistry([make_capability(tenant_scope=frozenset({"globex"}))])

    assert registry.discover(person()) == []
    assert registry.discover(person(tenant_id="globex")) != []


def test_discovery_hides_capabilities_above_the_caller_clearance() -> None:
    registry = CapabilityRegistry([make_capability(max_access_class=AccessClass.RESTRICTED)])

    assert registry.discover(person()) == []
    assert registry.discover(person(clearance=AccessClass.RESTRICTED)) != []


def test_discovery_hides_capabilities_the_caller_has_no_role_for() -> None:
    registry = CapabilityRegistry([make_capability(required_roles=frozenset({Role.ADMINISTRATOR}))])

    assert registry.discover(person()) == []
    assert registry.discover(person(roles=frozenset({Role.ADMINISTRATOR}))) != []


def test_discovery_is_sorted_by_qualified_name() -> None:
    registry = CapabilityRegistry(
        [make_capability(name="search"), make_capability(name="fetch")],
    )

    assert [c.qualified_name for c in registry.discover(person())] == [
        "web-research.fetch",
        "web-research.search",
    ]


def test_an_agent_only_discovers_capabilities_for_its_own_role() -> None:
    registry = CapabilityRegistry([make_capability()])

    assert registry.discover(person().for_agent(AgentRole.RESEARCHER)) != []
    assert registry.discover(person().for_agent(AgentRole.REPORTER)) == []


def test_the_planner_reads_metadata_but_cannot_execute() -> None:
    capability = make_capability()
    planner = person().for_agent(AgentRole.PLANNER)

    assert capability.is_visible_to(planner) is True
    assert capability.is_executable_by(planner) is False


def test_an_allowed_agent_may_execute_what_it_discovers() -> None:
    capability = make_capability()
    researcher = person().for_agent(AgentRole.RESEARCHER)

    assert capability.is_executable_by(researcher) is True


def test_resolving_an_invisible_capability_reports_it_as_missing() -> None:
    registry = CapabilityRegistry([make_capability(tenant_scope=frozenset({"globex"}))])

    with pytest.raises(CapabilityNotFound):
        registry.resolve_for(person(), "web-research", "fetch")


def test_resolving_an_unregistered_capability_reports_the_qualified_name() -> None:
    registry = CapabilityRegistry([make_capability()])

    with pytest.raises(CapabilityNotFound, match="web-research.crawl"):
        registry.get("web-research", "crawl")


def test_resolving_a_visible_capability_returns_it() -> None:
    registry = CapabilityRegistry([make_capability()])

    assert registry.resolve_for(person(), "web-research", "fetch").name == "fetch"


def test_the_default_catalogue_covers_the_specified_servers() -> None:
    registry = default_registry()

    assert registry.servers() == [
        "evidence",
        "filesystem",
        "github",
        "postgres",
        "python-analysis",
        "registry",
        "web-research",
    ]
    assert len(registry) == 11


def test_the_reporter_is_never_offered_a_side_effecting_capability() -> None:
    registry = default_registry()
    reporter = person(
        roles=frozenset({Role.ADMINISTRATOR}),
        clearance=AccessClass.RESTRICTED,
    ).for_agent(AgentRole.REPORTER)

    executable = [c for c in registry.discover(reporter) if c.is_executable_by(reporter)]

    assert executable != []
    assert all(not capability.side_effecting for capability in executable)


def test_administrative_capabilities_stay_hidden_from_a_requester() -> None:
    registry = default_registry()
    requester = person(clearance=AccessClass.RESTRICTED)

    discovered = {capability.qualified_name for capability in registry.discover(requester)}

    assert "registry.reload_policy" not in discovered
    assert "web-research.fetch" in discovered


def test_every_default_capability_is_scoped_to_all_tenants_by_default() -> None:
    for capability in default_registry().discover(
        person(roles=frozenset({Role.ADMINISTRATOR}), clearance=AccessClass.RESTRICTED)
    ):
        assert capability.tenant_scope == frozenset({ALL_TENANTS})
