from uuid import uuid4

import pytest

from research_platform.composition import build_gateway, build_policy_stack
from research_platform.domain.models import AccessClass, ResearchBudget
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role
from research_platform.mcp.gateway import CapabilityDenied, ExecutionRequest
from research_platform.mcp.opa import AllOfPolicyEngine
from research_platform.mcp.policy import RegistryPolicyEngine
from research_platform.settings import Settings


class StubExecutor:
    def __init__(self) -> None:
        self.calls: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> str:
        self.calls.append(request)
        return "The team plan costs 20 USD per seat."


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type,call-arg]


def test_without_an_authorization_service_only_the_registry_boundary_applies() -> None:
    stack = build_policy_stack(settings())

    assert stack.externally_enforced is False
    assert isinstance(stack.engine, RegistryPolicyEngine)
    assert "registry boundary only" in stack.description


def test_a_configured_authorization_service_is_layered_with_the_registry() -> None:
    stack = build_policy_stack(settings(opa_url="http://opa.test"))

    assert stack.externally_enforced is True
    assert isinstance(stack.engine, AllOfPolicyEngine)
    assert "Open Policy Agent" in stack.description


def test_the_allowed_domain_list_is_parsed_and_normalized() -> None:
    parsed = settings(web_allowed_domains=" Vendor.test, regulator.test ,, ")

    assert parsed.allowed_domains == frozenset({"vendor.test", "regulator.test"})


def test_no_allowed_domains_are_configured_by_default() -> None:
    assert settings().allowed_domains == frozenset()


def test_an_authorization_service_is_not_configured_by_default() -> None:
    assert settings().policy_is_externally_enforced is False


@pytest.mark.parametrize("timeout", [0, -1, 31])
def test_an_unusable_authorization_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError):
        settings(opa_timeout_seconds=timeout)


def test_the_assembled_gateway_enforces_the_registry_boundary() -> None:
    executor = StubExecutor()
    gateway = build_gateway(executor=executor, settings=settings())
    planner = Principal(
        tenant_id="acme",
        subject_id="user-1",
        roles=frozenset({Role.REQUESTER}),
        clearance=AccessClass.INTERNAL,
    ).for_agent(AgentRole.PLANNER)

    with pytest.raises(CapabilityDenied, match="may not execute"):
        gateway.invoke(
            principal=planner,
            job_id=uuid4(),
            task_id=uuid4(),
            server="web-research",
            capability_name="fetch",
            arguments={"url": "https://vendor.test/pricing"},
            budget=ResearchBudget(),
        )

    assert executor.calls == []


def test_the_assembled_gateway_permits_an_authorized_call() -> None:
    executor = StubExecutor()
    gateway = build_gateway(executor=executor, settings=settings())
    researcher = Principal(
        tenant_id="acme",
        subject_id="user-1",
        roles=frozenset({Role.REQUESTER}),
    ).for_agent(AgentRole.RESEARCHER)

    result = gateway.invoke(
        principal=researcher,
        job_id=uuid4(),
        task_id=uuid4(),
        server="web-research",
        capability_name="fetch",
        arguments={"url": "https://vendor.test/pricing"},
        budget=ResearchBudget(),
    )

    assert "20 USD" in result.content.text
    assert len(executor.calls) == 1
