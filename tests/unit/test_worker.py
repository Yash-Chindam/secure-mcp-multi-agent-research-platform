import pytest

from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import ExecutionRequest, UpstreamError
from research_platform.settings import Settings
from research_platform.worker import (
    UnconfiguredExecutor,
    build_job_activities,
    build_research_activities,
)
from research_platform.workflow.activities import JobActivities, ResearchActivities


def test_a_deployment_with_no_web_domains_configured_registers_no_capabilities() -> None:
    activities = build_research_activities(Settings())

    assert isinstance(activities, ResearchActivities)
    assert activities.registry.servers() == []


def test_a_deployment_with_web_domains_configured_registers_web_research() -> None:
    activities = build_research_activities(Settings(web_allowed_domains="vendor.test"))

    assert activities.registry.servers() == ["web-research"]


def test_an_unconfigured_executor_refuses_every_call_rather_than_defaulting() -> None:
    capability = default_registry().get("web-research", "fetch")
    principal = Principal(tenant_id="acme", subject_id="agent-1").for_agent(AgentRole.RESEARCHER)
    request = ExecutionRequest(capability=capability, principal=principal, arguments={})

    with pytest.raises(UpstreamError, match="no MCP backend is configured"):
        UnconfiguredExecutor().execute(request)


def test_build_job_activities_returns_a_working_job_activities_instance() -> None:
    activities = build_job_activities()

    assert isinstance(activities, JobActivities)
