import pytest

from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import ExecutionRequest, UpstreamError
from research_platform.settings import Settings
from research_platform.worker import (
    UnconfiguredExecutor,
    build_job_activities,
    build_job_service,
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


def test_a_worker_with_no_database_configured_falls_back_to_in_process_job_state() -> None:
    """Usable for a single-process demo; a deployment must configure a database."""
    activities = build_job_activities(Settings(database_url=None))

    assert isinstance(activities, JobActivities)


def test_a_worker_writes_its_audit_trail_and_job_state_to_one_shared_store() -> None:
    """What the gateway records and what the job activities persist must be one store."""
    settings = Settings(database_url=None, web_allowed_domains="vendor.test")
    jobs = build_job_service(settings)

    research = build_research_activities(settings, jobs)
    persistence = build_job_activities(settings, jobs)

    assert persistence.jobs is jobs
    assert research.gateway._audit == jobs.record_invocation
