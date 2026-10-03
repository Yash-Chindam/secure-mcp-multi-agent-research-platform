from uuid import UUID, uuid4

import pytest

from research_platform.application.jobs import (
    InMemoryJobRepository,
    JobNotFoundError,
    ResearchJobService,
)
from research_platform.domain.invocations import (
    AuthorizationDecision,
    InvocationOutcome,
    ToolInvocation,
)
from research_platform.domain.models import (
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)


def service() -> ResearchJobService:
    return ResearchJobService(InMemoryJobRepository())


def test_repository_never_returns_another_tenants_job() -> None:
    jobs = service()
    created = jobs.create("tenant-a", "requester", ResearchJobCreate(question="Question"))

    with pytest.raises(JobNotFoundError):
        jobs.get("tenant-b", created.id)


def test_evidence_is_scoped_to_owning_tenant_and_job() -> None:
    jobs = service()
    created = jobs.create("tenant-a", "requester", ResearchJobCreate(question="Question"))
    command = EvidenceRecordCreate(
        excerpt="Primary-source excerpt",
        source_uri="https://example.com/source",
        content_hash=f"sha256:{'a' * 64}",
        producing_task_id=uuid4(),
        tool_invocation_id=uuid4(),
    )

    evidence = jobs.add_evidence("tenant-a", created.id, command)

    assert jobs.list_evidence("tenant-a", created.id) == [evidence]
    with pytest.raises(JobNotFoundError):
        jobs.list_evidence("tenant-b", created.id)


def test_service_lists_and_transitions_only_the_tenants_jobs() -> None:
    jobs = service()
    tenant_job = jobs.create("tenant-a", "requester", ResearchJobCreate(question="Question A"))
    jobs.create("tenant-b", "requester", ResearchJobCreate(question="Question B"))

    planning = jobs.transition("tenant-a", tenant_job.id, JobStatus.PLANNING)

    assert planning.status is JobStatus.PLANNING
    assert jobs.list("tenant-a") == [planning]


def test_repository_rejects_updating_an_unknown_job() -> None:
    repository = InMemoryJobRepository()
    unknown = ResearchJob(tenant_id="tenant-a", requester_id="requester", question="Question")

    with pytest.raises(JobNotFoundError):
        repository.update(unknown)


def an_invocation(tenant_id: str, job_id: UUID) -> ToolInvocation:
    return ToolInvocation(
        job_id=job_id,
        task_id=uuid4(),
        tenant_id=tenant_id,
        mcp_server="web-research",
        capability="search",
        argument_digest=f"sha256:{'c' * 64}",
        policy_version="registry-boundary/1",
        authorization_decision=AuthorizationDecision.ALLOW,
        outcome=InvocationOutcome.SUCCEEDED,
        duration_ms=42,
    )


def test_the_service_records_a_workflow_checkpoint_without_changing_status() -> None:
    jobs = service()
    created = jobs.create("tenant-a", "requester", ResearchJobCreate(question="Question"))

    checkpointed = jobs.record_checkpoint(
        "tenant-a", created.id, workflow_id="research-job-1", workflow_run_id="run-1"
    )

    assert checkpointed.workflow_id == "research-job-1"
    assert checkpointed.workflow_run_id == "run-1"
    assert checkpointed.status is created.status
    assert jobs.get("tenant-a", created.id).workflow_id == "research-job-1"


def test_a_checkpoint_cannot_be_written_for_another_tenants_job() -> None:
    jobs = service()
    created = jobs.create("tenant-a", "requester", ResearchJobCreate(question="Question"))

    with pytest.raises(JobNotFoundError):
        jobs.record_checkpoint(
            "tenant-b", created.id, workflow_id="research-job-1", workflow_run_id="run-1"
        )


def test_the_service_returns_the_audit_trail_for_its_own_job_only() -> None:
    jobs = service()
    created = jobs.create("tenant-a", "requester", ResearchJobCreate(question="Question"))
    recorded = jobs.record_invocation(an_invocation("tenant-a", created.id))

    assert jobs.list_invocations("tenant-a", created.id) == [recorded]
    with pytest.raises(JobNotFoundError):
        jobs.list_invocations("tenant-b", created.id)
