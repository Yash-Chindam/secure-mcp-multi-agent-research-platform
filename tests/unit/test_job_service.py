from uuid import uuid4

import pytest

from research_platform.application.jobs import (
    InMemoryJobRepository,
    JobNotFoundError,
    ResearchJobService,
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
