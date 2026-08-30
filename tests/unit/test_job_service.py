from uuid import uuid4

import pytest

from research_platform.application.jobs import (
    InMemoryJobRepository,
    JobNotFoundError,
    ResearchJobService,
)
from research_platform.domain.models import EvidenceRecordCreate, ResearchJobCreate


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
