from uuid import uuid4

import pytest
from pydantic import ValidationError

from research_platform.domain.models import (
    Finding,
    InvalidStateTransition,
    JobStatus,
    ResearchJob,
)


def test_job_accepts_only_explicit_state_transitions() -> None:
    job = ResearchJob(tenant_id="tenant-a", requester_id="user-1", question="What changed?")

    planning = job.transition_to(JobStatus.PLANNING)

    assert planning.status is JobStatus.PLANNING
    assert job.status is JobStatus.CREATED


def test_job_rejects_skipping_workflow_stages() -> None:
    job = ResearchJob(tenant_id="tenant-a", requester_id="user-1", question="What changed?")

    with pytest.raises(InvalidStateTransition, match="cannot transition"):
        job.transition_to(JobStatus.COMPLETED)


def test_finding_requires_supporting_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="A factual claim", supporting_evidence_ids=[], confidence=0.5)


def test_evidence_cannot_support_and_contradict_same_finding() -> None:
    evidence_id = uuid4()

    with pytest.raises(ValidationError, match="both support and contradict"):
        Finding(
            claim="A factual claim",
            supporting_evidence_ids=[evidence_id],
            contradicting_evidence_ids=[evidence_id],
            confidence=0.5,
        )
