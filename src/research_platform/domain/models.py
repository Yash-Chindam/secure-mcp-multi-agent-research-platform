from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


NonEmptyText = Annotated[str, Field(min_length=1, max_length=4_000)]


class JobStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    REVIEW_REQUIRED = "review_required"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.CREATED: frozenset({JobStatus.PLANNING, JobStatus.FAILED}),
    JobStatus.PLANNING: frozenset({JobStatus.RESEARCHING, JobStatus.FAILED}),
    JobStatus.RESEARCHING: frozenset({JobStatus.ANALYZING, JobStatus.PARTIAL, JobStatus.FAILED}),
    JobStatus.ANALYZING: frozenset(
        {JobStatus.REVIEW_REQUIRED, JobStatus.REPORTING, JobStatus.PARTIAL, JobStatus.FAILED}
    ),
    JobStatus.REVIEW_REQUIRED: frozenset(
        {JobStatus.RESEARCHING, JobStatus.REPORTING, JobStatus.FAILED}
    ),
    JobStatus.REPORTING: frozenset({JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.PARTIAL: frozenset(),
}


class ResearchBudget(BaseModel):
    max_tool_calls: int = Field(default=50, ge=1, le=10_000)
    max_runtime_seconds: int = Field(default=3_600, ge=30, le=86_400)
    max_cost_usd: float = Field(default=10.0, gt=0, le=10_000)


class ResearchJobCreate(BaseModel):
    question: NonEmptyText
    constraints: list[NonEmptyText] = Field(default_factory=list, max_length=50)
    source_requirements: list[NonEmptyText] = Field(default_factory=list, max_length=50)
    budget: ResearchBudget = Field(default_factory=ResearchBudget)


class ResearchJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = Field(min_length=1, max_length=100)
    requester_id: str = Field(min_length=1, max_length=200)
    question: NonEmptyText
    constraints: list[str] = Field(default_factory=list)
    source_requirements: list[str] = Field(default_factory=list)
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    status: JobStatus = JobStatus.CREATED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def transition_to(self, target: JobStatus) -> ResearchJob:
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidStateTransition(self.status, target)
        return self.model_copy(update={"status": target, "updated_at": utc_now()})


class EvidenceRecordCreate(BaseModel):
    excerpt: NonEmptyText
    source_uri: HttpUrl | NonEmptyText
    title: str | None = Field(default=None, max_length=500)
    content_hash: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    producing_task_id: UUID
    tool_invocation_id: UUID


class EvidenceRecord(EvidenceRecordCreate):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    tenant_id: str = Field(min_length=1, max_length=100)
    retrieved_at: datetime = Field(default_factory=utc_now)


class Finding(BaseModel):
    claim: NonEmptyText
    supporting_evidence_ids: list[UUID] = Field(min_length=1)
    contradicting_evidence_ids: list[UUID] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def evidence_sets_must_not_overlap(self) -> Finding:
        supporting = set(self.supporting_evidence_ids)
        contradicting = set(self.contradicting_evidence_ids)
        if supporting & contradicting:
            raise ValueError("evidence cannot both support and contradict a finding")
        return self


class InvalidStateTransition(ValueError):
    def __init__(self, current: JobStatus, target: JobStatus) -> None:
        super().__init__(f"cannot transition research job from {current} to {target}")
        self.current = current
        self.target = target
