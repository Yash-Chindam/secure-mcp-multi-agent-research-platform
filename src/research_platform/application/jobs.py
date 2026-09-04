from __future__ import annotations

import builtins
from threading import RLock
from uuid import UUID

from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)


class JobNotFoundError(LookupError):
    pass


class InMemoryJobRepository:
    """Development repository that enforces tenant isolation at every lookup."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, UUID], ResearchJob] = {}
        self._evidence: dict[tuple[str, UUID], list[EvidenceRecord]] = {}
        self._lock = RLock()

    def add(self, job: ResearchJob) -> ResearchJob:
        with self._lock:
            self._jobs[(job.tenant_id, job.id)] = job
        return job

    def get(self, tenant_id: str, job_id: UUID) -> ResearchJob:
        with self._lock:
            job = self._jobs.get((tenant_id, job_id))
        if job is None:
            raise JobNotFoundError(str(job_id))
        return job

    def list(self, tenant_id: str) -> builtins.list[ResearchJob]:
        with self._lock:
            jobs = [job for (tenant, _), job in self._jobs.items() if tenant == tenant_id]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def update(self, job: ResearchJob) -> ResearchJob:
        with self._lock:
            key = (job.tenant_id, job.id)
            if key not in self._jobs:
                raise JobNotFoundError(str(job.id))
            self._jobs[key] = job
        return job

    def add_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord:
        self.get(evidence.tenant_id, evidence.job_id)
        with self._lock:
            self._evidence.setdefault((evidence.tenant_id, evidence.job_id), []).append(evidence)
        return evidence

    def list_evidence(self, tenant_id: str, job_id: UUID) -> builtins.list[EvidenceRecord]:
        self.get(tenant_id, job_id)
        with self._lock:
            return list(self._evidence.get((tenant_id, job_id), []))


class ResearchJobService:
    def __init__(self, repository: InMemoryJobRepository) -> None:
        self._repository = repository

    def create(self, tenant_id: str, requester_id: str, command: ResearchJobCreate) -> ResearchJob:
        job = ResearchJob(
            tenant_id=tenant_id,
            requester_id=requester_id,
            **command.model_dump(),
        )
        return self._repository.add(job)

    def get(self, tenant_id: str, job_id: UUID) -> ResearchJob:
        return self._repository.get(tenant_id, job_id)

    def list(self, tenant_id: str) -> builtins.list[ResearchJob]:
        return self._repository.list(tenant_id)

    def transition(self, tenant_id: str, job_id: UUID, target: JobStatus) -> ResearchJob:
        job = self.get(tenant_id, job_id)
        return self._repository.update(job.transition_to(target))

    def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        self.get(tenant_id, job_id)
        evidence = EvidenceRecord(tenant_id=tenant_id, job_id=job_id, **command.model_dump())
        return self._repository.add_evidence(evidence)

    def list_evidence(self, tenant_id: str, job_id: UUID) -> builtins.list[EvidenceRecord]:
        return self._repository.list_evidence(tenant_id, job_id)


class AsyncJobs:
    """Adapts the synchronous ``ResearchJobService`` to an async ``JobsPort``.

    ``research_platform.workflow.orchestration`` awaits its job-state calls, because a
    Temporal-driven run answers them with an activity. This in-process service has no
    activity to await - it is a plain in-memory write - so this adapter exists purely to
    satisfy that async shape, not to add any real asynchrony.
    """

    def __init__(self, service: ResearchJobService) -> None:
        self._service = service

    async def transition(self, tenant_id: str, job_id: UUID, target: JobStatus) -> ResearchJob:
        return self._service.transition(tenant_id, job_id, target)

    async def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        return self._service.add_evidence(tenant_id, job_id, command)
