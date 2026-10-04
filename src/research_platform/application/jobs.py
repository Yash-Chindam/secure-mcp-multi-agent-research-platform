"""The application's system of record, behind one storage-agnostic port.

``ResearchJobService`` holds the rules - a status transition is validated before it is
written, evidence cannot be attached to a job that does not exist, every read is scoped
to one tenant - and delegates storage to a ``JobRepository``. Two implement that port:
``InMemoryJobRepository`` here, for development and tests, and
``research_platform.persistence.PostgresJobRepository``, the durable store section 10
asks for.
"""

from __future__ import annotations

import builtins
from threading import RLock
from typing import Protocol
from uuid import UUID

from research_platform.application.publication import ReportPublication
from research_platform.domain.invocations import ToolInvocation
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    Finding,
    FindingRecord,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)


class JobNotFoundError(LookupError):
    pass


class JobRepository(Protocol):
    """Storage for the jobs, evidence and audit trail of one deployment.

    Every method takes the tenant explicitly rather than inferring it: a repository must
    not be able to answer a question that was never scoped to a tenant, which is what
    makes the cross-tenant guarantee in section 14 checkable at this boundary.
    """

    def add(self, job: ResearchJob) -> ResearchJob: ...

    def get(self, tenant_id: str, job_id: UUID) -> ResearchJob: ...

    def list(self, tenant_id: str) -> builtins.list[ResearchJob]: ...

    def update(self, job: ResearchJob) -> ResearchJob: ...

    def add_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord: ...

    def list_evidence(self, tenant_id: str, job_id: UUID) -> builtins.list[EvidenceRecord]: ...

    def record_invocation(self, invocation: ToolInvocation) -> ToolInvocation: ...

    def list_invocations(self, tenant_id: str, job_id: UUID) -> builtins.list[ToolInvocation]: ...

    def replace_findings(
        self, tenant_id: str, job_id: UUID, findings: builtins.list[FindingRecord]
    ) -> builtins.list[FindingRecord]: ...

    def list_findings(self, tenant_id: str, job_id: UUID) -> builtins.list[FindingRecord]: ...

    def record_publication(self, publication: ReportPublication) -> ReportPublication: ...

    def get_publication(self, tenant_id: str, job_id: UUID) -> ReportPublication | None: ...


class InMemoryJobRepository:
    """Development repository that enforces tenant isolation at every lookup."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, UUID], ResearchJob] = {}
        self._evidence: dict[tuple[str, UUID], list[EvidenceRecord]] = {}
        self._invocations: dict[tuple[str, UUID], list[ToolInvocation]] = {}
        self._findings: dict[tuple[str, UUID], list[FindingRecord]] = {}
        self._publications: dict[tuple[str, UUID], ReportPublication] = {}
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

    def record_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        """Append one audit record, without requiring the job it names to exist.

        An audit record is written whether or not its job is present - a denial for an
        unknown job is itself worth keeping - so this deliberately does not check that it
        exists. The write is idempotent on the record's own identifier, because an
        activity Temporal redelivers re-reports the audit record it already persisted and
        the trail must not count that call twice.
        """
        key = (invocation.tenant_id, invocation.job_id)
        with self._lock:
            recorded = self._invocations.setdefault(key, [])
            if any(existing.id == invocation.id for existing in recorded):
                return invocation
            recorded.append(invocation)
        return invocation

    def list_invocations(self, tenant_id: str, job_id: UUID) -> builtins.list[ToolInvocation]:
        with self._lock:
            recorded = list(self._invocations.get((tenant_id, job_id), []))
        return sorted(recorded, key=lambda invocation: invocation.started_at)

    def replace_findings(
        self, tenant_id: str, job_id: UUID, findings: builtins.list[FindingRecord]
    ) -> builtins.list[FindingRecord]:
        """Store the job's current findings in place of whatever was recorded before.

        Findings are re-derived each time the critic judges the analysis, and again when
        a reviewer decides, so the stored set is the latest one rather than an
        accumulation - which also makes a redelivered write harmless.
        """
        self.get(tenant_id, job_id)
        with self._lock:
            self._findings[(tenant_id, job_id)] = list(findings)
        return findings

    def list_findings(self, tenant_id: str, job_id: UUID) -> builtins.list[FindingRecord]:
        self.get(tenant_id, job_id)
        with self._lock:
            return list(self._findings.get((tenant_id, job_id), []))

    def record_publication(self, publication: ReportPublication) -> ReportPublication:
        """Store where a job's report was published, replacing an earlier publication.

        One job has one published report. Publishing again - a redelivered activity -
        points the job at the same artifacts, so the latest record simply wins.
        """
        self.get(publication.tenant_id, publication.job_id)
        with self._lock:
            self._publications[(publication.tenant_id, publication.job_id)] = publication
        return publication

    def get_publication(self, tenant_id: str, job_id: UUID) -> ReportPublication | None:
        self.get(tenant_id, job_id)
        with self._lock:
            return self._publications.get((tenant_id, job_id))


class ResearchJobService:
    def __init__(self, repository: JobRepository) -> None:
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

    def transition(
        self, tenant_id: str, job_id: UUID, target: JobStatus, detail: str | None = None
    ) -> ResearchJob:
        job = self.get(tenant_id, job_id)
        return self._repository.update(job.transition_to(target, detail))

    def record_checkpoint(
        self, tenant_id: str, job_id: UUID, *, workflow_id: str, workflow_run_id: str
    ) -> ResearchJob:
        """Store which durable execution owns this job, leaving its status untouched."""
        job = self.get(tenant_id, job_id)
        return self._repository.update(
            job.with_checkpoint(workflow_id=workflow_id, workflow_run_id=workflow_run_id)
        )

    def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        self.get(tenant_id, job_id)
        evidence = EvidenceRecord(tenant_id=tenant_id, job_id=job_id, **command.model_dump())
        return self._repository.add_evidence(evidence)

    def list_evidence(self, tenant_id: str, job_id: UUID) -> builtins.list[EvidenceRecord]:
        return self._repository.list_evidence(tenant_id, job_id)

    def record_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        return self._repository.record_invocation(invocation)

    def list_invocations(self, tenant_id: str, job_id: UUID) -> builtins.list[ToolInvocation]:
        """Return the audit trail for one job, oldest call first."""
        self.get(tenant_id, job_id)
        return self._repository.list_invocations(tenant_id, job_id)

    def record_findings(
        self, tenant_id: str, job_id: UUID, findings: builtins.list[Finding]
    ) -> builtins.list[FindingRecord]:
        """Store the job's current findings, replacing any earlier set."""
        self.get(tenant_id, job_id)
        records = [
            FindingRecord(tenant_id=tenant_id, job_id=job_id, **finding.model_dump())
            for finding in findings
        ]
        return self._repository.replace_findings(tenant_id, job_id, records)

    def list_findings(self, tenant_id: str, job_id: UUID) -> builtins.list[FindingRecord]:
        return self._repository.list_findings(tenant_id, job_id)

    def record_publication(self, publication: ReportPublication) -> ReportPublication:
        return self._repository.record_publication(publication)

    def get_publication(self, tenant_id: str, job_id: UUID) -> ReportPublication | None:
        """Where the job's report was published, or ``None`` if it has not been."""
        return self._repository.get_publication(tenant_id, job_id)


class AsyncJobs:
    """Adapts the synchronous ``ResearchJobService`` to an async ``JobsPort``.

    ``research_platform.workflow.orchestration`` awaits its job-state calls, because a
    Temporal-driven run answers them with an activity. This in-process service has no
    activity to await - it is a plain repository write - so this adapter exists purely to
    satisfy that async shape, not to add any real asynchrony.
    """

    def __init__(self, service: ResearchJobService) -> None:
        self._service = service

    async def transition(
        self, tenant_id: str, job_id: UUID, target: JobStatus, detail: str | None = None
    ) -> ResearchJob:
        return self._service.transition(tenant_id, job_id, target, detail)

    async def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        return self._service.add_evidence(tenant_id, job_id, command)

    async def record_findings(
        self, tenant_id: str, job_id: UUID, findings: builtins.list[Finding]
    ) -> builtins.list[FindingRecord]:
        return self._service.record_findings(tenant_id, job_id, findings)
