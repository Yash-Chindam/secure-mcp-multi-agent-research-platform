from enum import StrEnum
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel

from research_platform.agents.contracts import ResearchReport
from research_platform.api.dependencies import Identity
from research_platform.application.artifacts import ArtifactNotFound, ArtifactStore
from research_platform.application.jobs import JobNotFoundError, ResearchJobService
from research_platform.application.publication import ProvenanceManifest, ReportPublication
from research_platform.application.workflows import (
    WorkflowNotRunning,
    WorkflowStarter,
    WorkflowUnavailable,
)
from research_platform.domain.invocations import ToolInvocation
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    FindingRecord,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Role
from research_platform.mcp.registry import Capability, CapabilityRegistry
from research_platform.workflow.orchestration import ReviewerDecision

REVIEWERS = frozenset({Role.REVIEWER, Role.ADMINISTRATOR})


class ReviewSubmission(BaseModel):
    """A reviewer's decision on a job the critic sent for review."""

    decision: ReviewerDecision


class ReportFormat(StrEnum):
    JSON = "json"
    MARKDOWN = "markdown"


def create_router(
    service: ResearchJobService,
    registry: CapabilityRegistry,
    workflows: WorkflowStarter | None = None,
    artifacts: ArtifactStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["research-jobs"])

    def published(identity: Identity, job_id: UUID) -> ReportPublication:
        """The job's publication record, or the 404 that says which part is missing."""
        try:
            publication = service.get_publication(identity.tenant_id, job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error
        if publication is None:
            raise HTTPException(
                status_code=404, detail="this research job has not published a report"
            )
        return publication

    def stored(identity: Identity, name: str) -> bytes:
        if artifacts is None:
            raise HTTPException(status_code=503, detail="no artifact store is configured")
        try:
            return artifacts.get(identity.tenant_id, name)
        except ArtifactNotFound as error:
            raise HTTPException(
                status_code=404, detail="the published artifact is no longer stored"
            ) from error

    @router.get("/capabilities", response_model=list[Capability], tags=["mcp"])
    def discover_capabilities(
        identity: Identity,
        acting_agent: Annotated[AgentRole | None, Query()] = None,
    ) -> list[Capability]:
        """Reveal only the capabilities this identity is permitted to see."""
        principal = identity.principal
        if acting_agent is not None:
            principal = principal.for_agent(acting_agent)
        return registry.discover(principal)

    @router.post("/jobs", response_model=ResearchJob, status_code=status.HTTP_201_CREATED)
    async def create_job(command: ResearchJobCreate, identity: Identity) -> ResearchJob:
        """Record a research job and, when workflows are enabled, start working on it.

        The job is stored first and started second, so a job that could not be started
        is still on record: it is marked failed and reported as unavailable rather than
        left looking as if work were under way.
        """
        job = await run_in_threadpool(
            service.create, identity.tenant_id, identity.requester_id, command
        )
        if workflows is None:
            return job
        try:
            checkpoint = await workflows.start(job)
        except WorkflowUnavailable as error:
            await run_in_threadpool(service.transition, job.tenant_id, job.id, JobStatus.FAILED)
            raise HTTPException(
                status_code=503,
                detail=f"research job {job.id} was recorded but could not be started: {error}",
            ) from error
        return await run_in_threadpool(
            lambda: service.record_checkpoint(
                job.tenant_id,
                job.id,
                workflow_id=checkpoint.workflow_id,
                workflow_run_id=checkpoint.workflow_run_id,
            )
        )

    @router.post(
        "/jobs/{job_id}/review",
        response_model=ResearchJob,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def review_job(
        job_id: UUID, submission: ReviewSubmission, identity: Identity
    ) -> ResearchJob:
        """Deliver a reviewer's decision to a job that is waiting for one.

        Only a reviewer may decide, and never on a job they requested themselves - the
        approval checkpoint is a second person by design (an administrator may override
        that for a tenant with no second reviewer available).
        """
        if not identity.roles & REVIEWERS:
            raise HTTPException(status_code=403, detail="only a reviewer may decide a review")
        try:
            job = await run_in_threadpool(service.get, identity.tenant_id, job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error
        if job.requester_id == identity.requester_id and Role.ADMINISTRATOR not in identity.roles:
            raise HTTPException(
                status_code=403, detail="a requester cannot review their own research job"
            )
        if job.status is not JobStatus.REVIEW_REQUIRED:
            raise HTTPException(
                status_code=409, detail=f"research job is {job.status.value}, not awaiting review"
            )
        if workflows is None:
            raise HTTPException(status_code=503, detail="workflows are not enabled")
        try:
            await workflows.submit_reviewer_decision(job, submission.decision)
        except WorkflowNotRunning as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except WorkflowUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return job

    @router.get("/jobs", response_model=list[ResearchJob])
    def list_jobs(identity: Identity) -> list[ResearchJob]:
        return service.list(identity.tenant_id)

    @router.get("/jobs/{job_id}", response_model=ResearchJob)
    def get_job(job_id: UUID, identity: Identity) -> ResearchJob:
        try:
            return service.get(identity.tenant_id, job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error

    @router.post("/jobs/{job_id}/transitions", response_model=ResearchJob)
    def transition_job(
        job_id: UUID,
        target: Annotated[JobStatus, Query()],
        identity: Identity,
    ) -> ResearchJob:
        try:
            return service.transition(identity.tenant_id, job_id, target)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post(
        "/jobs/{job_id}/evidence",
        response_model=EvidenceRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def add_evidence(
        job_id: UUID,
        command: EvidenceRecordCreate,
        identity: Identity,
    ) -> EvidenceRecord:
        try:
            return service.add_evidence(identity.tenant_id, job_id, command)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error

    @router.get("/jobs/{job_id}/evidence", response_model=list[EvidenceRecord])
    def list_evidence(job_id: UUID, identity: Identity) -> list[EvidenceRecord]:
        try:
            return service.list_evidence(identity.tenant_id, job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error

    @router.get("/jobs/{job_id}/findings", response_model=list[FindingRecord])
    def list_findings(job_id: UUID, identity: Identity) -> list[FindingRecord]:
        """The job's claims, each with its evidence, critic verdict and reviewer status."""
        try:
            return service.list_findings(identity.tenant_id, job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error

    @router.get(
        "/jobs/{job_id}/report",
        response_model=ResearchReport,
        responses={200: {"content": {"text/markdown": {}}}},
    )
    def get_report(
        job_id: UUID,
        identity: Identity,
        format: Annotated[ReportFormat, Query()] = ReportFormat.JSON,
    ) -> Response:
        """The published report: structured JSON, or Markdown with numbered sources.

        ``X-Report-SHA256`` carries the hash the provenance manifest records for the JSON
        form, and ``X-Report-Partial`` says whether the result is a partial one.
        """
        publication = published(identity, job_id)
        headers = {
            "X-Report-SHA256": publication.report_sha256,
            "X-Report-Partial": str(publication.is_partial).lower(),
        }
        if format is ReportFormat.MARKDOWN:
            return Response(
                stored(identity, publication.markdown_key),
                media_type="text/markdown; charset=utf-8",
                headers=headers,
            )
        return Response(
            stored(identity, publication.report_key),
            media_type="application/json",
            headers=headers,
        )

    @router.get("/jobs/{job_id}/manifest", response_model=ProvenanceManifest)
    def get_manifest(job_id: UUID, identity: Identity) -> Response:
        """The provenance manifest: every claim, source, hash and tool call behind the report."""
        publication = published(identity, job_id)
        return Response(stored(identity, publication.manifest_key), media_type="application/json")

    @router.get("/jobs/{job_id}/invocations", response_model=list[ToolInvocation], tags=["mcp"])
    def list_invocations(job_id: UUID, identity: Identity) -> list[ToolInvocation]:
        """Return the audit trail for one job: every MCP call attempted on its behalf.

        The records carry sanitized arguments and an argument digest rather than the
        arguments themselves (section 11), so the trail can be read by a requester
        without exposing what a sensitive argument contained.
        """
        try:
            return service.list_invocations(identity.tenant_id, job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="research job not found") from error

    return router
