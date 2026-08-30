from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from research_platform.api.dependencies import Identity
from research_platform.application.jobs import JobNotFoundError, ResearchJobService
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole
from research_platform.mcp.registry import Capability, CapabilityRegistry


def create_router(service: ResearchJobService, registry: CapabilityRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["research-jobs"])

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
    def create_job(command: ResearchJobCreate, identity: Identity) -> ResearchJob:
        return service.create(identity.tenant_id, identity.requester_id, command)

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

    return router
