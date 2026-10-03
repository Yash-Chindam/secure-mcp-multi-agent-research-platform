"""The evidence MCP server: what the critic and the reporter are allowed to read.

Section 7 gives those two agents "evidence retrieval only". This is that retrieval. It
reads the job's recorded evidence from the system of record, scoped to the tenant the
gateway supplies, and returns each record with the identifier a claim must cite - so a
critic checks a claim against what was actually stored rather than against the excerpt a
previous agent chose to repeat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fastmcp import FastMCP

from research_platform.application.jobs import JobNotFoundError
from research_platform.domain.models import EvidenceRecord


class EvidenceSource(Protocol):
    def list_evidence(self, tenant_id: str, job_id: UUID) -> list[EvidenceRecord]: ...


@dataclass(frozen=True)
class EvidenceService:
    source: EvidenceSource

    def retrieve(
        self, tenant_id: str, job_id: str, evidence_id: str | None
    ) -> list[dict[str, object]]:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("an evidence call must identify its tenant")
        try:
            job = UUID(job_id)
            wanted = UUID(evidence_id) if evidence_id else None
        except ValueError as error:
            raise ValueError("job_id and evidence_id must be identifiers") from error
        try:
            records = self.source.list_evidence(tenant_id, job)
        except JobNotFoundError as error:
            # Reported the same way for a missing job and for another tenant's job, so
            # the answer cannot be used to learn which jobs exist.
            raise ValueError(f"no evidence is recorded for job {job_id}") from error
        return [
            {
                "id": str(record.id),
                "excerpt": record.excerpt,
                "source_uri": str(record.source_uri),
                "title": record.title,
                "published_at": record.published_at.isoformat() if record.published_at else None,
                "retrieved_at": record.retrieved_at.isoformat(),
                "trust_level": record.trust_level.value,
                "content_hash": record.content_hash,
            }
            for record in records
            if wanted is None or record.id == wanted
        ]


def build_evidence_server(service: EvidenceService) -> FastMCP:
    """Expose recorded evidence over MCP."""
    server: FastMCP = FastMCP(name="evidence")

    @server.tool
    def retrieve(tenant_id: str, job_id: str, evidence_id: str | None = None) -> str:
        """Retrieve the recorded evidence for a job, or one record by identifier."""
        try:
            records = service.retrieve(tenant_id, job_id, evidence_id)
        except ValueError as error:
            raise ValueError(f"validation error: {error}") from error
        return json.dumps(records, separators=(",", ":"))

    return server
