"""The last step of section 9: export the report and its provenance manifest.

Publication is one activity. It reads what the job recorded - evidence, findings, the
audit trail - looks at every cited source once more, and writes four artifacts under the
job's own prefix: the report as JSON and as Markdown, the provenance manifest, and the
evidence bundle the report rests on.

Looking at a source again is how section 12's "preserve the original evidence and flag
content drift" is met. The captured excerpt is never replaced. The source is re-read
through the same governed gateway that fetched it, and the manifest and the rendered
report say whether the excerpt is still there.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from temporalio import activity

from research_platform.agents.contracts import ResearchReport
from research_platform.agents.provenance import normalize_text
from research_platform.application.artifacts import ArtifactStore, job_artifact
from research_platform.application.jobs import ResearchJobService
from research_platform.application.publication import (
    DriftStatus,
    ReportPublication,
    SourceCheck,
    build_manifest,
    render_markdown,
    report_json,
    sha256_of,
)
from research_platform.domain.invocations import ToolInvocation
from research_platform.domain.models import EvidenceRecord, ResearchJob
from research_platform.domain.tasks import AgentRole
from research_platform.mcp.gateway import CapabilityDenied, CapabilityFailed, CapabilityGateway
from research_platform.mcp.registry import CapabilityNotFound
from research_platform.observability.metrics import PlatformMetrics, get_metrics
from research_platform.workflow.activities import principal_for

REPORT_JSON = "report.json"
REPORT_MARKDOWN = "report.md"
MANIFEST = "provenance-manifest.json"
EVIDENCE_BUNDLE = "evidence.json"


class SourceChecker(Protocol):
    """Reads one cited source again and says whether the captured excerpt is still there."""

    def check(
        self, job: ResearchJob, record: EvidenceRecord, invocation: ToolInvocation | None
    ) -> SourceCheck: ...


def refetch_arguments(
    record: EvidenceRecord, invocation: ToolInvocation | None
) -> dict[str, Any] | None:
    """The arguments that read a source again, or ``None`` when it has no address.

    Which capability captured the evidence comes from the audit trail, not from the shape
    of the source string. A page and a workspace document can be read again by address;
    a search, a query or a calculation is identified only by the call that produced it,
    and running that call again would be new research rather than a re-read.
    """
    if invocation is None:
        return None
    source = str(record.source_uri)
    key = (invocation.mcp_server, invocation.capability)
    if key == ("web-research", "fetch"):
        return {"url": source}
    if key == ("filesystem", "read_document"):
        return {"path": urlsplit(source).path.lstrip("/")}
    return None


@dataclass(frozen=True)
class GatewaySourceChecker:
    """Re-reads a source through the capability gateway, as the researcher that captured it.

    The re-read is an ordinary governed call: it is authorized, counted against the
    job's tool-call budget, and written to the audit trail like any other. A call that
    is refused or fails leaves the source ``unavailable`` rather than guessed at.
    """

    gateway: CapabilityGateway

    def check(
        self, job: ResearchJob, record: EvidenceRecord, invocation: ToolInvocation | None
    ) -> SourceCheck:
        arguments = refetch_arguments(record, invocation)
        if arguments is None or invocation is None:
            return SourceCheck(
                evidence_id=record.id,
                status=DriftStatus.NOT_CHECKED,
                detail="this kind of source has no address to read again",
            )
        try:
            result = self.gateway.invoke(
                principal=principal_for(job, AgentRole.RESEARCHER),
                job_id=job.id,
                task_id=record.producing_task_id,
                server=invocation.mcp_server,
                capability_name=invocation.capability,
                arguments=arguments,
                budget=job.budget,
            )
        except (CapabilityDenied, CapabilityFailed) as error:
            return SourceCheck(
                evidence_id=record.id, status=DriftStatus.UNAVAILABLE, detail=error.reason[:500]
            )
        except CapabilityNotFound as error:
            return SourceCheck(
                evidence_id=record.id, status=DriftStatus.UNAVAILABLE, detail=str(error)[:500]
            )
        if normalize_text(record.excerpt) in normalize_text(result.content.text):
            return SourceCheck(evidence_id=record.id, status=DriftStatus.UNCHANGED)
        return SourceCheck(
            evidence_id=record.id,
            status=DriftStatus.DRIFTED,
            detail="the captured excerpt is no longer present at the source",
        )


@dataclass(frozen=True)
class PublicationActivities:
    """Publishes a job's report. Safe to redeliver: every write lands on the same name."""

    jobs: ResearchJobService
    artifacts: ArtifactStore
    sources: SourceChecker | None = None
    metrics: PlatformMetrics | None = None

    @activity.defn(name="publish_report")
    async def publish(
        self, job: ResearchJob, report: ResearchReport, shortfalls: list[str]
    ) -> ReportPublication:
        # Re-reading sources and writing objects are blocking calls, so they run off the
        # event loop the worker's other activities share.
        return await asyncio.to_thread(self._publish, job, report, shortfalls)

    def _check_sources(
        self,
        job: ResearchJob,
        report: ResearchReport,
        evidence: list[EvidenceRecord],
        invocations: list[ToolInvocation],
    ) -> list[SourceCheck]:
        if self.sources is None:
            return []
        metrics = self.metrics or get_metrics()
        by_id = {invocation.id: invocation for invocation in invocations}
        checks: list[SourceCheck] = []
        for record in evidence:
            if record.id not in report.cited_evidence_ids:
                continue
            check = self.sources.check(job, record, by_id.get(record.tool_invocation_id))
            metrics.source_checks.add(1, {"tenant.id": job.tenant_id, "status": check.status.value})
            checks.append(check)
        return checks

    def _publish(
        self, job: ResearchJob, report: ResearchReport, shortfalls: list[str]
    ) -> ReportPublication:
        evidence = self.jobs.list_evidence(job.tenant_id, job.id)
        findings = self.jobs.list_findings(job.tenant_id, job.id)
        # The trail is read before the sources are re-read, so the manifest lists the
        # calls the research made rather than the publication's own re-reads.
        invocations = self.jobs.list_invocations(job.tenant_id, job.id)
        checks = self._check_sources(job, report, evidence, invocations)

        manifest = build_manifest(
            job,
            report,
            shortfalls=shortfalls,
            findings=findings,
            evidence=evidence,
            invocations=invocations,
            checks=checks,
        )
        markdown = render_markdown(
            job, report, shortfalls=shortfalls, evidence=evidence, checks=checks
        )
        bundle = json.dumps(
            [record.model_dump(mode="json") for record in evidence if record.is_publishable],
            indent=2,
        )
        body = report_json(report)

        def name(filename: str) -> str:
            return job_artifact(job.id, filename)

        tenant = job.tenant_id
        self.artifacts.put(tenant, name(REPORT_JSON), body, "application/json")
        self.artifacts.put(
            tenant, name(REPORT_MARKDOWN), markdown.encode("utf-8"), "text/markdown; charset=utf-8"
        )
        self.artifacts.put(
            tenant,
            name(MANIFEST),
            manifest.model_dump_json(indent=2).encode("utf-8"),
            "application/json",
        )
        self.artifacts.put(
            tenant, name(EVIDENCE_BUNDLE), bundle.encode("utf-8"), "application/json"
        )
        # The record pointing at the artifacts is written last, so a job is never shown
        # as published while one of its artifacts is still missing.
        return self.jobs.record_publication(
            ReportPublication(
                job_id=job.id,
                tenant_id=tenant,
                report_key=name(REPORT_JSON),
                markdown_key=name(REPORT_MARKDOWN),
                manifest_key=name(MANIFEST),
                evidence_key=name(EVIDENCE_BUNDLE),
                report_sha256=sha256_of(body),
                is_partial=manifest.is_partial,
                drifted_evidence_ids=manifest.drifted_evidence_ids,
            )
        )
