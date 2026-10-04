"""What a finished job publishes: findings, a report, and where every claim came from.

Section 9 ends with "export report and provenance manifest", and section 10 gives a
``Finding`` a critic verdict and a reviewer status. Nothing here calls a model or a
network. Each function turns what the pipeline already recorded - the analyst's
proposals, the critic's verdicts, a reviewer's decision, the evidence and the audit
trail - into the records a reader can check a report against.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from research_platform.agents.contracts import (
    CITATION,
    AnalysisResult,
    CriticReview,
    ResearchReport,
)
from research_platform.domain.invocations import (
    AuthorizationDecision,
    InvocationOutcome,
    ToolInvocation,
)
from research_platform.domain.models import (
    AccessClass,
    CriticVerdict,
    EvidenceRecord,
    Finding,
    FindingRecord,
    ResearchJob,
    ReviewerStatus,
    TrustLevel,
    utc_now,
)

MANIFEST_SCHEMA_VERSION = "1"


def derive_findings(
    analysis: AnalysisResult, critique: CriticReview, reviewer_status: ReviewerStatus
) -> list[Finding]:
    """Join each proposed claim with the critic's verdict and the reviewer's status.

    A claim the critic gave no verdict stays pending rather than being guessed at. A
    claim with evidence against it is never recorded as supported, whatever the verdict
    says: the conflict is kept and the finding is recorded as contradicted.
    """
    verdicts = {verdict.claim: verdict for verdict in critique.verdicts}
    findings: list[Finding] = []
    for proposed in analysis.findings:
        verdict = verdicts.get(proposed.claim)
        supporting = list(proposed.supporting_evidence_ids)
        conflicting = [
            *proposed.contradicting_evidence_ids,
            *(verdict.conflicting_evidence_ids if verdict is not None else []),
        ]
        contradicting = [
            identifier
            for identifier in dict.fromkeys(conflicting)
            if identifier not in set(supporting)
        ]
        critic_verdict = verdict.verdict if verdict is not None else CriticVerdict.PENDING
        if critic_verdict is CriticVerdict.SUPPORTED and contradicting:
            critic_verdict = CriticVerdict.CONTRADICTED
        findings.append(
            Finding(
                claim=proposed.claim,
                supporting_evidence_ids=supporting,
                contradicting_evidence_ids=contradicting,
                calculation_ids=list(proposed.calculation_ids),
                confidence=proposed.confidence,
                critic_verdict=critic_verdict,
                reviewer_status=reviewer_status,
            )
        )
    return findings


class DriftStatus(StrEnum):
    """Whether a cited source still says what was captured from it."""

    UNCHANGED = "unchanged"
    DRIFTED = "drifted"
    UNAVAILABLE = "unavailable"
    NOT_CHECKED = "not_checked"


DRIFT_NOTES = {
    DriftStatus.UNCHANGED: "source unchanged at publication",
    DriftStatus.DRIFTED: "**source content has changed since capture**",
    DriftStatus.UNAVAILABLE: "source could not be re-read at publication",
    DriftStatus.NOT_CHECKED: "source not re-checked",
}


class SourceCheck(BaseModel):
    """The result of looking at one cited source again at publication time."""

    model_config = {"frozen": True}

    evidence_id: UUID
    status: DriftStatus
    checked_at: datetime = Field(default_factory=utc_now)
    detail: str | None = Field(default=None, max_length=500)


class ManifestEvidence(BaseModel):
    """One evidence record as the manifest lists it: where it came from, not what it says.

    The excerpt itself is left out. Its hash is enough to check a copy against, and a
    manifest that carried the text would disclose internal content to anyone it was
    shared with.
    """

    model_config = {"frozen": True}

    id: UUID
    source_uri: str
    title: str | None
    author: str | None
    published_at: datetime | None
    retrieved_at: datetime
    content_hash: str
    trust_level: TrustLevel
    access_class: AccessClass
    producing_task_id: UUID
    tool_invocation_id: UUID
    cited: bool
    drift: DriftStatus
    drift_checked_at: datetime | None = None
    drift_detail: str | None = None


class ManifestInvocation(BaseModel):
    """One tool call as the manifest lists it: who allowed it, and how it ended."""

    model_config = {"frozen": True}

    id: UUID
    mcp_server: str
    capability: str
    argument_digest: str
    policy_version: str
    authorization_decision: AuthorizationDecision
    outcome: InvocationOutcome
    started_at: datetime


class ManifestFinding(BaseModel):
    model_config = {"frozen": True}

    claim: str
    supporting_evidence_ids: list[UUID]
    contradicting_evidence_ids: list[UUID]
    calculation_ids: list[UUID]
    confidence: float
    critic_verdict: CriticVerdict
    reviewer_status: ReviewerStatus


class ProvenanceManifest(BaseModel):
    """Everything needed to trace a published report back to its sources."""

    model_config = {"frozen": True}

    schema_version: str = MANIFEST_SCHEMA_VERSION
    job_id: UUID
    tenant_id: str
    requester_id: str
    question: str
    generated_at: datetime = Field(default_factory=utc_now)
    report_sha256: str
    is_partial: bool
    shortfalls: list[str]
    findings: list[ManifestFinding]
    evidence: list[ManifestEvidence]
    tool_invocations: list[ManifestInvocation]

    @property
    def drifted_evidence_ids(self) -> list[UUID]:
        return [record.id for record in self.evidence if record.drift is DriftStatus.DRIFTED]


class ReportPublication(BaseModel):
    """Where a job's published report and manifest were stored (section 10).

    Each key is the artifact's name within the tenant's own prefix, which is how the
    artifact store is addressed - never a raw storage key.
    """

    model_config = {"frozen": True}

    job_id: UUID
    tenant_id: str = Field(min_length=1, max_length=100)
    published_at: datetime = Field(default_factory=utc_now)
    report_key: str
    markdown_key: str
    manifest_key: str
    evidence_key: str
    report_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    is_partial: bool
    drifted_evidence_ids: list[UUID] = Field(default_factory=list)


def sha256_of(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def report_json(report: ResearchReport) -> bytes:
    """The report's canonical bytes: what is stored, and what its hash is taken over."""
    return report.model_dump_json(indent=2).encode("utf-8")


def build_manifest(
    job: ResearchJob,
    report: ResearchReport,
    *,
    shortfalls: list[str],
    findings: list[FindingRecord],
    evidence: list[EvidenceRecord],
    invocations: list[ToolInvocation],
    checks: list[SourceCheck],
) -> ProvenanceManifest:
    cited = report.cited_evidence_ids
    checked = {check.evidence_id: check for check in checks}

    def listed(record: EvidenceRecord) -> ManifestEvidence:
        check = checked.get(record.id)
        return ManifestEvidence(
            id=record.id,
            source_uri=str(record.source_uri),
            title=record.title,
            author=record.author,
            published_at=record.published_at,
            retrieved_at=record.retrieved_at,
            content_hash=record.content_hash,
            trust_level=record.trust_level,
            access_class=record.access_class,
            producing_task_id=record.producing_task_id,
            tool_invocation_id=record.tool_invocation_id,
            cited=record.id in cited,
            drift=check.status if check is not None else DriftStatus.NOT_CHECKED,
            drift_checked_at=check.checked_at if check is not None else None,
            drift_detail=check.detail if check is not None else None,
        )

    return ProvenanceManifest(
        job_id=job.id,
        tenant_id=job.tenant_id,
        requester_id=job.requester_id,
        question=job.question,
        report_sha256=sha256_of(report_json(report)),
        is_partial=report.is_partial or bool(shortfalls),
        shortfalls=list(shortfalls),
        findings=[
            ManifestFinding(**finding.model_dump(include=set(ManifestFinding.model_fields)))
            for finding in findings
        ],
        evidence=[listed(record) for record in evidence],
        tool_invocations=[
            ManifestInvocation(
                id=invocation.id,
                mcp_server=invocation.mcp_server,
                capability=invocation.capability,
                argument_digest=invocation.argument_digest,
                policy_version=invocation.policy_version,
                authorization_decision=invocation.authorization_decision,
                outcome=invocation.outcome,
                started_at=invocation.started_at,
            )
            for invocation in invocations
        ],
    )


def render_markdown(
    job: ResearchJob,
    report: ResearchReport,
    *,
    shortfalls: list[str],
    evidence: list[EvidenceRecord],
    checks: list[SourceCheck],
) -> str:
    """Render the report for a reader, with numbered sources in place of identifiers.

    Each evidence identifier becomes a footnote number in the order it is first cited,
    and the source list says, per source, when it was captured, its content hash, and
    whether it had changed by the time the report was published.
    """
    by_id = {record.id: record for record in evidence}
    checked = {check.evidence_id: check for check in checks}
    numbers: dict[UUID, int] = {}

    def footnote(match: re.Match[str]) -> str:
        number = numbers.setdefault(UUID(match.group("identifier")), len(numbers) + 1)
        return f"[{number}]"

    lines = [f"# {report.title}", "", f"**Question:** {job.question}", ""]
    if report.is_partial or shortfalls:
        lines += ["> **Partial result.** This report is incomplete:", ">"]
        lines += [f"> - {reason}" for reason in shortfalls or report.omitted_because]
        lines.append("")
    for section in report.sections:
        lines += [f"## {section.heading}", "", CITATION.sub(footnote, section.body), ""]

    lines += ["## Sources", ""]
    for identifier, number in numbers.items():
        record = by_id.get(identifier)
        if record is None:
            lines.append(f"{number}. Unrecorded evidence `{identifier}`")
            continue
        check = checked.get(identifier)
        note = DRIFT_NOTES[check.status if check is not None else DriftStatus.NOT_CHECKED]
        label = f"{record.title} - " if record.title else ""
        lines.append(
            f"{number}. {label}{record.source_uri} (retrieved "
            f"{record.retrieved_at.date().isoformat()}, {record.trust_level.value}, "
            f"`{record.content_hash}`; {note}) `{identifier}`"
        )
    if not numbers:
        lines.append("No sources were cited.")
    return "\n".join(lines) + "\n"
