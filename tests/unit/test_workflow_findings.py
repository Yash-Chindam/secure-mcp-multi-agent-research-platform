"""The pipeline records its findings as it goes, and exports the report before it finishes."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from research_platform.agents.contracts import (
    AnalysisResult,
    ClaimVerdict,
    CriticReview,
    EvidenceSubmission,
    PlannedTask,
    ProposedFinding,
    ReportSection,
    ResearchPlan,
    ResearchReport,
)
from research_platform.application.artifacts import InMemoryArtifactStore
from research_platform.application.jobs import AsyncJobs, InMemoryJobRepository, ResearchJobService
from research_platform.application.publication import ReportPublication
from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
    ReviewerStatus,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.workflow.orchestration import (
    OrchestrationActivities,
    ReviewerDecision,
    run_research_job,
)
from research_platform.workflow.publishing import PublicationActivities

pytestmark = pytest.mark.asyncio

TENANT = "acme"
CLAIM = "The vendor charges 20 USD per seat."

PLAN = ResearchPlan(
    tasks=[
        PlannedTask(
            objective="Collect the vendor pricing page",
            assigned_agent=AgentRole.RESEARCHER,
            evidence_requirements=["a dated pricing page"],
        )
    ],
    rationale="Pricing must be sourced before it can be compared.",
)


@dataclass
class Crew:
    """The five steps, with a reviewer who answers from a queue of decisions."""

    service: ResearchJobService
    job: ResearchJob
    gaps: list[str] = field(default_factory=list)
    decisions: list[ReviewerDecision] = field(default_factory=list)
    seen_while_waiting: list[list[ReviewerStatus]] = field(default_factory=list)
    status_at_publication: list[JobStatus] = field(default_factory=list)
    publish_error: Exception | None = None
    publishing: PublicationActivities | None = None

    async def plan(self, _job: ResearchJob) -> ResearchPlan:
        return PLAN

    async def research(self, _job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return EvidenceSubmission(
            records=[
                EvidenceRecordCreate(
                    excerpt=f"Vendor pricing is 20 USD per seat ({uuid4()}).",
                    source_uri="https://vendor.test/pricing",
                    content_hash=f"sha256:{uuid4().hex}{uuid4().hex}",
                    producing_task_id=task.id,
                    tool_invocation_id=uuid4(),
                )
            ]
        )

    async def analyze(self, _job: ResearchJob, evidence: list[EvidenceRecord]) -> AnalysisResult:
        return AnalysisResult(
            findings=[
                ProposedFinding(
                    claim=CLAIM,
                    supporting_evidence_ids=[record.id for record in evidence],
                    confidence=0.9,
                )
            ]
        )

    async def critique(
        self, _job: ResearchJob, _analysis: AnalysisResult, _evidence: list[EvidenceRecord]
    ) -> CriticReview:
        return CriticReview(
            verdicts=[
                ClaimVerdict(
                    claim=CLAIM, verdict=CriticVerdict.SUPPORTED, reasoning="Matches the source."
                )
            ],
            coverage_gaps=self.gaps,
        )

    async def report(
        self, _job: ResearchJob, _critique: CriticReview, evidence: list[EvidenceRecord]
    ) -> ResearchReport:
        body = f"{CLAIM[:-1]} [{evidence[0].id}]."
        return ResearchReport(
            title="Vendor pricing", sections=[ReportSection(heading="Pricing", body=body)]
        )

    async def reviewer(self, _job: ResearchJob, _critique: CriticReview) -> ReviewerDecision:
        self.seen_while_waiting.append(self.reviewer_statuses())
        return self.decisions.pop(0)

    async def publish(
        self, job: ResearchJob, report: ResearchReport, shortfalls: list[str]
    ) -> ReportPublication:
        self.status_at_publication.append(self.service.get(TENANT, job.id).status)
        if self.publish_error is not None:
            raise self.publish_error
        assert self.publishing is not None
        return await self.publishing.publish(job, report, shortfalls)

    def reviewer_statuses(self) -> list[ReviewerStatus]:
        return [
            finding.reviewer_status for finding in self.service.list_findings(TENANT, self.job.id)
        ]

    def activities(self, *, publishes: bool = True) -> OrchestrationActivities:
        return OrchestrationActivities(
            plan=self.plan,
            research=self.research,
            analyze=self.analyze,
            critique=self.critique,
            report=self.report,
            await_reviewer_decision=self.reviewer,
            publish=self.publish if publishes else None,
        )


@pytest.fixture
def crew() -> Crew:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    publishing = PublicationActivities(jobs=service, artifacts=InMemoryArtifactStore())
    return Crew(service=service, job=job, publishing=publishing)


async def run(crew: Crew, *, publishes: bool = True) -> tuple[JobStatus, UUID | None]:
    outcome = await run_research_job(
        crew.job, jobs=AsyncJobs(crew.service), activities=crew.activities(publishes=publishes)
    )
    published = outcome.publication.job_id if outcome.publication is not None else None
    return outcome.job.status, published


async def test_findings_are_stored_with_the_critics_verdict_and_their_evidence(crew: Crew) -> None:
    await run(crew)

    [finding] = crew.service.list_findings(TENANT, crew.job.id)
    [record] = crew.service.list_evidence(TENANT, crew.job.id)
    assert finding.claim == CLAIM
    assert finding.supporting_evidence_ids == [record.id]
    assert finding.critic_verdict is CriticVerdict.SUPPORTED
    assert finding.reviewer_status is ReviewerStatus.NOT_REQUIRED
    assert finding.job_id == crew.job.id


async def test_findings_are_pending_while_a_reviewer_decides_and_approved_after(
    crew: Crew,
) -> None:
    crew.gaps = ["enterprise pricing was not sourced"]
    crew.decisions = [ReviewerDecision.APPROVE]

    status, _ = await run(crew)

    assert status is JobStatus.COMPLETED
    assert crew.seen_while_waiting == [[ReviewerStatus.PENDING]]
    assert crew.reviewer_statuses() == [ReviewerStatus.APPROVED]


async def test_a_rejection_is_recorded_on_the_findings_and_nothing_is_published(
    crew: Crew,
) -> None:
    crew.gaps = ["enterprise pricing was not sourced"]
    crew.decisions = [ReviewerDecision.REJECT]

    status, published = await run(crew)

    assert status is JobStatus.FAILED
    assert crew.reviewer_statuses() == [ReviewerStatus.REJECTED]
    assert published is None
    assert crew.service.get_publication(TENANT, crew.job.id) is None


async def test_a_second_research_pass_replaces_the_findings_rather_than_adding_to_them(
    crew: Crew,
) -> None:
    crew.gaps = ["enterprise pricing was not sourced"]
    crew.decisions = [ReviewerDecision.REQUEST_MORE_RESEARCH, ReviewerDecision.APPROVE]

    await run(crew)

    [finding] = crew.service.list_findings(TENANT, crew.job.id)
    assert len(finding.supporting_evidence_ids) == 2
    assert finding.reviewer_status is ReviewerStatus.APPROVED
    assert crew.seen_while_waiting == [[ReviewerStatus.PENDING], [ReviewerStatus.PENDING]]


async def test_the_report_is_exported_before_the_job_is_called_complete(crew: Crew) -> None:
    status, published = await run(crew)

    assert status is JobStatus.COMPLETED
    assert published == crew.job.id
    assert crew.status_at_publication == [JobStatus.REPORTING]
    assert crew.service.get_publication(TENANT, crew.job.id) is not None


async def test_a_report_that_could_not_be_exported_fails_the_job_with_the_reason(
    crew: Crew,
) -> None:
    crew.publish_error = ConnectionError("object store unreachable")

    status, published = await run(crew)

    stored = crew.service.get(TENANT, crew.job.id)
    assert status is JobStatus.FAILED
    assert stored.status_detail == "ConnectionError: object store unreachable"
    assert published is None


async def test_a_pipeline_with_no_publisher_still_completes_and_keeps_its_findings(
    crew: Crew,
) -> None:
    status, published = await run(crew, publishes=False)

    assert status is JobStatus.COMPLETED
    assert published is None
    assert len(crew.service.list_findings(TENANT, crew.job.id)) == 1
