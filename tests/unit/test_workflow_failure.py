"""A job that cannot finish says so: it never stays in a working status with no work."""

from uuid import UUID, uuid4

import pytest
from temporalio.exceptions import ApplicationError

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
from research_platform.application.jobs import AsyncJobs, InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.workflow.orchestration import (
    OrchestrationActivities,
    ReviewerDecision,
    run_research_job,
)

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


def report_citing(identifier: UUID) -> ResearchReport:
    return ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body=f"{CLAIM[:-1]} [{identifier}].")],
    )


class Crew:
    """The five steps, each replaceable, so one test can break exactly one of them."""

    def __init__(self) -> None:
        self.unmet: list[str] = []
        self.plan_error: Exception | None = None
        self.analyze_error: Exception | None = None
        self.report_override: ResearchReport | None = None

    async def plan(self, _job: ResearchJob) -> ResearchPlan:
        if self.plan_error is not None:
            raise self.plan_error
        return PLAN

    async def research(self, _job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return EvidenceSubmission(
            records=[
                EvidenceRecordCreate(
                    excerpt="Vendor pricing is 20 USD per seat.",
                    source_uri="https://vendor.test/pricing",
                    content_hash=f"sha256:{'0' * 64}",
                    producing_task_id=task.id,
                    tool_invocation_id=uuid4(),
                )
            ],
            unmet_requirements=self.unmet,
        )

    async def analyze(self, _job: ResearchJob, evidence: list[EvidenceRecord]) -> AnalysisResult:
        if self.analyze_error is not None:
            raise self.analyze_error
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
            ]
        )

    async def report(
        self, _job: ResearchJob, _critique: CriticReview, evidence: list[EvidenceRecord]
    ) -> ResearchReport:
        return self.report_override or report_citing(evidence[0].id)

    async def reviewer(self, *_args: object) -> ReviewerDecision:
        raise AssertionError("no reviewer decision should have been awaited")

    @property
    def activities(self) -> OrchestrationActivities:
        return OrchestrationActivities(
            plan=self.plan,
            research=self.research,
            analyze=self.analyze,
            critique=self.critique,
            report=self.report,
            await_reviewer_decision=self.reviewer,
        )


def started() -> tuple[ResearchJobService, ResearchJob]:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    return service, job


async def test_a_step_that_cannot_be_completed_ends_the_job_as_failed_with_the_reason() -> None:
    service, job = started()
    crew = Crew()
    crew.analyze_error = RuntimeError("the model provider refused the request")

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert outcome.job.status is JobStatus.FAILED
    assert outcome.failure == "RuntimeError: the model provider refused the request"
    assert outcome.report is None
    stored = service.get(TENANT, job.id)
    assert stored.status is JobStatus.FAILED
    assert stored.status_detail == outcome.failure


async def test_a_failure_that_crossed_an_activity_boundary_is_named_once() -> None:
    """Temporal's wrapper already renders as "Type: message"; the reason must not double it."""
    service, job = started()
    crew = Crew()
    crew.analyze_error = ApplicationError(
        "the model provider refused the request", type="ProviderRefused"
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert outcome.failure == "ProviderRefused: the model provider refused the request"


async def test_a_failure_in_the_very_first_step_still_ends_the_job_as_failed() -> None:
    service, job = started()
    crew = Crew()
    crew.plan_error = ValueError("no plan could be produced")

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert outcome.job.status is JobStatus.FAILED
    assert service.get(TENANT, job.id).status is JobStatus.FAILED


async def test_a_long_failure_reason_is_stored_truncated_rather_than_refused() -> None:
    service, job = started()
    crew = Crew()
    crew.analyze_error = RuntimeError("x" * 5_000)

    await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert len(service.get(TENANT, job.id).status_detail or "") == 1_000


async def test_a_failure_that_cannot_even_be_recorded_is_raised_not_hidden() -> None:
    """If the store is what failed, pretending the job was marked failed would be a lie."""

    class BrokenJobs:
        async def transition(self, *_args: object, **_kwargs: object) -> ResearchJob:
            raise ConnectionError("database unreachable")

        async def add_evidence(self, *_args: object) -> EvidenceRecord:
            raise AssertionError("no evidence should be written")

    _service, job = started()

    with pytest.raises(ConnectionError, match="database unreachable"):
        await run_research_job(job, jobs=BrokenJobs(), activities=Crew().activities)


async def test_a_completed_job_carries_no_explanation() -> None:
    service, job = started()

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=Crew().activities)

    assert outcome.job.status is JobStatus.COMPLETED
    assert outcome.job.status_detail is None
    assert outcome.failure is None


async def test_a_partial_job_says_which_requirements_went_unmet() -> None:
    service, job = started()
    crew = Crew()
    crew.unmet = ["enterprise pricing"]

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.job.status_detail == "unmet requirement: enterprise pricing"


async def test_a_report_citing_unrecorded_evidence_is_never_called_complete() -> None:
    """The last check before completion, independent of the one inside the activity."""
    service, job = started()
    crew = Crew()
    invented = uuid4()
    crew.report_override = report_citing(invented)

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.job.status_detail == f"cites unrecorded evidence {invented}"


async def test_a_partial_report_says_what_it_omitted() -> None:
    service, job = started()
    crew = Crew()
    crew.report_override = ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body="This suggests pricing is unclear.")],
        is_partial=True,
        omitted_because=["no enterprise pricing was published"],
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=crew.activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.job.status_detail == "omitted: no enterprise pricing was published"
