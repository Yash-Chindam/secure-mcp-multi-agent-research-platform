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
from research_platform.application.jobs import AsyncJobs, InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.workflow.orchestration import (
    MAX_REVIEW_CYCLES,
    OrchestrationActivities,
    ReviewerDecision,
    run_research_job,
)

TENANT = "acme"


def new_job(jobs: ResearchJobService) -> ResearchJob:
    return jobs.create(
        TENANT, "requester-1", ResearchJobCreate(question="What does the vendor charge?")
    )


def evidence(job_id: UUID, task_id: UUID, note: str = "pricing") -> EvidenceRecordCreate:
    return EvidenceRecordCreate(
        excerpt=f"The {note} page states 20 USD per seat.",
        source_uri="https://vendor.test/pricing",
        content_hash=f"sha256:{'0' * 64}",
        producing_task_id=task_id,
        tool_invocation_id=uuid4(),
    )


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


def submission_for(task: ResearchTask) -> EvidenceSubmission:
    return EvidenceSubmission(records=[evidence(task.job_id, task.id)])


def analysis_for(evidence_ids: list[UUID]) -> AnalysisResult:
    return AnalysisResult(
        findings=[
            ProposedFinding(
                claim="The vendor charges 20 USD per seat.",
                supporting_evidence_ids=evidence_ids,
                confidence=0.9,
            )
        ]
    )


def supported_review(analysis: AnalysisResult) -> CriticReview:
    return CriticReview(
        verdicts=[
            ClaimVerdict(
                claim=finding.claim,
                verdict=CriticVerdict.SUPPORTED,
                reasoning="Matches the source.",
            )
            for finding in analysis.findings
        ]
    )


def review_needing_a_reviewer(analysis: AnalysisResult) -> CriticReview:
    return CriticReview(
        verdicts=[
            ClaimVerdict(
                claim=finding.claim,
                verdict=CriticVerdict.SUPPORTED,
                reasoning="Matches the source.",
            )
            for finding in analysis.findings
        ],
        coverage_gaps=["no evidence of enterprise-tier pricing"],
    )


def report_for(evidence_ids: list[UUID]) -> ResearchReport:
    citation = str(evidence_ids[0])
    return ResearchReport(
        title="Vendor pricing",
        sections=[
            ReportSection(
                heading="Pricing", body=f"The vendor charges 20 USD per seat [{citation}]."
            )
        ],
    )


def build_service() -> ResearchJobService:
    return ResearchJobService(InMemoryJobRepository())


async def _unreached_reviewer_call(*_args: object) -> ReviewerDecision:
    raise AssertionError("no reviewer decision should have been awaited")


@pytest.mark.asyncio
async def test_the_happy_path_completes_without_a_reviewer() -> None:
    service = build_service()
    job = new_job(service)

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return supported_review(analysis)

    async def report(
        _job: ResearchJob, _critique: CriticReview, evidence_records: list
    ) -> ResearchReport:  # type: ignore[type-arg]
        return report_for([record.id for record in evidence_records])

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=_unreached_reviewer_call,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.COMPLETED
    assert outcome.report is not None
    assert not outcome.report.is_partial
    assert len(outcome.evidence) == 1


@pytest.mark.asyncio
async def test_a_job_with_no_evidence_lands_on_partial_without_analysis() -> None:
    service = build_service()
    job = new_job(service)
    analyze_was_called = False

    async def research(_job: ResearchJob, _task: ResearchTask) -> EvidenceSubmission:
        raise RuntimeError("no approved source could be reached")

    async def analyze(*_args: object) -> AnalysisResult:
        nonlocal analyze_was_called
        analyze_was_called = True
        raise AssertionError("analysis should not run without evidence")

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,  # type: ignore[arg-type]
        critique=_unreached_critique,  # type: ignore[arg-type]
        report=_unreached_report,  # type: ignore[arg-type]
        await_reviewer_decision=_unreached_reviewer_call,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.report is None
    assert not analyze_was_called


@pytest.mark.asyncio
async def test_a_reviewer_who_approves_lets_the_job_complete() -> None:
    service = build_service()
    job = new_job(service)
    statuses: list[JobStatus] = []

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        current_job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        statuses.append(current_job.status)
        return review_needing_a_reviewer(analysis)

    async def await_reviewer_decision(
        current_job: ResearchJob, _critique: CriticReview
    ) -> ReviewerDecision:
        assert current_job.status is JobStatus.REVIEW_REQUIRED
        return ReviewerDecision.APPROVE

    async def report(
        _job: ResearchJob, _critique: CriticReview, evidence_records: list
    ) -> ResearchReport:  # type: ignore[type-arg]
        return report_for([record.id for record in evidence_records])

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=await_reviewer_decision,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.COMPLETED
    assert statuses == [JobStatus.ANALYZING]


@pytest.mark.asyncio
async def test_a_reviewer_who_rejects_fails_the_job() -> None:
    service = build_service()
    job = new_job(service)

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return review_needing_a_reviewer(analysis)

    async def await_reviewer_decision(*_args: object) -> ReviewerDecision:
        return ReviewerDecision.REJECT

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=_unreached_report,  # type: ignore[arg-type]
        await_reviewer_decision=await_reviewer_decision,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.FAILED
    assert outcome.report is None


@pytest.mark.asyncio
async def test_a_reviewer_requesting_more_research_gets_a_second_pass() -> None:
    service = build_service()
    job = new_job(service)
    research_calls = 0
    decisions = iter([ReviewerDecision.REQUEST_MORE_RESEARCH, ReviewerDecision.APPROVE])

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        nonlocal research_calls
        research_calls += 1
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return review_needing_a_reviewer(analysis)

    async def await_reviewer_decision(*_args: object) -> ReviewerDecision:
        return next(decisions)

    async def report(
        _job: ResearchJob, _critique: CriticReview, evidence_records: list
    ) -> ResearchReport:  # type: ignore[type-arg]
        return report_for([record.id for record in evidence_records])

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=await_reviewer_decision,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.COMPLETED
    assert research_calls == 2
    assert len(outcome.evidence) == 2


@pytest.mark.asyncio
async def test_repeated_requests_for_more_research_end_in_partial_not_a_loop() -> None:
    service = build_service()
    job = new_job(service)

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return review_needing_a_reviewer(analysis)

    async def await_reviewer_decision(*_args: object) -> ReviewerDecision:
        return ReviewerDecision.REQUEST_MORE_RESEARCH

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=_unreached_report,  # type: ignore[arg-type]
        await_reviewer_decision=await_reviewer_decision,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.report is None


@pytest.mark.asyncio
async def test_a_report_marked_partial_by_the_reporter_leaves_the_job_partial() -> None:
    service = build_service()
    job = new_job(service)

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return supported_review(analysis)

    async def report(
        _job: ResearchJob, _critique: CriticReview, evidence_records: list
    ) -> ResearchReport:  # type: ignore[type-arg]
        citation = str(evidence_records[0].id)
        return ResearchReport(
            title="Vendor pricing",
            sections=[
                ReportSection(
                    heading="Pricing", body=f"The vendor charges 20 USD per seat [{citation}]."
                )
            ],
            is_partial=True,
            omitted_because=["enterprise pricing was never approved for release"],
        )

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=_unreached_reviewer_call,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.report is not None
    assert outcome.report.is_partial


@pytest.mark.asyncio
async def test_evidence_records_are_persisted_against_the_job() -> None:
    service = build_service()
    job = new_job(service)

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return supported_review(analysis)

    async def report(
        _job: ResearchJob, _critique: CriticReview, evidence_records: list
    ) -> ResearchReport:  # type: ignore[type-arg]
        return report_for([record.id for record in evidence_records])

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN),
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=_unreached_reviewer_call,
    )

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert len(service.list_evidence(TENANT, job.id)) == 1


PLAN_WITH_TWO_RESEARCH_TASKS = ResearchPlan(
    tasks=[
        PlannedTask(
            objective="Collect the vendor pricing page",
            assigned_agent=AgentRole.RESEARCHER,
            evidence_requirements=["a dated pricing page"],
        ),
        PlannedTask(
            objective="Collect the enterprise pricing page",
            assigned_agent=AgentRole.RESEARCHER,
            evidence_requirements=["a dated enterprise pricing page"],
        ),
    ],
    rationale="Both tiers must be sourced before they can be compared.",
)

PLAN_WITH_NO_RESEARCH_TASKS = ResearchPlan(
    tasks=[
        PlannedTask(
            objective="Summarize what is already known",
            assigned_agent=AgentRole.ANALYST,
            evidence_requirements=["nothing new"],
        )
    ],
    rationale="No new evidence is required for this assignment.",
)


@pytest.mark.asyncio
async def test_a_plan_with_no_researcher_tasks_lands_on_partial() -> None:
    service = build_service()
    job = new_job(service)

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN_WITH_NO_RESEARCH_TASKS),
        research=_unreached_research,  # type: ignore[arg-type]
        analyze=_unreached_analyze,  # type: ignore[arg-type]
        critique=_unreached_critique,  # type: ignore[arg-type]
        report=_unreached_report,  # type: ignore[arg-type]
        await_reviewer_decision=_unreached_reviewer_call,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.evidence == []


@pytest.mark.asyncio
async def test_one_failing_task_does_not_stop_the_others() -> None:
    service = build_service()
    job = new_job(service)

    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        if "enterprise" in task.objective:
            raise RuntimeError("enterprise pricing page requires a login")
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence_records: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence_records])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return supported_review(analysis)

    async def report(
        _job: ResearchJob, _critique: CriticReview, evidence_records: list
    ) -> ResearchReport:  # type: ignore[type-arg]
        return report_for([record.id for record in evidence_records])

    activities = OrchestrationActivities(
        plan=lambda _job: _async(PLAN_WITH_TWO_RESEARCH_TASKS),
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=_unreached_reviewer_call,
    )

    outcome = await run_research_job(job, jobs=AsyncJobs(service), activities=activities)

    # The other task's evidence still produces a report, but the job stays labelled
    # partial: the enterprise pricing requirement genuinely went unmet (section 12).
    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.report is not None
    assert len(outcome.evidence) == 1


def test_the_review_cycle_limit_is_small_and_positive() -> None:
    assert 0 < MAX_REVIEW_CYCLES <= 5


async def _unreached_research(*_args: object) -> EvidenceSubmission:
    raise AssertionError("research should not have run")


async def _unreached_analyze(*_args: object) -> AnalysisResult:
    raise AssertionError("analysis should not have run")


async def _unreached_critique(*_args: object) -> CriticReview:
    raise AssertionError("critique should not have run")


async def _unreached_report(*_args: object) -> ResearchReport:
    raise AssertionError("report should not have run")


async def _async(value: ResearchPlan) -> ResearchPlan:
    return value
