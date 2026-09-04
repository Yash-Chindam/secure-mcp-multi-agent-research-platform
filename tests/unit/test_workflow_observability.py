from uuid import UUID, uuid4

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

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
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.observability.metrics import PlatformMetrics
from research_platform.workflow.orchestration import (
    OrchestrationActivities,
    ReviewerDecision,
    run_research_job,
)

TENANT = "acme"

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


def build_service() -> ResearchJobService:
    return ResearchJobService(InMemoryJobRepository())


def new_job(service: ResearchJobService) -> ResearchJob:
    return service.create(
        TENANT, "requester-1", ResearchJobCreate(question="What does the vendor charge?")
    )


def evidence_for(task: ResearchTask) -> EvidenceRecordCreate:
    return EvidenceRecordCreate(
        excerpt="The vendor charges 20 USD per seat.",
        source_uri="https://vendor.test/pricing",
        content_hash=f"sha256:{'0' * 64}",
        producing_task_id=task.id,
        tool_invocation_id=uuid4(),
    )


def submission_for(task: ResearchTask) -> EvidenceSubmission:
    return EvidenceSubmission(records=[evidence_for(task)])


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
            ClaimVerdict(claim=finding.claim, verdict=CriticVerdict.SUPPORTED, reasoning="Matches.")
            for finding in analysis.findings
        ]
    )


def review_needing_a_reviewer(analysis: AnalysisResult) -> CriticReview:
    return CriticReview(
        verdicts=[
            ClaimVerdict(claim=finding.claim, verdict=CriticVerdict.SUPPORTED, reasoning="Matches.")
            for finding in analysis.findings
        ],
        coverage_gaps=["no evidence of enterprise-tier pricing"],
    )


def report_citing(evidence_id: UUID) -> ResearchReport:
    return ResearchReport(
        title="Vendor pricing",
        sections=[
            ReportSection(
                heading="Pricing", body=f"The vendor charges 20 USD per seat [{evidence_id}]."
            )
        ],
    )


def build_metrics() -> tuple[PlatformMetrics, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return PlatformMetrics(meter=provider.get_meter("test")), reader


def metric_points(reader: InMemoryMetricReader, name: str) -> list[tuple[float, dict[str, object]]]:
    points: list[tuple[float, dict[str, object]]] = []
    data = reader.get_metrics_data()
    if data is None:
        return points
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    value = getattr(point, "value", None)
                    if value is None:
                        value = point.sum
                    points.append((value, dict(point.attributes or {})))
    return points


async def _happy_path_activities(
    *, requires_reviewer: bool = False, decision: ReviewerDecision = ReviewerDecision.APPROVE
) -> OrchestrationActivities:
    async def research(_job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return submission_for(task)

    async def analyze(_job: ResearchJob, evidence: list) -> AnalysisResult:  # type: ignore[type-arg]
        return analysis_for([record.id for record in evidence])

    async def critique(
        _job: ResearchJob, analysis: AnalysisResult, _evidence: list
    ) -> CriticReview:  # type: ignore[type-arg]
        return (
            review_needing_a_reviewer(analysis) if requires_reviewer else supported_review(analysis)
        )

    async def report(_job: ResearchJob, _critique: CriticReview, evidence: list) -> ResearchReport:  # type: ignore[type-arg]
        return report_citing(evidence[0].id)

    async def await_reviewer_decision(*_args: object) -> ReviewerDecision:
        return decision

    async def plan(_job: ResearchJob) -> ResearchPlan:
        return PLAN

    return OrchestrationActivities(
        plan=plan,
        research=research,
        analyze=analyze,
        critique=critique,
        report=report,
        await_reviewer_decision=await_reviewer_decision,
    )


@pytest.mark.asyncio
async def test_active_jobs_is_incremented_then_decremented() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities()

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    points = metric_points(reader, "research.jobs.active")
    assert sum(value for value, _attrs in points) == 0


@pytest.mark.asyncio
async def test_job_queue_age_is_recorded() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities()

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    points = metric_points(reader, "research.jobs.queue_age")
    assert len(points) == 1
    assert points[0][0] >= 0


@pytest.mark.asyncio
async def test_task_completions_are_recorded_by_outcome() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities()

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    points = metric_points(reader, "research.tasks.completed")
    assert points == [
        (1, {"tenant.id": TENANT, "agent_role": "researcher", "outcome": "succeeded"})
    ]


@pytest.mark.asyncio
async def test_a_failing_task_is_recorded_as_failed() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities()

    async def failing_research(_job: ResearchJob, _task: ResearchTask) -> EvidenceSubmission:
        raise RuntimeError("no source reachable")

    activities = OrchestrationActivities(
        plan=activities.plan,
        research=failing_research,
        analyze=activities.analyze,
        critique=activities.critique,
        report=activities.report,
        await_reviewer_decision=activities.await_reviewer_decision,
    )

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    points = metric_points(reader, "research.tasks.completed")
    assert points == [(1, {"tenant.id": TENANT, "agent_role": "researcher", "outcome": "failed"})]


@pytest.mark.asyncio
async def test_approval_wait_time_is_recorded_when_a_reviewer_is_needed() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities(
        requires_reviewer=True, decision=ReviewerDecision.APPROVE
    )

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    points = metric_points(reader, "research.review.wait_time")
    assert len(points) == 1
    assert points[0][0] >= 0


@pytest.mark.asyncio
async def test_no_approval_wait_time_is_recorded_when_no_reviewer_is_needed() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities(requires_reviewer=False)

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    assert metric_points(reader, "research.review.wait_time") == []


@pytest.mark.asyncio
async def test_a_fabricated_citation_is_counted_as_unsupported() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()

    async def report_with_a_fabricated_citation(
        _job: ResearchJob,
        _critique: CriticReview,
        _evidence: list,  # type: ignore[type-arg]
    ) -> ResearchReport:
        return report_citing(uuid4())

    activities = await _happy_path_activities()
    activities = OrchestrationActivities(
        plan=activities.plan,
        research=activities.research,
        analyze=activities.analyze,
        critique=activities.critique,
        report=report_with_a_fabricated_citation,
        await_reviewer_decision=activities.await_reviewer_decision,
    )

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    points = metric_points(reader, "research.report.unsupported_citations")
    assert points == [(1, {"tenant.id": TENANT})]


@pytest.mark.asyncio
async def test_a_correct_citation_is_not_counted_as_unsupported() -> None:
    service = build_service()
    job = new_job(service)
    metrics, reader = build_metrics()
    activities = await _happy_path_activities()

    await run_research_job(job, jobs=AsyncJobs(service), activities=activities, metrics=metrics)

    assert metric_points(reader, "research.report.unsupported_citations") == []
