"""The section 9 core workflow, expressed independently of Temporal.

Temporal decides *when* a step actually runs - inside a durable workflow, replayed from
history after a worker restart, suspended on a signal without holding a worker open
while a reviewer is away (section 12). This module decides what the steps *are*: plan,
discover, research in parallel, analyze, let the critic flag what needs a reviewer, wait
for that reviewer, report, and land on a clearly labelled partial result rather than a
false success when evidence or review runs out (section 12 again). Every step here is a
pure async function over the domain models and five injected activities, so the whole
pipeline - branching, status transitions, the bounded review-cycle limit - is tested
without a Temporal server. ``research_platform.workflow.temporal_workflow`` is the thin
adapter that runs each activity for real and lets a Temporal signal answer
``await_reviewer_decision``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Protocol
from uuid import UUID

from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    EvidenceSubmission,
    PlannedTask,
    ResearchPlan,
    ResearchReport,
    unsupported_citations,
)
from research_platform.application.publication import ReportPublication, derive_findings
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    Finding,
    FindingRecord,
    JobStatus,
    ResearchJob,
    ReviewerStatus,
    utc_now,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.observability.metrics import PlatformMetrics, get_metrics
from research_platform.observability.tracing import get_tracer


class JobsPort(Protocol):
    """The slice of ``ResearchJobService`` the orchestrator needs to persist state.

    Both methods are async so a Temporal-driven run can supply a proxy whose methods
    each call an activity, meaning job state changes travel through the same durable,
    replay-safe path as everything else the pipeline does. ``ResearchJobService`` is
    synchronous, in-process persistence with no durability of its own, so it is adapted
    with ``application.jobs.AsyncJobs`` rather than made to satisfy this directly.
    """

    async def transition(
        self, tenant_id: str, job_id: UUID, target: JobStatus, detail: str | None = None
    ) -> ResearchJob: ...

    async def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord: ...

    async def record_findings(
        self, tenant_id: str, job_id: UUID, findings: list[Finding]
    ) -> list[FindingRecord]: ...


class ReviewerDecision(StrEnum):
    """What a reviewer does with a job the critic sent for review."""

    APPROVE = "approve"
    REQUEST_MORE_RESEARCH = "request_more_research"
    REJECT = "reject"


@dataclass(frozen=True)
class OrchestrationActivities:
    """The five section 7 roles, as async callables the orchestrator drives.

    Each callable does whatever it takes to produce a validated contract - an LLM call
    through ``BoundedSchemaCorrection``, tool calls through the capability gateway - and
    is expected to run inside a durable Temporal activity in production. The
    orchestrator itself never makes an LLM call or a network call directly, so it stays
    deterministic and safe to drive from a Temporal workflow.
    """

    plan: Callable[[ResearchJob], Awaitable[ResearchPlan]]
    research: Callable[[ResearchJob, ResearchTask], Awaitable[EvidenceSubmission]]
    analyze: Callable[[ResearchJob, list[EvidenceRecord]], Awaitable[AnalysisResult]]
    critique: Callable[[ResearchJob, AnalysisResult, list[EvidenceRecord]], Awaitable[CriticReview]]
    report: Callable[[ResearchJob, CriticReview, list[EvidenceRecord]], Awaitable[ResearchReport]]
    await_reviewer_decision: Callable[[ResearchJob, CriticReview], Awaitable[ReviewerDecision]]
    publish: (
        Callable[[ResearchJob, ResearchReport, list[str]], Awaitable[ReportPublication]] | None
    ) = None
    """Exports the report and its provenance manifest (section 9's last step).

    ``None`` runs the pipeline without exporting anything, which is what a caller that
    has no artifact store - a unit test of the branching, mostly - wants.
    """


MAX_REVIEW_CYCLES = 2
"""How many times a reviewer may send a job back for more research before it stops.

Bounded for the reason a schema correction is bounded (section 12): a reviewer who keeps
asking for coverage a second research pass could not produce should end the job with a
clearly labelled partial result rather than loop it indefinitely.
"""


@dataclass(frozen=True)
class ResearchOutcome:
    """What one pipeline run produced, alongside the job's final status."""

    job: ResearchJob
    report: ResearchReport | None
    evidence: list[EvidenceRecord]
    failure: str | None = None
    publication: ReportPublication | None = None


async def _record_findings(
    jobs: JobsPort,
    job: ResearchJob,
    analysis: AnalysisResult,
    critique: CriticReview,
    reviewer_status: ReviewerStatus,
) -> None:
    await jobs.record_findings(
        job.tenant_id, job.id, derive_findings(analysis, critique, reviewer_status)
    )


def _task_from(planned: PlannedTask, *, job: ResearchJob) -> ResearchTask:
    return ResearchTask(
        job_id=job.id,
        tenant_id=job.tenant_id,
        objective=planned.objective,
        assigned_agent=planned.assigned_agent,
        evidence_requirements=list(planned.evidence_requirements),
        source_restrictions=list(planned.source_restrictions),
    )


async def run_research_job(
    job: ResearchJob,
    *,
    jobs: JobsPort,
    activities: OrchestrationActivities,
    metrics: PlatformMetrics | None = None,
) -> ResearchOutcome:
    """Drive one research job through the section 9 pipeline to a terminal status."""
    metrics = metrics or get_metrics()
    tracer = get_tracer()
    # Started without being made the current span. A workflow function does not own the
    # context it runs in: Temporal may evict it from a worker's cache mid-await and close
    # it from a different context, where detaching a context token raises. Nothing is
    # lost by this - the activities it drives run in their own contexts regardless.
    span = tracer.start_span("research.job")
    span.set_attribute("job.id", str(job.id))
    span.set_attribute("tenant.id", job.tenant_id)
    metrics.active_jobs.add(1, {"tenant.id": job.tenant_id})
    try:
        return await _run_research_job(job, jobs=jobs, activities=activities, metrics=metrics)
    except Exception as error:
        return await _fail(job, jobs, error)
    finally:
        metrics.active_jobs.add(-1, {"tenant.id": job.tenant_id})
        span.end()


def _describe(error: BaseException) -> str:
    """Name what went wrong without the wrapper an activity failure arrives in."""
    cause = getattr(error, "cause", None) or error
    # A failure that crossed an activity boundary keeps its original class name in
    # ``type``; the exception object itself is the transport's generic wrapper.
    name = getattr(cause, "type", None) or type(cause).__name__
    return f"{name}: {cause}"


async def _fail(job: ResearchJob, jobs: JobsPort, error: Exception) -> ResearchOutcome:
    """End the job as failed, with the reason, when a step could not be completed.

    A step fails here only after its own bounded retries are spent - schema correction
    inside the activity, then the activity's retry policy - so this is not a place to
    try again. What matters is that the job does not stay in a working status forever
    with nothing working on it: it is marked failed and says why. If even that cannot
    be recorded, the original error is raised so the failure is not hidden.
    """
    reason = _describe(error)
    try:
        failed = await jobs.transition(job.tenant_id, job.id, JobStatus.FAILED, reason)
    except Exception:
        raise error from None
    return ResearchOutcome(job=failed, report=None, evidence=[], failure=reason)


async def _run_research_job(
    job: ResearchJob,
    *,
    jobs: JobsPort,
    activities: OrchestrationActivities,
    metrics: PlatformMetrics,
) -> ResearchOutcome:
    job = await jobs.transition(job.tenant_id, job.id, JobStatus.PLANNING)
    metrics.job_queue_age.record(
        (utc_now() - job.created_at).total_seconds(), {"tenant.id": job.tenant_id}
    )
    plan = await activities.plan(job)

    job = await jobs.transition(job.tenant_id, job.id, JobStatus.RESEARCHING)
    researcher_tasks = [
        _task_from(planned, job=job)
        for planned in plan.tasks
        if planned.assigned_agent is AgentRole.RESEARCHER
    ]
    evidence, unmet = await _collect_evidence(
        job, researcher_tasks, jobs=jobs, activities=activities, metrics=metrics
    )

    if not evidence:
        job = await jobs.transition(
            job.tenant_id, job.id, JobStatus.PARTIAL, "no evidence could be collected"
        )
        return ResearchOutcome(job=job, report=None, evidence=evidence)

    critique: CriticReview
    review_cycle = 0
    while True:
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.ANALYZING)
        analysis = await activities.analyze(job, evidence)
        critique = await activities.critique(job, analysis, evidence)

        # Findings are stored as soon as the critic has judged them, so a reviewer - and
        # anyone reading the job while it waits - sees what is being decided.
        record = partial(_record_findings, jobs, job, analysis, critique)
        if not critique.requires_reviewer:
            await record(ReviewerStatus.NOT_REQUIRED)
            break
        await record(ReviewerStatus.PENDING)

        job = await jobs.transition(job.tenant_id, job.id, JobStatus.REVIEW_REQUIRED)
        review_started = utc_now()
        decision = await activities.await_reviewer_decision(job, critique)
        metrics.approval_wait_time.record(
            (utc_now() - review_started).total_seconds(), {"tenant.id": job.tenant_id}
        )

        if decision is ReviewerDecision.APPROVE:
            await record(ReviewerStatus.APPROVED)
            break
        if decision is ReviewerDecision.REJECT:
            await record(ReviewerStatus.REJECTED)
            job = await jobs.transition(
                job.tenant_id, job.id, JobStatus.FAILED, "rejected by the reviewer"
            )
            return ResearchOutcome(job=job, report=None, evidence=evidence)

        review_cycle += 1
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.RESEARCHING)
        if review_cycle > MAX_REVIEW_CYCLES:
            job = await jobs.transition(
                job.tenant_id,
                job.id,
                JobStatus.PARTIAL,
                f"the reviewer asked for more research more than {MAX_REVIEW_CYCLES} times",
            )
            return ResearchOutcome(job=job, report=None, evidence=evidence)

        more_evidence, unmet = await _collect_evidence(
            job, researcher_tasks, jobs=jobs, activities=activities, metrics=metrics
        )
        evidence = evidence + more_evidence

    job = await jobs.transition(job.tenant_id, job.id, JobStatus.REPORTING)
    report = await activities.report(job, critique, evidence)
    missing_citations = unsupported_citations(
        report, available_evidence_ids=frozenset(record.id for record in evidence)
    )
    if missing_citations:
        metrics.unsupported_citations.add(len(missing_citations), {"tenant.id": job.tenant_id})

    # The report activity already refuses a citation to evidence that was never
    # recorded. It is checked again here because this is the last point before a job is
    # called complete, and "every published claim is linked to evidence" (section 14)
    # should not depend on a single check having run.
    shortfalls = [
        *(f"unmet requirement: {requirement}" for requirement in unmet),
        *(f"omitted: {reason}" for reason in report.omitted_because),
        *(
            f"cites unrecorded evidence {identifier}"
            for identifier in sorted(missing_citations, key=str)
        ),
    ]
    # Exported before the job is given its final status: a job is not called complete
    # while its report and manifest are still unwritten.
    publication = (
        await activities.publish(job, report, shortfalls)
        if activities.publish is not None
        else None
    )
    if shortfalls:
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.PARTIAL, "; ".join(shortfalls))
    else:
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.COMPLETED)
    return ResearchOutcome(job=job, report=report, evidence=evidence, publication=publication)


async def _collect_evidence(
    job: ResearchJob,
    tasks: list[ResearchTask],
    *,
    jobs: JobsPort,
    activities: OrchestrationActivities,
    metrics: PlatformMetrics,
) -> tuple[list[EvidenceRecord], list[str]]:
    """Run every researcher task in parallel and persist what each one found.

    ``EvidenceSubmission`` requires at least one record, so a researcher who truly found
    nothing has nothing valid to return and its activity raises instead (bounded schema
    correction exhausted, a budget denial, or any other failure). That failure is caught
    here rather than allowed to fail the whole job: the task's own evidence requirements
    become unmet requirements, and every other task keeps running beside it (section 9's
    parallel evidence gathering, and section 12's requirement that a budget or a single
    failure stop that task's work, not the whole job).
    """
    if not tasks:
        return [], []
    results = await asyncio.gather(
        *(activities.research(job, task) for task in tasks), return_exceptions=True
    )
    records: list[EvidenceRecord] = []
    unmet: list[str] = []
    for task, result in zip(tasks, results, strict=True):
        attributes = {"tenant.id": job.tenant_id, "agent_role": task.assigned_agent.value}
        if isinstance(result, BaseException):
            unmet.extend(task.evidence_requirements or [task.objective])
            metrics.task_completions.add(1, attributes | {"outcome": "failed"})
            continue
        for record in result.records:
            records.append(await jobs.add_evidence(job.tenant_id, job.id, record))
        unmet.extend(result.unmet_requirements)
        metrics.task_completions.add(1, attributes | {"outcome": "succeeded"})
    return records, unmet
