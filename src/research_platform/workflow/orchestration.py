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
from typing import Protocol
from uuid import UUID

from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    EvidenceSubmission,
    PlannedTask,
    ResearchPlan,
    ResearchReport,
)
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
)
from research_platform.domain.tasks import AgentRole, ResearchTask


class JobsPort(Protocol):
    """The slice of ``ResearchJobService`` the orchestrator needs to persist state.

    Both methods are async so a Temporal-driven run can supply a proxy whose methods
    each call an activity, meaning job state changes travel through the same durable,
    replay-safe path as everything else the pipeline does. ``ResearchJobService`` is
    synchronous, in-process persistence with no durability of its own, so it is adapted
    with ``application.jobs.AsyncJobs`` rather than made to satisfy this directly.
    """

    async def transition(self, tenant_id: str, job_id: UUID, target: JobStatus) -> ResearchJob: ...

    async def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord: ...


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
) -> ResearchOutcome:
    """Drive one research job through the section 9 pipeline to a terminal status."""
    job = await jobs.transition(job.tenant_id, job.id, JobStatus.PLANNING)
    plan = await activities.plan(job)

    job = await jobs.transition(job.tenant_id, job.id, JobStatus.RESEARCHING)
    researcher_tasks = [
        _task_from(planned, job=job)
        for planned in plan.tasks
        if planned.assigned_agent is AgentRole.RESEARCHER
    ]
    evidence, unmet = await _collect_evidence(
        job, researcher_tasks, jobs=jobs, activities=activities
    )

    if not evidence:
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.PARTIAL)
        return ResearchOutcome(job=job, report=None, evidence=evidence)

    critique: CriticReview
    review_cycle = 0
    while True:
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.ANALYZING)
        analysis = await activities.analyze(job, evidence)
        critique = await activities.critique(job, analysis, evidence)

        if not critique.requires_reviewer:
            break

        job = await jobs.transition(job.tenant_id, job.id, JobStatus.REVIEW_REQUIRED)
        decision = await activities.await_reviewer_decision(job, critique)

        if decision is ReviewerDecision.APPROVE:
            break
        if decision is ReviewerDecision.REJECT:
            job = await jobs.transition(job.tenant_id, job.id, JobStatus.FAILED)
            return ResearchOutcome(job=job, report=None, evidence=evidence)

        review_cycle += 1
        job = await jobs.transition(job.tenant_id, job.id, JobStatus.RESEARCHING)
        if review_cycle > MAX_REVIEW_CYCLES:
            job = await jobs.transition(job.tenant_id, job.id, JobStatus.PARTIAL)
            return ResearchOutcome(job=job, report=None, evidence=evidence)

        more_evidence, unmet = await _collect_evidence(
            job, researcher_tasks, jobs=jobs, activities=activities
        )
        evidence = evidence + more_evidence

    job = await jobs.transition(job.tenant_id, job.id, JobStatus.REPORTING)
    report = await activities.report(job, critique, evidence)

    final_status = JobStatus.PARTIAL if (report.is_partial or unmet) else JobStatus.COMPLETED
    job = await jobs.transition(job.tenant_id, job.id, final_status)
    return ResearchOutcome(job=job, report=report, evidence=evidence)


async def _collect_evidence(
    job: ResearchJob,
    tasks: list[ResearchTask],
    *,
    jobs: JobsPort,
    activities: OrchestrationActivities,
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
        if isinstance(result, BaseException):
            unmet.extend(task.evidence_requirements or [task.objective])
            continue
        for record in result.records:
            records.append(await jobs.add_evidence(job.tenant_id, job.id, record))
        unmet.extend(result.unmet_requirements)
    return records, unmet
