"""The Temporal adapter for the section 9 orchestration.

Nothing here decides what the pipeline does - ``research_platform.workflow.orchestration``
does, and is fully tested without a Temporal server. This module only decides how each
step actually runs: an activity is looked up by the string name
``research_platform.workflow.activities`` registers it under, rather than imported
directly, so this file's own non-determinism surface stays small even though the class
itself opts out of Temporal's workflow sandbox (see ``ResearchJobWorkflow`` for why). A
reviewer's decision arrives through a signal and is waited for with
``workflow.wait_condition``, which suspends the workflow durably without holding a worker
open while a person is away (section 12).
"""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from temporalio import workflow
from temporalio.common import RetryPolicy

from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    EvidenceSubmission,
    ResearchPlan,
    ResearchReport,
)
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchJob,
)
from research_platform.domain.tasks import ResearchTask
from research_platform.workflow.orchestration import (
    OrchestrationActivities,
    ResearchOutcome,
    ReviewerDecision,
    run_research_job,
)

AGENT_ACTIVITY_TIMEOUT = timedelta(minutes=10)
"""How long one agent call - including its own bounded schema-correction retries - may
run before Temporal considers the activity itself to have failed."""

AGENT_RETRY_POLICY = RetryPolicy(maximum_attempts=3)
"""Temporal's own retry, for a transient failure such as a dropped connection to the LLM
provider. The bounded schema-correction retries inside the activity are separate and
much narrower (section 12) - this is not a second copy of that budget."""

JOB_ACTIVITY_TIMEOUT = timedelta(seconds=30)
"""Persisting a status transition or a piece of evidence is local, fast persistence."""

JOB_RETRY_POLICY = RetryPolicy(maximum_attempts=5)
"""Every activity call needs an explicit, bounded retry policy.

Temporal's default is to retry an activity indefinitely - with backoff, but with no
attempt limit - until its schedule-to-close timeout elapses. Leaving that default in
place turns any persistently failing call (a bug, not a transient blip) into an
unbounded retry storm rather than a clean failure, which is exactly the runaway behavior
section 12 asks every bounded retry in this platform to avoid."""


@workflow.defn(name="ResearchJobWorkflow", sandboxed=False)
class ResearchJobWorkflow:
    """Drives one research job through the section 9 pipeline, durably.

    Implements ``research_platform.workflow.orchestration.JobsPort`` itself, so job state
    changes travel through the ``transition_job``/``add_job_evidence`` activities rather
    than touching a repository directly from inside the (replayed) workflow function.

    ``sandboxed=False``: Temporal's workflow sandbox re-imports a workflow's module in an
    isolated environment to catch non-determinism, but its own import hooking collides
    with ``beartype.claw`` - a dependency pulled in transitively through CrewAI, which is
    already imported elsewhere in the same worker process - raising a circular-import
    error before the workflow ever runs. Determinism here does not depend on the sandbox
    to enforce it: every non-deterministic operation already lives behind
    ``execute_activity``, called by string name, and the pipeline logic itself
    (``research_platform.workflow.orchestration.run_research_job``) is pure and fully
    unit tested on its own.
    """

    def __init__(self) -> None:
        self._reviewer_decision: ReviewerDecision | None = None

    @workflow.run
    async def run(self, job: ResearchJob) -> ResearchOutcome:
        activities = OrchestrationActivities(
            plan=self._plan,
            research=self._research,
            analyze=self._analyze,
            critique=self._critique,
            report=self._report,
            await_reviewer_decision=self._await_reviewer_decision,
        )
        return await run_research_job(job, jobs=self, activities=activities)

    # -- JobsPort, backed by durable activities --

    async def transition(self, tenant_id: str, job_id: UUID, target: JobStatus) -> ResearchJob:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "transition_job",
            args=[tenant_id, job_id, target],
            start_to_close_timeout=JOB_ACTIVITY_TIMEOUT,
            retry_policy=JOB_RETRY_POLICY,
            result_type=ResearchJob,
        )

    async def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "add_job_evidence",
            args=[tenant_id, job_id, command],
            start_to_close_timeout=JOB_ACTIVITY_TIMEOUT,
            retry_policy=JOB_RETRY_POLICY,
            result_type=EvidenceRecord,
        )

    # -- the five section 7 roles, each a durable activity --

    async def _plan(self, job: ResearchJob) -> ResearchPlan:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "plan_research",
            args=[job],
            start_to_close_timeout=AGENT_ACTIVITY_TIMEOUT,
            retry_policy=AGENT_RETRY_POLICY,
            result_type=ResearchPlan,
        )

    async def _research(self, job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "research_task",
            args=[job, task],
            start_to_close_timeout=AGENT_ACTIVITY_TIMEOUT,
            retry_policy=AGENT_RETRY_POLICY,
            result_type=EvidenceSubmission,
        )

    async def _analyze(self, job: ResearchJob, evidence: list[EvidenceRecord]) -> AnalysisResult:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "analyze_evidence",
            args=[job, evidence],
            start_to_close_timeout=AGENT_ACTIVITY_TIMEOUT,
            retry_policy=AGENT_RETRY_POLICY,
            result_type=AnalysisResult,
        )

    async def _critique(
        self, job: ResearchJob, analysis: AnalysisResult, evidence: list[EvidenceRecord]
    ) -> CriticReview:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "critique_analysis",
            args=[job, analysis, evidence],
            start_to_close_timeout=AGENT_ACTIVITY_TIMEOUT,
            retry_policy=AGENT_RETRY_POLICY,
            result_type=CriticReview,
        )

    async def _report(
        self, job: ResearchJob, critique: CriticReview, evidence: list[EvidenceRecord]
    ) -> ResearchReport:
        return await workflow.execute_activity(  # type: ignore[no-any-return]
            "write_report",
            args=[job, critique, evidence],
            start_to_close_timeout=AGENT_ACTIVITY_TIMEOUT,
            retry_policy=AGENT_RETRY_POLICY,
            result_type=ResearchReport,
        )

    # -- the reviewer, answered by a signal instead of an activity --

    async def _await_reviewer_decision(
        self, job: ResearchJob, critique: CriticReview
    ) -> ReviewerDecision:
        """Suspend until ``submit_reviewer_decision`` is signalled.

        ``workflow.wait_condition`` parks the workflow durably rather than blocking a
        worker: a reviewer who takes a day to respond costs nothing but Temporal's own
        state storage for that day (section 12's "suspend durably without consuming an
        active worker").
        """
        self._reviewer_decision = None
        await workflow.wait_condition(lambda: self._reviewer_decision is not None)
        assert self._reviewer_decision is not None
        return self._reviewer_decision

    @workflow.signal(name="submit_reviewer_decision")
    def submit_reviewer_decision(self, decision: ReviewerDecision) -> None:
        self._reviewer_decision = decision
