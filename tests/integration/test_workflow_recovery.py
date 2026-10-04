"""Recovery after controlled failures (sections 12 and 14).

"Worker restarted: resume from Temporal history without duplicating completed side
effects." These tests stop a real worker partway through a job, bring up a different
one with nothing in memory, and check that the job finishes with every completed step
having run exactly once.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from crewai.tools import BaseTool
from support.scripted import StubExecutor, honest_crew
from temporalio import activity
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from research_platform.application.artifacts import InMemoryArtifactStore
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchBudget,
    ResearchJob,
    ReviewerStatus,
)
from research_platform.domain.tasks import AgentRole
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway
from research_platform.worker import registered_activities
from research_platform.workflow import research_workflow
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.orchestration import ReviewerDecision
from research_platform.workflow.publishing import GatewaySourceChecker, PublicationActivities
from research_platform.workflow.research_workflow import TASK_QUEUE, ResearchJobWorkflow

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TENANT = "acme"


class Deployment:
    """What outlives any one worker: the job store, the artifact store, and a tally."""

    def __init__(self, *, requires_reviewer: bool) -> None:
        repository = InMemoryJobRepository()
        self.jobs = ResearchJobService(repository)
        self.artifacts = InMemoryArtifactStore()
        self.agent_calls: Counter[str] = Counter()
        self.job = repository.add(
            ResearchJob(
                tenant_id=TENANT,
                requester_id="requester-1",
                question="What does the vendor charge?",
                budget=ResearchBudget(max_tool_calls=10),
            )
        )
        self._crew = honest_crew(requires_reviewer=requires_reviewer)

    def _build_agent(self, role: AgentRole, tools: list[BaseTool]) -> Any:
        agent = self._crew(role, tools)
        calls = self.agent_calls

        class Counted:
            def kickoff(self, message: str) -> Any:
                calls[role.value] += 1
                return agent.kickoff(message)

        return Counted()

    def activities(self, **replaced: Callable[..., Any]) -> list[Callable[..., Any]]:
        """A fresh set of activity objects, as a newly started worker process would build."""
        registry = default_registry()
        research = ResearchActivities(
            gateway=CapabilityGateway(
                registry=registry, executor=StubExecutor(), audit=self.jobs.record_invocation
            ),
            registry=registry,
            build_agent=self._build_agent,
        )
        persistence = JobActivities(jobs=self.jobs)
        publication = PublicationActivities(
            jobs=self.jobs, artifacts=self.artifacts, sources=GatewaySourceChecker(research.gateway)
        )
        return list((registered_activities(research, persistence, publication) | replaced).values())

    def worker(
        self, client: Client, *, stops_abruptly: bool = False, **replaced: Callable[..., Any]
    ) -> Worker:
        """A worker. One that ``stops_abruptly`` keeps no claim on the jobs it ran.

        Temporal routes a job's next step to the worker that last ran it, and falls
        back to any worker once that one has not answered for a few seconds. The test
        server does not let those seconds pass, so a worker that is about to be stopped
        is told not to hold jobs in the first place - which also means every step it
        runs is rebuilt from history, the thing these tests are about.
        """
        return Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[ResearchJobWorkflow],
            activities=self.activities(**replaced),
            **({"max_cached_workflows": 0} if stops_abruptly else {}),
        )

    def status(self) -> JobStatus:
        return self.jobs.get(TENANT, self.job.id).status

    async def start(self, client: Client) -> WorkflowHandle[Any, Any]:
        return await client.start_workflow(
            ResearchJobWorkflow.run,
            self.job,
            id=f"research-job-{self.job.id}",
            task_queue=TASK_QUEUE,
        )

    async def reaches(self, status: JobStatus) -> None:
        for _ in range(400):
            if self.status() is status:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"job never reached {status}; it is {self.status()}")


async def test_a_job_resumes_on_a_new_worker_without_repeating_completed_steps() -> None:
    deployment = Deployment(requires_reviewer=True)
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        # The first worker takes the job as far as the reviewer checkpoint, then stops.
        async with deployment.worker(env.client, stops_abruptly=True):
            handle = await deployment.start(env.client)
            await deployment.reaches(JobStatus.REVIEW_REQUIRED)
        before = Counter(deployment.agent_calls)
        evidence_before = deployment.jobs.list_evidence(TENANT, deployment.job.id)
        calls_before = deployment.jobs.list_invocations(TENANT, deployment.job.id)

        # A different worker, with nothing in memory, picks the job up from history.
        async with deployment.worker(env.client):
            await handle.signal(
                ResearchJobWorkflow.submit_reviewer_decision, ReviewerDecision.APPROVE
            )
            outcome = await handle.result()

    assert outcome.job.status is JobStatus.COMPLETED
    assert before == {"planner": 1, "researcher": 1, "analyst": 1, "critic": 1}
    # Only the step that had not run yet ran on the second worker.
    assert deployment.agent_calls == {**before, "reporter": 1}
    evidence = deployment.jobs.list_evidence(TENANT, deployment.job.id)
    assert [record.id for record in evidence] == [record.id for record in evidence_before]
    calls = deployment.jobs.list_invocations(TENANT, deployment.job.id)
    # The research call was not repeated; the one new call is publication's re-read.
    assert [call.id for call in calls[: len(calls_before)]] == [call.id for call in calls_before]
    assert len(calls) == len(calls_before) + 1
    [finding] = deployment.jobs.list_findings(TENANT, deployment.job.id)
    assert finding.reviewer_status is ReviewerStatus.APPROVED
    assert deployment.jobs.get_publication(TENANT, deployment.job.id) is not None


async def test_a_job_with_no_worker_running_waits_and_then_runs_to_completion() -> None:
    """Work submitted while every worker is down is not lost."""
    deployment = Deployment(requires_reviewer=False)
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        handle = await deployment.start(env.client)
        assert deployment.status() is JobStatus.CREATED

        async with deployment.worker(env.client):
            outcome = await handle.result()

    assert outcome.job.status is JobStatus.COMPLETED
    assert deployment.agent_calls == {
        "planner": 1,
        "researcher": 1,
        "analyst": 1,
        "critic": 1,
        "reporter": 1,
    }


async def test_a_write_that_landed_before_its_worker_died_is_not_written_twice() -> None:
    """The activity stored the evidence, then failed before reporting success.

    Temporal runs it again. The second attempt must recognise the record it already
    wrote rather than store the excerpt a second time.
    """
    deployment = Deployment(requires_reviewer=False)
    persistence = JobActivities(jobs=deployment.jobs)
    attempts = 0

    @activity.defn(name="add_job_evidence")
    async def add_evidence_then_crash_once(
        tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        nonlocal attempts
        attempts += 1
        stored = await persistence.add_evidence(tenant_id, job_id, command)
        if attempts == 1:
            raise ConnectionError("worker lost before the write was acknowledged")
        return stored

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        deployment.worker(env.client, add_evidence=add_evidence_then_crash_once),
    ):
        handle = await deployment.start(env.client)
        outcome = await handle.result()

    assert attempts == 2
    assert outcome.job.status is JobStatus.COMPLETED
    assert len(deployment.jobs.list_evidence(TENANT, deployment.job.id)) == 1
    assert deployment.agent_calls["researcher"] == 1


async def test_a_status_change_that_landed_before_its_worker_died_is_not_an_error() -> None:
    deployment = Deployment(requires_reviewer=False)
    persistence = JobActivities(jobs=deployment.jobs)
    crashed: list[JobStatus] = []

    @activity.defn(name="transition_job")
    async def transition_then_crash_once(
        tenant_id: str, job_id: UUID, target: JobStatus, detail: str | None = None
    ) -> ResearchJob:
        moved = await persistence.transition(tenant_id, job_id, target, detail)
        if target is JobStatus.RESEARCHING and not crashed:
            crashed.append(target)
            raise ConnectionError("worker lost before the transition was acknowledged")
        return moved

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        deployment.worker(env.client, transition=transition_then_crash_once),
    ):
        handle = await deployment.start(env.client)
        outcome = await handle.result()

    assert crashed == [JobStatus.RESEARCHING]
    assert outcome.job.status is JobStatus.COMPLETED


async def test_a_decision_made_before_the_workflow_was_waiting_for_it_is_not_lost() -> None:
    """The job shows as awaiting review as soon as that status is stored.

    A reviewer can therefore decide before the workflow has processed the stored status
    and begun to wait - certain if the worker stopped in between. The decision has to be
    acted on when a worker gets there.
    """
    deployment = Deployment(requires_reviewer=True)
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        async with deployment.worker(env.client, stops_abruptly=True):
            handle = await deployment.start(env.client)
            await deployment.reaches(JobStatus.REVIEW_REQUIRED)
        # No worker is running: the decision is recorded in history ahead of the wait.
        await handle.signal(ResearchJobWorkflow.submit_reviewer_decision, ReviewerDecision.REJECT)

        async with deployment.worker(env.client):
            outcome = await handle.result()

    assert outcome.job.status is JobStatus.FAILED
    assert outcome.job.status_detail == "rejected by the reviewer"


async def test_a_second_decision_in_one_round_does_not_answer_the_next_round() -> None:
    """Two clicks on "more research" are one request, not two."""
    deployment = Deployment(requires_reviewer=True)
    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        deployment.worker(env.client),
    ):
        handle = await deployment.start(env.client)
        await deployment.reaches(JobStatus.REVIEW_REQUIRED)
        for _ in range(2):
            await handle.signal(
                ResearchJobWorkflow.submit_reviewer_decision,
                ReviewerDecision.REQUEST_MORE_RESEARCH,
            )
        # The second research pass ends back at the reviewer, still waiting.
        for _ in range(200):
            if deployment.agent_calls["critic"] == 2 and (
                deployment.status() is JobStatus.REVIEW_REQUIRED
            ):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        waiting = deployment.status()
        await handle.signal(ResearchJobWorkflow.submit_reviewer_decision, ReviewerDecision.APPROVE)
        outcome = await handle.result()

    assert waiting is JobStatus.REVIEW_REQUIRED
    assert deployment.agent_calls["critic"] == 2
    assert outcome.job.status is JobStatus.COMPLETED


async def test_a_decision_signalled_in_the_same_step_the_review_opened_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The narrow case a restart produces: the signal is applied before the wait begins.

    Temporal hands a workflow its signals before it resumes the workflow's own code, so
    when the stored status and the decision are replayed together the decision is
    already there when the wait starts. It must not be thrown away as stale.
    """
    waited_on: list[bool] = []

    async def wait_condition(ready: Callable[[], bool]) -> None:
        waited_on.append(ready())

    monkeypatch.setattr(research_workflow.workflow, "wait_condition", wait_condition)
    running = ResearchJobWorkflow()
    running._review_is_open = True  # what sending the job to review_required does
    running.submit_reviewer_decision(ReviewerDecision.APPROVE)

    decision = await running._await_reviewer_decision(None, None)  # type: ignore[arg-type]

    assert decision is ReviewerDecision.APPROVE
    assert waited_on == [True]


async def test_a_decision_for_a_job_that_is_not_at_a_review_is_dropped() -> None:
    running = ResearchJobWorkflow()

    running.submit_reviewer_decision(ReviewerDecision.APPROVE)

    assert running._reviewer_decisions == []
