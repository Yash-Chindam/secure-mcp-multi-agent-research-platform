"""Exercises the real Temporal wiring: a workflow, its activities and a live signal.

Everything about the pipeline's logic is already covered in
tests/unit/test_workflow_orchestration.py without a Temporal server. This file's job is
narrower: prove that research_platform.workflow.research_workflow actually drives
research_platform.workflow.orchestration.run_research_job through a real Temporal
workflow and real (string-registered) activities, that a signal answers a reviewer
decision, and that the whole thing round-trips through Temporal's data converter.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from crewai.tools import BaseTool
from support.scripted import (
    SOURCE_TEXT,
    ScriptedAgent,
    ScriptedResearcher,
    StubExecutor,
    honest_crew,
)
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from research_platform.agents.contracts import AnalysisResult
from research_platform.agents.provenance import hash_content
from research_platform.application.artifacts import InMemoryArtifactStore
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import (
    EvidenceRecord,
    JobStatus,
    ResearchBudget,
    ResearchJob,
    TrustLevel,
)
from research_platform.domain.tasks import AgentRole
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway
from research_platform.worker import registered_activities
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.orchestration import ReviewerDecision
from research_platform.workflow.publishing import GatewaySourceChecker, PublicationActivities
from research_platform.workflow.research_workflow import TASK_QUEUE, ResearchJobWorkflow

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TENANT = "acme"

AgentFactory = Callable[[AgentRole, list[BaseTool]], ScriptedAgent]


def new_job() -> ResearchJob:
    return ResearchJob(
        tenant_id=TENANT,
        requester_id="requester-1",
        question="What does the vendor charge?",
        budget=ResearchBudget(max_tool_calls=10),
    )


def build_activities(
    job: ResearchJob, build_agent: AgentFactory
) -> tuple[ResearchActivities, JobActivities, ResearchJobService]:
    repository = InMemoryJobRepository()
    repository.add(job)
    jobs = ResearchJobService(repository)
    registry = default_registry()
    research = ResearchActivities(
        gateway=CapabilityGateway(
            registry=registry, executor=StubExecutor(), audit=jobs.record_invocation
        ),
        registry=registry,
        build_agent=build_agent,
    )
    return research, JobActivities(jobs=jobs), jobs


def registered(
    research: ResearchActivities, persistence: JobActivities, **replaced: Any
) -> Sequence[Callable[..., Any]]:
    """Every activity the workflow calls, with any named one swapped for a substitute."""
    publication = PublicationActivities(
        jobs=persistence.jobs,
        artifacts=InMemoryArtifactStore(),
        sources=GatewaySourceChecker(research.gateway),
    )
    activities = registered_activities(research, persistence, publication)
    return list((activities | replaced).values())


async def run_to_completion(job: ResearchJob, activities: Sequence[Callable[..., Any]]) -> Any:
    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ResearchJobWorkflow],
            activities=activities,
        ),
    ):
        return await env.client.execute_workflow(
            ResearchJobWorkflow.run, job, id=f"research-job-{job.id}", task_queue=TASK_QUEUE
        )


async def test_the_workflow_completes_the_happy_path_through_real_activities() -> None:
    job = new_job()
    research, persistence, jobs = build_activities(job, honest_crew())

    outcome = await run_to_completion(job, registered(research, persistence))

    assert outcome.job.status is JobStatus.COMPLETED
    assert outcome.report is not None
    assert len(jobs.list_evidence(TENANT, job.id)) == 1


async def test_recorded_evidence_carries_provenance_the_platform_established_itself() -> None:
    """Source, hash, classification and producing call all come from the tool call."""
    job = new_job()
    research, persistence, jobs = build_activities(job, honest_crew())

    await run_to_completion(job, registered(research, persistence))

    [evidence] = jobs.list_evidence(TENANT, job.id)
    # The first call captured the evidence; publication re-read the source after it.
    call, reread = jobs.list_invocations(TENANT, job.id)
    assert reread.capability == call.capability == "fetch"
    assert evidence.tool_invocation_id == call.id
    assert evidence.producing_task_id == call.task_id
    assert str(evidence.source_uri) == "https://vendor.test/pricing"
    assert evidence.content_hash == hash_content(SOURCE_TEXT)
    assert evidence.trust_level is TrustLevel.PRIMARY


async def test_a_researcher_that_quotes_what_no_tool_returned_collects_no_evidence() -> None:
    """The fabricated excerpt is refused, so the job ends partial rather than citing it."""

    def fabricating_crew(role: AgentRole, tools: list[BaseTool]) -> ScriptedAgent:
        if role is AgentRole.RESEARCHER:
            return ScriptedResearcher(tools, excerpt="Vendor pricing is 5 USD per seat.")
        return honest_crew()(role, tools)

    job = new_job()
    research, persistence, jobs = build_activities(job, fabricating_crew)

    outcome = await run_to_completion(job, registered(research, persistence))

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.job.status_detail == "no evidence could be collected"
    assert outcome.report is None
    assert jobs.list_evidence(TENANT, job.id) == []


async def test_a_report_citing_evidence_that_does_not_exist_fails_the_job_with_a_reason() -> None:
    def crew_with_an_inventive_reporter(role: AgentRole, tools: list[BaseTool]) -> ScriptedAgent:
        if role is AgentRole.REPORTER:
            invented = "11111111-2222-3333-4444-555555555555"
            body = f"The vendor charges 20 USD per seat [{invented}]."
            return ScriptedAgent(
                lambda _message: json.dumps(
                    {"title": "Vendor pricing", "sections": [{"heading": "Pricing", "body": body}]}
                )
            )
        return honest_crew()(role, tools)

    job = new_job()
    research, persistence, jobs = build_activities(job, crew_with_an_inventive_reporter)

    outcome = await run_to_completion(job, registered(research, persistence))

    assert outcome.job.status is JobStatus.FAILED
    assert outcome.report is None
    assert "never recorded" in (outcome.failure or "")
    assert jobs.get(TENANT, job.id).status_detail == outcome.failure


async def test_a_reviewer_signal_lets_a_flagged_job_proceed() -> None:
    job = new_job()
    research, persistence, jobs = build_activities(job, honest_crew(requires_reviewer=True))

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ResearchJobWorkflow],
            activities=registered(research, persistence),
        ),
    ):
        handle = await env.client.start_workflow(
            ResearchJobWorkflow.run, job, id=f"research-job-{job.id}", task_queue=TASK_QUEUE
        )

        for _ in range(200):
            if jobs.get(TENANT, job.id).status is JobStatus.REVIEW_REQUIRED:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the job never reached review_required")

        await handle.signal(ResearchJobWorkflow.submit_reviewer_decision, ReviewerDecision.APPROVE)
        outcome = await handle.result()

    assert outcome.job.status is JobStatus.COMPLETED
    assert outcome.report is not None


async def test_a_transient_activity_failure_recovers_via_temporals_own_retry() -> None:
    """Section 14's "recovery after controlled failures", proven end to end.

    ``research_workflow.AGENT_RETRY_POLICY`` bounds every agent activity to 3 attempts.
    This activity fails its first attempt and succeeds on the second, with no retry logic
    of this platform's own involved - Temporal's own mechanism recovers it.
    """
    job = new_job()
    research, persistence, _jobs = build_activities(job, honest_crew())
    attempts: list[int] = []

    @activity.defn(name="analyze_evidence")
    async def flaky_analyze(job: ResearchJob, evidence: list[EvidenceRecord]) -> AnalysisResult:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise RuntimeError("simulated transient failure")
        return await research.analyze(job, evidence)

    outcome = await run_to_completion(job, registered(research, persistence, analyze=flaky_analyze))

    assert outcome.job.status is JobStatus.COMPLETED
    assert outcome.report is not None
    assert len(attempts) == 2


async def test_an_output_the_agent_cannot_correct_is_not_retried_by_temporal() -> None:
    """A refusal is deterministic; re-running the same model calls would change nothing."""
    calls: list[str] = []

    def stubborn_crew(role: AgentRole, tools: list[BaseTool]) -> ScriptedAgent:
        if role is AgentRole.PLANNER:

            def respond(_message: str) -> str:
                calls.append("plan")
                return "not json"

            return ScriptedAgent(respond)
        return honest_crew()(role, tools)

    job = new_job()
    research, persistence, _jobs = build_activities(job, stubborn_crew)

    outcome = await run_to_completion(job, registered(research, persistence))

    assert outcome.job.status is JobStatus.FAILED
    assert "SchemaCorrectionExhausted" in (outcome.failure or "")
    assert len(calls) == 3  # one activity attempt, three bounded corrections
