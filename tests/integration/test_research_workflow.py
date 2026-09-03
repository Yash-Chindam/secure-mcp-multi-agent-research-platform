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
from collections.abc import Callable
from uuid import uuid4

import pytest
from crewai.lite_agent_output import LiteAgentOutput
from crewai.tools import BaseTool
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import JobStatus, ResearchBudget, ResearchJob
from research_platform.domain.tasks import AgentRole
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway, ExecutionRequest
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.orchestration import ReviewerDecision
from research_platform.workflow.research_workflow import ResearchJobWorkflow

TENANT = "acme"

VALID_PLAN = json.dumps(
    {
        "tasks": [
            {
                "objective": "Collect the vendor pricing page",
                "assigned_agent": AgentRole.RESEARCHER.value,
                "evidence_requirements": ["a dated pricing page"],
            }
        ],
        "rationale": "Pricing must be sourced before it can be compared.",
    }
)


def submission_for(*_args: object) -> str:
    return json.dumps(
        {
            "records": [
                {
                    "excerpt": "Vendor pricing is 20 USD per seat.",
                    "source_uri": "https://vendor.test/pricing",
                    "content_hash": f"sha256:{'0' * 64}",
                    "producing_task_id": str(uuid4()),
                    "tool_invocation_id": str(uuid4()),
                }
            ]
        }
    )


def analysis_response(evidence_id: str) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "claim": "The vendor charges 20 USD per seat.",
                    "supporting_evidence_ids": [evidence_id],
                    "confidence": 0.9,
                }
            ]
        }
    )


def review_response(*, requires_reviewer: bool) -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "claim": "The vendor charges 20 USD per seat.",
                    "verdict": "supported",
                    "reasoning": "Matches the source.",
                }
            ],
            "coverage_gaps": ["enterprise pricing was not sourced"] if requires_reviewer else [],
        }
    )


def report_response(evidence_id: str) -> str:
    return json.dumps(
        {
            "title": "Vendor pricing",
            "sections": [
                {
                    "heading": "Pricing",
                    "body": f"The vendor charges 20 USD per seat [{evidence_id}].",
                }
            ],
        }
    )


class StubExecutor:
    def execute(self, request: ExecutionRequest) -> str:
        return "Vendor pricing is 20 USD per seat."


class ScriptedAgent:
    """Returns whatever ``respond`` computes for the message it was given."""

    def __init__(self, respond: Callable[[str], str]) -> None:
        self._respond = respond
        self.tools: list[BaseTool] = []

    def kickoff(self, message: str) -> LiteAgentOutput:
        return LiteAgentOutput(raw=self._respond(message), agent_role="agent")


def new_job() -> ResearchJob:
    return ResearchJob(
        tenant_id=TENANT,
        requester_id="requester-1",
        question="What does the vendor charge?",
        budget=ResearchBudget(max_tool_calls=10),
    )


def build_activities(
    *, job: ResearchJob, evidence_id_holder: list[str], requires_reviewer: bool
) -> tuple[ResearchActivities, JobActivities, ResearchJobService]:
    gateway = CapabilityGateway(registry=default_registry(), executor=StubExecutor())
    repository = InMemoryJobRepository()
    repository.add(job)
    jobs = ResearchJobService(repository)

    def build_agent(role: AgentRole, _tools: list[BaseTool]) -> ScriptedAgent:
        if role is AgentRole.PLANNER:
            return ScriptedAgent(lambda _msg: VALID_PLAN)
        if role is AgentRole.RESEARCHER:
            return ScriptedAgent(submission_for)
        if role is AgentRole.ANALYST:
            return ScriptedAgent(lambda _msg: analysis_response(evidence_id_holder[0]))
        if role is AgentRole.CRITIC:
            return ScriptedAgent(lambda _msg: review_response(requires_reviewer=requires_reviewer))
        return ScriptedAgent(lambda _msg: report_response(evidence_id_holder[0]))

    research_activities = ResearchActivities(
        gateway=gateway, registry=default_registry(), build_agent=build_agent
    )
    job_activities = JobActivities(jobs=jobs)
    return research_activities, job_activities, jobs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_workflow_completes_the_happy_path_through_real_activities() -> None:
    job = new_job()
    evidence_id_holder: list[str] = []
    research_activities, job_activities, jobs = build_activities(
        job=job, evidence_id_holder=evidence_id_holder, requires_reviewer=False
    )

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue="research-jobs",
            workflows=[ResearchJobWorkflow],
            activities=[
                research_activities.plan,
                research_activities.research,
                research_activities.analyze,
                research_activities.critique,
                research_activities.report,
                job_activities.transition,
                job_activities.add_evidence,
            ],
        ),
    ):
        # The analyst's evidence identifier is only known once the researcher runs,
        # so it cannot be baked into the plan up front; the earliest a real deployment
        # ever forges an id is with the record's own row, and every test agent here
        # only ever needs one, so seed it before the workflow starts.
        evidence_id_holder.append(str(uuid4()))
        outcome = await env.client.execute_workflow(
            ResearchJobWorkflow.run,
            job,
            id=f"research-job-{job.id}",
            task_queue="research-jobs",
        )

    assert outcome.job.status is JobStatus.COMPLETED
    assert outcome.report is not None
    assert len(jobs.list_evidence(TENANT, job.id)) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_reviewer_signal_lets_a_flagged_job_proceed() -> None:
    job = new_job()
    evidence_id_holder: list[str] = [str(uuid4())]
    research_activities, job_activities, jobs = build_activities(
        job=job, evidence_id_holder=evidence_id_holder, requires_reviewer=True
    )

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue="research-jobs",
            workflows=[ResearchJobWorkflow],
            activities=[
                research_activities.plan,
                research_activities.research,
                research_activities.analyze,
                research_activities.critique,
                research_activities.report,
                job_activities.transition,
                job_activities.add_evidence,
            ],
        ),
    ):
        handle = await env.client.start_workflow(
            ResearchJobWorkflow.run,
            job,
            id=f"research-job-{job.id}",
            task_queue="research-jobs",
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
