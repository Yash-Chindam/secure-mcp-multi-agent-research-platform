"""The pipeline with real CrewAI agents: only the model itself is canned.

Every other test drives the pipeline with scripted stand-ins for the agents. Those
cannot show whether a real ``crewai.Agent`` works where the platform runs it: inside an
event loop, in a Temporal activity, calling the platform's tools through CrewAI's own
tool loop. These tests use the agents ``build_agent`` creates and replace only the model
provider.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from crewai.tools import BaseTool
from support.model import SOURCE_TEXT, SOURCE_URL, CannedModel
from support.scripted import StubExecutor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from research_platform.agents.contracts import ResearchPlan
from research_platform.agents.crew import (
    build_agent,
    request_agent_output,
    request_agent_output_async,
)
from research_platform.agents.usage import AgentCallStats
from research_platform.agents.validation import SchemaCorrectionExhausted
from research_platform.application.artifacts import InMemoryArtifactStore
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import JobStatus, ResearchBudget, ResearchJob
from research_platform.domain.tasks import AgentRole
from research_platform.evaluation.suite import CORPUS_DOMAIN, Scenario, run_scenario
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway
from research_platform.mcp.servers.backends import SourceDocument
from research_platform.worker import registered_activities
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.publishing import GatewaySourceChecker, PublicationActivities
from research_platform.workflow.research_workflow import TASK_QUEUE, ResearchJobWorkflow

pytestmark = pytest.mark.integration

TENANT = "acme"


def agents(model: CannedModel) -> Any:
    def factory(role: AgentRole, tools: list[BaseTool]) -> Any:
        return build_agent(role, llm=model, tools=tools)

    return factory


def test_a_real_agent_produces_its_contract_from_plain_synchronous_code() -> None:
    model = CannedModel()

    plan = request_agent_output(
        build_agent(AgentRole.PLANNER, llm=model), AgentRole.PLANNER, instructions="Plan."
    )

    assert isinstance(plan, ResearchPlan)
    assert model.calls == ["planner"]


@pytest.mark.asyncio
async def test_a_real_agent_produces_its_contract_from_inside_an_event_loop() -> None:
    """Where every Temporal activity runs.

    Asked to ``kickoff`` directly from here, a CrewAI agent returns an unawaited
    coroutine rather than an answer. Running the step as a Flow is what makes it answer.
    """
    model = CannedModel()
    agent = build_agent(AgentRole.PLANNER, llm=model)
    assert asyncio.iscoroutine(direct := agent.kickoff("Plan."))
    direct.close()

    plan = await request_agent_output_async(agent, AgentRole.PLANNER, instructions="Plan.")

    assert isinstance(plan, ResearchPlan)


@pytest.mark.asyncio
async def test_a_real_agents_rejected_response_is_corrected_through_the_flow() -> None:
    model = CannedModel(first_plan_is_malformed=True)
    stats = AgentCallStats()

    plan = await request_agent_output_async(
        build_agent(AgentRole.PLANNER, llm=model),
        AgentRole.PLANNER,
        instructions="Plan.",
        stats=stats,
    )

    assert isinstance(plan, ResearchPlan)
    assert model.calls == ["planner", "planner"]
    assert stats.corrections == 1


@pytest.mark.asyncio
async def test_a_real_agent_that_never_conforms_exhausts_its_attempts() -> None:
    class Stubborn(CannedModel):
        def _planner(self, _conversation: str) -> str:
            return "Thought: done\nFinal Answer: not json at all"

    model = Stubborn()

    with pytest.raises(SchemaCorrectionExhausted, match="ResearchPlan was not produced in 3"):
        await request_agent_output_async(
            build_agent(AgentRole.PLANNER, llm=model), AgentRole.PLANNER, instructions="Plan."
        )

    assert len(model.calls) == 3


@pytest.mark.asyncio
async def test_real_agents_carry_a_job_to_a_published_report_under_temporal() -> None:
    """The whole of section 9 with real agents, real tool calls and a real workflow."""
    model = CannedModel()
    repository = InMemoryJobRepository()
    jobs = ResearchJobService(repository)
    job = repository.add(
        ResearchJob(
            tenant_id=TENANT,
            requester_id="requester-1",
            question="What does the vendor charge?",
            budget=ResearchBudget(max_tool_calls=10),
        )
    )
    registry = default_registry()
    research = ResearchActivities(
        gateway=CapabilityGateway(
            registry=registry, executor=StubExecutor(), audit=jobs.record_invocation
        ),
        registry=registry,
        build_agent=agents(model),
    )
    persistence = JobActivities(jobs=jobs, budgets=research.gateway.budgets)
    artifacts = InMemoryArtifactStore()
    publication = PublicationActivities(
        jobs=jobs, artifacts=artifacts, sources=GatewaySourceChecker(research.gateway)
    )

    async with (
        await WorkflowEnvironment.start_time_skipping(
            data_converter=pydantic_data_converter
        ) as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ResearchJobWorkflow],
            activities=list(registered_activities(research, persistence, publication).values()),
        ),
    ):
        outcome = await env.client.execute_workflow(
            ResearchJobWorkflow.run, job, id=f"research-job-{job.id}", task_queue=TASK_QUEUE
        )

    stored = jobs.get(TENANT, job.id)
    assert outcome.job.status is JobStatus.COMPLETED, stored.status_detail
    # The researcher called its tool through CrewAI's own tool loop, so it was asked
    # twice: once to act, once to answer with what the tool returned.
    assert model.calls == ["planner", "researcher", "researcher", "analyst", "critic", "reporter"]
    [evidence] = jobs.list_evidence(TENANT, job.id)
    assert evidence.excerpt == SOURCE_TEXT
    assert str(evidence.source_uri) == SOURCE_URL
    first_call = jobs.list_invocations(TENANT, job.id)[0]
    assert evidence.tool_invocation_id == first_call.id
    assert (first_call.mcp_server, first_call.capability) == ("web-research", "fetch")
    assert stored.usage.agent_calls == 5
    assert stored.usage.tool_calls == 2
    assert jobs.get_publication(TENANT, job.id) is not None
    assert b"20 USD per seat [1]." in artifacts.get(TENANT, f"jobs/{job.id}/report.md")


@pytest.mark.asyncio
async def test_the_evaluation_suite_runs_with_real_agents() -> None:
    url = f"https://{CORPUS_DOMAIN}/pricing"

    class CorpusModel(CannedModel):
        def _researcher(self, conversation: str) -> str:
            return super()._researcher(conversation).replace(SOURCE_URL, url)

    scenario = Scenario(
        name="pricing",
        question="What does the vendor charge per seat?",
        documents=(SourceDocument(url=url, title="Pricing", text=SOURCE_TEXT),),
        expected_tools=frozenset({"web-research.fetch"}),
    )

    result = await run_scenario(scenario, agents(CorpusModel()))

    assert result.status is JobStatus.COMPLETED, result.status_detail
    assert result.citation_correctness == 1.0
    assert result.tool_selection_accuracy == 1.0
    assert (result.tool_calls, result.invalid_tool_calls) == (1, 0)
