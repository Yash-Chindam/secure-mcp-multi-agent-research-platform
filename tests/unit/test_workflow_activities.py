import json
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import pytest
from crewai.lite_agent_output import LiteAgentOutput
from crewai.tools import BaseTool
from support.scripted import ScriptedResearcher
from temporalio.testing import ActivityEnvironment

from research_platform.agents.contracts import (
    AnalysisResult,
    ClaimVerdict,
    CriticReview,
    EvidenceSubmission,
    ResearchPlan,
    ResearchReport,
)
from research_platform.agents.provenance import hash_content
from research_platform.agents.validation import SchemaCorrectionExhausted
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import (
    AccessClass,
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    ResearchBudget,
    ResearchJob,
    ResearchJobCreate,
    TrustLevel,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway, ExecutionRequest
from research_platform.workflow.activities import (
    JobActivities,
    ResearchActivities,
    principal_for,
)

TENANT = "acme"
EVIDENCE_ID = uuid4()

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


class StubExecutor:
    def __init__(self, response: str = "Vendor pricing is 20 USD per seat.") -> None:
        self.response = response
        self.calls: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> str:
        self.calls.append(request)
        return self.response


class FakeAgent:
    """Stands in for a crewai.Agent, returning one scripted kickoff response."""

    def __init__(self, response: str, *, tools: list[BaseTool] | None = None) -> None:
        self.response = response
        self.tools = tools or []
        self.messages: list[str] = []

    def kickoff(self, message: str) -> LiteAgentOutput:
        self.messages.append(message)
        return LiteAgentOutput(raw=self.response, agent_role="agent")


def new_job() -> ResearchJob:
    return ResearchJob(
        tenant_id=TENANT,
        requester_id="requester-1",
        question="What does the vendor charge?",
        budget=ResearchBudget(max_tool_calls=10),
    )


def build_activities(
    response: str, *, executor: StubExecutor | None = None
) -> tuple[ResearchActivities, list[FakeAgent]]:
    gateway = CapabilityGateway(registry=default_registry(), executor=executor or StubExecutor())
    captured: list[FakeAgent] = []

    def build_agent(_role: AgentRole, tools: list[BaseTool]) -> FakeAgent:
        agent = FakeAgent(response, tools=tools)
        captured.append(agent)
        return agent

    activities = ResearchActivities(
        gateway=gateway, registry=default_registry(), build_agent=build_agent
    )
    return activities, captured


@pytest.mark.asyncio
async def test_plan_asks_for_a_research_plan_and_parses_it() -> None:
    activities, captured = build_activities(VALID_PLAN)
    env = ActivityEnvironment()

    plan = await env.run(activities.plan, new_job())

    assert isinstance(plan, ResearchPlan)
    assert plan.tasks[0].assigned_agent is AgentRole.RESEARCHER


@pytest.mark.asyncio
async def test_plan_gives_the_planner_no_tools() -> None:
    activities, captured = build_activities(VALID_PLAN)
    env = ActivityEnvironment()

    await env.run(activities.plan, new_job())

    (agent,) = captured
    assert agent.tools == []


@pytest.mark.asyncio
async def test_plan_describes_the_available_capabilities_in_the_prompt() -> None:
    activities, captured = build_activities(VALID_PLAN)
    env = ActivityEnvironment()

    await env.run(activities.plan, new_job())

    (agent,) = captured
    assert "web-research.search" in agent.messages[0]


def research_task(job: ResearchJob) -> ResearchTask:
    return ResearchTask(
        job_id=job.id,
        tenant_id=job.tenant_id,
        objective="Collect the vendor pricing page",
        assigned_agent=AgentRole.RESEARCHER,
        evidence_requirements=["a dated pricing page"],
    )


def researching(
    build_agent: Callable[[AgentRole, list[BaseTool]], Any],
) -> ResearchActivities:
    registry = default_registry()
    return ResearchActivities(
        gateway=CapabilityGateway(registry=registry, executor=StubExecutor()),
        registry=registry,
        build_agent=build_agent,
    )


@pytest.mark.asyncio
async def test_research_gives_the_researcher_only_researcher_tools() -> None:
    captured: list[ScriptedResearcher] = []

    def build_agent(_role: AgentRole, tools: list[BaseTool]) -> ScriptedResearcher:
        captured.append(ScriptedResearcher(tools))
        return captured[-1]

    job = new_job()

    submission = await ActivityEnvironment().run(
        researching(build_agent).research, job, research_task(job)
    )

    assert isinstance(submission, EvidenceSubmission)
    tool_names = {tool.name for tool in captured[0].tools}
    assert "web_research_search" in tool_names
    assert "postgres_run_analytical_query" not in tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("clearance", "reads_internal_documents"),
    [(AccessClass.PUBLIC, False), (AccessClass.INTERNAL, True), (AccessClass.RESTRICTED, True)],
)
async def test_a_researcher_is_offered_internal_tools_only_on_a_job_cleared_for_them(
    clearance: AccessClass, reads_internal_documents: bool
) -> None:
    """The job carries its requester's clearance, and its agents act under exactly that."""
    captured: list[ScriptedResearcher] = []

    def build_agent(_role: AgentRole, tools: list[BaseTool]) -> ScriptedResearcher:
        captured.append(ScriptedResearcher(tools))
        return captured[-1]

    job = new_job().model_copy(update={"clearance": clearance})

    await ActivityEnvironment().run(researching(build_agent).research, job, research_task(job))

    tool_names = {tool.name for tool in captured[0].tools}
    assert "web_research_fetch" in tool_names
    assert ("filesystem_read_document" in tool_names) is reads_internal_documents
    assert ("github_read_repository" in tool_names) is reads_internal_documents


@pytest.mark.parametrize("role", list(AgentRole))
def test_an_agent_acts_for_the_jobs_tenant_with_the_jobs_clearance(role: AgentRole) -> None:
    job = new_job().model_copy(update={"clearance": AccessClass.INTERNAL})

    principal = principal_for(job, role)

    assert principal.tenant_id == job.tenant_id
    assert principal.clearance is AccessClass.INTERNAL
    assert principal.agent_role is role
    assert principal.subject_id == f"job:{job.id}"


def test_a_job_is_public_unless_its_requester_was_cleared_for_more() -> None:
    assert principal_for(new_job(), AgentRole.RESEARCHER).clearance is AccessClass.PUBLIC


@pytest.mark.asyncio
async def test_research_establishes_each_records_provenance_from_the_tool_call() -> None:
    job = new_job()
    task = research_task(job)

    submission = await ActivityEnvironment().run(
        researching(lambda _role, tools: ScriptedResearcher(tools)).research, job, task
    )

    [record] = submission.records
    assert str(record.source_uri) == "https://vendor.test/pricing"
    assert record.content_hash == hash_content("Vendor pricing is 20 USD per seat.")
    assert record.producing_task_id == task.id
    assert record.trust_level is TrustLevel.PRIMARY


@pytest.mark.asyncio
async def test_research_refuses_an_excerpt_no_tool_returned() -> None:
    job = new_job()
    fabricating = researching(
        lambda _role, tools: ScriptedResearcher(tools, excerpt="Pricing is 5 USD per seat.")
    )

    with pytest.raises(SchemaCorrectionExhausted, match="does not appear in the output"):
        await ActivityEnvironment().run(fabricating.research, job, research_task(job))


@pytest.mark.asyncio
async def test_research_refuses_evidence_attributed_to_a_tool_call_that_never_happened() -> None:
    job = new_job()
    invented = json.dumps(
        {
            "claims": [
                {
                    "excerpt": "Vendor pricing is 20 USD per seat.",
                    "tool_invocation_id": str(uuid4()),
                }
            ]
        }
    )
    activities, _captured = build_activities(invented)

    with pytest.raises(SchemaCorrectionExhausted, match="not a tool call this task made"):
        await ActivityEnvironment().run(activities.research, job, research_task(job))


@pytest.mark.asyncio
async def test_analyze_cites_the_evidence_identifiers_in_the_prompt() -> None:
    job = new_job()
    evidence = [
        EvidenceRecord(
            job_id=job.id,
            tenant_id=job.tenant_id,
            excerpt="Vendor pricing is 20 USD per seat.",
            source_uri="https://vendor.test/pricing",
            content_hash=f"sha256:{'0' * 64}",
            producing_task_id=uuid4(),
            tool_invocation_id=uuid4(),
            trust_level=TrustLevel.PRIMARY,
        )
    ]
    analysis_response = json.dumps(
        {
            "findings": [
                {
                    "claim": "The vendor charges 20 USD per seat.",
                    "supporting_evidence_ids": [str(evidence[0].id)],
                    "confidence": 0.9,
                }
            ]
        }
    )
    activities, captured = build_activities(analysis_response)
    env = ActivityEnvironment()

    result = await env.run(activities.analyze, job, evidence)

    assert isinstance(result, AnalysisResult)
    (agent,) = captured
    assert str(evidence[0].id) in agent.messages[0]
    assert "primary" in agent.messages[0]


@pytest.mark.asyncio
async def test_critique_lists_the_proposed_findings_in_the_prompt() -> None:
    job = new_job()
    analysis = AnalysisResult.model_validate(
        {
            "findings": [
                {
                    "claim": "The vendor charges 20 USD per seat.",
                    "supporting_evidence_ids": [str(uuid4())],
                    "confidence": 0.9,
                }
            ]
        }
    )
    review_response = json.dumps(
        {
            "verdicts": [
                {
                    "claim": "The vendor charges 20 USD per seat.",
                    "verdict": CriticVerdict.SUPPORTED.value,
                    "reasoning": "Matches the source.",
                }
            ]
        }
    )
    activities, captured = build_activities(review_response)
    env = ActivityEnvironment()

    review = await env.run(activities.critique, job, analysis, [])

    assert isinstance(review, CriticReview)
    (agent,) = captured
    assert "The vendor charges 20 USD per seat." in agent.messages[0]


@pytest.mark.asyncio
async def test_report_names_the_critic_supported_claims_in_the_prompt() -> None:
    job = new_job()
    recorded = EvidenceRecord(
        job_id=job.id,
        tenant_id=job.tenant_id,
        excerpt="Vendor pricing is 20 USD per seat.",
        source_uri="https://vendor.test/pricing",
        content_hash=f"sha256:{'0' * 64}",
        producing_task_id=uuid4(),
        tool_invocation_id=uuid4(),
    )
    citation = recorded.id
    critique = CriticReview(
        verdicts=[
            ClaimVerdict(
                claim="The vendor charges 20 USD per seat.",
                verdict=CriticVerdict.SUPPORTED,
                reasoning="Matches the source.",
            )
        ]
    )
    report_response = json.dumps(
        {
            "title": "Vendor pricing",
            "sections": [
                {
                    "heading": "Pricing",
                    "body": f"The vendor charges 20 USD per seat [{citation}].",
                }
            ],
        }
    )
    activities, captured = build_activities(report_response)
    env = ActivityEnvironment()

    report = await env.run(activities.report, job, critique, [recorded])

    assert isinstance(report, ResearchReport)
    (agent,) = captured
    assert "The vendor charges 20 USD per seat." in agent.messages[0]


@pytest.mark.asyncio
async def test_every_activity_is_registered_with_a_temporal_name() -> None:
    activities, captured = build_activities(VALID_PLAN)

    names = {
        getattr(method, "__temporal_activity_definition").name
        for method in (
            activities.plan,
            activities.research,
            activities.analyze,
            activities.critique,
            activities.report,
        )
    }

    assert names == {
        "plan_research",
        "research_task",
        "analyze_evidence",
        "critique_analysis",
        "write_report",
    }


@pytest.mark.asyncio
async def test_transition_job_applies_a_new_status() -> None:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    activities = JobActivities(jobs=service)
    env = ActivityEnvironment()

    updated = await env.run(activities.transition, TENANT, job.id, JobStatus.PLANNING)

    assert updated.status is JobStatus.PLANNING


@pytest.mark.asyncio
async def test_transition_job_is_a_no_op_when_already_at_the_target_status() -> None:
    """A retried activity call must not fail the very recovery it exists to support."""
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    activities = JobActivities(jobs=service)
    env = ActivityEnvironment()
    await env.run(activities.transition, TENANT, job.id, JobStatus.PLANNING)

    repeated = await env.run(activities.transition, TENANT, job.id, JobStatus.PLANNING)

    assert repeated.status is JobStatus.PLANNING


@pytest.mark.asyncio
async def test_transition_job_still_rejects_a_genuinely_invalid_transition() -> None:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    activities = JobActivities(jobs=service)
    env = ActivityEnvironment()

    with pytest.raises(Exception, match="cannot transition"):
        await env.run(activities.transition, TENANT, job.id, JobStatus.COMPLETED)


def evidence_command(excerpt_hash: str = "0") -> EvidenceRecordCreate:
    return EvidenceRecordCreate(
        excerpt="Vendor pricing is 20 USD per seat.",
        source_uri="https://vendor.test/pricing",
        content_hash=f"sha256:{excerpt_hash * 64}",
        producing_task_id=UUID(int=1),
        tool_invocation_id=UUID(int=2),
    )


@pytest.mark.asyncio
async def test_a_redelivered_evidence_write_does_not_store_the_record_twice() -> None:
    """Section 12: a restarted worker must not duplicate a completed side effect."""
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    activities = JobActivities(jobs=service)
    env = ActivityEnvironment()

    first = await env.run(activities.add_evidence, TENANT, job.id, evidence_command())
    redelivered = await env.run(activities.add_evidence, TENANT, job.id, evidence_command())

    assert redelivered.id == first.id
    assert len(service.list_evidence(TENANT, job.id)) == 1


@pytest.mark.asyncio
async def test_different_content_from_the_same_source_is_still_stored_separately() -> None:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    activities = JobActivities(jobs=service)
    env = ActivityEnvironment()

    await env.run(activities.add_evidence, TENANT, job.id, evidence_command("0"))
    await env.run(activities.add_evidence, TENANT, job.id, evidence_command("1"))

    assert len(service.list_evidence(TENANT, job.id)) == 2
