"""Tokens, cost and working time per job: counted, budgeted, stored and measured."""

import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from crewai.lite_agent_output import LiteAgentOutput
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from temporalio.testing import ActivityEnvironment

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
from research_platform.agents.crew import request_agent_output
from research_platform.agents.usage import AgentCallStats, TokenPricing
from research_platform.agents.validation import SchemaCorrectionExhausted
from research_platform.application.jobs import AsyncJobs, InMemoryJobRepository, ResearchJobService
from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    JobStatus,
    JobUsage,
    ResearchBudget,
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.mcp.breaker import (
    BudgetExhausted,
    BudgetLedger,
    CircuitBreaker,
    CircuitState,
)
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway, ExecutionRequest
from research_platform.observability.metrics import PlatformMetrics
from research_platform.workflow import research_workflow
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.orchestration import (
    OrchestrationActivities,
    ReviewerDecision,
    run_research_job,
)
from research_platform.workflow.research_workflow import ReplaySafeMetrics

TENANT = "acme"
CLAIM = "The vendor charges 20 USD per seat."

VALID_PLAN = json.dumps(
    {
        "tasks": [
            {
                "objective": "Collect the vendor pricing page",
                "assigned_agent": "researcher",
                "evidence_requirements": ["a dated pricing page"],
            }
        ],
        "rationale": "Pricing must be sourced before it can be compared.",
    }
)


# -- measuring ----------------------------------------------------------------------------------


def build_metrics() -> tuple[PlatformMetrics, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    return PlatformMetrics(meter=MeterProvider(metric_readers=[reader]).get_meter("test")), reader


def points(reader: InMemoryMetricReader, name: str) -> list[tuple[float, dict[str, Any]]]:
    """Every data point of one metric, as (value, attributes)."""
    found: list[tuple[float, dict[str, Any]]] = []
    data = reader.get_metrics_data()
    for resource in data.resource_metrics if data is not None else []:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    value = getattr(point, "value", None)
                    found.append(
                        (point.sum if value is None else value, dict(point.attributes or {}))
                    )
    return found


# -- one agent call -----------------------------------------------------------------------------


def test_tokens_are_counted_as_the_provider_reported_them() -> None:
    stats = AgentCallStats()

    stats.observe({"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150})
    stats.observe({"prompt_tokens": 200, "completion_tokens": 50})

    assert (stats.prompt_tokens, stats.completion_tokens) == (320, 80)
    assert stats.attempts == 2
    assert stats.corrections == 1


@pytest.mark.parametrize(
    "usage",
    [None, {}, {"prompt_tokens": None}, {"prompt_tokens": "many"}, {"prompt_tokens": -5}],
)
def test_a_missing_or_malformed_usage_report_counts_the_attempt_but_no_tokens(
    usage: dict[str, Any] | None,
) -> None:
    stats = AgentCallStats()

    stats.observe(usage)

    assert stats.attempts == 1
    assert (stats.prompt_tokens, stats.completion_tokens) == (0, 0)


def test_a_boolean_is_not_mistaken_for_a_token_count() -> None:
    stats = AgentCallStats()

    stats.observe({"prompt_tokens": True, "completion_tokens": False})

    assert stats.prompt_tokens == 0


def test_cost_is_tokens_times_the_configured_price() -> None:
    pricing = TokenPricing(input_per_million_usd=0.15, output_per_million_usd=0.60)

    assert pricing.cost_of(1_000_000, 0) == pytest.approx(0.15)
    assert pricing.cost_of(0, 500_000) == pytest.approx(0.30)
    assert TokenPricing().cost_of(1_000_000, 1_000_000) == 0


def test_a_negative_price_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        TokenPricing(input_per_million_usd=-1)


class Agent:
    """Returns scripted responses in order, each with the token usage a provider would report."""

    def __init__(self, *responses: str, usage: dict[str, int] | None = None) -> None:
        self._responses = list(responses)
        self._usage = usage
        self.calls = 0

    def kickoff(self, _message: str) -> LiteAgentOutput:
        self.calls += 1
        raw = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return LiteAgentOutput(raw=raw, agent_role="agent", usage_metrics=self._usage)


def test_every_attempt_of_a_corrected_call_is_counted_with_its_tokens() -> None:
    agent = Agent("not json", VALID_PLAN, usage={"prompt_tokens": 100, "completion_tokens": 10})
    stats = AgentCallStats()

    request_agent_output(agent, AgentRole.PLANNER, instructions="Plan.", stats=stats)

    assert stats.attempts == 2
    assert stats.corrections == 1
    assert (stats.prompt_tokens, stats.completion_tokens) == (200, 20)


# -- the ledger -----------------------------------------------------------------------------------


def test_usage_accumulates_across_agent_calls() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()

    ledger.record_agent_call(job_id, prompt_tokens=100, completion_tokens=20, cost_usd=0.5)
    total = ledger.record_agent_call(
        job_id, prompt_tokens=50, completion_tokens=5, active_seconds=2.5, corrections=1
    )

    assert total == JobUsage(
        agent_calls=2,
        schema_corrections=1,
        prompt_tokens=150,
        completion_tokens=25,
        cost_usd=0.5,
        active_seconds=2.5,
    )
    assert total.total_tokens == 175
    assert ledger.usage(uuid4()) == JobUsage()


def test_recorded_usage_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        BudgetLedger().record_agent_call(uuid4(), prompt_tokens=-1)


def test_what_a_caller_reads_from_the_ledger_cannot_change_the_ledger() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()
    ledger.record_agent_call(job_id, prompt_tokens=10)

    ledger.usage(job_id).prompt_tokens = 0

    assert ledger.usage(job_id).prompt_tokens == 10


@pytest.mark.parametrize(
    ("spent", "message"),
    [
        ({"prompt_tokens": 600, "completion_tokens": 400}, "reached its 1000 token budget"),
        ({"cost_usd": 2.0}, "reached its 2 USD cost budget"),
        ({"active_seconds": 60.0}, "reached its 60 second runtime budget"),
    ],
)
def test_a_job_that_has_spent_a_budget_is_refused_new_agent_work(
    spent: dict[str, Any], message: str
) -> None:
    budget = ResearchBudget(max_tokens=1_000, max_cost_usd=2.0, max_runtime_seconds=60)
    ledger = BudgetLedger()
    job_id = uuid4()
    ledger.ensure_within(job_id, budget)

    ledger.record_agent_call(job_id, **spent)

    with pytest.raises(BudgetExhausted, match=message):
        ledger.ensure_within(job_id, budget)


def test_one_jobs_spending_never_exhausts_another_jobs_budget() -> None:
    budget = ResearchBudget(max_tokens=1_000)
    ledger = BudgetLedger()
    ledger.record_agent_call(uuid4(), prompt_tokens=5_000)

    ledger.ensure_within(uuid4(), budget)


def test_the_tool_call_budget_keeps_its_own_wording() -> None:
    ledger = BudgetLedger()
    job_id = uuid4()
    budget = ResearchBudget(max_tool_calls=1)
    ledger.reserve_call(job_id, budget)

    with pytest.raises(BudgetExhausted, match="reached its 1 tool-call budget") as refused:
        ledger.reserve_call(job_id, budget)

    assert refused.value.kind == "tool-call"


def test_a_breaker_reports_the_state_of_every_server_it_has_seen() -> None:
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_success("web-research")
    breaker.record_failure("github")

    assert breaker.states() == {"web-research": CircuitState.CLOSED, "github": CircuitState.OPEN}


# -- the activities -------------------------------------------------------------------------------


class NoTools:
    def execute(self, _request: ExecutionRequest) -> str:
        raise AssertionError("no tool should be called")


def activities_for(
    agent: Agent, metrics: PlatformMetrics | None = None
) -> tuple[ResearchActivities, BudgetLedger]:
    registry = default_registry()
    gateway = CapabilityGateway(registry=registry, executor=NoTools())
    research = ResearchActivities(
        gateway=gateway,
        registry=registry,
        build_agent=lambda _role, _tools: agent,
        pricing=TokenPricing(input_per_million_usd=1.0, output_per_million_usd=2.0),
        metrics=metrics,
    )
    return research, gateway.budgets


def a_job(**budget: Any) -> ResearchJob:
    return ResearchJob(
        tenant_id=TENANT,
        requester_id="requester-1",
        question="What does the vendor charge?",
        budget=ResearchBudget(**budget),
    )


@pytest.mark.asyncio
async def test_an_agent_call_is_recorded_against_its_job_with_tokens_cost_and_time() -> None:
    agent = Agent(VALID_PLAN, usage={"prompt_tokens": 1_000, "completion_tokens": 500})
    research, budgets = activities_for(agent)
    job = a_job()

    await ActivityEnvironment().run(research.plan, job)

    spent = budgets.usage(job.id)
    assert spent.agent_calls == 1
    assert (spent.prompt_tokens, spent.completion_tokens) == (1_000, 500)
    assert spent.cost_usd == pytest.approx(0.002)
    assert spent.active_seconds >= 0
    assert spent.schema_corrections == 0


@pytest.mark.asyncio
async def test_a_job_over_its_token_budget_is_refused_before_the_model_is_called() -> None:
    agent = Agent(VALID_PLAN, usage={"prompt_tokens": 1_000, "completion_tokens": 0})
    research, budgets = activities_for(agent)
    job = a_job(max_tokens=1_000)
    budgets.record_agent_call(job.id, prompt_tokens=1_000)

    with pytest.raises(BudgetExhausted, match="token budget"):
        await ActivityEnvironment().run(research.plan, job)

    assert agent.calls == 0


@pytest.mark.asyncio
async def test_a_call_that_never_produced_its_contract_still_costs_what_it_used() -> None:
    agent = Agent("not json", usage={"prompt_tokens": 100, "completion_tokens": 10})
    research, budgets = activities_for(agent)
    job = a_job()

    with pytest.raises(SchemaCorrectionExhausted):
        await ActivityEnvironment().run(research.plan, job)

    spent = budgets.usage(job.id)
    assert spent.prompt_tokens == 300
    assert spent.schema_corrections == 2
    assert spent.agent_calls == 1


@pytest.mark.asyncio
async def test_tokens_cost_and_corrections_are_measured_by_agent_role() -> None:
    metrics, reader = build_metrics()
    agent = Agent("not json", VALID_PLAN, usage={"prompt_tokens": 100, "completion_tokens": 10})
    research, _budgets = activities_for(agent, metrics)

    await ActivityEnvironment().run(research.plan, a_job())

    tokens = {attrs["token.type"]: value for value, attrs in points(reader, "research.llm.tokens")}
    assert tokens == {"input": 200, "output": 20}
    [(cost, cost_attributes)] = points(reader, "research.llm.cost")
    assert cost == pytest.approx(0.00024)
    assert cost_attributes == {"tenant.id": TENANT, "agent_role": "planner"}
    assert [value for value, _ in points(reader, "research.agent.schema_corrections")] == [1]
    assert points(reader, "research.activity.retries") == []


@pytest.mark.asyncio
async def test_a_retried_activity_is_counted_as_a_retry() -> None:
    metrics, reader = build_metrics()
    research, _budgets = activities_for(Agent(VALID_PLAN), metrics)
    environment = ActivityEnvironment()
    environment.info = environment.info.__class__(  # type: ignore[call-arg]
        **{**environment.info.__dict__, "attempt": 2}
    )

    await environment.run(research.plan, a_job())

    [(retries, attributes)] = points(reader, "research.activity.retries")
    assert retries == 1
    assert attributes["agent_role"] == "planner"


@pytest.mark.asyncio
async def test_each_status_a_job_reaches_is_stored_with_what_it_had_spent() -> None:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    budgets = BudgetLedger()
    budgets.reserve_call(job.id, job.budget)
    budgets.record_agent_call(job.id, prompt_tokens=400, completion_tokens=100, cost_usd=0.25)
    persistence = JobActivities(jobs=service, budgets=budgets)

    stored = await ActivityEnvironment().run(
        persistence.transition, TENANT, job.id, JobStatus.PLANNING, None
    )

    assert stored.usage.tool_calls == 1
    assert stored.usage.total_tokens == 500
    assert stored.usage.cost_usd == 0.25
    assert service.get(TENANT, job.id).usage == stored.usage


@pytest.mark.asyncio
async def test_a_job_with_no_ledger_keeps_the_usage_it_already_had() -> None:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    service.transition(TENANT, job.id, JobStatus.PLANNING, usage=JobUsage(prompt_tokens=7))

    stored = await ActivityEnvironment().run(
        JobActivities(jobs=service).transition, TENANT, job.id, JobStatus.RESEARCHING, None
    )

    assert stored.usage.prompt_tokens == 7


# -- the pipeline ---------------------------------------------------------------------------------

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


class Pipeline:
    """The five steps, any one of which can be made to find the budget spent."""

    def __init__(self, service: ResearchJobService, job: ResearchJob) -> None:
        self.service = service
        self.job = job
        self.exhausted_at: str | None = None
        self.contradicted = False

    def _spend(self, step: str) -> None:
        if self.exhausted_at == step:
            raise BudgetExhausted(self.job.id, 1_000, "token")

    async def plan(self, _job: ResearchJob) -> ResearchPlan:
        self._spend("plan")
        return PLAN

    async def research(self, _job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        self._spend("research")
        return EvidenceSubmission(
            records=[
                EvidenceRecordCreate(
                    excerpt="Vendor pricing is 20 USD per seat.",
                    source_uri="https://vendor.test/pricing",
                    content_hash=f"sha256:{'0' * 64}",
                    producing_task_id=task.id,
                    tool_invocation_id=uuid4(),
                )
            ]
        )

    async def analyze(self, _job: ResearchJob, evidence: list[EvidenceRecord]) -> AnalysisResult:
        self._spend("analyze")
        return AnalysisResult(
            findings=[
                ProposedFinding(
                    claim=CLAIM,
                    supporting_evidence_ids=[record.id for record in evidence],
                    confidence=0.9,
                )
            ]
        )

    async def critique(
        self, _job: ResearchJob, _analysis: AnalysisResult, _evidence: list[EvidenceRecord]
    ) -> CriticReview:
        self._spend("critique")
        return CriticReview(
            verdicts=[
                ClaimVerdict(
                    claim=CLAIM,
                    verdict=CriticVerdict.UNSUPPORTED
                    if self.contradicted
                    else CriticVerdict.SUPPORTED,
                    reasoning="Checked against the source.",
                )
            ]
        )

    async def report(
        self, _job: ResearchJob, _critique: CriticReview, evidence: list[EvidenceRecord]
    ) -> ResearchReport:
        self._spend("report")
        return ResearchReport(
            title="Vendor pricing",
            sections=[ReportSection(heading="Pricing", body=f"{CLAIM[:-1]} [{evidence[0].id}].")],
        )

    async def reviewer(self, *_args: object) -> ReviewerDecision:
        raise AssertionError("no reviewer decision should have been awaited")

    async def run(self, metrics: PlatformMetrics | None = None) -> Any:
        return await run_research_job(
            self.job,
            jobs=AsyncJobs(self.service),
            activities=OrchestrationActivities(
                plan=self.plan,
                research=self.research,
                analyze=self.analyze,
                critique=self.critique,
                report=self.report,
                await_reviewer_decision=self.reviewer,
            ),
            metrics=metrics,
        )


@pytest.fixture
def pipeline() -> Pipeline:
    service = ResearchJobService(InMemoryJobRepository())
    job = service.create(TENANT, "requester-1", ResearchJobCreate(question="What does it cost?"))
    return Pipeline(service, job)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["analyze", "critique", "report"])
async def test_a_budget_spent_after_evidence_was_collected_ends_the_job_partial(
    pipeline: Pipeline, step: str
) -> None:
    """Section 12: stop new work and return clearly labelled partial findings."""
    pipeline.exhausted_at = step

    outcome = await pipeline.run()

    reason = f"BudgetExhausted: job {pipeline.job.id} reached its 1000 token budget"
    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.job.status_detail == reason
    assert outcome.failure == reason
    assert len(pipeline.service.list_evidence(TENANT, pipeline.job.id)) == 1


@pytest.mark.asyncio
async def test_a_budget_spent_before_anything_was_found_ends_the_job_failed(
    pipeline: Pipeline,
) -> None:
    """There is nothing partial to return from a job that never got past planning."""
    pipeline.exhausted_at = "plan"

    outcome = await pipeline.run()

    assert outcome.job.status is JobStatus.FAILED
    assert "token budget" in (outcome.job.status_detail or "")


@pytest.mark.asyncio
async def test_a_researcher_out_of_budget_leaves_the_job_partial_with_no_evidence(
    pipeline: Pipeline,
) -> None:
    pipeline.exhausted_at = "research"

    outcome = await pipeline.run()

    assert outcome.job.status is JobStatus.PARTIAL
    assert outcome.job.status_detail == "no evidence could be collected"


@pytest.mark.asyncio
async def test_a_completed_job_reports_its_usage_and_its_settled_findings(
    pipeline: Pipeline,
) -> None:
    metrics, reader = build_metrics()

    outcome = await pipeline.run(metrics)

    assert outcome.job.status is JobStatus.COMPLETED
    [(tokens, attributes)] = points(reader, "research.job.tokens")
    assert tokens == 0
    assert attributes == {"tenant.id": TENANT, "status": "completed"}
    assert len(points(reader, "research.job.cost")) == 1
    assert len(points(reader, "research.job.active_time")) == 1
    [(count, finding)] = points(reader, "research.findings")
    assert count == 1
    assert finding["critic_verdict"] == "supported"
    assert finding["reviewer_status"] == "not_required"


@pytest.mark.asyncio
async def test_an_unsupported_claim_is_counted_as_unsupported(pipeline: Pipeline) -> None:
    metrics, reader = build_metrics()
    pipeline.contradicted = True

    await pipeline.run(metrics)

    [(_, finding)] = points(reader, "research.findings")
    assert finding["critic_verdict"] == "unsupported"


# -- circuit state, and replay ------------------------------------------------------------


def test_circuit_state_is_reported_per_server_from_the_breaker_itself() -> None:
    metrics, reader = build_metrics()
    breaker = CircuitBreaker(failure_threshold=1)
    metrics.observe_circuits(breaker.states)
    breaker.record_success("web-research")
    breaker.record_failure("github")

    states = {attrs["mcp.server"]: value for value, attrs in points(reader, "mcp.circuit.state")}

    assert states == {"web-research": 0, "github": 2}


def test_a_replayed_workflow_does_not_count_what_it_already_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, reader = build_metrics()
    guarded = ReplaySafeMetrics(metrics)
    replaying = True
    monkeypatch.setattr(
        research_workflow.workflow.unsafe, "is_replaying", lambda: replaying, raising=True
    )

    guarded.task_completions.add(1, {"outcome": "succeeded"})
    guarded.approval_wait_time.record(30.0, {"tenant.id": TENANT})
    replaying = False
    guarded.task_completions.add(1, {"outcome": "succeeded"})
    guarded.approval_wait_time.record(45.0, {"tenant.id": TENANT})

    assert [value for value, _ in points(reader, "research.tasks.completed")] == [1]
    assert [value for value, _ in points(reader, "research.review.wait_time")] == [45.0]


def test_active_jobs_is_counted_on_replay_because_the_worker_holds_the_job_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics, reader = build_metrics()
    monkeypatch.setattr(research_workflow.workflow.unsafe, "is_replaying", lambda: True)

    ReplaySafeMetrics(metrics).active_jobs.add(1, {"tenant.id": TENANT})

    assert [value for value, _ in points(reader, "research.jobs.active")] == [1]


def test_job_usage_defaults_to_nothing_spent() -> None:
    job = a_job()

    assert job.usage == JobUsage()
    assert job.budget.max_tokens == 2_000_000
    assert isinstance(job.id, UUID)
