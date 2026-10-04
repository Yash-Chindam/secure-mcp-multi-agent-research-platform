"""Run the real pipeline over labelled scenarios and score it against section 14.

``scoring.py`` defines each metric. This module produces the numbers: it stands up the
platform as a deployment would - the capability gateway, a web research MCP server over a
fixed corpus, the five agent activities, publication - runs every scenario to a terminal
status, and reads the scores off what the job actually recorded.

Nothing here is specific to a model. The agents are built by whatever factory is passed
in, so the same suite scores a real model (``python -m research_platform.evaluation``)
and, with a scripted crew, checks in CI that the properties the platform itself
guarantees hold: every published citation resolves, a partial result is labelled, and no
cross-tenant read succeeds.

Recovery after a worker restart is the one section 14 item not scored here. It is a
property of a running Temporal workflow rather than of a finished job, and is proven by
``tests/integration/test_workflow_recovery.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from pydantic import BaseModel

from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    EvidenceSubmission,
    ResearchPlan,
    ResearchReport,
)
from research_platform.agents.crew import KickoffAgent
from research_platform.agents.provenance import hash_content
from research_platform.agents.usage import TokenPricing
from research_platform.application.artifacts import (
    ArtifactNotFound,
    InMemoryArtifactStore,
    job_artifact,
)
from research_platform.application.jobs import (
    AsyncJobs,
    InMemoryJobRepository,
    JobNotFoundError,
    ResearchJobService,
)
from research_platform.application.publication import ReportPublication
from research_platform.domain.invocations import ErrorClass
from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    Finding,
    JobStatus,
    ResearchBudget,
    ResearchJob,
    ResearchJobCreate,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.evaluation.scoring import (
    citation_correctness,
    claim_support_rate,
    research_coverage,
    task_completion_rate,
    tool_call_validity_rate,
    tool_selection_accuracy,
)
from research_platform.identity import Principal
from research_platform.mcp.catalogue import DEFAULT_CAPABILITIES
from research_platform.mcp.fastmcp_executor import FastMCPExecutor
from research_platform.mcp.gateway import CapabilityDenied, CapabilityFailed, CapabilityGateway
from research_platform.mcp.registry import CapabilityNotFound, CapabilityRegistry
from research_platform.mcp.servers.backends import SourceDocument, StaticWebBackend
from research_platform.mcp.servers.deployment import build_servers
from research_platform.mcp.servers.web_boundary import DomainPolicy
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.orchestration import (
    OrchestrationActivities,
    ReviewerDecision,
    run_research_job,
)
from research_platform.workflow.publishing import (
    REPORT_JSON,
    GatewaySourceChecker,
    PublicationActivities,
)

AgentFactory = Callable[[AgentRole, list[Any]], KickoffAgent]

TENANT = "evaluation"
CORPUS_DOMAIN = "corpus.test"

SCHEMA_VALID_TOOL_CALL_TARGET = 0.95
"""Section 14: at least 95% schema-valid tool calls on the evaluation suite."""


@dataclass(frozen=True)
class Scenario:
    """One labelled research assignment: a question, its corpus, and the ground truth."""

    name: str
    question: str
    documents: tuple[SourceDocument, ...]
    expected_tools: frozenset[str]
    known_contradictions: tuple[str, ...] = ()
    """Phrases the critic should flag. A contradiction counts as caught when some claim
    the critic marked contradicted contains the phrase, ignoring case - a model words a
    claim its own way, so the ground truth names the fact rather than the sentence."""
    budget: ResearchBudget = field(default_factory=ResearchBudget)


class ScenarioResult(BaseModel):
    """What one scenario's job recorded, scored."""

    name: str
    status: JobStatus
    status_detail: str | None
    completed: bool
    tools_used: list[str]
    tool_selection_accuracy: float
    citation_correctness: float | None
    claim_support_rate: float | None
    research_coverage: float | None
    contradiction_recall: float | None
    tool_calls: int
    invalid_tool_calls: int
    total_tokens: int
    cost_usd: float
    active_seconds: float
    published: bool
    partial_is_labelled: bool


class CrossTenantProbe(BaseModel):
    """One attempt by another tenant to reach a job that is not theirs."""

    attempt: str
    blocked: bool


class EvaluationReport(BaseModel):
    """The section 14 scores for one run of the suite, and whether each target was met."""

    scenarios: list[ScenarioResult]
    cross_tenant_probes: list[CrossTenantProbe]
    task_completion_rate: float
    tool_selection_accuracy: float
    tool_call_validity_rate: float
    citation_correctness: float | None
    claim_support_rate: float | None
    research_coverage: float | None
    contradiction_recall: float | None
    cross_tenant_successes: int
    mean_cost_usd_per_completed_report: float | None
    mean_active_seconds_per_completed_report: float | None
    targets: dict[str, bool]

    @property
    def meets_targets(self) -> bool:
        return all(self.targets.values())


@dataclass
class _Observed:
    """What the agents returned along the way, kept so it can be scored afterwards."""

    plans: list[ResearchPlan] = field(default_factory=list)
    reviews: list[CriticReview] = field(default_factory=list)
    requested: int = 0
    unmet: int = 0


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _contradiction_recall(review: CriticReview, known: tuple[str, ...]) -> float:
    flagged = [
        verdict.claim.lower()
        for verdict in review.verdicts
        if verdict.verdict is CriticVerdict.CONTRADICTED
    ]
    caught = sum(1 for phrase in known if any(phrase.lower() in claim for claim in flagged))
    return caught / len(known)


async def run_scenario(
    scenario: Scenario, build_agent: AgentFactory, *, pricing: TokenPricing | None = None
) -> ScenarioResult:
    """Run one scenario through the real pipeline and score what it recorded."""
    jobs = ResearchJobService(InMemoryJobRepository())
    job = jobs.create(
        TENANT,
        "evaluator",
        ResearchJobCreate(question=scenario.question, budget=scenario.budget),
    )
    servers = build_servers(
        web_backend=StaticWebBackend(
            documents={document.url: document for document in scenario.documents}
        ),
        web_policy=DomainPolicy(domains=frozenset({CORPUS_DOMAIN})),
        web_requests_per_minute=10_000,
    )
    registry = CapabilityRegistry(
        [capability for capability in DEFAULT_CAPABILITIES if capability.server in servers]
    )
    gateway = CapabilityGateway(
        registry=registry, executor=FastMCPExecutor(servers), audit=jobs.record_invocation
    )
    research = ResearchActivities(
        gateway=gateway,
        registry=registry,
        build_agent=build_agent,
        pricing=pricing or TokenPricing(),
    )
    persistence = JobActivities(jobs=jobs, budgets=gateway.budgets)
    publication = PublicationActivities(
        jobs=jobs, artifacts=InMemoryArtifactStore(), sources=GatewaySourceChecker(gateway)
    )
    observed = _Observed()

    async def plan(target: ResearchJob) -> ResearchPlan:
        produced = await research.plan(target)
        observed.plans.append(produced)
        return produced

    async def collect(target: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        requirements = len(task.evidence_requirements) or 1
        observed.requested += requirements
        try:
            submission = await research.research(target, task)
        except Exception:
            observed.unmet += requirements
            raise
        observed.unmet += min(requirements, len(submission.unmet_requirements))
        return submission

    async def critique(
        target: ResearchJob, analysis: AnalysisResult, evidence: list[EvidenceRecord]
    ) -> CriticReview:
        review = await research.critique(target, analysis, evidence)
        observed.reviews.append(review)
        return review

    async def approve(_job: ResearchJob, _review: CriticReview) -> ReviewerDecision:
        # The suite has no reviewer. Approving lets a flagged job reach its report, so
        # the report can be scored; the flag itself is still counted through the critic.
        return ReviewerDecision.APPROVE

    made_by_agents: set[UUID] | None = None

    async def publish(
        target: ResearchJob, report: ResearchReport, shortfalls: list[str]
    ) -> ReportPublication:
        # Publication re-reads cited sources through the gateway. Those calls are the
        # platform's, not the agents', so the agents' calls are noted before it starts.
        nonlocal made_by_agents
        made_by_agents = {call.id for call in jobs.list_invocations(TENANT, target.id)}
        return await publication.publish(target, report, shortfalls)

    class Jobs(AsyncJobs):
        """Job state written through the same activities a worker registers."""

        async def transition(self, *args: Any, **kwargs: Any) -> ResearchJob:
            return await persistence.transition(*args, **kwargs)

        async def add_evidence(self, *args: Any) -> EvidenceRecord:
            return await persistence.add_evidence(*args)

    outcome = await run_research_job(
        job,
        jobs=Jobs(jobs),
        activities=OrchestrationActivities(
            plan=plan,
            research=collect,
            analyze=research.analyze,
            critique=critique,
            report=research.report,
            await_reviewer_decision=approve,
            publish=publish,
        ),
    )

    final = jobs.get(TENANT, job.id)
    evidence = jobs.list_evidence(TENANT, job.id)
    invocations = jobs.list_invocations(TENANT, job.id)
    agent_calls = [
        call for call in invocations if made_by_agents is None or call.id in made_by_agents
    ]
    used = frozenset(f"{call.mcp_server}.{call.capability}" for call in agent_calls)
    review = observed.reviews[-1] if observed.reviews else None
    report: ResearchReport | None = outcome.report
    return ScenarioResult(
        name=scenario.name,
        status=final.status,
        status_detail=final.status_detail,
        completed=final.status is JobStatus.COMPLETED,
        tools_used=sorted(used),
        tool_selection_accuracy=tool_selection_accuracy(
            used=used, expected=scenario.expected_tools
        ),
        citation_correctness=(
            citation_correctness(
                report, available_evidence_ids=frozenset(record.id for record in evidence)
            )
            if report is not None
            else None
        ),
        claim_support_rate=claim_support_rate(review) if review is not None else None,
        research_coverage=(
            research_coverage(requested=observed.requested, unmet=observed.unmet)
            if observed.requested
            else None
        ),
        contradiction_recall=(
            _contradiction_recall(review, scenario.known_contradictions)
            if review is not None and scenario.known_contradictions
            else None
        ),
        tool_calls=len(agent_calls),
        invalid_tool_calls=sum(
            1 for call in agent_calls if call.error_class is ErrorClass.INVALID_ARGUMENTS
        ),
        total_tokens=final.usage.total_tokens,
        cost_usd=final.usage.cost_usd,
        active_seconds=final.usage.active_seconds,
        published=outcome.publication is not None,
        # A job that could not fully complete must say so: a status other than
        # completed always carries the reason (section 14's last design target).
        partial_is_labelled=final.status is JobStatus.COMPLETED or bool(final.status_detail),
    )


def probe_cross_tenant_access() -> list[CrossTenantProbe]:
    """Have a second tenant try every way of reaching another tenant's job.

    The job is real - evidence, findings, an audit record and a published report - so a
    probe that got through would have something to read. Each probe is blocked when it
    is refused or comes back empty; section 14's target is that none succeeds.
    """
    # Imported here: the API imports the workflow package this module is used by.
    from research_platform.main import create_app
    from research_platform.settings import Settings

    client = TestClient(create_app(Settings()))
    jobs: ResearchJobService = client.app.state.job_service  # type: ignore[attr-defined]
    artifacts: InMemoryArtifactStore = client.app.state.artifacts  # type: ignore[attr-defined]
    owner = {"X-Tenant-ID": "tenant-a", "X-Requester-ID": "owner", "X-Clearance": "restricted"}
    intruder = {
        "X-Tenant-ID": "tenant-b",
        "X-Requester-ID": "intruder",
        "X-Clearance": "restricted",
        "X-Roles": "requester,reviewer,administrator",
    }

    created = client.post("/api/v1/jobs", headers=owner, json={"question": "What is the price?"})
    job = jobs.get("tenant-a", UUID(created.json()["id"]))
    excerpt = "The negotiated price is 14 USD per seat."
    record = jobs.add_evidence(
        "tenant-a",
        job.id,
        EvidenceRecordCreate(
            excerpt=excerpt,
            source_uri=f"https://{CORPUS_DOMAIN}/contract",
            content_hash=hash_content(excerpt),
            producing_task_id=uuid4(),
            tool_invocation_id=uuid4(),
        ),
    )
    jobs.record_findings(
        "tenant-a",
        job.id,
        [
            Finding(
                claim=excerpt,
                supporting_evidence_ids=[record.id],
                confidence=0.9,
                critic_verdict=CriticVerdict.SUPPORTED,
            )
        ],
    )
    report = ResearchReport.model_validate(
        {
            "title": "Negotiated pricing",
            "sections": [{"heading": "Price", "body": f"{excerpt[:-1]} [{record.id}]."}],
        }
    )
    PublicationActivities(jobs=jobs, artifacts=artifacts)._publish(job, report, [])

    base = f"/api/v1/jobs/{job.id}"
    probes: list[CrossTenantProbe] = []

    def http(attempt: str, method: str, path: str, **kwargs: Any) -> None:
        response = client.request(method, path, headers=intruder, **kwargs)
        # A read gets through only if it returns something of the job's; a write gets
        # through if it is accepted at all.
        exposed = excerpt in response.text or str(job.id) in response.text
        got_through = response.is_success and (method != "GET" or exposed)
        probes.append(CrossTenantProbe(attempt=attempt, blocked=not got_through))

    http("read the job", "GET", base)
    http("read its evidence", "GET", f"{base}/evidence")
    http("read its findings", "GET", f"{base}/findings")
    http("read its audit trail", "GET", f"{base}/invocations")
    http("read its report", "GET", f"{base}/report")
    http("read its report as markdown", "GET", f"{base}/report", params={"format": "markdown"})
    http("read its manifest", "GET", f"{base}/manifest")
    http("move it to another status", "POST", f"{base}/transitions", params={"target": "failed"})
    http("decide its review", "POST", f"{base}/review", json={"decision": "approve"})
    http(
        "attach evidence to it",
        "POST",
        f"{base}/evidence",
        json={
            "excerpt": "Planted.",
            "source_uri": "https://attacker.test/",
            "content_hash": hash_content("Planted."),
            "producing_task_id": str(uuid4()),
            "tool_invocation_id": str(uuid4()),
        },
    )

    listed = client.get("/api/v1/jobs", headers=intruder).json()
    probes.append(CrossTenantProbe(attempt="find it in a job listing", blocked=listed == []))

    def direct(attempt: str, call: Callable[[], object]) -> None:
        try:
            found = call()
        except (JobNotFoundError, ArtifactNotFound):
            probes.append(CrossTenantProbe(attempt=attempt, blocked=True))
        else:
            probes.append(CrossTenantProbe(attempt=attempt, blocked=not found))

    direct("read the job from the service", lambda: jobs.get("tenant-b", job.id))
    direct("read its evidence from the service", lambda: jobs.list_evidence("tenant-b", job.id))
    direct("read its findings from the service", lambda: jobs.list_findings("tenant-b", job.id))
    direct(
        "read its publication from the service", lambda: jobs.get_publication("tenant-b", job.id)
    )
    direct(
        "read its report from the artifact store",
        lambda: artifacts.get("tenant-b", job_artifact(job.id, REPORT_JSON)),
    )

    # An agent working for the other tenant asks the evidence server for this job's
    # evidence by naming the job. The server is given the caller's tenant by the
    # platform, never by the agent, so the job is simply not there for it.
    servers = build_servers(evidence_source=jobs)
    registry = CapabilityRegistry(
        [capability for capability in DEFAULT_CAPABILITIES if capability.server in servers]
    )
    gateway = CapabilityGateway(registry=registry, executor=FastMCPExecutor(servers))
    agent = Principal(
        tenant_id="tenant-b",
        subject_id="job:intruder",
        clearance=job.clearance,
    ).for_agent(AgentRole.CRITIC)
    try:
        result = gateway.invoke(
            principal=agent,
            job_id=job.id,
            task_id=uuid4(),
            server="evidence",
            capability_name="retrieve",
            arguments={"job_id": str(job.id)},
            budget=ResearchBudget(),
        )
    except (CapabilityDenied, CapabilityFailed, CapabilityNotFound):
        probes.append(CrossTenantProbe(attempt="retrieve its evidence as an agent", blocked=True))
    else:
        probes.append(
            CrossTenantProbe(
                attempt="retrieve its evidence as an agent",
                blocked=excerpt not in result.content.text,
            )
        )

    # None of that may have changed the job it was aimed at.
    untouched = jobs.get("tenant-a", job.id).status is JobStatus.CREATED
    probes.append(CrossTenantProbe(attempt="leave the job changed", blocked=untouched))
    probes.append(
        CrossTenantProbe(
            attempt="leave planted evidence on the job",
            blocked=len(jobs.list_evidence("tenant-a", job.id)) == 1,
        )
    )
    return probes


async def run_suite(
    scenarios: Sequence[Scenario],
    build_agent: AgentFactory,
    *,
    pricing: TokenPricing | None = None,
) -> EvaluationReport:
    """Run every scenario and the cross-tenant probes, and score the run as a whole."""
    if not scenarios:
        raise ValueError("an evaluation needs at least one scenario")
    results = [await run_scenario(scenario, build_agent, pricing=pricing) for scenario in scenarios]
    probes = probe_cross_tenant_access()
    completed = [result for result in results if result.completed]
    published = [result for result in results if result.citation_correctness is not None]
    total_calls = sum(result.tool_calls for result in results)
    validity = tool_call_validity_rate(
        calls=total_calls, invalid=sum(result.invalid_tool_calls for result in results)
    )
    successes = sum(1 for probe in probes if not probe.blocked)
    return EvaluationReport(
        scenarios=results,
        cross_tenant_probes=probes,
        task_completion_rate=task_completion_rate(completed=len(completed), total=len(results)),
        tool_selection_accuracy=fmean(result.tool_selection_accuracy for result in results),
        tool_call_validity_rate=validity,
        citation_correctness=_mean([result.citation_correctness for result in results]),
        claim_support_rate=_mean([result.claim_support_rate for result in results]),
        research_coverage=_mean([result.research_coverage for result in results]),
        contradiction_recall=_mean([result.contradiction_recall for result in results]),
        cross_tenant_successes=successes,
        mean_cost_usd_per_completed_report=_mean([result.cost_usd for result in completed]),
        mean_active_seconds_per_completed_report=_mean(
            [result.active_seconds for result in completed]
        ),
        targets={
            "at least 95% schema-valid tool calls": validity >= SCHEMA_VALID_TOOL_CALL_TARGET,
            "every published factual claim linked to evidence": all(
                result.citation_correctness == 1.0 for result in published
            ),
            "zero successful cross-tenant access attempts": successes == 0,
            "partial output clearly identified": all(
                result.partial_is_labelled for result in results
            ),
        },
    )
