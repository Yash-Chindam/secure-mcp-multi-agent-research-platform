"""Temporal activities that run the section 7 agents for real.

Everything non-deterministic - an LLM call through CrewAI, a tool call through the
capability gateway - lives here rather than in the workflow itself. Temporal replays a
workflow's own code from history on every worker restart, and that replay must produce
the same result every time; an activity is not replayed, Temporal records that it
already ran and hands the workflow its recorded result instead (section 12's requirement
that a restarted worker resumes without repeating a completed side effect).

``build_agent`` is injected rather than called directly so a test can hand these
activities a fake agent instead of one backed by a real LLM (see
``research_platform.agents.crew.build_agent`` for the production factory).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from temporalio import activity

from research_platform.agents.checks import check_analysis, check_report, check_review
from research_platform.agents.contracts import (
    AnalysisResult,
    CriticReview,
    EvidenceSubmission,
    ResearchPlan,
    ResearchReport,
)
from research_platform.agents.crew import KickoffAgent, request_agent_output
from research_platform.agents.provenance import EvidenceClaims, EvidenceLedger, verify_claims
from research_platform.agents.tools import (
    ApprovalProvider,
    build_agent_tools,
    describe_visible_capabilities,
    no_approval,
)
from research_platform.application.jobs import ResearchJobService
from research_platform.domain.models import (
    EvidenceRecord,
    EvidenceRecordCreate,
    Finding,
    FindingRecord,
    JobStatus,
    ResearchJob,
)
from research_platform.domain.tasks import AgentRole, ResearchTask
from research_platform.identity import Principal
from research_platform.mcp.gateway import CapabilityGateway
from research_platform.mcp.registry import CapabilityRegistry


def principal_for(job: ResearchJob, role: AgentRole) -> Principal:
    return Principal(
        tenant_id=job.tenant_id, subject_id=f"job:{job.id}", roles=frozenset()
    ).for_agent(role)


def _describe_evidence(evidence: list[EvidenceRecord]) -> str:
    if not evidence:
        return "No evidence has been collected yet."
    return "\n".join(
        f"[{record.id}] {record.excerpt} "
        f"(source: {record.source_uri}, trust: {record.trust_level.value})"
        for record in evidence
    )


@dataclass(frozen=True)
class ResearchActivities:
    """Bound activities for one deployment: one gateway, one registry, one agent factory."""

    gateway: CapabilityGateway
    registry: CapabilityRegistry
    build_agent: Callable[[AgentRole, list[Any]], KickoffAgent]
    approval_provider: ApprovalProvider = field(default=no_approval)

    @activity.defn(name="plan_research")
    async def plan(self, job: ResearchJob) -> ResearchPlan:
        principal = principal_for(job, AgentRole.PLANNER)
        catalogue = "\n".join(describe_visible_capabilities(self.registry, principal))
        agent = self.build_agent(AgentRole.PLANNER, [])
        instructions = (
            f"Research question: {job.question}\n"
            f"Constraints: {', '.join(job.constraints) or 'none'}\n"
            f"Source requirements: {', '.join(job.source_requirements) or 'none'}\n\n"
            f"Capabilities available to the crew:\n{catalogue}\n\n"
            "Decompose this into a ResearchPlan."
        )
        result = request_agent_output(agent, AgentRole.PLANNER, instructions=instructions)
        assert isinstance(result, ResearchPlan)
        return result

    @activity.defn(name="research_task")
    async def research(self, job: ResearchJob, task: ResearchTask) -> EvidenceSubmission:
        principal = principal_for(job, AgentRole.RESEARCHER)
        ledger = EvidenceLedger()
        tools = build_agent_tools(
            gateway=self.gateway,
            registry=self.registry,
            principal=principal,
            job_id=job.id,
            task_id=task.id,
            budget=job.budget,
            approval_provider=self.approval_provider,
            ledger=ledger,
        )
        agent = self.build_agent(AgentRole.RESEARCHER, tools)
        instructions = (
            f"Objective: {task.objective}\n"
            f"Evidence requirements: {', '.join(task.evidence_requirements) or 'none stated'}\n"
            f"Source restrictions: {', '.join(task.source_restrictions) or 'none'}\n\n"
            "Collect evidence with your tools and produce EvidenceClaims. Each claim "
            "quotes an excerpt exactly as a tool returned it and gives the "
            "tool_invocation_id printed at the top of that tool result. Record any "
            "requirement you could not meet instead of guessing at it."
        )
        claims = request_agent_output(
            agent,
            AgentRole.RESEARCHER,
            instructions=instructions,
            verify=lambda claimed: verify_claims(claimed, ledger=ledger, task_id=task.id),
        )
        assert isinstance(claims, EvidenceClaims)
        # The agent supplied excerpts and the calls it says they came from. The source,
        # the hash and the classification of each record are the platform's own.
        return verify_claims(claims, ledger=ledger, task_id=task.id)

    @activity.defn(name="analyze_evidence")
    async def analyze(self, job: ResearchJob, evidence: list[EvidenceRecord]) -> AnalysisResult:
        principal = principal_for(job, AgentRole.ANALYST)
        tools = build_agent_tools(
            gateway=self.gateway,
            registry=self.registry,
            principal=principal,
            job_id=job.id,
            task_id=uuid4(),
            budget=job.budget,
            approval_provider=self.approval_provider,
        )
        agent = self.build_agent(AgentRole.ANALYST, tools)
        instructions = (
            f"Research question: {job.question}\n\nEvidence collected so far:\n"
            f"{_describe_evidence(evidence)}\n\n"
            "Compare this evidence and produce an AnalysisResult. Every finding must cite "
            "the evidence identifiers above; do not invent one."
        )
        result = request_agent_output(
            agent,
            AgentRole.ANALYST,
            instructions=instructions,
            verify=lambda proposed: check_analysis(proposed, evidence),
        )
        assert isinstance(result, AnalysisResult)
        return result

    @activity.defn(name="critique_analysis")
    async def critique(
        self, job: ResearchJob, analysis: AnalysisResult, evidence: list[EvidenceRecord]
    ) -> CriticReview:
        principal = principal_for(job, AgentRole.CRITIC)
        tools = build_agent_tools(
            gateway=self.gateway,
            registry=self.registry,
            principal=principal,
            job_id=job.id,
            task_id=uuid4(),
            budget=job.budget,
            approval_provider=self.approval_provider,
        )
        agent = self.build_agent(AgentRole.CRITIC, tools)
        findings = "\n".join(
            f"- {finding.claim} (supported by {[str(i) for i in finding.supporting_evidence_ids]})"
            for finding in analysis.findings
        )
        instructions = (
            f"Proposed findings:\n{findings}\n\nEvidence collected:\n"
            f"{_describe_evidence(evidence)}\n\n"
            "Judge each claim against the evidence and produce a CriticReview."
        )
        result = request_agent_output(
            agent,
            AgentRole.CRITIC,
            instructions=instructions,
            verify=lambda review: check_review(review, analysis, evidence),
        )
        assert isinstance(result, CriticReview)
        return result

    @activity.defn(name="write_report")
    async def report(
        self, job: ResearchJob, critique: CriticReview, evidence: list[EvidenceRecord]
    ) -> ResearchReport:
        principal = principal_for(job, AgentRole.REPORTER)
        tools = build_agent_tools(
            gateway=self.gateway,
            registry=self.registry,
            principal=principal,
            job_id=job.id,
            task_id=uuid4(),
            budget=job.budget,
            approval_provider=self.approval_provider,
        )
        agent = self.build_agent(AgentRole.REPORTER, tools)
        supported = "\n".join(f"- {claim}" for claim in critique.supported_claims)
        gaps = ", ".join(critique.coverage_gaps) or "none"
        instructions = (
            f"Research question: {job.question}\n\nClaims the critic supported:\n{supported}\n\n"
            f"Evidence collected:\n{_describe_evidence(evidence)}\n\n"
            "Write the ResearchReport. Cite evidence for every factual sentence, using the "
            "identifiers above in square brackets. If the coverage gaps below mean the "
            f"report cannot be complete, mark it partial: {gaps}."
        )
        result = request_agent_output(
            agent,
            AgentRole.REPORTER,
            instructions=instructions,
            verify=lambda written: check_report(written, evidence),
        )
        assert isinstance(result, ResearchReport)
        return result


@dataclass(frozen=True)
class JobActivities:
    """Persists job status transitions and evidence as durable activities.

    ``research_platform.workflow.orchestration`` only ever sees the ``JobsPort``
    protocol; when Temporal drives it, ``research_platform.workflow.research_workflow``
    implements that protocol by calling these activities, so a status transition, a
    piece of evidence or a set of findings survives a worker restart the same way every
    other step in the pipeline does (section 12). All are safe to redeliver: Temporal
    runs an activity at least once, so each recognises a call that already landed instead
    of repeating it.
    """

    jobs: ResearchJobService

    @activity.defn(name="transition_job")
    async def transition(
        self, tenant_id: str, job_id: UUID, target: JobStatus, detail: str | None = None
    ) -> ResearchJob:
        """Persist a status transition, tolerating a call that already landed.

        Temporal delivers an activity at least once: a worker can vanish after this
        method has already returned but before that success was reported back, and the
        replacement worker Temporal hands the retry to will call this again. Treating
        "already at the target status" as success rather than an ``InvalidStateTransition``
        keeps this activity safe to retry - the very thing a second worker's recovery
        depends on should not itself be a new failure mode.
        """
        current = self.jobs.get(tenant_id, job_id)
        if current.status is target:
            return current
        return self.jobs.transition(tenant_id, job_id, target, detail)

    @activity.defn(name="add_job_evidence")
    async def add_evidence(
        self, tenant_id: str, job_id: UUID, command: EvidenceRecordCreate
    ) -> EvidenceRecord:
        """Persist one piece of evidence, tolerating a call that already landed.

        The same at-least-once delivery ``transition`` guards against applies here, and
        an unguarded retry would store the excerpt a second time under a new identifier -
        a duplicated side effect section 12 rules out. A record is the same record when
        the same task captured the same content from the same source through the same
        tool call, so that is what a redelivery is recognised by.
        """
        for existing in self.jobs.list_evidence(tenant_id, job_id):
            if (
                existing.content_hash == command.content_hash
                and existing.producing_task_id == command.producing_task_id
                and existing.tool_invocation_id == command.tool_invocation_id
                and str(existing.source_uri) == str(command.source_uri)
            ):
                return existing
        return self.jobs.add_evidence(tenant_id, job_id, command)

    @activity.defn(name="record_job_findings")
    async def record_findings(
        self, tenant_id: str, job_id: UUID, findings: list[Finding]
    ) -> list[FindingRecord]:
        """Store the job's current findings. Replacing the set makes a redelivery harmless."""
        return self.jobs.record_findings(tenant_id, job_id, findings)
