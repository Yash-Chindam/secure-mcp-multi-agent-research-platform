"""The schemas every agent must produce.

Section 7 requires that all agent outputs use Pydantic schemas and that an invalid
response cannot advance the workflow state. These models are therefore the gate rather
than a convenience: the rules an agent could otherwise talk its way around — a plan whose
dependencies do not resolve, a finding with no supporting evidence, a report sentence with
no citation — are validation errors here.
"""

from __future__ import annotations

import re
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecordCreate,
    NonEmptyText,
    TrustLevel,
)
from research_platform.domain.tasks import AgentRole

CITATION = re.compile(r"\[(?P<identifier>[0-9a-fA-F-]{36})\]")
"""A claim cites evidence by writing its identifier in square brackets.

The citation must fall inside the sentence it supports, before the terminal
punctuation: "the team plan costs 20 USD [<id>]." A citation left dangling between
two sentences is ambiguous to a reader as well as to the validator, so it does not
satisfy the preceding sentence.
"""

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

HEDGES = (
    "we recommend",
    "this suggests",
    "in our view",
    "further research",
)
"""Openings that introduce interpretation rather than a factual assertion."""


class PlannedTask(BaseModel):
    """One task the planner proposes, before it is given an identifier."""

    model_config = {"frozen": True}

    objective: NonEmptyText
    assigned_agent: AgentRole
    evidence_requirements: list[NonEmptyText] = Field(min_length=1, max_length=20)
    source_restrictions: list[NonEmptyText] = Field(default_factory=list, max_length=20)
    depends_on: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def only_collecting_agents_may_be_assigned_evidence(self) -> PlannedTask:
        if self.assigned_agent is AgentRole.PLANNER:
            raise ValueError("the planner cannot assign work to itself")
        return self


class ResearchPlan(BaseModel):
    """The planner's decomposition of an assignment into ordered tasks."""

    model_config = {"frozen": True}

    tasks: list[PlannedTask] = Field(min_length=1, max_length=50)
    rationale: NonEmptyText

    @model_validator(mode="after")
    def dependencies_must_resolve_without_a_cycle(self) -> ResearchPlan:
        count = len(self.tasks)
        for position, task in enumerate(self.tasks):
            for dependency in task.depends_on:
                if dependency >= count:
                    raise ValueError(
                        f"task {position} depends on task {dependency}, which does not exist"
                    )
                if dependency == position:
                    raise ValueError(f"task {position} depends on itself")
        self._reject_cycles()
        return self

    def _reject_cycles(self) -> None:
        placed: set[int] = set()
        remaining = set(range(len(self.tasks)))
        while remaining:
            ready = {
                position
                for position in remaining
                if placed.issuperset(self.tasks[position].depends_on)
            }
            if not ready:
                raise ValueError("the planned tasks contain a dependency cycle")
            placed |= ready
            remaining -= ready

    @property
    def collects_evidence(self) -> bool:
        return any(task.assigned_agent is AgentRole.RESEARCHER for task in self.tasks)


class EvidenceSubmission(BaseModel):
    """A researcher's collected evidence for one task."""

    model_config = {"frozen": True}

    records: list[EvidenceRecordCreate] = Field(min_length=1, max_length=100)
    unmet_requirements: list[NonEmptyText] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def records_must_not_repeat_the_same_excerpt(self) -> EvidenceSubmission:
        hashes = [record.content_hash for record in self.records]
        if len(set(hashes)) != len(hashes):
            raise ValueError("the same excerpt was submitted more than once")
        return self

    @property
    def is_complete(self) -> bool:
        return not self.unmet_requirements


class Calculation(BaseModel):
    """A reproducible calculation the analyst performed."""

    model_config = {"frozen": True}

    id: UUID
    description: NonEmptyText
    expression: NonEmptyText
    result: NonEmptyText
    input_evidence_ids: list[UUID] = Field(min_length=1, max_length=50)


class ProposedFinding(BaseModel):
    """A claim the analyst proposes, with the evidence it rests on."""

    model_config = {"frozen": True}

    claim: NonEmptyText
    supporting_evidence_ids: list[UUID] = Field(min_length=1, max_length=50)
    contradicting_evidence_ids: list[UUID] = Field(default_factory=list, max_length=50)
    calculation_ids: list[UUID] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def evidence_cannot_both_support_and_contradict(self) -> ProposedFinding:
        if set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids):
            raise ValueError("evidence cannot both support and contradict a claim")
        return self


class AnalysisResult(BaseModel):
    """The analyst's findings and the calculations behind them."""

    model_config = {"frozen": True}

    findings: list[ProposedFinding] = Field(min_length=1, max_length=100)
    calculations: list[Calculation] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def cited_calculations_must_be_present(self) -> AnalysisResult:
        available = {calculation.id for calculation in self.calculations}
        for finding in self.findings:
            missing = sorted(str(cited) for cited in set(finding.calculation_ids) - available)
            if missing:
                raise ValueError(f"finding cites calculations that were not reported: {missing}")
        return self


class ClaimVerdict(BaseModel):
    """The critic's judgement on one claim."""

    model_config = {"frozen": True}

    claim: NonEmptyText
    verdict: CriticVerdict
    reasoning: NonEmptyText
    conflicting_evidence_ids: list[UUID] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def a_contradicted_claim_must_name_the_conflict(self) -> ClaimVerdict:
        if self.verdict is CriticVerdict.CONTRADICTED and not self.conflicting_evidence_ids:
            raise ValueError("a contradicted claim must name the conflicting evidence")
        if self.verdict is CriticVerdict.PENDING:
            raise ValueError("the critic must reach a verdict rather than leaving it pending")
        return self


class CriticReview(BaseModel):
    """The critic's verdicts, plus what the research did not cover."""

    model_config = {"frozen": True}

    verdicts: list[ClaimVerdict] = Field(min_length=1, max_length=100)
    coverage_gaps: list[NonEmptyText] = Field(default_factory=list, max_length=20)

    @property
    def supported_claims(self) -> list[str]:
        return [
            verdict.claim for verdict in self.verdicts if verdict.verdict is CriticVerdict.SUPPORTED
        ]

    @property
    def requires_reviewer(self) -> bool:
        """A conflict or a coverage gap is for a person to resolve, not the crew."""
        return bool(self.coverage_gaps) or any(
            verdict.verdict is CriticVerdict.CONTRADICTED for verdict in self.verdicts
        )


class ReportSection(BaseModel):
    """One section of the final report."""

    model_config = {"frozen": True}

    heading: NonEmptyText
    body: NonEmptyText

    @property
    def cited_evidence_ids(self) -> frozenset[UUID]:
        return frozenset(UUID(match.group("identifier")) for match in CITATION.finditer(self.body))

    def sentences(self) -> list[str]:
        """Split the body into sentences, keeping a trailing citation with its claim.

        Both "costs 20 USD [id]." and "costs 20 USD. [id]" are natural ways to cite, so a
        fragment that is nothing but citations is attached to the sentence it supports
        rather than being treated as an uncited assertion of its own.
        """
        fragments: list[str] = []
        for fragment in SENTENCE_BOUNDARY.split(self.body.strip()):
            candidate = fragment.strip()
            if not candidate:
                continue
            if fragments and not CITATION.sub("", candidate).strip():
                fragments[-1] = f"{fragments[-1]} {candidate}"
                continue
            fragments.append(candidate)
        return fragments

    def uncited_sentences(self) -> list[str]:
        """Return the factual sentences that cite nothing.

        A sentence that opens with a recommendation or an explicit interpretation is not a
        factual assertion, so it is not required to carry a citation.
        """
        return [
            sentence
            for sentence in self.sentences()
            if not CITATION.search(sentence) and not sentence.lower().startswith(HEDGES)
        ]


class ResearchReport(BaseModel):
    """The reporter's output, which may not introduce uncited factual content."""

    model_config = {"frozen": True}

    title: NonEmptyText
    sections: list[ReportSection] = Field(min_length=1, max_length=50)
    is_partial: bool = False
    omitted_because: list[NonEmptyText] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def every_factual_sentence_must_cite_evidence(self) -> ResearchReport:
        for section in self.sections:
            uncited = section.uncited_sentences()
            if uncited:
                raise ValueError(f"section {section.heading!r} states uncited facts: {uncited[:3]}")
        return self

    @model_validator(mode="after")
    def a_partial_report_must_say_what_is_missing(self) -> ResearchReport:
        """Section 12 requires partial output to be clearly labelled."""
        if self.is_partial and not self.omitted_because:
            raise ValueError("a partial report must state what could not be completed")
        if not self.is_partial and self.omitted_because:
            raise ValueError("a complete report cannot list omissions")
        return self

    @property
    def cited_evidence_ids(self) -> frozenset[UUID]:
        return frozenset().union(*(section.cited_evidence_ids for section in self.sections))


def unsupported_citations(
    report: ResearchReport,
    *,
    available_evidence_ids: frozenset[UUID],
) -> frozenset[UUID]:
    """Return citations the report makes to evidence that was never recorded.

    A model can produce a well-formed identifier for evidence that does not exist, so the
    citations are checked against what was actually collected rather than trusted.
    """
    return report.cited_evidence_ids - available_evidence_ids


def publishable_trust_levels(minimum: TrustLevel) -> frozenset[TrustLevel]:
    """The trust levels at or above a required minimum."""
    order = (TrustLevel.UNVERIFIED, TrustLevel.SECONDARY, TrustLevel.PRIMARY)
    return frozenset(order[order.index(minimum) :])
