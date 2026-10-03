"""What a contract cannot check on its own: whether its references are real.

A schema proves an agent returned well-formed identifiers. It cannot prove they identify
anything - a model produces a plausible UUID as easily as a real one. Each check here
compares an agent's output with what the job actually recorded, and raises ``ValueError``
with a description the bounded correction loop hands back to the agent (section 12). An
identifier is safe to echo back; a claim's text is not, so problems are reported by
position and identifier only.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from research_platform.agents.contracts import AnalysisResult, CriticReview, ResearchReport
from research_platform.domain.models import EvidenceRecord


def _unknown(cited: Iterable[UUID], known: frozenset[UUID]) -> list[str]:
    return sorted(str(identifier) for identifier in set(cited) - known)


def check_analysis(analysis: AnalysisResult, evidence: list[EvidenceRecord]) -> None:
    """Every finding and calculation must rest on evidence that was actually recorded."""
    known = frozenset(record.id for record in evidence)
    problems: list[str] = []
    for index, finding in enumerate(analysis.findings):
        cited = [*finding.supporting_evidence_ids, *finding.contradicting_evidence_ids]
        if unknown := _unknown(cited, known):
            problems.append(f"findings.{index}: cites evidence that was never recorded: {unknown}")
    for index, calculation in enumerate(analysis.calculations):
        if unknown := _unknown(calculation.input_evidence_ids, known):
            problems.append(
                f"calculations.{index}: uses evidence that was never recorded: {unknown}"
            )
    if problems:
        raise ValueError("; ".join(problems))


def check_review(
    review: CriticReview, analysis: AnalysisResult, evidence: list[EvidenceRecord]
) -> None:
    """The critic must judge exactly the proposed claims, against recorded evidence."""
    known = frozenset(record.id for record in evidence)
    proposed = [finding.claim for finding in analysis.findings]
    judged = [verdict.claim for verdict in review.verdicts]
    problems: list[str] = []
    for index, verdict in enumerate(review.verdicts):
        if verdict.claim not in proposed:
            problems.append(
                f"verdicts.{index}: judges a claim the analyst did not propose; repeat each "
                "proposed claim word for word"
            )
        if unknown := _unknown(verdict.conflicting_evidence_ids, known):
            problems.append(
                f"verdicts.{index}: names conflicting evidence that was never recorded: {unknown}"
            )
    for index, claim in enumerate(proposed):
        if claim not in judged:
            problems.append(f"proposed finding {index} was given no verdict")
    if problems:
        raise ValueError("; ".join(problems))


def check_report(report: ResearchReport, evidence: list[EvidenceRecord]) -> None:
    """A report may cite only recorded evidence that is cleared to be published."""
    by_id = {record.id: record for record in evidence}
    cited = report.cited_evidence_ids
    problems: list[str] = []
    if unknown := _unknown(cited, frozenset(by_id)):
        problems.append(f"cites evidence that was never recorded: {unknown}")
    restricted = sorted(
        str(identifier)
        for identifier in cited
        if identifier in by_id and not by_id[identifier].is_publishable
    )
    if restricted:
        problems.append(f"cites restricted evidence that must not be published: {restricted}")
    if problems:
        raise ValueError("; ".join(problems))
