from uuid import uuid4

import pytest

from research_platform.agents.contracts import (
    ClaimVerdict,
    CriticReview,
    ReportSection,
    ResearchReport,
)
from research_platform.domain.models import CriticVerdict
from research_platform.evaluation.scoring import (
    citation_correctness,
    claim_support_rate,
    contradiction_recall,
    research_coverage,
    schema_validity_rate,
    tool_selection_accuracy,
)

EVIDENCE_A = uuid4()
EVIDENCE_B = uuid4()


def report_citing(*evidence_ids: object) -> ResearchReport:
    citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
    return ResearchReport(
        title="Vendor pricing",
        sections=[
            ReportSection(
                heading="Pricing", body=f"The vendor charges 20 USD per seat {citations}."
            )
        ],
    )


def verdict(
    claim: str, outcome: CriticVerdict, conflicts: list[object] | None = None
) -> ClaimVerdict:
    return ClaimVerdict(
        claim=claim,
        verdict=outcome,
        reasoning="Because the evidence says so.",
        conflicting_evidence_ids=conflicts
        or ([uuid4()] if outcome is CriticVerdict.CONTRADICTED else []),
    )


def test_citation_correctness_is_perfect_when_every_citation_is_genuine() -> None:
    report = report_citing(EVIDENCE_A, EVIDENCE_B)

    score = citation_correctness(report, available_evidence_ids=frozenset({EVIDENCE_A, EVIDENCE_B}))

    assert score == 1.0


def test_citation_correctness_penalizes_a_fabricated_citation() -> None:
    fabricated = uuid4()
    report = report_citing(EVIDENCE_A, fabricated)

    score = citation_correctness(report, available_evidence_ids=frozenset({EVIDENCE_A}))

    assert score == 0.5


def test_citation_correctness_is_perfect_for_a_report_with_no_citations_to_check() -> None:
    # ResearchReport itself would refuse an uncited factual sentence; a report with
    # nothing but hedged, non-factual sentences has no citations to be wrong about.
    report = ResearchReport(
        title="Vendor pricing",
        sections=[
            ReportSection(heading="Pricing", body="We recommend confirming pricing directly.")
        ],
    )

    assert citation_correctness(report, available_evidence_ids=frozenset()) == 1.0


def test_claim_support_rate_counts_only_supported_verdicts() -> None:
    review = CriticReview(
        verdicts=[
            verdict("A", CriticVerdict.SUPPORTED),
            verdict("B", CriticVerdict.SUPPORTED),
            verdict("C", CriticVerdict.UNSUPPORTED),
            verdict("D", CriticVerdict.CONTRADICTED),
        ]
    )

    assert claim_support_rate(review) == 0.5


def test_research_coverage_rejects_more_unmet_than_requested() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        research_coverage(requested=1, unmet=2)


def test_research_coverage_rejects_zero_requested() -> None:
    with pytest.raises(ValueError, match="at least one"):
        research_coverage(requested=0, unmet=0)


def test_research_coverage_is_the_fraction_met() -> None:
    assert research_coverage(requested=4, unmet=1) == 0.75


def test_contradiction_recall_counts_only_caught_contradictions() -> None:
    review = CriticReview(
        verdicts=[
            verdict("A", CriticVerdict.CONTRADICTED),
            verdict("B", CriticVerdict.SUPPORTED),
        ]
    )

    recall = contradiction_recall(review, known_contradictions=frozenset({"A", "C"}))

    assert recall == 0.5


def test_contradiction_recall_is_perfect_when_nothing_was_expected() -> None:
    review = CriticReview(verdicts=[verdict("A", CriticVerdict.SUPPORTED)])

    assert contradiction_recall(review, known_contradictions=frozenset()) == 1.0


def test_tool_selection_accuracy_is_the_jaccard_index() -> None:
    used = frozenset({"web-research.search", "web-research.fetch"})
    expected = frozenset({"web-research.fetch", "filesystem.read_document"})

    assert tool_selection_accuracy(used=used, expected=expected) == pytest.approx(1 / 3)


def test_tool_selection_accuracy_is_perfect_when_nothing_was_called_or_expected() -> None:
    assert tool_selection_accuracy(used=frozenset(), expected=frozenset()) == 1.0


def test_tool_selection_accuracy_is_zero_when_disjoint() -> None:
    used = frozenset({"web-research.search"})
    expected = frozenset({"postgres.describe_schema"})

    assert tool_selection_accuracy(used=used, expected=expected) == 0.0


def test_schema_validity_rate_meets_the_section_14_target_when_perfect() -> None:
    assert schema_validity_rate(attempts=20, exhausted=0) == 1.0


def test_schema_validity_rate_reflects_exhausted_attempts() -> None:
    assert schema_validity_rate(attempts=20, exhausted=1) == 0.95


def test_schema_validity_rate_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        schema_validity_rate(attempts=5, exhausted=6)
    with pytest.raises(ValueError, match="at least one"):
        schema_validity_rate(attempts=0, exhausted=0)
