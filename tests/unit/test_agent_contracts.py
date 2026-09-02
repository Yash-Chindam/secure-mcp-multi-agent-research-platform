from uuid import UUID, uuid4

import pytest

from research_platform.agents.contracts import (
    AnalysisResult,
    Calculation,
    ClaimVerdict,
    CriticReview,
    EvidenceSubmission,
    PlannedTask,
    ProposedFinding,
    ReportSection,
    ResearchPlan,
    ResearchReport,
    publishable_trust_levels,
    unsupported_citations,
)
from research_platform.domain.models import CriticVerdict, TrustLevel
from research_platform.domain.tasks import AgentRole

HASH = "sha256:" + "a" * 64
EVIDENCE_ID = uuid4()
OTHER_EVIDENCE_ID = uuid4()


def planned(**overrides: object) -> PlannedTask:
    defaults: dict[str, object] = {
        "objective": "Collect the vendor pricing page",
        "assigned_agent": AgentRole.RESEARCHER,
        "evidence_requirements": ["a dated pricing page"],
    }
    return PlannedTask(**(defaults | overrides))  # type: ignore[arg-type]


def plan(tasks: list[PlannedTask] | None = None) -> ResearchPlan:
    return ResearchPlan(
        tasks=tasks or [planned()],
        rationale="Pricing must be sourced before it can be compared.",
    )


def test_a_plan_orders_tasks_and_reports_that_it_collects_evidence() -> None:
    built = plan([planned(), planned(assigned_agent=AgentRole.ANALYST, depends_on=[0])])

    assert built.collects_evidence is True
    assert len(built.tasks) == 2


def test_a_task_must_state_what_evidence_it_needs() -> None:
    with pytest.raises(ValueError):
        planned(evidence_requirements=[])


def test_the_planner_cannot_assign_work_to_itself() -> None:
    with pytest.raises(ValueError, match="cannot assign work to itself"):
        planned(assigned_agent=AgentRole.PLANNER)


def test_a_plan_must_contain_a_task() -> None:
    with pytest.raises(ValueError):
        ResearchPlan(tasks=[], rationale="Nothing to do.")


def test_a_dependency_on_a_missing_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        plan([planned(depends_on=[5])])


def test_a_task_cannot_depend_on_itself() -> None:
    with pytest.raises(ValueError, match="depends on itself"):
        plan([planned(depends_on=[0])])


def test_a_dependency_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        plan([planned(depends_on=[1]), planned(depends_on=[0])])


def evidence(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "excerpt": "The team plan costs 20 USD per seat.",
        "source_uri": "https://vendor.test/pricing",
        "content_hash": HASH,
        "producing_task_id": uuid4(),
        "tool_invocation_id": uuid4(),
    }
    return defaults | overrides


def test_a_submission_reports_completeness() -> None:
    submission = EvidenceSubmission.model_validate({"records": [evidence()]})

    assert submission.is_complete is True


def test_a_submission_with_unmet_requirements_is_incomplete() -> None:
    submission = EvidenceSubmission.model_validate(
        {"records": [evidence()], "unmet_requirements": ["no dated source was found"]}
    )

    assert submission.is_complete is False


def test_a_submission_must_contain_a_record() -> None:
    with pytest.raises(ValueError):
        EvidenceSubmission.model_validate({"records": []})


def test_the_same_excerpt_cannot_be_submitted_twice() -> None:
    with pytest.raises(ValueError, match="more than once"):
        EvidenceSubmission.model_validate({"records": [evidence(), evidence()]})


def test_distinct_excerpts_are_accepted() -> None:
    submission = EvidenceSubmission.model_validate(
        {"records": [evidence(), evidence(content_hash="sha256:" + "b" * 64)]}
    )

    assert len(submission.records) == 2


def finding(**overrides: object) -> ProposedFinding:
    defaults: dict[str, object] = {
        "claim": "The vendor charges 20 USD per seat.",
        "supporting_evidence_ids": [EVIDENCE_ID],
        "confidence": 0.9,
    }
    return ProposedFinding(**(defaults | overrides))  # type: ignore[arg-type]


def test_a_finding_must_rest_on_evidence() -> None:
    with pytest.raises(ValueError):
        finding(supporting_evidence_ids=[])


def test_evidence_cannot_both_support_and_contradict_a_claim() -> None:
    with pytest.raises(ValueError, match="both support and contradict"):
        finding(contradicting_evidence_ids=[EVIDENCE_ID])


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValueError):
        finding(confidence=1.5)


def calculation(identifier: UUID) -> Calculation:
    return Calculation(
        id=identifier,
        description="Annualize the seat price",
        expression="20 * 12",
        result="240",
        input_evidence_ids=[EVIDENCE_ID],
    )


def test_an_analysis_may_report_findings_without_calculations() -> None:
    result = AnalysisResult(findings=[finding()])

    assert result.calculations == []


def test_a_finding_cannot_cite_a_calculation_that_was_not_reported() -> None:
    with pytest.raises(ValueError, match="calculations that were not reported"):
        AnalysisResult(findings=[finding(calculation_ids=[uuid4()])])


def test_a_finding_may_cite_a_reported_calculation() -> None:
    identifier = uuid4()

    result = AnalysisResult(
        findings=[finding(calculation_ids=[identifier])],
        calculations=[calculation(identifier)],
    )

    assert result.calculations[0].id == identifier


def verdict(**overrides: object) -> ClaimVerdict:
    defaults: dict[str, object] = {
        "claim": "The vendor charges 20 USD per seat.",
        "verdict": CriticVerdict.SUPPORTED,
        "reasoning": "The dated pricing page states the figure directly.",
    }
    return ClaimVerdict(**(defaults | overrides))  # type: ignore[arg-type]


def test_the_critic_must_reach_a_verdict() -> None:
    with pytest.raises(ValueError, match="rather than leaving it pending"):
        verdict(verdict=CriticVerdict.PENDING)


def test_a_contradicted_claim_must_name_the_conflicting_evidence() -> None:
    with pytest.raises(ValueError, match="must name the conflicting evidence"):
        verdict(verdict=CriticVerdict.CONTRADICTED)


def test_a_contradicted_claim_with_a_named_conflict_is_accepted() -> None:
    judged = verdict(
        verdict=CriticVerdict.CONTRADICTED,
        conflicting_evidence_ids=[OTHER_EVIDENCE_ID],
    )

    assert judged.conflicting_evidence_ids == [OTHER_EVIDENCE_ID]


def test_a_review_lists_the_supported_claims() -> None:
    review = CriticReview(
        verdicts=[verdict(), verdict(claim="Unsupported.", verdict=CriticVerdict.UNSUPPORTED)]
    )

    assert review.supported_claims == ["The vendor charges 20 USD per seat."]
    assert review.requires_reviewer is False


def test_a_coverage_gap_requires_a_reviewer() -> None:
    review = CriticReview(verdicts=[verdict()], coverage_gaps=["no EU pricing was found"])

    assert review.requires_reviewer is True


def test_a_contradiction_requires_a_reviewer() -> None:
    review = CriticReview(
        verdicts=[
            verdict(
                verdict=CriticVerdict.CONTRADICTED,
                conflicting_evidence_ids=[OTHER_EVIDENCE_ID],
            )
        ]
    )

    assert review.requires_reviewer is True


def cited(text: str, evidence_id: UUID = EVIDENCE_ID) -> str:
    """Cite inside the sentence, before its terminal punctuation."""
    return f"{text.rstrip('.')} [{evidence_id}]."


def test_a_cited_report_is_accepted() -> None:
    report = ResearchReport(
        title="Vendor pricing comparison",
        sections=[ReportSection(heading="Pricing", body=cited("The team plan costs 20 USD."))],
    )

    assert report.cited_evidence_ids == frozenset({EVIDENCE_ID})


def test_a_citation_left_between_two_sentences_does_not_count() -> None:
    """The convention is to cite inside the sentence; a dangling citation is ambiguous."""
    body = f"The team plan costs 20 USD. [{EVIDENCE_ID}] It is the cheapest tier."

    with pytest.raises(ValueError, match="states uncited facts"):
        ResearchReport(
            title="Vendor pricing comparison",
            sections=[ReportSection(heading="Pricing", body=body)],
        )


def test_a_citation_only_fragment_supports_the_sentence_before_it() -> None:
    body = f"The team plan costs 20 USD. [{EVIDENCE_ID}]"

    report = ResearchReport(
        title="Vendor pricing comparison",
        sections=[ReportSection(heading="Pricing", body=body)],
    )

    assert report.cited_evidence_ids == frozenset({EVIDENCE_ID})


def test_an_uncited_factual_sentence_is_rejected() -> None:
    with pytest.raises(ValueError, match="states uncited facts"):
        ResearchReport(
            title="Vendor pricing comparison",
            sections=[ReportSection(heading="Pricing", body="The team plan costs 20 USD.")],
        )


def test_one_uncited_sentence_among_cited_ones_is_still_rejected() -> None:
    body = f"{cited('The team plan costs 20 USD.')} The vendor is the market leader."

    with pytest.raises(ValueError, match="states uncited facts"):
        ResearchReport(
            title="Vendor pricing comparison",
            sections=[ReportSection(heading="Pricing", body=body)],
        )


def test_an_explicit_recommendation_does_not_require_a_citation() -> None:
    body = f"{cited('The team plan costs 20 USD.')} We recommend the team plan."

    report = ResearchReport(
        title="Vendor pricing comparison",
        sections=[ReportSection(heading="Pricing", body=body)],
    )

    assert report.cited_evidence_ids == frozenset({EVIDENCE_ID})


def test_a_partial_report_must_state_what_is_missing() -> None:
    with pytest.raises(ValueError, match="what could not be completed"):
        ResearchReport(
            title="Vendor pricing comparison",
            sections=[ReportSection(heading="Pricing", body=cited("The plan costs 20 USD."))],
            is_partial=True,
        )


def test_a_complete_report_cannot_list_omissions() -> None:
    with pytest.raises(ValueError, match="cannot list omissions"):
        ResearchReport(
            title="Vendor pricing comparison",
            sections=[ReportSection(heading="Pricing", body=cited("The plan costs 20 USD."))],
            omitted_because=["the budget was exhausted"],
        )


def test_a_labelled_partial_report_is_accepted() -> None:
    report = ResearchReport(
        title="Vendor pricing comparison",
        sections=[ReportSection(heading="Pricing", body=cited("The plan costs 20 USD."))],
        is_partial=True,
        omitted_because=["the tool-call budget was exhausted"],
    )

    assert report.is_partial is True


def test_a_citation_to_evidence_that_was_never_recorded_is_reported() -> None:
    """A model can invent a well-formed identifier, so citations are checked, not trusted."""
    invented = uuid4()
    report = ResearchReport(
        title="Vendor pricing comparison",
        sections=[
            ReportSection(heading="Pricing", body=cited("The plan costs 20 USD.", invented)),
        ],
    )

    assert unsupported_citations(report, available_evidence_ids=frozenset({EVIDENCE_ID})) == (
        frozenset({invented})
    )


def test_a_report_citing_only_recorded_evidence_has_no_unsupported_citations() -> None:
    report = ResearchReport(
        title="Vendor pricing comparison",
        sections=[ReportSection(heading="Pricing", body=cited("The plan costs 20 USD."))],
    )

    assert (
        unsupported_citations(
            report, available_evidence_ids=frozenset({EVIDENCE_ID, OTHER_EVIDENCE_ID})
        )
        == frozenset()
    )


def test_trust_levels_at_or_above_a_minimum_are_returned() -> None:
    assert publishable_trust_levels(TrustLevel.PRIMARY) == frozenset({TrustLevel.PRIMARY})
    assert publishable_trust_levels(TrustLevel.SECONDARY) == frozenset(
        {TrustLevel.SECONDARY, TrustLevel.PRIMARY}
    )
    assert publishable_trust_levels(TrustLevel.UNVERIFIED) == frozenset(TrustLevel)
