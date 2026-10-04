"""Findings, the provenance manifest and the rendered report, from what a job recorded."""

from uuid import UUID, uuid4

from research_platform.agents.contracts import (
    AnalysisResult,
    ClaimVerdict,
    CriticReview,
    ProposedFinding,
    ReportSection,
    ResearchReport,
)
from research_platform.application.publication import (
    DriftStatus,
    SourceCheck,
    build_manifest,
    derive_findings,
    render_markdown,
    report_json,
    sha256_of,
)
from research_platform.domain.invocations import (
    AuthorizationDecision,
    InvocationOutcome,
    ToolInvocation,
)
from research_platform.domain.models import (
    AccessClass,
    CriticVerdict,
    EvidenceRecord,
    FindingRecord,
    ResearchJob,
    ReviewerStatus,
    TrustLevel,
)

CLAIM = "The vendor charges 20 USD per seat."
OTHER_CLAIM = "The vendor offers a free tier."

JOB = ResearchJob(tenant_id="acme", requester_id="requester-1", question="What does it cost?")


def evidence(
    excerpt: str = "Vendor pricing is 20 USD per seat.", **changes: object
) -> EvidenceRecord:
    fields: dict[str, object] = {
        "tenant_id": JOB.tenant_id,
        "job_id": JOB.id,
        "excerpt": excerpt,
        "source_uri": "https://vendor.test/pricing",
        "title": "Pricing",
        "trust_level": TrustLevel.PRIMARY,
        "content_hash": f"sha256:{'a' * 64}",
        "producing_task_id": uuid4(),
        "tool_invocation_id": uuid4(),
    }
    return EvidenceRecord.model_validate(fields | changes)


def proposed(
    claim: str, supporting: list[UUID], against: list[UUID] | None = None
) -> ProposedFinding:
    return ProposedFinding(
        claim=claim,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=against or [],
        confidence=0.8,
    )


def verdict(
    claim: str, outcome: CriticVerdict, conflicting: list[UUID] | None = None
) -> ClaimVerdict:
    return ClaimVerdict(
        claim=claim,
        verdict=outcome,
        reasoning="Checked against the source.",
        conflicting_evidence_ids=conflicting or [],
    )


def report_citing(*identifiers: UUID) -> ResearchReport:
    cited = " ".join(f"[{identifier}]" for identifier in identifiers)
    return ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body=f"{CLAIM[:-1]} {cited}.")],
    )


# -- findings -----------------------------------------------------------------------------


def test_a_finding_joins_the_analysts_claim_with_the_critics_verdict() -> None:
    supporting = uuid4()
    analysis = AnalysisResult(findings=[proposed(CLAIM, [supporting])])
    critique = CriticReview(verdicts=[verdict(CLAIM, CriticVerdict.SUPPORTED)])

    [finding] = derive_findings(analysis, critique, ReviewerStatus.NOT_REQUIRED)

    assert finding.claim == CLAIM
    assert finding.supporting_evidence_ids == [supporting]
    assert finding.confidence == 0.8
    assert finding.critic_verdict is CriticVerdict.SUPPORTED
    assert finding.reviewer_status is ReviewerStatus.NOT_REQUIRED
    assert finding.is_publishable


def test_a_claim_the_critic_gave_no_verdict_stays_pending() -> None:
    analysis = AnalysisResult(
        findings=[proposed(CLAIM, [uuid4()]), proposed(OTHER_CLAIM, [uuid4()])]
    )
    critique = CriticReview(verdicts=[verdict(CLAIM, CriticVerdict.SUPPORTED)])

    _, unjudged = derive_findings(analysis, critique, ReviewerStatus.NOT_REQUIRED)

    assert unjudged.critic_verdict is CriticVerdict.PENDING
    assert not unjudged.is_publishable


def test_the_critics_conflicting_evidence_is_kept_on_the_finding() -> None:
    supporting, analyst_conflict, critic_conflict = uuid4(), uuid4(), uuid4()
    analysis = AnalysisResult(findings=[proposed(CLAIM, [supporting], [analyst_conflict])])
    critique = CriticReview(
        verdicts=[verdict(CLAIM, CriticVerdict.CONTRADICTED, [critic_conflict, analyst_conflict])]
    )

    [finding] = derive_findings(analysis, critique, ReviewerStatus.PENDING)

    assert finding.contradicting_evidence_ids == [analyst_conflict, critic_conflict]
    assert finding.critic_verdict is CriticVerdict.CONTRADICTED
    assert not finding.is_publishable


def test_a_claim_with_evidence_against_it_is_never_recorded_as_supported() -> None:
    """Whatever the verdict says, the conflict the analyst recorded is not dropped."""
    analysis = AnalysisResult(findings=[proposed(CLAIM, [uuid4()], [uuid4()])])
    critique = CriticReview(verdicts=[verdict(CLAIM, CriticVerdict.SUPPORTED)])

    [finding] = derive_findings(analysis, critique, ReviewerStatus.NOT_REQUIRED)

    assert finding.critic_verdict is CriticVerdict.CONTRADICTED


def test_evidence_the_critic_calls_conflicting_cannot_also_be_left_supporting() -> None:
    supporting = uuid4()
    analysis = AnalysisResult(findings=[proposed(CLAIM, [supporting])])
    critique = CriticReview(verdicts=[verdict(CLAIM, CriticVerdict.CONTRADICTED, [supporting])])

    [finding] = derive_findings(analysis, critique, ReviewerStatus.PENDING)

    assert finding.supporting_evidence_ids == [supporting]
    assert finding.contradicting_evidence_ids == []
    assert finding.critic_verdict is CriticVerdict.CONTRADICTED


def test_a_rejected_finding_is_not_publishable_even_when_the_critic_supported_it() -> None:
    analysis = AnalysisResult(findings=[proposed(CLAIM, [uuid4()])])
    critique = CriticReview(verdicts=[verdict(CLAIM, CriticVerdict.SUPPORTED)])

    [finding] = derive_findings(analysis, critique, ReviewerStatus.REJECTED)

    assert not finding.is_publishable


# -- manifest -----------------------------------------------------------------------------


def invocation_for(record: EvidenceRecord) -> ToolInvocation:
    return ToolInvocation(
        id=record.tool_invocation_id,
        job_id=JOB.id,
        task_id=record.producing_task_id,
        tenant_id=JOB.tenant_id,
        mcp_server="web-research",
        capability="fetch",
        sanitized_arguments={"url": "https://vendor.test/pricing"},
        argument_digest=f"sha256:{'b' * 64}",
        policy_version="registry-boundary/1",
        authorization_decision=AuthorizationDecision.ALLOW,
        outcome=InvocationOutcome.SUCCEEDED,
    )


def test_the_manifest_traces_the_report_to_its_sources_without_carrying_their_text() -> None:
    cited, uncited = evidence(), evidence("An unrelated passage.")
    report = report_citing(cited.id)
    finding = FindingRecord(
        tenant_id=JOB.tenant_id,
        job_id=JOB.id,
        claim=CLAIM,
        supporting_evidence_ids=[cited.id],
        confidence=0.9,
        critic_verdict=CriticVerdict.SUPPORTED,
    )

    manifest = build_manifest(
        JOB,
        report,
        shortfalls=[],
        findings=[finding],
        evidence=[cited, uncited],
        invocations=[invocation_for(cited)],
        checks=[SourceCheck(evidence_id=cited.id, status=DriftStatus.UNCHANGED)],
    )

    assert manifest.job_id == JOB.id
    assert manifest.question == JOB.question
    assert manifest.report_sha256 == sha256_of(report_json(report))
    assert manifest.is_partial is False
    assert [(listed.id, listed.cited) for listed in manifest.evidence] == [
        (cited.id, True),
        (uncited.id, False),
    ]
    assert manifest.evidence[0].content_hash == cited.content_hash
    assert manifest.evidence[0].drift is DriftStatus.UNCHANGED
    assert manifest.evidence[1].drift is DriftStatus.NOT_CHECKED
    assert manifest.findings[0].claim == CLAIM
    assert manifest.tool_invocations[0].id == cited.tool_invocation_id
    assert manifest.tool_invocations[0].policy_version == "registry-boundary/1"
    assert cited.excerpt not in manifest.model_dump_json()


def test_the_manifest_names_drifted_sources_and_what_a_partial_report_lacks() -> None:
    record = evidence()

    manifest = build_manifest(
        JOB,
        report_citing(record.id),
        shortfalls=["unmet requirement: enterprise pricing"],
        findings=[],
        evidence=[record],
        invocations=[],
        checks=[SourceCheck(evidence_id=record.id, status=DriftStatus.DRIFTED, detail="changed")],
    )

    assert manifest.is_partial is True
    assert manifest.shortfalls == ["unmet requirement: enterprise pricing"]
    assert manifest.drifted_evidence_ids == [record.id]
    assert manifest.evidence[0].drift_detail == "changed"


# -- rendered report ------------------------------------------------------------------------


def test_the_rendered_report_numbers_its_sources_in_the_order_they_are_cited() -> None:
    first, second = evidence(), evidence("A second passage.", title=None)
    report = ResearchReport(
        title="Vendor pricing",
        sections=[
            ReportSection(heading="Pricing", body=f"Seats cost 20 USD [{second.id}]."),
            ReportSection(
                heading="Detail", body=f"The price is per seat [{first.id}] [{second.id}]."
            ),
        ],
    )

    markdown = render_markdown(JOB, report, shortfalls=[], evidence=[first, second], checks=[])

    assert markdown.startswith("# Vendor pricing\n")
    assert "**Question:** What does it cost?" in markdown
    assert "Seats cost 20 USD [1]." in markdown
    assert "The price is per seat [2] [1]." in markdown
    sources = markdown.split("## Sources\n\n")[1].splitlines()
    assert sources[0].startswith("1. https://vendor.test/pricing (retrieved ")
    assert sources[1].startswith("2. Pricing - https://vendor.test/pricing (retrieved ")
    assert f"`{first.content_hash}`" in sources[1]
    assert "Partial result" not in markdown


def test_the_rendered_report_flags_a_source_that_changed_since_it_was_captured() -> None:
    record = evidence()

    markdown = render_markdown(
        JOB,
        report_citing(record.id),
        shortfalls=[],
        evidence=[record],
        checks=[SourceCheck(evidence_id=record.id, status=DriftStatus.DRIFTED)],
    )

    assert "**source content has changed since capture**" in markdown
    assert record.excerpt not in markdown


def test_a_partial_report_is_labelled_as_one_at_the_top() -> None:
    record = evidence()

    markdown = render_markdown(
        JOB,
        report_citing(record.id),
        shortfalls=["unmet requirement: enterprise pricing"],
        evidence=[record],
        checks=[],
    )

    banner = markdown.index("**Partial result.**")
    assert banner < markdown.index("## Pricing")
    assert "> - unmet requirement: enterprise pricing" in markdown


def test_a_report_that_declares_its_own_omissions_lists_them() -> None:
    report = ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body="This suggests pricing is unclear.")],
        is_partial=True,
        omitted_because=["no enterprise pricing was published"],
    )

    markdown = render_markdown(JOB, report, shortfalls=[], evidence=[], checks=[])

    assert "> - no enterprise pricing was published" in markdown
    assert markdown.rstrip().endswith("No sources were cited.")


def test_a_citation_to_unrecorded_evidence_is_shown_as_unrecorded_not_hidden() -> None:
    invented = uuid4()

    markdown = render_markdown(JOB, report_citing(invented), shortfalls=[], evidence=[], checks=[])

    assert f"1. Unrecorded evidence `{invented}`" in markdown


def test_restricted_evidence_keeps_its_classification_in_the_manifest() -> None:
    record = evidence(access_class=AccessClass.RESTRICTED)

    manifest = build_manifest(
        JOB,
        report_citing(uuid4()),
        shortfalls=[],
        findings=[],
        evidence=[record],
        invocations=[],
        checks=[],
    )

    assert manifest.evidence[0].access_class is AccessClass.RESTRICTED
    assert manifest.evidence[0].cited is False
