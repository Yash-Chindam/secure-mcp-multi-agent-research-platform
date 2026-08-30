from datetime import timedelta
from uuid import uuid4

import pytest

from research_platform.domain.models import (
    AccessClass,
    CriticVerdict,
    EvidenceRecord,
    Finding,
    ReviewerStatus,
    TrustLevel,
    utc_now,
)

HASH = "sha256:" + "a" * 64


def make_evidence(**overrides: object) -> EvidenceRecord:
    defaults: dict[str, object] = {
        "job_id": uuid4(),
        "tenant_id": "acme",
        "excerpt": "Vendor pricing starts at 20 USD per seat.",
        "source_uri": "https://vendor.test/pricing",
        "content_hash": HASH,
        "producing_task_id": uuid4(),
        "tool_invocation_id": uuid4(),
    }
    return EvidenceRecord(**(defaults | overrides))  # type: ignore[arg-type]


def make_finding(**overrides: object) -> Finding:
    defaults: dict[str, object] = {
        "claim": "The vendor charges 20 USD per seat.",
        "supporting_evidence_ids": [uuid4()],
        "confidence": 0.9,
    }
    return Finding(**(defaults | overrides))  # type: ignore[arg-type]


def test_evidence_defaults_to_unverified_public_classification() -> None:
    evidence = make_evidence()

    assert evidence.trust_level is TrustLevel.UNVERIFIED
    assert evidence.access_class is AccessClass.PUBLIC
    assert evidence.is_publishable is True


def test_restricted_evidence_is_not_publishable() -> None:
    assert make_evidence(access_class=AccessClass.RESTRICTED).is_publishable is False


def test_internal_evidence_remains_publishable() -> None:
    assert make_evidence(access_class=AccessClass.INTERNAL).is_publishable is True


def test_evidence_rejects_a_future_publication_date() -> None:
    with pytest.raises(ValueError, match="future publication date"):
        make_evidence(published_at=utc_now() + timedelta(days=1))


def test_evidence_accepts_a_past_publication_date_and_author() -> None:
    published = utc_now() - timedelta(days=30)

    evidence = make_evidence(author="A. Analyst", published_at=published)

    assert evidence.author == "A. Analyst"
    assert evidence.published_at == published


def test_evidence_detects_content_drift_without_losing_the_excerpt() -> None:
    evidence = make_evidence()

    assert evidence.has_drifted_from("sha256:" + "b" * 64) is True
    assert evidence.has_drifted_from(HASH) is False
    assert evidence.excerpt.startswith("Vendor pricing")


def test_a_supported_finding_cannot_retain_contradicting_evidence() -> None:
    with pytest.raises(ValueError, match="supported finding cannot retain"):
        make_finding(
            contradicting_evidence_ids=[uuid4()],
            critic_verdict=CriticVerdict.SUPPORTED,
        )


def test_an_unreviewed_supported_finding_is_publishable() -> None:
    assert make_finding(critic_verdict=CriticVerdict.SUPPORTED).is_publishable is True


@pytest.mark.parametrize(
    ("verdict", "reviewer_status"),
    [
        (CriticVerdict.PENDING, ReviewerStatus.NOT_REQUIRED),
        (CriticVerdict.UNSUPPORTED, ReviewerStatus.NOT_REQUIRED),
        (CriticVerdict.CONTRADICTED, ReviewerStatus.APPROVED),
        (CriticVerdict.SUPPORTED, ReviewerStatus.REJECTED),
        (CriticVerdict.SUPPORTED, ReviewerStatus.PENDING),
    ],
)
def test_unsupported_or_unresolved_findings_are_not_publishable(
    verdict: CriticVerdict, reviewer_status: ReviewerStatus
) -> None:
    finding = make_finding(critic_verdict=verdict, reviewer_status=reviewer_status)

    assert finding.is_publishable is False


def test_a_reviewer_approved_supported_finding_is_publishable() -> None:
    finding = make_finding(
        critic_verdict=CriticVerdict.SUPPORTED,
        reviewer_status=ReviewerStatus.APPROVED,
        calculation_ids=[uuid4()],
    )

    assert finding.is_publishable is True
