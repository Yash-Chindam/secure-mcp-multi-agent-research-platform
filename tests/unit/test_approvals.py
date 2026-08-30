from datetime import timedelta
from uuid import uuid4

import pytest

from research_platform.domain.approvals import (
    ApprovalMismatch,
    ApprovalStatus,
    request_sensitive_tool_approval,
)
from research_platform.domain.models import utc_now

JOB_ID = uuid4()
ARGUMENTS = {"repository": "acme/private", "ref": "main"}
NOW = utc_now()


def make_request(**overrides: object):  # type: ignore[no-untyped-def]
    defaults: dict[str, object] = {
        "job_id": JOB_ID,
        "tenant_id": "acme",
        "mcp_server": "github",
        "capability": "read_repository",
        "resource_id": "acme/private",
        "arguments": ARGUMENTS,
        "reason": "Repository is outside the default allowlist",
        "now": NOW,
    }
    return request_sensitive_tool_approval(**(defaults | overrides))  # type: ignore[arg-type]


def granted():  # type: ignore[no-untyped-def]
    return make_request().decide(reviewer_id="reviewer-1", granted=True, at=NOW)


def test_a_new_request_is_pending_and_expires_in_the_future() -> None:
    request = make_request()

    assert request.status is ApprovalStatus.PENDING
    assert request.expires_at > request.requested_at
    assert request.reviewer_id is None


def test_granting_records_the_reviewer_and_decision_time() -> None:
    decision = granted()

    assert decision.status is ApprovalStatus.GRANTED
    assert decision.reviewer_id == "reviewer-1"
    assert decision.decided_at == NOW


def test_rejecting_blocks_the_call() -> None:
    decision = make_request().decide(reviewer_id="reviewer-1", granted=False, at=NOW)

    assert decision.status is ApprovalStatus.REJECTED
    assert not decision.is_valid_for(
        mcp_server="github",
        capability="read_repository",
        resource_id="acme/private",
        arguments=ARGUMENTS,
        at=NOW,
    )


def test_a_request_cannot_be_decided_twice() -> None:
    decision = granted()

    with pytest.raises(ValueError, match="already"):
        decision.decide(reviewer_id="reviewer-2", granted=False, at=NOW)


def test_deciding_after_the_deadline_expires_the_request() -> None:
    request = make_request(valid_for=timedelta(minutes=5))

    decision = request.decide(reviewer_id="reviewer-1", granted=True, at=NOW + timedelta(hours=1))

    assert decision.status is ApprovalStatus.EXPIRED


def test_a_grant_authorizes_exactly_the_call_it_described() -> None:
    granted().authorize(
        mcp_server="github",
        capability="read_repository",
        resource_id="acme/private",
        arguments=dict(reversed(list(ARGUMENTS.items()))),
        at=NOW,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mcp_server", "filesystem"),
        ("capability", "write_repository"),
        ("resource_id", "acme/other"),
    ],
)
def test_a_grant_does_not_transfer_to_a_different_call(field: str, value: str) -> None:
    call = {
        "mcp_server": "github",
        "capability": "read_repository",
        "resource_id": "acme/private",
        field: value,
    }

    with pytest.raises(ApprovalMismatch):
        granted().authorize(arguments=ARGUMENTS, at=NOW, **call)  # type: ignore[arg-type]


def test_a_grant_does_not_transfer_to_different_arguments() -> None:
    with pytest.raises(ApprovalMismatch):
        granted().authorize(
            mcp_server="github",
            capability="read_repository",
            resource_id="acme/private",
            arguments={"repository": "acme/private", "ref": "attacker-branch"},
            at=NOW,
        )


def test_an_expired_grant_no_longer_authorizes_the_call() -> None:
    with pytest.raises(ApprovalMismatch):
        granted().authorize(
            mcp_server="github",
            capability="read_repository",
            resource_id="acme/private",
            arguments=ARGUMENTS,
            at=NOW + timedelta(days=2),
        )
