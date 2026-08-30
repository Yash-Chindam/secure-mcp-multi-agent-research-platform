from uuid import uuid4

import pytest

from research_platform.domain.invocations import (
    REDACTED,
    AuthorizationDecision,
    ErrorClass,
    InvocationOutcome,
    ToolInvocation,
    argument_digest,
    record_denied_invocation,
    sanitize_arguments,
)

JOB_ID = uuid4()
TASK_ID = uuid4()


def make_invocation(**overrides: object) -> ToolInvocation:
    defaults: dict[str, object] = {
        "job_id": JOB_ID,
        "task_id": TASK_ID,
        "tenant_id": "acme",
        "mcp_server": "web-research",
        "capability": "fetch",
        "sanitized_arguments": {"url": "https://example.com"},
        "argument_digest": argument_digest({"url": "https://example.com"}),
        "policy_version": "2026-08-01",
        "authorization_decision": AuthorizationDecision.ALLOW,
    }
    return ToolInvocation(**(defaults | overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    ["api_key", "Authorization", "github_token", "db_password", "client_secret"],
)
def test_sanitize_redacts_credential_shaped_arguments(name: str) -> None:
    assert sanitize_arguments({name: "value"}) == {name: REDACTED}


def test_sanitize_preserves_ordinary_arguments() -> None:
    assert sanitize_arguments({"url": "https://example.com"}) == {"url": "https://example.com"}


def test_argument_digest_is_stable_across_key_order() -> None:
    assert argument_digest({"a": 1, "b": 2}) == argument_digest({"b": 2, "a": 1})


def test_argument_digest_changes_with_the_arguments() -> None:
    assert argument_digest({"url": "https://a.test"}) != argument_digest({"url": "https://b.test"})


def test_invocation_refuses_unsanitized_arguments() -> None:
    with pytest.raises(ValueError, match="must be sanitized"):
        make_invocation(sanitized_arguments={"api_key": "live-secret"})


def test_denied_invocation_cannot_report_success() -> None:
    with pytest.raises(ValueError, match="denied invocation cannot report"):
        make_invocation(
            authorization_decision=AuthorizationDecision.DENY,
            outcome=InvocationOutcome.SUCCEEDED,
        )


def test_denied_outcome_must_record_the_policy_error_class() -> None:
    with pytest.raises(ValueError, match="policy_denied"):
        make_invocation(
            authorization_decision=AuthorizationDecision.DENY,
            outcome=InvocationOutcome.DENIED,
            error_class=ErrorClass.TIMEOUT,
        )


def test_successful_invocation_cannot_carry_an_error_class() -> None:
    with pytest.raises(ValueError, match="cannot carry an error class"):
        make_invocation(outcome=InvocationOutcome.SUCCEEDED, error_class=ErrorClass.TIMEOUT)


def test_failed_invocation_must_classify_its_error() -> None:
    with pytest.raises(ValueError, match="must record an error class"):
        make_invocation(outcome=InvocationOutcome.FAILED)


@pytest.mark.parametrize(
    ("error_class", "retryable"),
    [
        (ErrorClass.TIMEOUT, True),
        (ErrorClass.UPSTREAM_UNAVAILABLE, True),
        (ErrorClass.POLICY_DENIED, False),
        (ErrorClass.INVALID_ARGUMENTS, False),
        (ErrorClass.BUDGET_EXHAUSTED, False),
    ],
)
def test_only_transient_failures_are_retryable(error_class: ErrorClass, retryable: bool) -> None:
    invocation = make_invocation(outcome=InvocationOutcome.FAILED, error_class=error_class)

    assert invocation.is_retryable is retryable


def test_denied_record_redacts_credentials_and_binds_the_digest() -> None:
    arguments = {"repository": "acme/private", "token": "ghp_live"}

    invocation = record_denied_invocation(
        job_id=JOB_ID,
        task_id=TASK_ID,
        tenant_id="acme",
        mcp_server="github",
        capability="read_repository",
        arguments=arguments,
        policy_version="2026-08-01",
    )

    assert invocation.sanitized_arguments == {"repository": "acme/private", "token": REDACTED}
    assert invocation.argument_digest == argument_digest(arguments)
    assert invocation.outcome is InvocationOutcome.DENIED
    assert invocation.error_class is ErrorClass.POLICY_DENIED
    assert invocation.is_retryable is False
