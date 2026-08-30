"""Audit records for every capability an agent invokes through MCP."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from research_platform.domain.models import NonEmptyText, utc_now

REDACTED = "[redacted]"

SENSITIVE_ARGUMENT_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "password",
        "secret",
        "token",
    }
)


class AuthorizationDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class InvocationOutcome(StrEnum):
    """What happened to the call.

    ``DENIED`` means the call was refused before anything left the platform, whether by
    policy, by a missing approval, by the job budget or by an open circuit. ``FAILED``
    means the call reached the server and did not succeed.
    """

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"


class ErrorClass(StrEnum):
    """Failure taxonomy used to decide whether a retry can help."""

    NONE = "none"
    POLICY_DENIED = "policy_denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    TIMEOUT = "timeout"
    RESULT_TOO_LARGE = "result_too_large"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNEXPECTED = "unexpected"


RETRYABLE_ERRORS = frozenset({ErrorClass.UPSTREAM_UNAVAILABLE, ErrorClass.TIMEOUT})


def is_sensitive_argument(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in SENSITIVE_ARGUMENT_NAMES)


def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace credential-shaped argument values so they never reach logs or traces."""
    return {
        name: REDACTED if is_sensitive_argument(name) else value
        for name, value in arguments.items()
    }


def argument_digest(arguments: dict[str, Any]) -> str:
    """Return a stable digest binding an approval to one exact argument set."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


class ToolInvocation(BaseModel):
    """One authorized capability call, recorded for audit and observability."""

    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    task_id: UUID
    tenant_id: str = Field(min_length=1, max_length=100)
    mcp_server: NonEmptyText
    capability: NonEmptyText
    sanitized_arguments: dict[str, Any] = Field(default_factory=dict)
    argument_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    policy_version: NonEmptyText
    authorization_decision: AuthorizationDecision
    outcome: InvocationOutcome = InvocationOutcome.PENDING
    error_class: ErrorClass = ErrorClass.NONE
    started_at: datetime = Field(default_factory=utc_now)
    duration_ms: Annotated[int, Field(ge=0, le=86_400_000)] | None = None

    @model_validator(mode="after")
    def denied_calls_cannot_report_success(self) -> ToolInvocation:
        if self.authorization_decision is AuthorizationDecision.DENY:
            if self.outcome is not InvocationOutcome.DENIED:
                raise ValueError("a policy-denied invocation must record a denied outcome")
            if self.error_class is not ErrorClass.POLICY_DENIED:
                raise ValueError("a policy-denied invocation must record a policy_denied error")
        if self.outcome is InvocationOutcome.DENIED and self.error_class is ErrorClass.NONE:
            raise ValueError("a refused invocation must record why it was refused")
        if self.outcome is InvocationOutcome.SUCCEEDED and self.error_class is not ErrorClass.NONE:
            raise ValueError("a successful invocation cannot carry an error class")
        if self.outcome is InvocationOutcome.FAILED and self.error_class is ErrorClass.NONE:
            raise ValueError("a failed invocation must record an error class")
        if self.sanitized_arguments != sanitize_arguments(self.sanitized_arguments):
            raise ValueError("invocation arguments must be sanitized before they are recorded")
        return self

    @property
    def is_retryable(self) -> bool:
        return self.error_class in RETRYABLE_ERRORS


def record_denied_invocation(
    *,
    job_id: UUID,
    task_id: UUID,
    tenant_id: str,
    mcp_server: str,
    capability: str,
    arguments: dict[str, Any],
    policy_version: str,
) -> ToolInvocation:
    """Build the audit record for a call policy refused before execution."""
    sanitized = sanitize_arguments(arguments)
    return ToolInvocation(
        job_id=job_id,
        task_id=task_id,
        tenant_id=tenant_id,
        mcp_server=mcp_server,
        capability=capability,
        sanitized_arguments=sanitized,
        argument_digest=argument_digest(arguments),
        policy_version=policy_version,
        authorization_decision=AuthorizationDecision.DENY,
        outcome=InvocationOutcome.DENIED,
        error_class=ErrorClass.POLICY_DENIED,
        duration_ms=0,
    )
