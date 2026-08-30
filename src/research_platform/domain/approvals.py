"""Human approval checkpoints for sensitive actions and publication."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from research_platform.domain.invocations import argument_digest
from research_platform.domain.models import NonEmptyText, utc_now


class ApprovalKind(StrEnum):
    SCOPE = "scope"
    SENSITIVE_TOOL = "sensitive_tool"
    PUBLICATION = "publication"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalMismatch(ValueError):
    """Raised when a granted approval does not describe the call being attempted."""


class ApprovalRequest(BaseModel):
    """A reviewer decision bound to one exact tool, resource and argument digest."""

    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    tenant_id: str = Field(min_length=1, max_length=100)
    kind: ApprovalKind
    mcp_server: NonEmptyText
    capability: NonEmptyText
    resource_id: NonEmptyText
    argument_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    reason: NonEmptyText
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    reviewer_id: str | None = Field(default=None, max_length=200)
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def decided_requests_need_a_reviewer(self) -> ApprovalRequest:
        if self.expires_at <= self.requested_at:
            raise ValueError("an approval request must expire after it is requested")
        decided = self.status in {ApprovalStatus.GRANTED, ApprovalStatus.REJECTED}
        if decided and (self.reviewer_id is None or self.decided_at is None):
            raise ValueError("a decided approval must record its reviewer and decision time")
        return self

    def decide(self, *, reviewer_id: str, granted: bool, at: datetime) -> ApprovalRequest:
        if self.status is not ApprovalStatus.PENDING:
            raise ValueError(f"approval {self.id} was already {self.status}")
        if at >= self.expires_at:
            return self.model_copy(update={"status": ApprovalStatus.EXPIRED, "decided_at": at})
        return self.model_copy(
            update={
                "status": ApprovalStatus.GRANTED if granted else ApprovalStatus.REJECTED,
                "reviewer_id": reviewer_id,
                "decided_at": at,
            }
        )

    def is_valid_for(
        self,
        *,
        mcp_server: str,
        capability: str,
        resource_id: str,
        arguments: dict[str, Any],
        at: datetime,
    ) -> bool:
        """Report whether this approval authorizes exactly the call being attempted."""
        return (
            self.status is ApprovalStatus.GRANTED
            and at < self.expires_at
            and self.mcp_server == mcp_server
            and self.capability == capability
            and self.resource_id == resource_id
            and self.argument_digest == argument_digest(arguments)
        )

    def authorize(
        self,
        *,
        mcp_server: str,
        capability: str,
        resource_id: str,
        arguments: dict[str, Any],
        at: datetime,
    ) -> None:
        """Raise unless this approval authorizes exactly the call being attempted."""
        if not self.is_valid_for(
            mcp_server=mcp_server,
            capability=capability,
            resource_id=resource_id,
            arguments=arguments,
            at=at,
        ):
            raise ApprovalMismatch(
                f"approval {self.id} does not authorize {mcp_server}.{capability} on {resource_id}"
            )


def request_sensitive_tool_approval(
    *,
    job_id: UUID,
    tenant_id: str,
    mcp_server: str,
    capability: str,
    resource_id: str,
    arguments: dict[str, Any],
    reason: str,
    valid_for: timedelta = timedelta(hours=24),
    now: datetime | None = None,
) -> ApprovalRequest:
    """Open an approval bound to the digest of the arguments the agent proposed."""
    requested_at = now or utc_now()
    return ApprovalRequest(
        job_id=job_id,
        tenant_id=tenant_id,
        kind=ApprovalKind.SENSITIVE_TOOL,
        mcp_server=mcp_server,
        capability=capability,
        resource_id=resource_id,
        argument_digest=argument_digest(arguments),
        reason=reason,
        requested_at=requested_at,
        expires_at=requested_at + valid_for,
    )
