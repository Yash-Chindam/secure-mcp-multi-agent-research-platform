"""The single governed path from an agent to an MCP capability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel

from research_platform.domain.approvals import ApprovalMismatch, ApprovalRequest
from research_platform.domain.invocations import (
    AuthorizationDecision,
    ErrorClass,
    InvocationOutcome,
    ToolInvocation,
    argument_digest,
    sanitize_arguments,
)
from research_platform.domain.models import ResearchBudget, utc_now
from research_platform.identity import Principal
from research_platform.mcp.breaker import BudgetExhausted, BudgetLedger, CircuitBreaker, CircuitOpen
from research_platform.mcp.policy import AuthorizationRequest, PolicyEngine, RegistryPolicyEngine
from research_platform.mcp.registry import Capability, CapabilityRegistry
from research_platform.mcp.sanitizer import UntrustedContent, sanitize_result


class CapabilityDenied(PermissionError):
    """Raised when policy, approval or budget refuses a call before it executes."""

    def __init__(self, reason: str, invocation: ToolInvocation) -> None:
        super().__init__(reason)
        self.reason = reason
        self.invocation = invocation


class CapabilityFailed(RuntimeError):
    """Raised when an authorized call reached the server and did not succeed."""

    def __init__(self, reason: str, invocation: ToolInvocation) -> None:
        super().__init__(reason)
        self.reason = reason
        self.invocation = invocation


class UpstreamError(RuntimeError):
    """Raised by an executor when the MCP server itself fails."""

    def __init__(self, message: str, error_class: ErrorClass) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class ExecutionRequest:
    capability: Capability
    principal: Principal
    arguments: dict[str, Any]


class CapabilityExecutor(Protocol):
    """The transport seam a FastMCP Streamable HTTP client will implement."""

    def execute(self, request: ExecutionRequest) -> str: ...


class InvocationResult(BaseModel):
    """A completed call: the audit record and the untrusted content it produced."""

    model_config = {"frozen": True}

    invocation: ToolInvocation
    content: UntrustedContent

    @property
    def is_suspicious(self) -> bool:
        return self.content.is_suspicious


@dataclass(frozen=True)
class _AuditContext:
    """The invariant parts of the audit record for one attempted call."""

    job_id: UUID
    task_id: UUID
    tenant_id: str
    capability: Capability
    sanitized_arguments: dict[str, Any]
    argument_digest: str
    policy_version: str

    def record(
        self,
        *,
        decision: AuthorizationDecision,
        outcome: InvocationOutcome,
        error_class: ErrorClass,
        duration_ms: int,
    ) -> ToolInvocation:
        return ToolInvocation(
            job_id=self.job_id,
            task_id=self.task_id,
            tenant_id=self.tenant_id,
            mcp_server=self.capability.server,
            capability=self.capability.name,
            sanitized_arguments=self.sanitized_arguments,
            argument_digest=self.argument_digest,
            policy_version=self.policy_version,
            authorization_decision=decision,
            outcome=outcome,
            error_class=error_class,
            duration_ms=duration_ms,
        )

    def denied(self, reason: str, error_class: ErrorClass) -> CapabilityDenied:
        return CapabilityDenied(
            reason,
            self.record(
                decision=AuthorizationDecision.DENY,
                outcome=InvocationOutcome.DENIED,
                error_class=error_class,
                duration_ms=0,
            ),
        )

    def with_policy_version(self, policy_version: str) -> _AuditContext:
        return _AuditContext(
            job_id=self.job_id,
            task_id=self.task_id,
            tenant_id=self.tenant_id,
            capability=self.capability,
            sanitized_arguments=self.sanitized_arguments,
            argument_digest=self.argument_digest,
            policy_version=policy_version,
        )


class CapabilityGateway:
    """Authorize, meter, execute and audit every MCP call in one place.

    Nothing reaches an MCP server except through this gateway, so the security rules in
    section 11 hold regardless of what an agent reasoned its way to asking for.
    """

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        executor: CapabilityExecutor,
        policy: PolicyEngine | None = None,
        budgets: BudgetLedger | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._policy = policy or RegistryPolicyEngine()
        self._budgets = budgets or BudgetLedger()
        self._breaker = breaker or CircuitBreaker()

    @property
    def budgets(self) -> BudgetLedger:
        return self._budgets

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def invoke(
        self,
        *,
        principal: Principal,
        job_id: UUID,
        task_id: UUID,
        server: str,
        capability_name: str,
        arguments: dict[str, Any],
        budget: ResearchBudget,
        approval: ApprovalRequest | None = None,
    ) -> InvocationResult:
        """Run one capability, or refuse it before anything leaves the platform."""
        capability = self._registry.resolve_for(principal, server, capability_name)
        audit = _AuditContext(
            job_id=job_id,
            task_id=task_id,
            tenant_id=principal.tenant_id,
            capability=capability,
            sanitized_arguments=sanitize_arguments(arguments),
            argument_digest=argument_digest(arguments),
            policy_version=self._policy.policy_version,
        )

        decision = self._policy.evaluate(
            AuthorizationRequest(
                principal=principal,
                capability=capability,
                job_id=job_id,
                task_id=task_id,
                arguments=arguments,
            )
        )
        audit = audit.with_policy_version(decision.policy_version)
        if not decision.allowed:
            raise audit.denied(decision.reason, ErrorClass.POLICY_DENIED)

        if capability.requires_approval:
            self._check_approval(audit, arguments, approval)

        try:
            self._budgets.reserve_call(job_id, budget)
        except BudgetExhausted as error:
            raise audit.denied(str(error), ErrorClass.BUDGET_EXHAUSTED) from error

        try:
            self._breaker.ensure_closed(capability.server)
        except CircuitOpen as error:
            raise audit.denied(str(error), ErrorClass.UPSTREAM_UNAVAILABLE) from error

        return self._execute(audit, principal, arguments)

    def _execute(
        self,
        audit: _AuditContext,
        principal: Principal,
        arguments: dict[str, Any],
    ) -> InvocationResult:
        capability = audit.capability
        started = utc_now()
        try:
            raw = self._executor.execute(
                ExecutionRequest(capability=capability, principal=principal, arguments=arguments)
            )
        except UpstreamError as error:
            self._breaker.record_failure(capability.server)
            raise CapabilityFailed(
                str(error),
                audit.record(
                    decision=AuthorizationDecision.ALLOW,
                    outcome=InvocationOutcome.FAILED,
                    error_class=error.error_class,
                    duration_ms=_elapsed_ms(started),
                ),
            ) from error

        self._breaker.record_success(capability.server)
        content = sanitize_result(raw, max_bytes=capability.max_result_bytes)
        outcome = InvocationOutcome.SUCCEEDED
        error_class = ErrorClass.NONE
        if content.truncated:
            outcome = InvocationOutcome.FAILED
            error_class = ErrorClass.RESULT_TOO_LARGE

        return InvocationResult(
            invocation=audit.record(
                decision=AuthorizationDecision.ALLOW,
                outcome=outcome,
                error_class=error_class,
                duration_ms=_elapsed_ms(started),
            ),
            content=content,
        )

    @staticmethod
    def _check_approval(
        audit: _AuditContext,
        arguments: dict[str, Any],
        approval: ApprovalRequest | None,
    ) -> None:
        capability = audit.capability
        if approval is None:
            raise audit.denied(
                f"{capability.qualified_name} requires reviewer approval",
                ErrorClass.POLICY_DENIED,
            )
        try:
            approval.authorize(
                mcp_server=capability.server,
                capability=capability.name,
                resource_id=approval.resource_id,
                arguments=arguments,
                at=utc_now(),
            )
        except ApprovalMismatch as error:
            raise audit.denied(str(error), ErrorClass.POLICY_DENIED) from error


def _elapsed_ms(started: datetime) -> int:
    return max(0, int((utc_now() - started).total_seconds() * 1000))
