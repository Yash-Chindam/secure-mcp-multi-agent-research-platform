"""Section 10's ToolInvocation trail: every decided call reaches the audit sink."""

from uuid import uuid4

import pytest

from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.domain.invocations import (
    AuthorizationDecision,
    ErrorClass,
    InvocationOutcome,
    ToolInvocation,
)
from research_platform.domain.models import ResearchBudget
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role
from research_platform.mcp.gateway import (
    CapabilityDenied,
    CapabilityFailed,
    CapabilityGateway,
    ExecutionRequest,
    InvocationSink,
    UpstreamError,
)
from research_platform.mcp.registry import Capability, CapabilityRegistry

JOB_ID = uuid4()
TASK_ID = uuid4()


class StubExecutor:
    def __init__(self) -> None:
        self.failure: UpstreamError | None = None

    def execute(self, request: ExecutionRequest) -> str:
        if self.failure is not None:
            raise self.failure
        return "Vendor pricing is 20 USD per seat."


def principal(agent: AgentRole = AgentRole.RESEARCHER) -> Principal:
    return Principal(
        tenant_id="acme", subject_id="user-1", roles=frozenset({Role.REQUESTER})
    ).for_agent(agent)


def gateway_with(
    audit: InvocationSink | None, executor: StubExecutor | None = None
) -> CapabilityGateway:
    capability = Capability(
        server="web-research",
        name="fetch",
        description="Fetch an approved public URL.",
        required_roles=frozenset({Role.REQUESTER}),
        allowed_agents=frozenset({AgentRole.RESEARCHER}),
    )
    return CapabilityGateway(
        registry=CapabilityRegistry([capability]),
        executor=executor or StubExecutor(),
        audit=audit,
    )


def fetch(gateway: CapabilityGateway, *, budget: ResearchBudget | None = None) -> object:
    return gateway.invoke(
        principal=principal(),
        job_id=JOB_ID,
        task_id=TASK_ID,
        server="web-research",
        capability_name="fetch",
        arguments={"url": "https://vendor.test/pricing", "api_key": "s3cret"},
        budget=budget or ResearchBudget(max_tool_calls=5),
    )


def test_an_allowed_call_is_written_to_the_audit_trail() -> None:
    recorded: list[ToolInvocation] = []

    fetch(gateway_with(recorded.append))

    assert len(recorded) == 1
    assert recorded[0].outcome is InvocationOutcome.SUCCEEDED
    assert recorded[0].authorization_decision is AuthorizationDecision.ALLOW
    assert recorded[0].job_id == JOB_ID


def test_the_stored_record_never_contains_a_secret_argument() -> None:
    recorded: list[ToolInvocation] = []

    fetch(gateway_with(recorded.append))

    assert "s3cret" not in str(recorded[0].sanitized_arguments)
    assert recorded[0].sanitized_arguments["url"] == "https://vendor.test/pricing"


def test_a_refused_call_is_written_to_the_audit_trail_too() -> None:
    recorded: list[ToolInvocation] = []
    gateway = gateway_with(recorded.append)
    exhausted = ResearchBudget(max_tool_calls=1)
    fetch(gateway, budget=exhausted)

    with pytest.raises(CapabilityDenied):
        fetch(gateway, budget=exhausted)

    assert [record.outcome for record in recorded] == [
        InvocationOutcome.SUCCEEDED,
        InvocationOutcome.DENIED,
    ]
    assert recorded[1].error_class is ErrorClass.BUDGET_EXHAUSTED


def test_a_failed_upstream_call_is_written_to_the_audit_trail_too() -> None:
    recorded: list[ToolInvocation] = []
    executor = StubExecutor()
    executor.failure = UpstreamError("server down", ErrorClass.UPSTREAM_UNAVAILABLE)

    with pytest.raises(CapabilityFailed):
        fetch(gateway_with(recorded.append, executor))

    assert recorded[0].outcome is InvocationOutcome.FAILED


def test_a_sink_that_cannot_store_the_record_does_not_change_the_calls_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken_sink(_invocation: ToolInvocation) -> None:
        raise ConnectionError("database unreachable")

    result = fetch(gateway_with(broken_sink))

    assert result.invocation.outcome is InvocationOutcome.SUCCEEDED  # type: ignore[attr-defined]
    assert "could not be persisted" in caplog.text


def test_a_gateway_with_no_sink_still_returns_the_record_to_its_caller() -> None:
    result = fetch(gateway_with(None))

    assert result.invocation.outcome is InvocationOutcome.SUCCEEDED  # type: ignore[attr-defined]


def test_the_job_service_is_a_working_audit_sink() -> None:
    """The wiring a worker uses: the gateway writes straight into the system of record."""
    repository = InMemoryJobRepository()
    jobs = ResearchJobService(repository)

    fetch(gateway_with(jobs.record_invocation))

    trail = repository.list_invocations("acme", JOB_ID)
    assert [record.capability for record in trail] == ["fetch"]
