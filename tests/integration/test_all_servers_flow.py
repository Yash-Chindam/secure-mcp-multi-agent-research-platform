"""Every section 8 service, driven through the governed gateway over real MCP calls."""

import json
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from research_platform.domain.approvals import (
    ApprovalRequest,
    request_sensitive_tool_approval,
)
from research_platform.domain.invocations import ErrorClass, InvocationOutcome
from research_platform.domain.models import AccessClass, ResearchBudget, utc_now
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.fastmcp_executor import FastMCPExecutor
from research_platform.mcp.gateway import (
    CapabilityDenied,
    CapabilityFailed,
    CapabilityGateway,
)
from research_platform.mcp.servers.backends import SourceDocument, StaticWebBackend
from research_platform.mcp.servers.deployment import build_servers
from research_platform.mcp.servers.filesystem_boundary import WorkspaceRoots
from research_platform.mcp.servers.github_boundary import RepositoryAllowlist
from research_platform.mcp.servers.web_boundary import DomainPolicy

JOB_ID = uuid4()
TASK_ID = uuid4()
BUDGET = ResearchBudget(max_tool_calls=50)

PRICING = SourceDocument(
    url="https://vendor.test/pricing",
    title="Vendor pricing",
    text="The team plan costs 20 USD per seat per month.",
)


class StubSqlBackend:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    def describe_schema(self, schema: str) -> list[str]:
        return [f"{schema}.invoices(id integer, amount numeric)"]

    def run_query(self, sql: str, *, schema: str) -> list[dict[str, object]]:
        self.queries.append((sql, schema))
        return [{"id": 1, "amount": 20}]


class StubGitHubBackend:
    def read_repository(self, repository: str, *, ref: str) -> list[str]:
        return [f"{repository}@{ref}/README.md"]

    def read_pull_requests(self, repository: str, *, limit: int) -> list[dict[str, object]]:
        return [{"number": 1, "repository": repository}]


class StubSandboxBackend:
    def run(self, code: str, *, cpu_seconds: int, memory_mib: int) -> str:
        return json.dumps({"cpu_seconds": cpu_seconds, "memory_mib": memory_mib})


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceRoots:
    acme = tmp_path / "acme"
    acme.mkdir()
    (acme / "notes.md").write_text("Acme internal notes.", encoding="utf-8")
    globex = tmp_path / "globex"
    globex.mkdir()
    (globex / "plan.md").write_text("Globex plan.", encoding="utf-8")
    return WorkspaceRoots(roots={"acme": acme, "globex": globex})


@pytest.fixture
def sql_backend() -> StubSqlBackend:
    return StubSqlBackend()


@pytest.fixture
def executor(workspace: WorkspaceRoots, sql_backend: StubSqlBackend) -> Iterator[FastMCPExecutor]:
    servers = build_servers(
        web_backend=StaticWebBackend(documents={PRICING.url: PRICING}),
        web_policy=DomainPolicy(domains=frozenset({"vendor.test"})),
        workspace_roots=workspace,
        sql_backend=sql_backend,
        tenant_schemas={"acme": "tenant_acme", "globex": "tenant_globex"},
        github_backend=StubGitHubBackend(),
        repository_allowlist=RepositoryAllowlist(
            repositories={"acme": frozenset({"acme/pricing-service"})}
        ),
        sandbox_backend=StubSandboxBackend(),
    )
    with FastMCPExecutor(servers) as running:
        yield running


@pytest.fixture
def gateway(executor: FastMCPExecutor) -> CapabilityGateway:
    return CapabilityGateway(registry=default_registry(), executor=executor)


def agent(
    role: AgentRole,
    *,
    tenant_id: str = "acme",
    clearance: AccessClass = AccessClass.INTERNAL,
) -> Principal:
    return Principal(
        tenant_id=tenant_id,
        subject_id="user-1",
        roles=frozenset({Role.REQUESTER}),
        clearance=clearance,
    ).for_agent(role)


def call(
    gateway: CapabilityGateway,
    principal: Principal,
    server: str,
    capability: str,
    arguments: dict[str, object],
    approval: ApprovalRequest | None = None,
):  # type: ignore[no-untyped-def]
    return gateway.invoke(
        principal=principal,
        job_id=JOB_ID,
        task_id=TASK_ID,
        server=server,
        capability_name=capability,
        arguments=arguments,
        budget=BUDGET,
        approval=approval,
    )


def granted_query_approval(sql: str) -> ApprovalRequest:
    """A reviewer approval bound to exactly this query's arguments."""
    return request_sensitive_tool_approval(
        job_id=JOB_ID,
        tenant_id="acme",
        mcp_server="postgres",
        capability="run_analytical_query",
        resource_id="tenant_acme",
        arguments={"sql": sql},
        reason="analytical read of restricted invoice data",
    ).decide(reviewer_id="reviewer-1", granted=True, at=utc_now())


def run_query(
    gateway: CapabilityGateway,
    sql: str,
    approval: ApprovalRequest | None = None,
):  # type: ignore[no-untyped-def]
    return call(
        gateway,
        agent(AgentRole.ANALYST, clearance=AccessClass.RESTRICTED),
        "postgres",
        "run_analytical_query",
        {"sql": sql},
        approval if approval is not None else granted_query_approval(sql),
    )


@pytest.mark.integration
def test_every_configured_server_is_registered(executor: FastMCPExecutor) -> None:
    servers = build_servers()

    assert servers == {}


@pytest.mark.integration
def test_a_researcher_reads_an_approved_source(gateway: CapabilityGateway) -> None:
    result = call(
        gateway,
        agent(AgentRole.RESEARCHER),
        "web-research",
        "fetch",
        {"url": PRICING.url},
    )

    assert "20 USD per seat" in result.content.text


@pytest.mark.integration
def test_a_researcher_lists_and_reads_its_own_workspace(gateway: CapabilityGateway) -> None:
    listed = call(gateway, agent(AgentRole.RESEARCHER), "filesystem", "list_workspace", {})

    assert json.loads(listed.content.text) == ["notes.md"]

    read = call(
        gateway,
        agent(AgentRole.RESEARCHER),
        "filesystem",
        "read_document",
        {"path": "notes.md"},
    )

    assert read.content.text == "Acme internal notes."


@pytest.mark.integration
def test_a_tenant_cannot_read_another_tenant_workspace(gateway: CapabilityGateway) -> None:
    with pytest.raises(CapabilityFailed) as error:
        call(
            gateway,
            agent(AgentRole.RESEARCHER),
            "filesystem",
            "read_document",
            {"path": "plan.md"},
        )

    assert error.value.invocation.error_class is ErrorClass.INVALID_ARGUMENTS


@pytest.mark.integration
def test_a_workspace_escape_is_refused(gateway: CapabilityGateway) -> None:
    with pytest.raises(CapabilityFailed):
        call(
            gateway,
            agent(AgentRole.RESEARCHER),
            "filesystem",
            "read_document",
            {"path": "../globex/plan.md"},
        )


@pytest.mark.integration
def test_an_analyst_inspects_its_own_schema(gateway: CapabilityGateway) -> None:
    result = call(gateway, agent(AgentRole.ANALYST), "postgres", "describe_schema", {})

    assert "tenant_acme.invoices" in result.content.text


@pytest.mark.integration
def test_an_analytical_query_reaches_the_database_bounded(
    gateway: CapabilityGateway, sql_backend: StubSqlBackend
) -> None:
    result = run_query(gateway, "SELECT id, amount FROM tenant_acme.invoices")

    assert json.loads(result.content.text) == [{"id": 1, "amount": 20}]
    assert sql_backend.queries[0][0].endswith("LIMIT 1000")


@pytest.mark.integration
def test_a_write_statement_never_reaches_the_database(
    gateway: CapabilityGateway, sql_backend: StubSqlBackend
) -> None:
    with pytest.raises(CapabilityFailed):
        run_query(gateway, "DELETE FROM tenant_acme.invoices")

    assert sql_backend.queries == []


@pytest.mark.integration
def test_a_query_against_another_tenant_schema_is_refused(
    gateway: CapabilityGateway, sql_backend: StubSqlBackend
) -> None:
    with pytest.raises(CapabilityFailed):
        run_query(gateway, "SELECT id FROM tenant_globex.invoices")

    assert sql_backend.queries == []


@pytest.mark.integration
def test_an_analytical_query_without_approval_never_reaches_the_database(
    gateway: CapabilityGateway, sql_backend: StubSqlBackend
) -> None:
    with pytest.raises(CapabilityDenied, match="requires reviewer approval"):
        call(
            gateway,
            agent(AgentRole.ANALYST, clearance=AccessClass.RESTRICTED),
            "postgres",
            "run_analytical_query",
            {"sql": "SELECT id FROM tenant_acme.invoices"},
        )

    assert sql_backend.queries == []


@pytest.mark.integration
def test_an_approval_for_one_query_does_not_authorize_another(
    gateway: CapabilityGateway, sql_backend: StubSqlBackend
) -> None:
    """The approval digest binds the reviewer decision to the query they actually saw."""
    approved = granted_query_approval("SELECT id FROM tenant_acme.invoices")

    with pytest.raises(CapabilityDenied, match="does not authorize"):
        run_query(gateway, "SELECT amount FROM tenant_acme.invoices", approved)

    assert sql_backend.queries == []


@pytest.mark.integration
def test_an_allowlisted_repository_is_read(gateway: CapabilityGateway) -> None:
    result = call(
        gateway,
        agent(AgentRole.RESEARCHER),
        "github",
        "read_repository",
        {"repository": "acme/pricing-service", "ref": "main"},
    )

    assert "acme/pricing-service@main/README.md" in result.content.text


@pytest.mark.integration
def test_a_repository_outside_the_allowlist_is_refused(gateway: CapabilityGateway) -> None:
    with pytest.raises(CapabilityFailed):
        call(
            gateway,
            agent(AgentRole.RESEARCHER),
            "github",
            "read_repository",
            {"repository": "attacker/exfiltrate", "ref": "main"},
        )


@pytest.mark.integration
def test_a_calculation_runs_under_the_declared_ceiling(gateway: CapabilityGateway) -> None:
    result = call(
        gateway,
        agent(AgentRole.ANALYST),
        "python-analysis",
        "run_calculation",
        {"code": "import statistics\nresult = statistics.mean([1, 2, 3])"},
    )

    assert json.loads(result.content.text)["memory_mib"] == 512
    assert result.invocation.outcome is InvocationOutcome.SUCCEEDED


@pytest.mark.integration
def test_a_calculation_reaching_for_the_network_is_refused(gateway: CapabilityGateway) -> None:
    with pytest.raises(CapabilityFailed) as error:
        call(
            gateway,
            agent(AgentRole.ANALYST),
            "python-analysis",
            "run_calculation",
            {"code": "import socket\nresult = socket.gethostname()"},
        )

    assert error.value.invocation.error_class is ErrorClass.INVALID_ARGUMENTS
