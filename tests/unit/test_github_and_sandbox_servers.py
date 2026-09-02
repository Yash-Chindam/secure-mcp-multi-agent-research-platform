import json

import pytest

from research_platform.mcp.servers.github_boundary import (
    RepositoryAllowlist,
    RepositoryNotAllowed,
)
from research_platform.mcp.servers.github_server import GitHubService, build_github_server
from research_platform.mcp.servers.sandbox_boundary import (
    CalculationNotAllowed,
    SandboxLimits,
)
from research_platform.mcp.servers.sandbox_server import (
    SandboxService,
    SandboxUnavailable,
    build_sandbox_server,
)

ALLOWLIST = RepositoryAllowlist(
    repositories={
        "acme": frozenset({"acme/pricing-service"}),
        "globex": frozenset({"globex/platform"}),
    }
)


class StubGitHubBackend:
    def __init__(self) -> None:
        self.repository_reads: list[tuple[str, str]] = []
        self.pull_request_reads: list[tuple[str, int]] = []

    def read_repository(self, repository: str, *, ref: str) -> list[str]:
        self.repository_reads.append((repository, ref))
        return ["README.md", "src/pricing.py"]

    def read_pull_requests(self, repository: str, *, limit: int) -> list[dict[str, object]]:
        self.pull_request_reads.append((repository, limit))
        return [{"number": 1, "title": "Adjust seat pricing"}]


def github(backend: StubGitHubBackend | None = None) -> tuple[GitHubService, StubGitHubBackend]:
    used = backend or StubGitHubBackend()
    return GitHubService(backend=used, allowlist=ALLOWLIST), used


def test_an_allowlisted_repository_is_read_at_the_requested_ref() -> None:
    service, backend = github()

    paths = service.read_repository("acme", "acme/pricing-service", "release/2026-08")

    assert paths == ["README.md", "src/pricing.py"]
    assert backend.repository_reads == [("acme/pricing-service", "release/2026-08")]


def test_a_repository_outside_the_tenant_allowlist_is_never_read() -> None:
    service, backend = github()

    with pytest.raises(RepositoryNotAllowed, match="not an approved repository"):
        service.read_repository("acme", "globex/platform", "main")

    assert backend.repository_reads == []


def test_a_malformed_ref_is_refused_before_the_backend_is_called() -> None:
    service, backend = github()

    with pytest.raises(RepositoryNotAllowed, match="not a valid git ref"):
        service.read_repository("acme", "acme/pricing-service", "main/")

    assert backend.repository_reads == []


def test_a_github_call_must_identify_its_tenant() -> None:
    service, backend = github()

    with pytest.raises(RepositoryNotAllowed, match="must identify its tenant"):
        service.read_repository("  ", "acme/pricing-service", "main")

    assert backend.repository_reads == []


def test_the_pull_request_limit_is_bounded() -> None:
    service, backend = github()

    service.read_pull_requests("acme", "acme/pricing-service", 5_000)

    assert backend.pull_request_reads == [("acme/pricing-service", 50)]


def test_a_non_positive_pull_request_limit_is_raised_to_one() -> None:
    service, backend = github()

    service.read_pull_requests("acme", "acme/pricing-service", 0)

    assert backend.pull_request_reads == [("acme/pricing-service", 1)]


def test_the_github_server_exposes_both_read_tools() -> None:
    service, _ = github()

    server = build_github_server(service)

    assert server.name == "github"


class StubSandboxBackend:
    def __init__(self, output: str = "result = 3.5\n") -> None:
        self.output = output
        self.runs: list[tuple[str, int, int]] = []

    def run(self, code: str, *, cpu_seconds: int, memory_mib: int) -> str:
        self.runs.append((code, cpu_seconds, memory_mib))
        return self.output


def test_a_screened_calculation_runs_under_the_declared_ceiling() -> None:
    backend = StubSandboxBackend()
    service = SandboxService(backend=backend, limits=SandboxLimits(cpu_seconds=5, memory_mib=256))

    output = service.run("acme", "import statistics\nresult = statistics.mean([1, 2, 3])")

    assert output == "result = 3.5\n"
    assert backend.runs[0][1:] == (5, 256)


def test_withheld_capabilities_are_refused_before_a_container_starts() -> None:
    backend = StubSandboxBackend()
    service = SandboxService(backend=backend)

    with pytest.raises(CalculationNotAllowed, match="must not import socket"):
        service.run("acme", "import socket")

    assert backend.runs == []


def test_a_calculation_must_identify_its_tenant() -> None:
    backend = StubSandboxBackend()
    service = SandboxService(backend=backend)

    with pytest.raises(CalculationNotAllowed, match="must identify its tenant"):
        service.run("", "result = 1")

    assert backend.runs == []


def test_without_an_isolated_runtime_nothing_is_executed() -> None:
    """A screen is not containment, so a missing runtime must refuse rather than run."""
    service = SandboxService(backend=None)

    with pytest.raises(SandboxUnavailable, match="not executed"):
        service.run("acme", "result = 1")


def test_oversized_output_is_refused() -> None:
    backend = StubSandboxBackend(output="x" * 5_000)
    service = SandboxService(backend=backend, limits=SandboxLimits(max_output_bytes=1_024))

    with pytest.raises(CalculationNotAllowed, match="more than 1024 bytes"):
        service.run("acme", "result = 1")


def test_the_sandbox_server_is_named_for_its_capability() -> None:
    server = build_sandbox_server(SandboxService(backend=StubSandboxBackend()))

    assert server.name == "python-analysis"


def test_the_default_sandbox_ceiling_has_no_network() -> None:
    service = SandboxService(backend=StubSandboxBackend())

    assert service.limits.network_enabled is False


def test_repository_reads_serialize_as_json() -> None:
    service, _ = github()

    assert json.loads(
        json.dumps(service.read_repository("acme", "acme/pricing-service", "main"))
    ) == ["README.md", "src/pricing.py"]
