"""The GitHub, sandbox and evidence backends, and the gateway's service tokens."""

import subprocess
from typing import Any
from uuid import uuid4

import httpx
import pytest

from research_platform.application.jobs import InMemoryJobRepository
from research_platform.domain.models import EvidenceRecord, ResearchJob
from research_platform.mcp.servers.docker_sandbox import CalculationFailed, DockerSandboxBackend
from research_platform.mcp.servers.evidence_server import EvidenceService
from research_platform.mcp.servers.github_api import GitHubApiBackend, github_client
from research_platform.mcp.service_tokens import ClientCredentialsTokens, TokenUnavailable

# -- GitHub -----------------------------------------------------------------------------


def github(handler: Any) -> GitHubApiBackend:
    return GitHubApiBackend(
        httpx.Client(base_url="https://api.github.test", transport=httpx.MockTransport(handler))
    )


def test_a_repository_is_listed_as_its_file_paths_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/acme/research/git/trees/main"
        assert request.url.params["recursive"] == "1"
        return httpx.Response(
            200,
            json={
                "tree": [
                    {"path": "src", "type": "tree"},
                    {"path": "src/main.py", "type": "blob"},
                    {"path": "README.md", "type": "blob"},
                ]
            },
        )

    assert github(handler).read_repository("acme/research", ref="main") == [
        "README.md",
        "src/main.py",
    ]


def test_pull_requests_are_reduced_to_their_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["per_page"] == "1"
        return httpx.Response(
            200,
            json=[
                {
                    "number": 7,
                    "title": "Add pricing table",
                    "state": "closed",
                    "user": {"login": "octocat"},
                    "created_at": "2026-01-02T00:00:00Z",
                    "merged_at": "2026-01-03T00:00:00Z",
                    "html_url": "https://github.test/acme/research/pull/7",
                    "body": "A long description that is deliberately not returned.",
                },
                {"number": 8, "title": "Second", "user": None},
            ],
        )

    [pull] = github(handler).read_pull_requests("acme/research", limit=1)

    assert pull == {
        "number": 7,
        "title": "Add pricing table",
        "state": "closed",
        "author": "octocat",
        "created_at": "2026-01-02T00:00:00Z",
        "merged_at": "2026-01-03T00:00:00Z",
        "url": "https://github.test/acme/research/pull/7",
    }


def test_a_repository_the_token_cannot_read_is_reported_as_not_found() -> None:
    with pytest.raises(LookupError, match="not found"):
        github(lambda _request: httpx.Response(404)).read_repository("acme/secret", ref="main")


def test_a_github_failure_is_reported_with_its_status() -> None:
    with pytest.raises(RuntimeError, match="503"):
        github(lambda _request: httpx.Response(503)).read_pull_requests("acme/research", limit=5)


def test_an_unreachable_github_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(RuntimeError, match="could not be reached"):
        github(handler).read_repository("acme/research", ref="main")


def test_the_token_travels_as_a_header_and_never_in_the_url() -> None:
    client = github_client("ghs_secret", base_url="https://api.github.test")

    assert client.headers["Authorization"] == "Bearer ghs_secret"
    assert "ghs_secret" not in str(client.base_url)


# -- sandbox ----------------------------------------------------------------------------


class RecordingRunner:
    def __init__(self, result: subprocess.CompletedProcess[str] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, options))
        if isinstance(self.result, Exception) and len(self.calls) == 1:
            raise self.result
        if isinstance(self.result, Exception):
            return subprocess.CompletedProcess(command, 0, "", "")
        return self.result


def completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_a_calculation_runs_in_a_container_with_every_isolation_flag() -> None:
    runner = RecordingRunner(completed(stdout="42\n"))
    sandbox = DockerSandboxBackend(runner=runner)

    output = sandbox.run("print(6 * 7)", cpu_seconds=10, memory_mib=256)

    command, options = runner.calls[0]
    joined = " ".join(command)
    assert output == "42\n"
    assert "--rm" in command
    assert "--network none" in joined
    assert "--memory 256m" in joined
    assert "--memory-swap 256m" in joined
    assert "--pids-limit 64" in joined
    assert "--ulimit cpu=10:10" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--user 65534:65534" in joined
    assert options["timeout"] == sandbox.wall_clock_seconds


def test_the_code_is_passed_on_standard_input_never_on_the_command_line() -> None:
    runner = RecordingRunner(completed(stdout="ok"))
    code = "print('; rm -rf / #')"

    DockerSandboxBackend(runner=runner).run(code, cpu_seconds=5, memory_mib=128)

    command, options = runner.calls[0]
    assert options["input"] == code
    assert all(code not in argument for argument in command)


def test_each_calculation_gets_its_own_container() -> None:
    runner = RecordingRunner(completed(stdout="ok"))
    sandbox = DockerSandboxBackend(runner=runner)

    sandbox.run("1", cpu_seconds=5, memory_mib=128)
    sandbox.run("1", cpu_seconds=5, memory_mib=128)

    names = [command[command.index("--name") + 1] for command, _ in runner.calls]
    assert len(set(names)) == 2


def test_a_calculation_that_runs_too_long_is_killed_by_name() -> None:
    runner = RecordingRunner(subprocess.TimeoutExpired(cmd="docker", timeout=30))

    with pytest.raises(CalculationFailed, match="was stopped"):
        DockerSandboxBackend(runner=runner).run("while True: pass", cpu_seconds=5, memory_mib=128)

    started, _ = runner.calls[0]
    killed, _ = runner.calls[1]
    assert killed[:2] == ["docker", "kill"]
    assert killed[2] == started[started.index("--name") + 1]


def test_a_failing_calculation_reports_its_error_output() -> None:
    runner = RecordingRunner(completed(returncode=1, stderr="ZeroDivisionError: division by zero"))

    with pytest.raises(CalculationFailed, match="ZeroDivisionError"):
        DockerSandboxBackend(runner=runner).run("1 / 0", cpu_seconds=5, memory_mib=128)


def test_a_missing_container_runtime_is_reported_not_worked_around() -> None:
    runner = RecordingRunner(FileNotFoundError("docker"))

    with pytest.raises(CalculationFailed, match="no container runtime"):
        DockerSandboxBackend(runner=runner).run("1", cpu_seconds=5, memory_mib=128)


# -- evidence ---------------------------------------------------------------------------


def stored_evidence() -> tuple[InMemoryJobRepository, ResearchJob, EvidenceRecord]:
    repository = InMemoryJobRepository()
    job = repository.add(
        ResearchJob(tenant_id="acme", requester_id="requester-1", question="What does it cost?")
    )
    record = repository.add_evidence(
        EvidenceRecord(
            tenant_id="acme",
            job_id=job.id,
            excerpt="Vendor pricing is 20 USD per seat.",
            source_uri="https://vendor.test/pricing",
            content_hash=f"sha256:{'0' * 64}",
            producing_task_id=uuid4(),
            tool_invocation_id=uuid4(),
        )
    )
    return repository, job, record


def test_evidence_is_returned_with_the_identifier_a_claim_must_cite() -> None:
    repository, job, record = stored_evidence()

    [found] = EvidenceService(repository).retrieve("acme", str(job.id), None)

    assert found["id"] == str(record.id)
    assert found["excerpt"] == "Vendor pricing is 20 USD per seat."
    assert found["content_hash"] == record.content_hash


def test_one_record_can_be_retrieved_by_identifier() -> None:
    repository, job, record = stored_evidence()
    service = EvidenceService(repository)

    assert len(service.retrieve("acme", str(job.id), str(record.id))) == 1
    assert service.retrieve("acme", str(job.id), str(uuid4())) == []


def test_another_tenants_evidence_is_indistinguishable_from_a_missing_job() -> None:
    repository, job, _record = stored_evidence()
    service = EvidenceService(repository)

    with pytest.raises(ValueError, match="no evidence is recorded") as other_tenant:
        service.retrieve("globex", str(job.id), None)
    with pytest.raises(ValueError, match="no evidence is recorded") as missing:
        service.retrieve("acme", str(uuid4()), None)

    assert str(other_tenant.value).split("job")[0] == str(missing.value).split("job")[0]


@pytest.mark.parametrize(("tenant", "job_id"), [("", str(uuid4())), ("acme", "not-a-uuid")])
def test_an_unidentified_or_malformed_evidence_request_is_refused(tenant: str, job_id: str) -> None:
    repository, _job, _record = stored_evidence()

    with pytest.raises(ValueError):
        EvidenceService(repository).retrieve(tenant, job_id, None)


# -- service tokens ---------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def tokens(handler: Any, clock: Clock | None = None) -> ClientCredentialsTokens:
    return ClientCredentialsTokens(
        token_url="https://issuer.test/token",
        client_id="gateway",
        client_secret="s3cret",
        audience="research-platform",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock or Clock(),
    )


def test_a_service_token_is_requested_with_client_credentials() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"access_token": "token-1", "expires_in": 300})

    assert tokens(handler)() == "token-1"
    body = seen[0].content.decode()
    assert "grant_type=client_credentials" in body
    assert "audience=research-platform" in body
    assert seen[0].headers["Authorization"].startswith("Basic ")
    assert "s3cret" not in body


def test_a_token_is_reused_until_shortly_before_it_expires_then_replaced() -> None:
    issued: list[str] = []
    clock = Clock()

    def handler(_request: httpx.Request) -> httpx.Response:
        issued.append(f"token-{len(issued) + 1}")
        return httpx.Response(200, json={"access_token": issued[-1], "expires_in": 300})

    provider = tokens(handler, clock)

    assert provider() == "token-1"
    clock.now = 200.0
    assert provider() == "token-1"
    clock.now = 271.0
    assert provider() == "token-2"


def test_a_refused_token_request_never_echoes_the_issuers_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid client secret s3cret")

    with pytest.raises(TokenUnavailable) as refused:
        tokens(handler)()

    assert "s3cret" not in str(refused.value)


def test_the_client_secret_is_not_in_the_providers_repr() -> None:
    assert "s3cret" not in repr(tokens(lambda _request: httpx.Response(200)))
