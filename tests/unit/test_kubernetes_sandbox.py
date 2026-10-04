"""The sandbox backend that runs each calculation as a Kubernetes Job."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from research_platform.mcp.servers.configured import build_local_servers
from research_platform.mcp.servers.docker_sandbox import CalculationFailed
from research_platform.mcp.servers.kubernetes_sandbox import (
    KubernetesSandboxBackend,
    in_cluster_sandbox,
)
from research_platform.settings import Settings

NAMESPACE = "research"
JOBS = f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs"
PODS = f"/api/v1/namespaces/{NAMESPACE}/pods"


class ApiServer:
    """Enough of the Kubernetes API to follow one Job from creation to deletion."""

    def __init__(
        self,
        *,
        statuses: list[dict[str, Any]] | None = None,
        log: str = "42\n",
        pods: bool = True,
        refuse: set[str] | None = None,
    ) -> None:
        self.statuses = statuses or [{"conditions": [{"type": "Complete", "status": "True"}]}]
        self.log = log
        self.pods = pods
        self.refuse = refuse or set()
        self.requests: list[httpx.Request] = []
        self.created: dict[str, Any] = {}
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        if method in self.refuse:
            return httpx.Response(403, json={"message": "forbidden"})
        if method == "POST" and path == JOBS:
            import json

            self.created = json.loads(request.content)
            return httpx.Response(201, json=self.created)
        name = self.created.get("metadata", {}).get("name", "")
        if method == "GET" and path == f"{JOBS}/{name}":
            status = self.statuses[min(self.polls, len(self.statuses) - 1)]
            self.polls += 1
            return httpx.Response(200, json={"status": status})
        if method == "GET" and path == PODS:
            assert request.url.params["labelSelector"] == f"job-name={name}"
            items = [{"metadata": {"name": f"{name}-abcde"}}] if self.pods else []
            return httpx.Response(200, json={"items": items})
        if method == "GET" and path == f"{PODS}/{name}-abcde/log":
            return httpx.Response(200, text=self.log)
        if method == "DELETE" and path == f"{JOBS}/{name}":
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"message": f"unexpected {method} {path}"})

    def methods(self) -> list[str]:
        return [request.method for request in self.requests]


def backend(api: ApiServer, **options: Any) -> KubernetesSandboxBackend:
    return KubernetesSandboxBackend(
        client=httpx.Client(base_url="https://kubernetes.test", transport=httpx.MockTransport(api)),
        namespace=NAMESPACE,
        token=lambda: "service-account-token",
        sleep=lambda _seconds: None,
        **options,
    )


def run(sandbox: KubernetesSandboxBackend, code: str = "print(6 * 7)") -> str:
    return sandbox.run(code, cpu_seconds=10, memory_mib=256)


def test_a_calculation_runs_as_a_job_and_its_output_is_returned() -> None:
    api = ApiServer()

    assert run(backend(api)) == "42\n"
    assert api.methods() == ["POST", "GET", "GET", "GET", "DELETE"]


def test_every_request_carries_the_service_account_token() -> None:
    api = ApiServer()

    run(backend(api))

    assert {request.headers["Authorization"] for request in api.requests} == {
        "Bearer service-account-token"
    }


def test_a_rotated_token_is_picked_up_without_a_restart() -> None:
    api = ApiServer()
    issued = iter(f"token-{number}" for number in range(10))
    sandbox = backend(api)
    sandbox.token = lambda: next(issued)

    run(sandbox)

    assert [request.headers["Authorization"] for request in api.requests][:2] == [
        "Bearer token-0",
        "Bearer token-1",
    ]


def test_the_code_is_passed_as_a_value_and_never_put_on_a_command_line() -> None:
    api = ApiServer()
    code = "print('; rm -rf / #')"

    run(backend(api), code)

    container = api.created["spec"]["template"]["spec"]["containers"][0]
    assert {"name": "CALCULATION", "value": code} in container["env"]
    assert code not in " ".join(container["command"])


def test_the_job_is_ephemeral_and_never_retried() -> None:
    api = ApiServer()

    run(backend(api, wall_clock_seconds=30, isolation_seconds=15))

    spec = api.created["spec"]
    assert spec["backoffLimit"] == 0
    assert spec["activeDeadlineSeconds"] == 45
    assert spec["ttlSecondsAfterFinished"] > 0
    assert spec["template"]["spec"]["restartPolicy"] == "Never"


def test_the_pod_is_labelled_so_the_network_policy_isolates_it() -> None:
    api = ApiServer()

    run(backend(api))

    assert api.created["spec"]["template"]["metadata"]["labels"] == {
        "research-platform/sandbox": "calculation"
    }


def test_the_pod_has_no_privileges_and_no_cluster_credentials() -> None:
    api = ApiServer()

    run(backend(api))

    pod = api.created["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert pod["automountServiceAccountToken"] is False
    assert pod["enableServiceLinks"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 65534
    assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }


def test_the_pod_runs_under_the_requested_resource_limits() -> None:
    api = ApiServer()

    backend(api).run("print(1)", cpu_seconds=7, memory_mib=128)

    container = api.created["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"] == {
        "cpu": "1",
        "memory": "128Mi",
        "ephemeral-storage": "64Mi",
    }
    assert {"name": "CALCULATION_CPU_SECONDS", "value": "7"} in container["env"]
    assert "RLIMIT_CPU" in container["command"][-1]


def test_a_runtime_class_is_set_only_when_one_is_named() -> None:
    plain, sandboxed = ApiServer(), ApiServer()

    run(backend(plain))
    run(backend(sandboxed, runtime_class="gvisor"))

    assert "runtimeClassName" not in plain.created["spec"]["template"]["spec"]
    assert sandboxed.created["spec"]["template"]["spec"]["runtimeClassName"] == "gvisor"


def test_the_job_is_polled_until_it_finishes() -> None:
    api = ApiServer(
        statuses=[
            {},
            {"active": 1},
            {"conditions": [{"type": "Complete", "status": "False"}]},
            {"conditions": [{"type": "Complete", "status": "True"}]},
        ]
    )

    assert run(backend(api)) == "42\n"
    assert api.polls == 4


def test_a_failed_calculation_reports_what_it_printed_and_is_still_removed() -> None:
    api = ApiServer(
        statuses=[
            {"conditions": [{"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}]}
        ],
        log="ZeroDivisionError: division by zero\n",
    )

    with pytest.raises(CalculationFailed, match="ZeroDivisionError"):
        run(backend(api))

    assert api.methods()[-1] == "DELETE"


def test_the_calculation_waits_until_the_pod_has_verified_it_has_no_network() -> None:
    api = ApiServer()

    run(backend(api, isolation_seconds=9))

    container = api.created["spec"]["template"]["spec"]["containers"][0]
    script = container["command"][-1]
    assert {"name": "CALCULATION_ISOLATION_SECONDS", "value": "9"} in container["env"]
    assert script.index("create_connection") < script.index("exec(compile(")
    assert "sys.exit('sandbox-not-isolated')" in script


def test_the_entrypoint_is_valid_python_and_hides_its_imports_from_the_calculation() -> None:
    script = backend(ApiServer()).manifest("job", "print(1)", cpu_seconds=1, memory_mib=64)["spec"][
        "template"
    ]["spec"]["containers"][0]["command"][-1]

    compile(script, "<entrypoint>", "exec")
    assert "{'__name__': '__main__'})" in script


def test_a_pod_the_cluster_does_not_isolate_refuses_to_run_and_says_why() -> None:
    api = ApiServer(
        statuses=[
            {"conditions": [{"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}]}
        ],
        log="sandbox-not-isolated\n",
    )

    with pytest.raises(CalculationFailed, match="was not run.*NetworkPolicy"):
        run(backend(api))


def test_a_calculation_past_its_deadline_is_reported_as_stopped() -> None:
    api = ApiServer(
        statuses=[
            {"conditions": [{"type": "Failed", "status": "True", "reason": "DeadlineExceeded"}]}
        ]
    )

    with pytest.raises(CalculationFailed, match="exceeded 30 seconds"):
        run(backend(api))

    assert api.methods()[-1] == "DELETE"


def test_a_job_that_never_finishes_is_given_up_on_and_removed() -> None:
    api = ApiServer(statuses=[{"active": 1}])
    ticks = iter(range(0, 10_000, 20))
    sandbox = backend(api, wall_clock_seconds=30, startup_seconds=60)
    sandbox.clock = lambda: float(next(ticks))

    with pytest.raises(CalculationFailed, match="did not finish in time"):
        run(sandbox)

    assert api.methods()[-1] == "DELETE"


def test_a_job_whose_pod_was_never_created_fails_with_no_output() -> None:
    api = ApiServer(
        statuses=[
            {"conditions": [{"type": "Failed", "status": "True", "reason": "PodFailurePolicy"}]}
        ],
        pods=False,
    )

    with pytest.raises(CalculationFailed, match="did not finish successfully"):
        run(backend(api))


def test_an_api_server_that_refuses_the_job_fails_the_calculation() -> None:
    api = ApiServer(refuse={"POST"})

    with pytest.raises(CalculationFailed, match="no calculation job could be started"):
        run(backend(api))

    assert api.methods() == ["POST"]


def test_a_job_that_cannot_be_followed_fails_the_calculation_and_is_still_removed() -> None:
    api = ApiServer(refuse={"GET"})

    with pytest.raises(CalculationFailed, match="could not be followed"):
        run(backend(api))

    assert api.methods()[-1] == "DELETE"


def test_a_failed_cleanup_does_not_fail_a_finished_calculation() -> None:
    api = ApiServer(refuse={"DELETE"})

    assert run(backend(api)) == "42\n"


# -- in-cluster configuration -------------------------------------------------------------


@pytest.fixture
def service_account(tmp_path: Path) -> Path:
    (tmp_path / "token").write_text("first-token\n", encoding="utf-8")
    (tmp_path / "namespace").write_text("research\n", encoding="utf-8")
    # An empty CA bundle is enough to build the client; nothing here connects.
    (tmp_path / "ca.crt").write_text("", encoding="utf-8")
    return tmp_path


def in_cluster(service_account: Path, **options: Any) -> KubernetesSandboxBackend:
    return in_cluster_sandbox(
        image="python:3.12-alpine",
        wall_clock_seconds=30,
        service_account=service_account,
        **options,
    )


def test_the_api_server_and_namespace_come_from_the_pod(
    monkeypatch: pytest.MonkeyPatch, service_account: Path
) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    monkeypatch.setattr(httpx, "Client", lambda **options: options)

    sandbox = in_cluster(service_account)

    assert sandbox.client["base_url"] == "https://10.43.0.1:443"  # type: ignore[index]
    assert sandbox.client["verify"] == str(service_account / "ca.crt")  # type: ignore[index]
    assert sandbox.namespace == "research"
    assert sandbox.token() == "first-token"

    (service_account / "token").write_text("second-token\n", encoding="utf-8")
    assert sandbox.token() == "second-token"


def test_an_ipv6_api_server_address_is_bracketed(
    monkeypatch: pytest.MonkeyPatch, service_account: Path
) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "fd00::1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "6443")
    monkeypatch.setattr(httpx, "Client", lambda **options: options)

    sandbox = in_cluster(service_account, namespace="calculations", runtime_class="gvisor")

    assert sandbox.client["base_url"] == "https://[fd00::1]:6443"  # type: ignore[index]
    assert sandbox.namespace == "calculations"
    assert sandbox.runtime_class == "gvisor"


def test_the_kubernetes_runtime_refuses_to_start_outside_a_cluster(
    monkeypatch: pytest.MonkeyPatch, service_account: Path
) -> None:
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    with pytest.raises(RuntimeError, match="inside a cluster"):
        in_cluster(service_account)


def test_the_configured_runtime_decides_which_backend_runs_calculations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen: dict[str, Any] = {}

    def fake(**options: Any) -> object:
        chosen.update(options)
        return object()

    monkeypatch.setattr("research_platform.mcp.servers.configured.in_cluster_sandbox", fake)
    settings = Settings(
        sandbox_image="python:3.12-alpine",
        sandbox_runtime="kubernetes",
        sandbox_namespace="calculations",
        sandbox_runtime_class="gvisor",
    )

    assert sorted(build_local_servers(settings)) == ["python-analysis"]
    assert chosen == {
        "image": "python:3.12-alpine",
        "wall_clock_seconds": 30,
        "namespace": "calculations",
        "runtime_class": "gvisor",
    }
