"""The Kubernetes sandbox backend against a real cluster.

Skipped unless a cluster is named. The cluster needs the chart's sandbox Role,
RoleBinding and NetworkPolicy applied in the namespace, a token for the service account
they are bound to, and a network plugin that enforces NetworkPolicy (k3s does):

    RESEARCH_TEST_KUBERNETES_URL         https://127.0.0.1:6443
    RESEARCH_TEST_KUBERNETES_TOKEN_FILE  a file holding the service account token
    RESEARCH_TEST_KUBERNETES_CA_FILE     the cluster's CA certificate
    RESEARCH_TEST_KUBERNETES_NAMESPACE   defaults to "research"
"""

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from research_platform.mcp.servers.docker_sandbox import CalculationFailed
from research_platform.mcp.servers.kubernetes_sandbox import KubernetesSandboxBackend
from research_platform.mcp.servers.sandbox_boundary import SandboxLimits
from research_platform.mcp.servers.sandbox_server import SandboxService

URL = os.environ.get("RESEARCH_TEST_KUBERNETES_URL")
NAMESPACE = os.environ.get("RESEARCH_TEST_KUBERNETES_NAMESPACE", "research")
IMAGE = "python:3.12-alpine"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="RESEARCH_TEST_KUBERNETES_URL is not set"),
]


@pytest.fixture(scope="module")
def client() -> Iterator[httpx.Client]:
    assert URL is not None
    with httpx.Client(
        base_url=URL, verify=os.environ["RESEARCH_TEST_KUBERNETES_CA_FILE"], timeout=15.0
    ) as connected:
        yield connected


def token() -> str:
    return Path(os.environ["RESEARCH_TEST_KUBERNETES_TOKEN_FILE"]).read_text("utf-8").strip()


@pytest.fixture(scope="module")
def sandbox(client: httpx.Client) -> KubernetesSandboxBackend:
    # A generous start-up allowance: the first calculation pulls the image.
    return KubernetesSandboxBackend(
        client=client,
        namespace=NAMESPACE,
        token=token,
        image=IMAGE,
        wall_clock_seconds=30,
        startup_seconds=240,
    )


def test_a_calculation_really_runs_as_a_job_and_returns_its_output(
    sandbox: KubernetesSandboxBackend,
) -> None:
    service = SandboxService(backend=sandbox, limits=SandboxLimits())

    assert service.run("acme", "print(sum(range(101)))").strip() == "5050"


def test_the_job_is_removed_once_its_output_is_read(
    sandbox: KubernetesSandboxBackend, client: httpx.Client
) -> None:
    sandbox.run("print('done')", cpu_seconds=5, memory_mib=128)

    # The service account may not list Jobs, which is the point of its Role; the pods a
    # Job leaves behind are what it can see.
    pods = client.get(
        f"/api/v1/namespaces/{NAMESPACE}/pods",
        params={"labelSelector": "research-platform/sandbox=calculation"},
        headers={"Authorization": f"Bearer {token()}"},
    ).json()["items"]

    assert [pod for pod in pods if "deletionTimestamp" not in pod["metadata"]] == []


def test_the_pod_has_no_network(sandbox: KubernetesSandboxBackend) -> None:
    """Run the backend directly: the screen would refuse this code before it got here."""
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('reached')\n"
        "except OSError as error:\n"
        "    print('blocked')\n"
    )

    assert sandbox.run(code, cpu_seconds=5, memory_mib=128).strip() == "blocked"


def test_the_pod_cannot_reach_the_api_server(sandbox: KubernetesSandboxBackend) -> None:
    code = (
        "import os, socket\n"
        "print('token' if os.path.exists("
        "'/var/run/secrets/kubernetes.io/serviceaccount/token') else 'no token')\n"
        "try:\n"
        "    socket.create_connection((os.environ['KUBERNETES_SERVICE_HOST'], 443), timeout=3)\n"
        "    print('reached')\n"
        "except OSError as error:\n"
        "    print('blocked')\n"
    )

    assert sandbox.run(code, cpu_seconds=5, memory_mib=128).split() == ["no", "token", "blocked"]


def test_the_pod_cannot_write_to_its_filesystem(sandbox: KubernetesSandboxBackend) -> None:
    code = (
        "try:\n"
        "    open('/home/escape.txt', 'w').write('x')\n"
        "    print('written')\n"
        "except OSError:\n"
        "    print('read-only')\n"
    )

    assert sandbox.run(code, cpu_seconds=5, memory_mib=128).strip() == "read-only"


def test_the_calculation_runs_as_an_unprivileged_user(sandbox: KubernetesSandboxBackend) -> None:
    assert sandbox.run("import os; print(os.getuid())", cpu_seconds=5, memory_mib=128).strip() == (
        "65534"
    )


def test_a_failing_calculation_reports_its_error(sandbox: KubernetesSandboxBackend) -> None:
    with pytest.raises(CalculationFailed, match="ZeroDivisionError"):
        sandbox.run("print(1 / 0)", cpu_seconds=5, memory_mib=128)


def test_a_calculation_that_exceeds_its_memory_ceiling_is_stopped(
    sandbox: KubernetesSandboxBackend,
) -> None:
    with pytest.raises(CalculationFailed):
        sandbox.run("x = bytearray(512 * 1024 * 1024); print(len(x))", cpu_seconds=5, memory_mib=64)


def test_a_calculation_that_burns_cpu_is_stopped_by_its_cpu_limit(
    sandbox: KubernetesSandboxBackend,
) -> None:
    with pytest.raises(CalculationFailed):
        sandbox.run("while True: pass", cpu_seconds=2, memory_mib=64)


def test_a_calculation_that_never_finishes_is_stopped_at_the_deadline(
    client: httpx.Client,
) -> None:
    impatient = KubernetesSandboxBackend(
        client=client, namespace=NAMESPACE, token=token, image=IMAGE, wall_clock_seconds=5
    )

    with pytest.raises(CalculationFailed, match="exceeded 5 seconds"):
        impatient.run("import time; time.sleep(120)", cpu_seconds=60, memory_mib=64)


def test_the_service_account_can_do_nothing_else(client: httpx.Client) -> None:
    headers = {"Authorization": f"Bearer {token()}"}

    assert client.get(f"/api/v1/namespaces/{NAMESPACE}/secrets", headers=headers).status_code == 403
    assert client.get("/api/v1/namespaces/kube-system/pods", headers=headers).status_code == 403
    assert (
        client.post(f"/api/v1/namespaces/{NAMESPACE}/pods", headers=headers, json={}).status_code
        == 403
    )
