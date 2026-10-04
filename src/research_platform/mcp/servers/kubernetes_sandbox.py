"""The sandbox backend for a cluster: one Kubernetes Job per calculation.

``DockerSandboxBackend`` needs a container runtime socket, and mounting one into a pod
hands that pod the node. This backend asks the API server instead, as the pod's own
service account, which needs only to create, read and delete Jobs and read pod logs in
one namespace.

Each calculation gets a fresh Job with the same properties section 8 asks for:

- ephemeral: one pod, never restarted, never retried, deleted once its output is read,
  with a TTL so a Job orphaned by a crash is still collected;
- no network: the pod carries the ``research-platform/sandbox: calculation`` label, which
  the chart's NetworkPolicy denies all ingress and egress for. A network plugin applies
  a policy to a new pod a moment after the pod starts, and some plugins do not enforce
  policies at all, so the container does not trust it: before any submitted code runs,
  it tries to reach the API server and starts the calculation only once that fails. A
  pod that can still reach it when the allowance runs out exits without running anything;
- resource limits: CPU, memory and ephemeral storage limits, a deadline on the Job, and
  a CPU-time limit set inside the container before the calculation starts;
- no way up: non-root, read-only root filesystem, all capabilities dropped, no privilege
  escalation, the default seccomp profile, and no service account token mounted.

A runtime class (gVisor, Kata) can be named for a stronger kernel boundary.

The code travels as an environment variable value in the pod specification, never
interpolated into a command line. A pod has one log stream, so on success the caller
receives what the calculation printed to standard output and standard error together.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from research_platform.mcp.servers.docker_sandbox import (
    MAX_ERROR_CHARACTERS,
    CalculationFailed,
)

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
SANDBOX_LABEL = "research-platform/sandbox"
SANDBOX_LABEL_VALUE = "calculation"
CODE_VARIABLE = "CALCULATION"
CPU_VARIABLE = "CALCULATION_CPU_SECONDS"
UNPRIVILEGED_USER = 65534
ORPHAN_TTL_SECONDS = 300

ISOLATION_VARIABLE = "CALCULATION_ISOLATION_SECONDS"
NOT_ISOLATED = "sandbox-not-isolated"

# Runs in the pod ahead of the submitted code. The API server is the probe target
# because every pod is told where it is and it always listens, so a connection that
# fails means the pod's traffic is being stopped. Two failures in a row are required.
# The calculation is executed in a namespace of its own, so nothing imported here is
# within its reach.
ENTRYPOINT = f"""
import os, resource, socket, sys, time
target = (os.environ['KUBERNETES_SERVICE_HOST'], int(os.environ['KUBERNETES_SERVICE_PORT']))
give_up = time.monotonic() + float(os.environ.pop('{ISOLATION_VARIABLE}'))
blocked = 0
while blocked < 2:
    try:
        socket.create_connection(target, timeout=1).close()
    except OSError:
        blocked += 1
        continue
    blocked = 0
    if time.monotonic() >= give_up:
        sys.exit('{NOT_ISOLATED}')
    time.sleep(0.25)
limit = int(os.environ.pop('{CPU_VARIABLE}'))
resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
code = os.environ.pop('{CODE_VARIABLE}')
exec(compile(code, '<calculation>', 'exec'), {{'__name__': '__main__'}})
"""


@dataclass
class KubernetesSandboxBackend:
    client: httpx.Client
    namespace: str
    token: Callable[[], str]
    image: str = "python:3.12-alpine"
    wall_clock_seconds: int = 30
    runtime_class: str | None = None
    # Scheduling and pulling the image are not the calculation's time; they get their
    # own allowance on top of the wall clock before the Job is given up on.
    startup_seconds: int = 60
    # How long a pod waits for the network policy to reach it before refusing to run.
    isolation_seconds: int = 15
    poll_seconds: float = 0.5
    sleep: Callable[[float], None] = field(default=time.sleep)
    clock: Callable[[], float] = field(default=time.monotonic)

    def manifest(
        self, name: str, code: str, *, cpu_seconds: int, memory_mib: int
    ) -> dict[str, Any]:
        labels = {SANDBOX_LABEL: SANDBOX_LABEL_VALUE}
        pod: dict[str, Any] = {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": UNPRIVILEGED_USER,
                "runAsGroup": UNPRIVILEGED_USER,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "calculation",
                    "image": self.image,
                    "command": ["python", "-I", "-c", ENTRYPOINT],
                    "env": [
                        {"name": CODE_VARIABLE, "value": code},
                        {"name": CPU_VARIABLE, "value": str(cpu_seconds)},
                        {"name": ISOLATION_VARIABLE, "value": str(self.isolation_seconds)},
                    ],
                    "resources": {
                        "limits": {
                            "cpu": "1",
                            "memory": f"{memory_mib}Mi",
                            "ephemeral-storage": "64Mi",
                        },
                        "requests": {"cpu": "100m", "memory": "64Mi"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": [{"name": "scratch", "mountPath": "/tmp"}],
                }
            ],
            "volumes": [{"name": "scratch", "emptyDir": {"medium": "Memory", "sizeLimit": "64Mi"}}],
        }
        if self.runtime_class:
            pod["runtimeClassName"] = self.runtime_class
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": self.wall_clock_seconds + self.isolation_seconds,
                "ttlSecondsAfterFinished": ORPHAN_TTL_SECONDS,
                "template": {"metadata": {"labels": labels}, "spec": pod},
            },
        }

    def run(self, code: str, *, cpu_seconds: int, memory_mib: int) -> str:
        name = f"research-calculation-{uuid4().hex}"
        jobs = f"/apis/batch/v1/namespaces/{self.namespace}/jobs"
        try:
            self._request(
                "POST",
                jobs,
                json=self.manifest(name, code, cpu_seconds=cpu_seconds, memory_mib=memory_mib),
            )
        except httpx.HTTPError as error:
            raise CalculationFailed(f"no calculation job could be started: {error}") from error

        try:
            succeeded = self._wait(f"{jobs}/{name}")
            output = self._output(name)
        except httpx.HTTPError as error:
            raise CalculationFailed(
                f"the calculation job could not be followed: {error}"
            ) from error
        finally:
            self._delete(f"{jobs}/{name}")

        if not succeeded and output.strip().endswith(NOT_ISOLATED):
            raise CalculationFailed(
                "the calculation was not run: the cluster does not cut calculation pods off "
                "from the network. The sandbox NetworkPolicy must exist in this namespace "
                "and the network plugin must enforce it."
            )
        if not succeeded:
            detail = output.strip()[-MAX_ERROR_CHARACTERS:]
            raise CalculationFailed(f"the calculation did not finish successfully: {detail}")
        return output

    def _wait(self, job: str) -> bool:
        deadline = (
            self.clock() + self.wall_clock_seconds + self.isolation_seconds + self.startup_seconds
        )
        while True:
            status = self._request("GET", job).json().get("status", {})
            for condition in status.get("conditions", []):
                if condition.get("status") != "True":
                    continue
                if condition.get("type") == "Complete":
                    return True
                if condition.get("type") == "Failed":
                    if condition.get("reason") == "DeadlineExceeded":
                        raise CalculationFailed(
                            f"the calculation exceeded {self.wall_clock_seconds} seconds "
                            "and was stopped"
                        )
                    return False
            if self.clock() >= deadline:
                raise CalculationFailed(
                    "the calculation job did not finish in time and was removed"
                )
            self.sleep(self.poll_seconds)

    def _output(self, name: str) -> str:
        pods = f"/api/v1/namespaces/{self.namespace}/pods"
        found = self._request("GET", pods, params={"labelSelector": f"job-name={name}"}).json()
        items = found.get("items", [])
        if not items:
            return ""
        pod = items[0]["metadata"]["name"]
        return self._request("GET", f"{pods}/{pod}/log").text

    def _delete(self, job: str) -> None:
        # Background propagation removes the pod with the Job. A failure here is not the
        # calculation's failure: the Job's TTL collects whatever is left behind.
        try:
            self._request("DELETE", job, params={"propagationPolicy": "Background"})
        except httpx.HTTPError:
            return

    def _request(self, method: str, path: str, **options: Any) -> httpx.Response:
        # The token is read for every request because a projected service account token
        # is rotated on disk while the process runs.
        response = self.client.request(
            method, path, headers={"Authorization": f"Bearer {self.token()}"}, **options
        )
        response.raise_for_status()
        return response


def in_cluster_sandbox(
    *,
    image: str,
    wall_clock_seconds: int,
    namespace: str | None = None,
    runtime_class: str | None = None,
    service_account: Path = SERVICE_ACCOUNT,
) -> KubernetesSandboxBackend:
    """A backend that talks to the API server of the cluster this process runs in."""
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if not host:
        raise RuntimeError(
            "the kubernetes sandbox runtime needs to run inside a cluster: "
            "KUBERNETES_SERVICE_HOST is not set"
        )
    address = f"[{host}]" if ":" in host else host
    token_file = service_account / "token"
    return KubernetesSandboxBackend(
        client=httpx.Client(
            base_url=f"https://{address}:{port}",
            verify=str(service_account / "ca.crt"),
            timeout=10.0,
        ),
        namespace=namespace or (service_account / "namespace").read_text(encoding="utf-8").strip(),
        token=lambda: token_file.read_text(encoding="utf-8").strip(),
        image=image,
        wall_clock_seconds=wall_clock_seconds,
        runtime_class=runtime_class,
    )
