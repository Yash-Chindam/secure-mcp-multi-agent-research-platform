"""The production sandbox backend: one ephemeral, network-less container per calculation.

``screen_calculation`` refuses the obvious escapes before code arrives here, but a screen
cannot contain arbitrary Python - the container is what does. Section 8 asks for an
ephemeral container, a disabled network and resource limits; every flag below is one of
those, and none is optional:

- ``--rm`` and a fresh container per call: nothing survives between calculations.
- ``--network none``: no interface but loopback, so no exfiltration and no callbacks.
- ``--memory`` with an equal ``--memory-swap``, ``--cpus``, ``--pids-limit`` and a CPU
  ``--ulimit``: bounded memory, no swap, no fork bomb, bounded compute.
- ``--read-only`` with a small ``noexec`` tmpfs: nothing can be written and then run.
- ``--cap-drop ALL``, ``no-new-privileges`` and an unprivileged user: no way back up.

The code is passed on standard input, never interpolated into a command line. This
backend needs a container runtime it can reach; in Kubernetes, where mounting a runtime
socket into a pod is itself a privilege, use ``KubernetesSandboxBackend`` instead.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

# The calculation gets a namespace of its own, so ``sys`` here is not within its reach.
ENTRYPOINT = (
    "import sys; exec(compile(sys.stdin.read(), '<calculation>', 'exec'), {'__name__': '__main__'})"
)
MAX_ERROR_CHARACTERS = 2_000


class CalculationFailed(RuntimeError):
    """The container ran the calculation and it did not finish successfully."""


@dataclass
class DockerSandboxBackend:
    image: str = "python:3.12-alpine"
    wall_clock_seconds: int = 30
    docker: str = "docker"
    runner: Runner = field(default=subprocess.run)

    def command(self, name: str, *, cpu_seconds: int, memory_mib: int) -> list[str]:
        return [
            self.docker,
            "run",
            "--rm",
            "--interactive",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            f"{memory_mib}m",
            "--memory-swap",
            f"{memory_mib}m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--ulimit",
            f"cpu={cpu_seconds}:{cpu_seconds}",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            self.image,
            "python",
            "-I",
            "-c",
            ENTRYPOINT,
        ]

    def run(self, code: str, *, cpu_seconds: int, memory_mib: int) -> str:
        name = f"research-calculation-{uuid4().hex}"
        options: dict[str, Any] = {
            "input": code,
            "capture_output": True,
            "text": True,
            "timeout": self.wall_clock_seconds,
            "check": False,
        }
        try:
            completed = self.runner(
                self.command(name, cpu_seconds=cpu_seconds, memory_mib=memory_mib), **options
            )
        except subprocess.TimeoutExpired as error:
            # The client timing out does not stop the container; it has to be killed by
            # name or it would keep its CPU and memory until it finished on its own.
            self.runner([self.docker, "kill", name], capture_output=True, text=True, check=False)
            raise CalculationFailed(
                f"the calculation exceeded {self.wall_clock_seconds} seconds and was stopped"
            ) from error
        except OSError as error:
            raise CalculationFailed(f"no container runtime could be started: {error}") from error

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()[-MAX_ERROR_CHARACTERS:]
            raise CalculationFailed(
                f"the calculation exited with status {completed.returncode}: {detail}"
            )
        return completed.stdout
