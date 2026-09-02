"""FastMCP tools for reproducible calculations in a network-isolated sandbox.

The service screens the submitted code and passes the declared resource ceiling to the
backend. It never executes the code in this process: the backend is responsible for
starting an ephemeral, network-disabled container, because a screen alone cannot contain
arbitrary Python.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp import FastMCP

from research_platform.mcp.servers.backends import SandboxBackend
from research_platform.mcp.servers.sandbox_boundary import (
    CalculationNotAllowed,
    SandboxLimits,
    screen_calculation,
)


class SandboxUnavailable(RuntimeError):
    """Raised when no isolated runtime is configured to contain a calculation."""


@dataclass(frozen=True)
class SandboxService:
    """Screen a calculation, then hand it to an isolated runtime under a fixed ceiling."""

    backend: SandboxBackend | None
    limits: SandboxLimits = SandboxLimits()

    def run(self, tenant_id: str, code: str) -> str:
        if not tenant_id or not tenant_id.strip():
            raise CalculationNotAllowed("a calculation must identify its tenant")
        screened = screen_calculation(code)
        if self.backend is None:
            raise SandboxUnavailable(
                "no isolated runtime is configured, so the calculation was not executed"
            )
        output = self.backend.run(
            screened,
            cpu_seconds=self.limits.cpu_seconds,
            memory_mib=self.limits.memory_mib,
        )
        if len(output.encode("utf-8")) > self.limits.max_output_bytes:
            raise CalculationNotAllowed(
                f"the calculation produced more than {self.limits.max_output_bytes} bytes"
            )
        return output


def build_sandbox_server(service: SandboxService) -> FastMCP:
    """Expose sandboxed calculation over MCP."""
    server: FastMCP = FastMCP(name="python-analysis")

    @server.tool
    def run_calculation(tenant_id: str, code: str) -> str:
        """Run a reproducible calculation in a network-isolated sandbox."""
        try:
            return service.run(tenant_id, code)
        except CalculationNotAllowed as error:
            raise ValueError(f"validation error: {error}") from error

    return server
