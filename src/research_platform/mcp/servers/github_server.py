"""FastMCP tools for allowlisted repository reads."""

from __future__ import annotations

import json
from dataclasses import dataclass

from fastmcp import FastMCP

from research_platform.mcp.servers.backends import GitHubBackend
from research_platform.mcp.servers.github_boundary import (
    RepositoryAllowlist,
    RepositoryNotAllowed,
    ensure_readable_ref,
    resolve_repository,
)

MAX_PULL_REQUESTS = 50


@dataclass(frozen=True)
class GitHubService:
    """Resolve every repository against the tenant allowlist before reading it."""

    backend: GitHubBackend
    allowlist: RepositoryAllowlist

    def read_repository(self, tenant_id: str, repository: str, ref: str) -> list[str]:
        resolved = self._resolve(tenant_id, repository)
        return self.backend.read_repository(resolved, ref=ensure_readable_ref(ref))

    def read_pull_requests(
        self, tenant_id: str, repository: str, limit: int
    ) -> list[dict[str, object]]:
        resolved = self._resolve(tenant_id, repository)
        bounded = max(1, min(limit, MAX_PULL_REQUESTS))
        return self.backend.read_pull_requests(resolved, limit=bounded)

    def _resolve(self, tenant_id: str, repository: str) -> str:
        if not tenant_id or not tenant_id.strip():
            raise RepositoryNotAllowed("a GitHub call must identify its tenant")
        return resolve_repository(repository, tenant_id=tenant_id, allowlist=self.allowlist)


def build_github_server(service: GitHubService) -> FastMCP:
    """Expose allowlisted repository reads over MCP."""
    server: FastMCP = FastMCP(name="github")

    @server.tool
    def read_repository(tenant_id: str, repository: str, ref: str = "main") -> str:
        """List file paths in an allowlisted repository at a branch or tag."""
        try:
            paths = service.read_repository(tenant_id, repository, ref)
        except RepositoryNotAllowed as error:
            raise ValueError(f"validation error: {error}") from error
        return json.dumps(paths, separators=(",", ":"))

    @server.tool
    def read_pull_requests(tenant_id: str, repository: str, limit: int = 10) -> str:
        """Read pull request and issue metadata from an allowlisted repository."""
        try:
            records = service.read_pull_requests(tenant_id, repository, limit)
        except RepositoryNotAllowed as error:
            raise ValueError(f"validation error: {error}") from error
        return json.dumps(records, default=str, separators=(",", ":"))

    return server
