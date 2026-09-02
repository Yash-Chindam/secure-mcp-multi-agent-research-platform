"""Assemble the five research MCP servers section 8 describes.

A server is registered only when the deployment gave it something to read. An
unconfigured service is left absent rather than mounted against a placeholder, so a call
to it fails as an unreachable server and opens its circuit instead of quietly returning
nothing an agent would treat as an answer.
"""

from __future__ import annotations

from fastmcp import FastMCP

from research_platform.mcp.servers.backends import (
    GitHubBackend,
    SandboxBackend,
    SqlBackend,
    WebBackend,
)
from research_platform.mcp.servers.filesystem_boundary import WorkspaceRoots
from research_platform.mcp.servers.filesystem_server import (
    FilesystemService,
    build_filesystem_server,
)
from research_platform.mcp.servers.github_boundary import RepositoryAllowlist
from research_platform.mcp.servers.github_server import GitHubService, build_github_server
from research_platform.mcp.servers.postgres_server import PostgresService, build_postgres_server
from research_platform.mcp.servers.sandbox_boundary import SandboxLimits
from research_platform.mcp.servers.sandbox_server import SandboxService, build_sandbox_server
from research_platform.mcp.servers.web_boundary import DomainPolicy, SlidingWindowRateLimiter
from research_platform.mcp.servers.web_research import WebResearchService, build_web_research_server


def build_servers(
    *,
    web_backend: WebBackend | None = None,
    web_policy: DomainPolicy | None = None,
    web_requests_per_minute: int = 30,
    workspace_roots: WorkspaceRoots | None = None,
    sql_backend: SqlBackend | None = None,
    tenant_schemas: dict[str, str] | None = None,
    github_backend: GitHubBackend | None = None,
    repository_allowlist: RepositoryAllowlist | None = None,
    sandbox_backend: SandboxBackend | None = None,
    sandbox_limits: SandboxLimits | None = None,
) -> dict[str, FastMCP]:
    """Build every server the deployment has configured a backend and boundary for."""
    servers: dict[str, FastMCP] = {}

    if web_backend is not None and web_policy is not None:
        servers["web-research"] = build_web_research_server(
            WebResearchService(
                backend=web_backend,
                policy=web_policy,
                limiter=SlidingWindowRateLimiter(limit=web_requests_per_minute),
            )
        )

    if workspace_roots is not None:
        servers["filesystem"] = build_filesystem_server(FilesystemService(roots=workspace_roots))

    if sql_backend is not None and tenant_schemas:
        servers["postgres"] = build_postgres_server(
            PostgresService(backend=sql_backend, tenant_schemas=tenant_schemas)
        )

    if github_backend is not None and repository_allowlist is not None:
        servers["github"] = build_github_server(
            GitHubService(backend=github_backend, allowlist=repository_allowlist)
        )

    if sandbox_backend is not None:
        servers["python-analysis"] = build_sandbox_server(
            SandboxService(
                backend=sandbox_backend,
                limits=sandbox_limits or SandboxLimits(),
            )
        )

    return servers
