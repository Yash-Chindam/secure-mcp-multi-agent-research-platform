"""Build the MCP servers a deployment actually configured, with production backends.

The rule ``deployment.build_servers`` already applies is kept end to end: a service is
built only when its backend *and* its boundary are both configured. A GitHub token with
no repository allowlist builds nothing; an analytical database with no tenant schema
builds nothing. An agent therefore never sees a capability the deployment cannot back,
and never gets a capability whose boundary was left open by omission.

A server named in ``RESEARCH_MCP_SERVER_URLS`` is reached over Streamable HTTP instead of
being built here, which is how one is scaled and deployed independently (section 15).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from psycopg import Connection
from psycopg_pool import ConnectionPool

from research_platform.mcp.fastmcp_executor import Target, TokenProvider
from research_platform.mcp.servers.backends import (
    GitHubBackend,
    SandboxBackend,
    SqlBackend,
    WebBackend,
)
from research_platform.mcp.servers.deployment import build_servers
from research_platform.mcp.servers.docker_sandbox import DockerSandboxBackend
from research_platform.mcp.servers.evidence_server import EvidenceSource
from research_platform.mcp.servers.filesystem_boundary import WorkspaceRoots
from research_platform.mcp.servers.github_api import GitHubApiBackend, github_client
from research_platform.mcp.servers.github_boundary import RepositoryAllowlist
from research_platform.mcp.servers.http_web import HttpWebBackend
from research_platform.mcp.servers.postgres_backend import PostgresSqlBackend
from research_platform.mcp.servers.sandbox_boundary import SandboxLimits
from research_platform.mcp.servers.web_boundary import (
    DomainPolicy,
    RateLimiter,
    SourceNotAllowed,
    normalize_source_url,
)
from research_platform.mcp.service_tokens import ClientCredentialsTokens
from research_platform.persistence.redis_state import RedisRateLimiter, connect
from research_platform.settings import Settings

logger = logging.getLogger(__name__)

KNOWN_SERVERS = frozenset(
    {"web-research", "filesystem", "postgres", "github", "python-analysis", "evidence"}
)


@dataclass(frozen=True)
class ConfiguredServers:
    """Where each configured server is reached, and how a remote one is authenticated."""

    targets: dict[str, Target]
    token_provider: TokenProvider | None

    @property
    def names(self) -> list[str]:
        return sorted(self.targets)


def _web_limiter(settings: Settings) -> RateLimiter | None:
    """The request rate limit, shared through Redis when one is configured.

    Without Redis each replica of the web research server counts its own window, so the
    effective limit is the configured one times the number of replicas.
    """
    if not settings.redis_url:
        return None
    return RedisRateLimiter(connect(settings.redis_url), limit=settings.web_requests_per_minute)


def _web(settings: Settings) -> tuple[WebBackend | None, DomainPolicy | None]:
    if not settings.allowed_domains:
        return None, None
    policy = DomainPolicy(domains=settings.allowed_domains)

    def is_permitted(url: str) -> bool:
        try:
            normalize_source_url(url, policy=policy)
        except SourceNotAllowed:
            return False
        return True

    backend = HttpWebBackend(
        client=httpx.Client(timeout=20.0),
        is_permitted=is_permitted,
        search_url=settings.web_search_url,
    )
    return backend, policy


def _sql(settings: Settings) -> SqlBackend | None:
    if not settings.analytics_database_url or not settings.tenant_schemas:
        return None
    pool: ConnectionPool[Connection[Any]] = ConnectionPool(
        settings.analytics_database_url, min_size=1, max_size=5, open=True
    )
    return PostgresSqlBackend(pool)


def _github(settings: Settings) -> GitHubBackend | None:
    if settings.github_token is None or not settings.tenant_repositories:
        return None
    return GitHubApiBackend(
        github_client(settings.github_token.get_secret_value(), base_url=settings.github_api_url)
    )


def _sandbox(settings: Settings, limits: SandboxLimits) -> SandboxBackend | None:
    if not settings.sandbox_image:
        return None
    return DockerSandboxBackend(
        image=settings.sandbox_image, wall_clock_seconds=limits.wall_clock_seconds
    )


def build_local_servers(
    settings: Settings,
    *,
    evidence_source: EvidenceSource | None = None,
    only: frozenset[str] | None = None,
) -> dict[str, FastMCP]:
    """Build every server this process is configured to run itself.

    ``only`` restricts construction to the named servers, so a process serving one
    server does not open the other servers' database pools and HTTP clients.
    """

    def wanted(name: str) -> bool:
        return only is None or name in only

    web_backend, web_policy = _web(settings) if wanted("web-research") else (None, None)
    roots = settings.tenant_workspace_roots if wanted("filesystem") else {}
    repositories = settings.tenant_repositories if wanted("github") else {}
    limits = SandboxLimits()
    return build_servers(
        web_backend=web_backend,
        web_policy=web_policy,
        web_requests_per_minute=settings.web_requests_per_minute,
        web_limiter=_web_limiter(settings) if web_backend is not None else None,
        workspace_roots=(
            WorkspaceRoots({tenant: Path(root) for tenant, root in roots.items()})
            if roots
            else None
        ),
        sql_backend=_sql(settings) if wanted("postgres") else None,
        tenant_schemas=settings.tenant_schemas,
        github_backend=_github(settings) if repositories else None,
        repository_allowlist=RepositoryAllowlist(repositories) if repositories else None,
        sandbox_backend=_sandbox(settings, limits) if wanted("python-analysis") else None,
        sandbox_limits=limits,
        evidence_source=evidence_source if wanted("evidence") else None,
    )


def build_token_provider(settings: Settings) -> TokenProvider | None:
    """The gateway's service identity, when one is configured.

    Without it a remote server is called unauthenticated, which only a server running
    without a token issuer will accept - the pairing a development stack uses.
    """
    if not settings.mcp_client_id or settings.mcp_client_secret is None:
        return None
    return ClientCredentialsTokens(
        token_url=settings.token_url,
        client_id=settings.mcp_client_id,
        client_secret=settings.mcp_client_secret.get_secret_value(),
        audience=settings.oidc_audience,
    )


def configure_servers(
    settings: Settings, *, evidence_source: EvidenceSource | None = None
) -> ConfiguredServers:
    """Resolve every configured server to an in-process instance or a remote URL."""
    remote = settings.remote_mcp_servers
    local = build_local_servers(
        settings,
        evidence_source=evidence_source,
        only=frozenset(KNOWN_SERVERS - remote.keys()),
    )
    targets: dict[str, Target] = {**local, **remote}
    token_provider = build_token_provider(settings) if remote else None
    if remote and token_provider is None:
        logger.warning(
            "remote MCP servers %s are called without a service token; configure "
            "RESEARCH_MCP_CLIENT_ID and RESEARCH_MCP_CLIENT_SECRET",
            sorted(remote),
        )
    return ConfiguredServers(targets=targets, token_provider=token_provider)
