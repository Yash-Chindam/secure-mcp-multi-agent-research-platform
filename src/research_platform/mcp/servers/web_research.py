"""The web research MCP server.

Every tool applies the domain policy and the rate limit before the backend is touched, so
an unapproved source is refused inside the server itself and not only at the gateway.
Enforcing at both layers means a future direct client of this server inherits the same
boundary.

The tenant is an explicit argument because the platform's only caller is the governed
gateway, which supplies it from the authenticated principal rather than from anything an
agent produced. The Keycloak milestone moves this into a verified token so the server
stops depending on its caller for that fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fastmcp import FastMCP

from research_platform.mcp.servers.backends import SourceDocument, WebBackend
from research_platform.mcp.servers.web_boundary import (
    DomainPolicy,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
    SourceNotAllowed,
    normalize_source_url,
    rate_limit_key,
)

MAX_SEARCH_RESULTS = 10


@dataclass
class WebResearchService:
    """The enforcement logic behind the web research server's tools."""

    backend: WebBackend
    policy: DomainPolicy
    limiter: SlidingWindowRateLimiter

    def search(self, tenant_id: str, query: str, limit: int) -> list[SourceDocument]:
        _require_tenant(tenant_id)
        if not query.strip():
            raise ValueError("a search query must not be empty")
        bounded = max(1, min(limit, MAX_SEARCH_RESULTS))
        self.limiter.acquire(f"{tenant_id}|search")
        return [
            document
            for document in self.backend.search(query, limit=bounded)
            if self._is_fetchable(document.url)
        ]

    def fetch(self, tenant_id: str, url: str) -> SourceDocument:
        _require_tenant(tenant_id)
        normalized = normalize_source_url(url, policy=self.policy)
        self.limiter.acquire(rate_limit_key(tenant_id, normalized))
        document = self.backend.fetch(normalized)
        return SourceDocument(url=normalized, title=document.title, text=document.text)

    def _is_fetchable(self, url: str) -> bool:
        """Drop a result the backend offered but policy would refuse to fetch."""
        try:
            normalize_source_url(url, policy=self.policy)
        except SourceNotAllowed:
            return False
        return True


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id or not tenant_id.strip():
        raise ValueError("a web research call must identify its tenant")


def build_web_research_server(service: WebResearchService) -> FastMCP:
    """Expose the web research capabilities over MCP."""
    server: FastMCP = FastMCP(name="web-research")

    @server.tool
    def search(tenant_id: str, query: str, limit: int = 5) -> str:
        """Search approved public sources and return the matching documents as JSON."""
        try:
            documents = service.search(tenant_id, query, limit)
        except RateLimitExceeded as error:
            raise ValueError(f"rate limited: {error}") from error
        return json.dumps(
            [{"url": document.url, "title": document.title} for document in documents],
            separators=(",", ":"),
        )

    @server.tool
    def fetch(tenant_id: str, url: str) -> str:
        """Fetch readable text from an approved public URL."""
        try:
            document = service.fetch(tenant_id, url)
        except SourceNotAllowed as error:
            raise ValueError(f"source refused: {error}") from error
        except RateLimitExceeded as error:
            raise ValueError(f"rate limited: {error}") from error
        return document.text

    return server
