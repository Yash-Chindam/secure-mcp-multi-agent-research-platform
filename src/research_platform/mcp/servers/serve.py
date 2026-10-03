"""Run one MCP server as its own Streamable HTTP service (sections 8 and 15).

    python -m research_platform.mcp.servers.serve web-research --port 8000

This is the "independently scalable FastMCP service" deployment unit. The server is the
same one a worker would otherwise run in process, built from the same settings with the
same boundary; what changes is that its caller now arrives over the network. So when a
token issuer is configured the server verifies a bearer token on every request - the
gateway's short-lived service token - and without one it says, loudly, that it is
accepting any caller that can reach it.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier

from research_platform.composition import build_job_repository
from research_platform.mcp.servers.configured import KNOWN_SERVERS, build_local_servers
from research_platform.settings import Settings, load_settings

logger = logging.getLogger(__name__)

MCP_PATH = "/mcp"


class ServerNotConfigured(RuntimeError):
    """The named server has no backend or boundary configured in this deployment."""


def build_server(name: str, settings: Settings) -> FastMCP:
    """Build one named server, verifying callers when a token issuer is configured."""
    if name not in KNOWN_SERVERS:
        raise ServerNotConfigured(f"{name} is not an MCP server this platform provides")
    evidence_source = build_job_repository(settings) if name == "evidence" else None
    servers = build_local_servers(settings, evidence_source=evidence_source, only=frozenset({name}))
    if name not in servers:
        raise ServerNotConfigured(
            f"{name} has no backend and boundary configured, so it was not started"
        )
    server = servers[name]
    if settings.tokens_are_verified:
        server.auth = JWTVerifier(
            jwks_uri=settings.jwks_uri,
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
        )
    else:
        logger.warning(
            "%s is serving without a token issuer configured: any caller that can reach "
            "it is accepted, and only the network policy separates it from one",
            name,
        )
    return server


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve one MCP server over Streamable HTTP.")
    parser.add_argument("server", choices=sorted(KNOWN_SERVERS))
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - a container listens on all
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    server = build_server(arguments.server, load_settings())
    logger.info("serving %s on %s:%s%s", arguments.server, arguments.host, arguments.port, MCP_PATH)
    server.run(transport="http", host=arguments.host, port=arguments.port, path=MCP_PATH)


if __name__ == "__main__":
    main()
