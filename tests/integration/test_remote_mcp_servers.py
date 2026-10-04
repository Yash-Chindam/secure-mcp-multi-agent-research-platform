"""Section 8: a remote MCP service over Streamable HTTP, called by the governed gateway.

A real server is started on a local port and reached over the network path a deployment
uses - HTTP transport, bearer token verification and all. Nothing is substituted but the
upstream website the web server would fetch.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import jwt
import pytest
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier
from starlette.testclient import TestClient

from research_platform.domain.invocations import ErrorClass, InvocationOutcome
from research_platform.domain.models import AccessClass, ResearchBudget
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role
from research_platform.mcp.catalogue import DEFAULT_CAPABILITIES
from research_platform.mcp.fastmcp_executor import FastMCPExecutor
from research_platform.mcp.gateway import CapabilityFailed, CapabilityGateway
from research_platform.mcp.registry import CapabilityRegistry
from research_platform.mcp.servers.configured import build_local_servers, configure_servers
from research_platform.mcp.servers.serve import MCP_PATH, ServerNotConfigured, build_server
from research_platform.settings import Settings

pytestmark = pytest.mark.integration

ISSUER = "https://keycloak.test/realms/research"
AUDIENCE = "research-platform"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_KEY = (
    SIGNING_KEY.public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)


def service_token(*, audience: str = AUDIENCE, lifetime: timedelta = timedelta(minutes=5)) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": ISSUER,
            "aud": audience,
            "sub": "service-gateway",
            "iat": now,
            "exp": now + lifetime,
        },
        SIGNING_KEY,
        algorithm="RS256",
    )


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port: int = listener.getsockname()[1]
        return port


def serve(server: FastMCP) -> Iterator[str]:
    port = free_port()
    running = uvicorn.Server(
        uvicorn.Config(
            server.http_app(path=MCP_PATH), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not running.started:
        if time.monotonic() > deadline:
            raise TimeoutError("the MCP server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}{MCP_PATH}"
    finally:
        running.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "acme"
    root.mkdir()
    (root / "notes.md").write_text("Vendor pricing is 20 USD per seat.", encoding="utf-8")
    return root


@pytest.fixture
def filesystem_url(workspace: Path) -> Iterator[str]:
    """A filesystem server that verifies a bearer token on every request."""
    settings = Settings(workspace_roots=f"acme={workspace}")
    server = build_local_servers(settings)["filesystem"]
    server.auth = JWTVerifier(public_key=PUBLIC_KEY, issuer=ISSUER, audience=AUDIENCE)
    yield from serve(server)


def gateway_for(url: str, token: str | None) -> CapabilityGateway:
    registry = CapabilityRegistry(
        [capability for capability in DEFAULT_CAPABILITIES if capability.server == "filesystem"]
    )
    provider = (lambda: token) if token is not None else None
    return CapabilityGateway(
        registry=registry,
        executor=FastMCPExecutor({"filesystem": url}, token_provider=provider),
    )


def read_notes(gateway: CapabilityGateway, tenant: str = "acme") -> str:
    principal = Principal(
        tenant_id=tenant,
        subject_id="user-1",
        roles=frozenset({Role.REQUESTER}),
        clearance=AccessClass.INTERNAL,
    ).for_agent(AgentRole.RESEARCHER)
    result = gateway.invoke(
        principal=principal,
        job_id=uuid4(),
        task_id=uuid4(),
        server="filesystem",
        capability_name="read_document",
        arguments={"path": "notes.md"},
        budget=ResearchBudget(max_tool_calls=5),
    )
    return result.content.text


def test_the_gateway_reads_through_a_remote_server_with_a_service_token(
    filesystem_url: str,
) -> None:
    assert "20 USD per seat" in read_notes(gateway_for(filesystem_url, service_token()))


def test_a_remote_server_refuses_a_caller_with_no_token(filesystem_url: str) -> None:
    with pytest.raises(CapabilityFailed) as refused:
        read_notes(gateway_for(filesystem_url, None))

    assert refused.value.invocation.outcome is InvocationOutcome.FAILED


@pytest.mark.parametrize(
    "token",
    [
        service_token(audience="some-other-service"),
        service_token(lifetime=timedelta(minutes=-5)),
        "not-a-token",
    ],
    ids=["wrong-audience", "expired", "malformed"],
)
def test_a_remote_server_refuses_a_token_it_cannot_verify(filesystem_url: str, token: str) -> None:
    with pytest.raises(CapabilityFailed):
        read_notes(gateway_for(filesystem_url, token))


def test_the_tenant_boundary_still_holds_across_the_network(filesystem_url: str) -> None:
    """An authenticated gateway acting for another tenant reads nothing of this one's."""
    with pytest.raises(CapabilityFailed):
        read_notes(gateway_for(filesystem_url, service_token()), tenant="globex")


def test_a_token_that_cannot_be_obtained_fails_the_call_as_unavailable(
    filesystem_url: str,
) -> None:
    def broken() -> str:
        raise ConnectionError("issuer unreachable")

    gateway = CapabilityGateway(
        registry=CapabilityRegistry([c for c in DEFAULT_CAPABILITIES if c.server == "filesystem"]),
        executor=FastMCPExecutor({"filesystem": filesystem_url}, token_provider=broken),
    )

    with pytest.raises(CapabilityFailed) as failed:
        read_notes(gateway)

    assert failed.value.invocation.error_class is ErrorClass.UPSTREAM_UNAVAILABLE


def test_a_server_built_for_serving_verifies_tokens_when_an_issuer_is_configured(
    workspace: Path,
) -> None:
    settings = Settings(
        workspace_roots=f"acme={workspace}", oidc_issuer=ISSUER, oidc_audience=AUDIENCE
    )

    assert isinstance(build_server("filesystem", settings).auth, JWTVerifier)


def test_a_server_built_without_an_issuer_says_it_accepts_any_caller(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = build_server("filesystem", Settings(workspace_roots=f"acme={workspace}"))

    assert server.auth is None
    assert "any caller that can reach it is accepted" in caplog.text


def test_a_served_server_answers_a_liveness_probe_without_a_token(workspace: Path) -> None:
    """A container or pod probe carries no token; the tools themselves still need one."""
    settings = Settings(
        workspace_roots=f"acme={workspace}", oidc_issuer=ISSUER, oidc_audience=AUDIENCE
    )
    server = build_server("filesystem", settings)

    with TestClient(server.http_app(path=MCP_PATH)) as probe:
        health = probe.get("/health")
        tools = probe.post(MCP_PATH, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "server": "filesystem"}
    assert tools.status_code == 401


def test_a_server_with_no_backend_configured_refuses_to_start() -> None:
    with pytest.raises(ServerNotConfigured, match="no backend and boundary"):
        build_server("github", Settings())


def test_an_unknown_server_name_is_refused() -> None:
    with pytest.raises(ServerNotConfigured, match="not an MCP server"):
        build_server("shell", Settings())


def test_a_server_named_as_remote_is_called_by_url_and_not_built_locally(
    workspace: Path,
) -> None:
    settings = Settings(
        workspace_roots=f"acme={workspace}",
        mcp_server_urls="filesystem=http://filesystem:8000/mcp",
        mcp_client_id="gateway",
        mcp_client_secret="s3cret",
        oidc_issuer=ISSUER,
        oidc_audience=AUDIENCE,
    )

    configured = configure_servers(settings)

    assert configured.targets == {"filesystem": "http://filesystem:8000/mcp"}
    assert configured.token_provider is not None
