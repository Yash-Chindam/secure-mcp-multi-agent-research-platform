"""Which MCP servers a deployment gets, decided from its settings alone."""

from pathlib import Path

import pytest

from research_platform.application.jobs import InMemoryJobRepository
from research_platform.mcp.servers.configured import (
    build_local_servers,
    build_token_provider,
    configure_servers,
)
from research_platform.settings import Settings, parse_pairs


def test_an_unconfigured_deployment_builds_no_servers_at_all() -> None:
    assert build_local_servers(Settings()) == {}


def test_allowed_domains_alone_are_enough_to_build_the_web_server() -> None:
    servers = build_local_servers(Settings(web_allowed_domains="vendor.test"))

    assert sorted(servers) == ["web-research"]


def test_a_workspace_root_builds_the_filesystem_server(tmp_path: Path) -> None:
    servers = build_local_servers(Settings(workspace_roots=f"acme={tmp_path}"))

    assert sorted(servers) == ["filesystem"]


def test_a_github_token_without_an_allowlist_builds_nothing() -> None:
    """A backend with no boundary is not a server with an open boundary."""
    assert build_local_servers(Settings(github_token="ghs_token")) == {}


def test_a_github_token_and_an_allowlist_build_the_github_server() -> None:
    settings = Settings(
        github_token="ghs_token", github_repositories="acme=acme/research|acme/docs"
    )

    assert sorted(build_local_servers(settings)) == ["github"]
    assert settings.tenant_repositories == {"acme": frozenset({"acme/research", "acme/docs"})}


def test_an_analytics_database_without_a_tenant_schema_builds_nothing() -> None:
    settings = Settings(analytics_database_url="postgresql://reader@db/analytics")

    assert build_local_servers(settings) == {}


def test_a_sandbox_image_builds_the_analysis_server() -> None:
    assert sorted(build_local_servers(Settings(sandbox_image="python:3.12-alpine"))) == [
        "python-analysis"
    ]


def test_an_evidence_source_builds_the_evidence_server() -> None:
    servers = build_local_servers(Settings(), evidence_source=InMemoryJobRepository())

    assert sorted(servers) == ["evidence"]


def test_only_the_named_servers_are_built_when_a_process_serves_one(tmp_path: Path) -> None:
    settings = Settings(
        web_allowed_domains="vendor.test",
        workspace_roots=f"acme={tmp_path}",
        sandbox_image="python:3.12-alpine",
    )

    assert sorted(build_local_servers(settings, only=frozenset({"filesystem"}))) == ["filesystem"]


def test_a_remote_server_without_a_service_identity_is_called_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configured = configure_servers(Settings(mcp_server_urls="github=http://github:8000/mcp"))

    assert configured.names == ["github"]
    assert configured.token_provider is None
    assert "without a service token" in caplog.text


def test_no_token_provider_is_built_for_a_purely_in_process_deployment() -> None:
    settings = Settings(
        web_allowed_domains="vendor.test", mcp_client_id="gateway", mcp_client_secret="s3cret"
    )

    assert configure_servers(settings).token_provider is None


def test_the_service_identity_needs_both_a_client_and_a_secret() -> None:
    assert build_token_provider(Settings(mcp_client_id="gateway")) is None


def test_the_token_endpoint_follows_the_realm_convention_unless_set() -> None:
    derived = Settings(oidc_issuer="https://keycloak.test/realms/research/")
    explicit = Settings(oidc_token_url="https://issuer.test/token")

    assert derived.token_url == (
        "https://keycloak.test/realms/research/protocol/openid-connect/token"
    )
    assert explicit.token_url == "https://issuer.test/token"
    with pytest.raises(ValueError, match="token issuer"):
        _ = Settings().token_url


def test_pairs_are_parsed_and_trimmed() -> None:
    assert parse_pairs(" acme = analytics , globex=reports ,", setting="s") == {
        "acme": "analytics",
        "globex": "reports",
    }


@pytest.mark.parametrize("raw", ["acme", "=analytics", "acme=", "acme=a,acme=b"])
def test_a_malformed_boundary_setting_is_a_startup_error_not_a_dropped_entry(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_pairs(raw, setting="analytics_tenant_schemas")


def test_the_github_token_is_not_in_the_settings_repr() -> None:
    assert "ghs_token" not in repr(Settings(github_token="ghs_token"))
