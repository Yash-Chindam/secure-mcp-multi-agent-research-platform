"""Runtime configuration, read from the environment.

Secrets are never defaulted to a working value. An unset authorization service means the
platform runs on its own registry boundary alone, which is stated explicitly at startup
rather than inferred from a silent fallback.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for one running instance of the platform."""

    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    oidc_issuer: str | None = Field(
        default=None,
        description="Issuer of the access tokens to accept; unset falls back to dev headers.",
    )
    oidc_audience: str | None = Field(
        default=None,
        description="Audience the access tokens must be issued for.",
    )
    oidc_jwks_uri: str | None = Field(
        default=None,
        description="Where the issuer publishes its signing keys; derived from the issuer "
        "when unset.",
    )
    opa_url: str | None = Field(
        default=None,
        description="Base URL of the Open Policy Agent deployment; unset disables it.",
    )
    opa_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    max_tool_calls_per_job: int = Field(default=50, ge=1, le=10_000)
    web_allowed_domains: str = Field(
        default="",
        description="Comma-separated domains the web research server may fetch.",
    )
    web_requests_per_minute: int = Field(default=30, ge=1, le=10_000)
    web_search_url: str | None = Field(
        default=None,
        description="A SearXNG-compatible JSON search endpoint; unset refuses web search.",
    )
    workspace_roots: str = Field(
        default="",
        description="Per-tenant filesystem roots, as tenant=/absolute/path pairs.",
    )
    analytics_database_url: str | None = Field(
        default=None,
        description="The analytical PostgreSQL the SQL server reads; connect as a read-only role.",
    )
    analytics_tenant_schemas: str = Field(
        default="",
        description="The one schema each tenant may query, as tenant=schema pairs.",
    )
    github_token: SecretStr | None = Field(
        default=None,
        description="A read-only token scoped to the allowlisted repositories.",
    )
    github_api_url: str = Field(default="https://api.github.com")
    github_repositories: str = Field(
        default="",
        description="Per-tenant repository allowlists, as tenant=owner/a|owner/b pairs.",
    )
    sandbox_image: str | None = Field(
        default=None,
        description="Container image calculations run in; unset disables the sandbox.",
    )
    mcp_server_urls: str = Field(
        default="",
        description="MCP servers reached over Streamable HTTP instead of in process, as "
        "server=url pairs.",
    )
    mcp_client_id: str | None = Field(
        default=None,
        description="The OAuth client the gateway authenticates to remote MCP servers as.",
    )
    mcp_client_secret: SecretStr | None = Field(default=None)
    oidc_token_url: str | None = Field(
        default=None,
        description="Where service tokens are issued; derived from the issuer when unset.",
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="Where to export traces and metrics; unset records them without "
        "sending them anywhere.",
    )
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL system of record; unset keeps job state in process memory.",
    )
    artifact_endpoint: str | None = Field(
        default=None,
        description="URL of the MinIO/S3 object store reports are exported to; unset keeps "
        "artifacts in process memory.",
    )
    artifact_access_key: str | None = Field(default=None)
    artifact_secret_key: SecretStr | None = Field(default=None)
    artifact_bucket: str = Field(default="research-artifacts", min_length=3, max_length=63)
    workflows_enabled: bool = Field(
        default=False,
        description="Start a durable workflow for each new job; off leaves a created job "
        "for a caller to drive by hand.",
    )
    temporal_target_host: str = Field(
        default="localhost:7233",
        description="Address of the Temporal frontend service the worker connects to.",
    )
    temporal_namespace: str = Field(default="default")
    agent_llm: str = Field(
        default="gpt-4o-mini",
        description="The model identifier crewai.Agent is built with; not a secret.",
    )

    @property
    def tokens_are_verified(self) -> bool:
        return bool(self.oidc_issuer and self.oidc_audience)

    @property
    def jwks_uri(self) -> str:
        """Where to fetch signing keys, following the Keycloak realm convention."""
        if self.oidc_jwks_uri:
            return self.oidc_jwks_uri
        if not self.oidc_issuer:
            raise ValueError("a token issuer must be configured before keys can be fetched")
        return f"{self.oidc_issuer.rstrip('/')}/protocol/openid-connect/certs"

    @property
    def token_url(self) -> str:
        """Where to request service tokens, following the Keycloak realm convention."""
        if self.oidc_token_url:
            return self.oidc_token_url
        if not self.oidc_issuer:
            raise ValueError("a token issuer must be configured before tokens can be issued")
        return f"{self.oidc_issuer.rstrip('/')}/protocol/openid-connect/token"

    @property
    def tenant_workspace_roots(self) -> dict[str, str]:
        return parse_pairs(self.workspace_roots, setting="workspace_roots")

    @property
    def tenant_schemas(self) -> dict[str, str]:
        return parse_pairs(self.analytics_tenant_schemas, setting="analytics_tenant_schemas")

    @property
    def tenant_repositories(self) -> dict[str, frozenset[str]]:
        return {
            tenant: frozenset(entry.strip() for entry in entries.split("|") if entry.strip())
            for tenant, entries in parse_pairs(
                self.github_repositories, setting="github_repositories"
            ).items()
        }

    @property
    def remote_mcp_servers(self) -> dict[str, str]:
        return parse_pairs(self.mcp_server_urls, setting="mcp_server_urls")

    @property
    def state_is_durable(self) -> bool:
        return bool(self.database_url)

    @property
    def artifacts_are_durable(self) -> bool:
        return bool(self.artifact_endpoint)

    @property
    def policy_is_externally_enforced(self) -> bool:
        return bool(self.opa_url)

    @property
    def allowed_domains(self) -> frozenset[str]:
        return frozenset(
            domain.strip().lower()
            for domain in self.web_allowed_domains.split(",")
            if domain.strip()
        )


def parse_pairs(raw: str, *, setting: str) -> dict[str, str]:
    """Parse ``key=value,key=value``, refusing an entry that is not exactly that.

    A malformed boundary setting is a startup error rather than a silently dropped entry:
    a tenant whose allowlist failed to parse must not quietly end up with a different one.
    """
    pairs: dict[str, str] = {}
    for entry in raw.split(","):
        if not entry.strip():
            continue
        key, separator, value = entry.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"{setting} entry {entry!r} is not a key=value pair")
        if key.strip() in pairs:
            raise ValueError(f"{setting} names {key.strip()!r} more than once")
        pairs[key.strip()] = value.strip()
    return pairs


def load_settings() -> Settings:
    return Settings()
