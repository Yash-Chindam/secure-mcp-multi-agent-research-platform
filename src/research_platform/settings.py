"""Runtime configuration, read from the environment.

Secrets are never defaulted to a working value. An unset authorization service means the
platform runs on its own registry boundary alone, which is stated explicitly at startup
rather than inferred from a silent fallback.
"""

from __future__ import annotations

from pydantic import Field
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
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="Where to export traces and metrics; unset records them without "
        "sending them anywhere.",
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
    def policy_is_externally_enforced(self) -> bool:
        return bool(self.opa_url)

    @property
    def allowed_domains(self) -> frozenset[str]:
        return frozenset(
            domain.strip().lower()
            for domain in self.web_allowed_domains.split(",")
            if domain.strip()
        )


def load_settings() -> Settings:
    return Settings()
