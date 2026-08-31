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
