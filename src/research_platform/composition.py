"""Assemble the governed platform from configuration.

Wiring lives here rather than in the API module so a worker can build the same governed
gateway the HTTP boundary uses, and so the policy layering is decided in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_platform.application.artifacts import ArtifactStore, InMemoryArtifactStore
from research_platform.application.jobs import InMemoryJobRepository, JobRepository
from research_platform.auth import KeyResolver, TokenVerifier
from research_platform.mcp.breaker import BudgetLedger, Budgets
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityExecutor, CapabilityGateway, InvocationSink
from research_platform.mcp.opa import AllOfPolicyEngine, OpaPolicyEngine
from research_platform.mcp.policy import PolicyEngine, RegistryPolicyEngine
from research_platform.mcp.registry import CapabilityRegistry
from research_platform.persistence.object_store import (
    build_artifact_store as connect_artifact_store,
)
from research_platform.persistence.postgres import build_repository
from research_platform.persistence.redis_state import RedisBudgetLedger
from research_platform.persistence.redis_state import connect as connect_redis
from research_platform.settings import Settings


@dataclass(frozen=True)
class PolicyStack:
    """The engine the gateway will use, and how it was assembled."""

    engine: PolicyEngine
    externally_enforced: bool

    @property
    def description(self) -> str:
        if self.externally_enforced:
            return "registry boundary and Open Policy Agent bundle"
        return "registry boundary only (no authorization service configured)"


def build_policy_stack(settings: Settings) -> PolicyStack:
    """Layer the registry boundary with the policy bundle when one is configured.

    Both must permit a call, so a misconfigured bundle cannot widen access beyond the
    registered capability boundary and the boundary cannot override a narrowed policy.
    """
    registry_engine = RegistryPolicyEngine()
    if not settings.opa_url:
        return PolicyStack(engine=registry_engine, externally_enforced=False)

    opa = OpaPolicyEngine(settings.opa_url, timeout_seconds=settings.opa_timeout_seconds)
    return PolicyStack(
        engine=AllOfPolicyEngine([registry_engine, opa]),
        externally_enforced=True,
    )


def build_token_verifier(settings: Settings) -> TokenVerifier | None:
    """Build the token verifier when an issuer and audience are both configured.

    Both are required: verifying a signature without pinning the audience would accept a
    token the realm issued for a different service.
    """
    if not settings.tokens_are_verified:
        return None
    assert settings.oidc_issuer is not None
    assert settings.oidc_audience is not None
    return TokenVerifier(
        issuer=settings.oidc_issuer,
        audience=settings.oidc_audience,
        keys=KeyResolver(settings.jwks_uri),
    )


def describe_identity(settings: Settings) -> str:
    if settings.tokens_are_verified:
        return "verified OAuth access tokens"
    return "development identity headers (no token issuer configured)"


def build_gateway(
    *,
    executor: CapabilityExecutor,
    settings: Settings,
    registry: CapabilityRegistry | None = None,
    audit: InvocationSink | None = None,
) -> CapabilityGateway:
    """Build the single governed path to the MCP capabilities."""
    return CapabilityGateway(
        registry=registry or default_registry(),
        executor=executor,
        policy=build_policy_stack(settings).engine,
        budgets=build_budget_ledger(settings),
        audit=audit,
    )


def build_budget_ledger(settings: Settings) -> Budgets:
    """The ledger jobs spend against: shared through Redis, or this process's own.

    A per-process ledger lets each worker grant a job its whole allowance, so a
    deployment with more than one worker must configure Redis for a budget to mean
    what it says.
    """
    if not settings.redis_url:
        return BudgetLedger()
    return RedisBudgetLedger(connect_redis(settings.redis_url))


def describe_limits(settings: Settings) -> str:
    if settings.limits_are_shared:
        return "budgets and rate limits shared through Redis"
    return "per-process budgets and rate limits (no Redis configured)"


def build_job_repository(settings: Settings) -> JobRepository:
    """Return the durable system of record, or the in-process one when none is configured.

    The in-process repository is a development convenience and is not shared between an
    API process and a worker, so a deployment that runs both must configure a database.
    ``describe_persistence`` states which of the two is in use at startup rather than
    leaving it to be inferred.
    """
    if not settings.database_url:
        return InMemoryJobRepository()
    return build_repository(settings.database_url)


def describe_persistence(settings: Settings) -> str:
    if settings.state_is_durable:
        return "PostgreSQL system of record"
    return "in-process job state (no database configured; not shared between processes)"


def build_artifact_store(settings: Settings) -> ArtifactStore:
    """Return the object store reports are exported to, or the in-process one.

    Like the in-process repository, the in-process store is not shared: a report a worker
    publishes there cannot be read back through the API, so a deployment that runs both
    must configure an object store.
    """
    if not settings.artifact_endpoint:
        return InMemoryArtifactStore()
    if not settings.artifact_access_key or settings.artifact_secret_key is None:
        raise ValueError(
            "RESEARCH_ARTIFACT_ACCESS_KEY and RESEARCH_ARTIFACT_SECRET_KEY are required "
            "when RESEARCH_ARTIFACT_ENDPOINT is set"
        )
    return connect_artifact_store(
        settings.artifact_endpoint,
        access_key=settings.artifact_access_key,
        secret_key=settings.artifact_secret_key.get_secret_value(),
        bucket=settings.artifact_bucket,
    )


def describe_artifacts(settings: Settings) -> str:
    if settings.artifacts_are_durable:
        return f"object store bucket {settings.artifact_bucket!r}"
    return "in-process artifacts (no object store configured; not shared between processes)"
