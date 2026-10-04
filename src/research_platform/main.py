from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from research_platform.api.routes import create_router
from research_platform.application.artifacts import ArtifactStore
from research_platform.application.jobs import ResearchJobService
from research_platform.application.workflows import WorkflowStarter
from research_platform.composition import (
    build_artifact_store,
    build_job_repository,
    build_policy_stack,
    build_token_verifier,
    describe_artifacts,
    describe_identity,
    describe_limits,
    describe_persistence,
)
from research_platform.mcp.catalogue import default_registry
from research_platform.observability.metrics import configure_metrics
from research_platform.observability.tracing import configure_tracing
from research_platform.settings import Settings, load_settings
from research_platform.workflow.starter import TemporalWorkflowStarter


def create_app(
    settings: Settings | None = None,
    workflows: WorkflowStarter | None = None,
    artifacts: ArtifactStore | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Secure MCP Multi-Agent Research Platform",
        version="0.10.0",
    )
    resolved = settings or load_settings()
    service = ResearchJobService(build_job_repository(resolved))
    registry = default_registry()
    artifacts = artifacts or build_artifact_store(resolved)
    policy = build_policy_stack(resolved)
    tracing = configure_tracing(resolved)
    metrics = configure_metrics(resolved)
    if workflows is None and resolved.workflows_enabled:
        workflows = TemporalWorkflowStarter.for_service(
            resolved.temporal_target_host, namespace=resolved.temporal_namespace
        )

    app.state.settings = resolved
    app.state.job_service = service
    app.state.capability_registry = registry
    app.state.policy_stack = policy
    app.state.token_verifier = build_token_verifier(resolved)
    app.state.workflows = workflows
    app.state.artifacts = artifacts
    app.include_router(create_router(service, registry, workflows, artifacts))

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        """Report readiness and how authorization and observability are configured."""
        return {
            "status": "ok",
            "identity": describe_identity(resolved),
            "authorization": policy.description,
            "persistence": describe_persistence(resolved),
            "artifacts": describe_artifacts(resolved),
            "limits": describe_limits(resolved),
            "workflows": (
                "each new job starts a durable workflow"
                if workflows is not None
                else "disabled (jobs are recorded but not executed)"
            ),
            "tracing": tracing.description,
            "metrics": metrics.description,
        }

    @app.get("/auth/config", tags=["operations"])
    def sign_in_configuration() -> dict[str, str]:
        """Tell the requester interface how its users are identified.

        Nothing here is secret: an issuer URL and a public client identifier are what
        any browser-based client is configured with.
        """
        if not resolved.tokens_are_verified:
            return {"mode": "development-headers"}
        assert resolved.oidc_issuer is not None
        configuration = {"mode": "tokens", "issuer": resolved.oidc_issuer.rstrip("/")}
        if resolved.oidc_ui_client_id:
            configuration["client_id"] = resolved.oidc_ui_client_id
        return configuration

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8")

    return app


app = create_app()
