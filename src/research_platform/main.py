from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from research_platform.api.routes import create_router
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.composition import (
    build_policy_stack,
    build_token_verifier,
    describe_identity,
)
from research_platform.mcp.catalogue import default_registry
from research_platform.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Secure MCP Multi-Agent Research Platform",
        version="0.1.0",
    )
    resolved = settings or load_settings()
    service = ResearchJobService(InMemoryJobRepository())
    registry = default_registry()
    policy = build_policy_stack(resolved)

    app.state.settings = resolved
    app.state.job_service = service
    app.state.capability_registry = registry
    app.state.policy_stack = policy
    app.state.token_verifier = build_token_verifier(resolved)
    app.include_router(create_router(service, registry))

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        """Report readiness and how authorization is being enforced."""
        return {
            "status": "ok",
            "identity": describe_identity(resolved),
            "authorization": policy.description,
        }

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8")

    return app


app = create_app()
