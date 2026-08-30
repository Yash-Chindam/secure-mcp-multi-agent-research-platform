from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from research_platform.api.routes import create_router
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService


def create_app() -> FastAPI:
    app = FastAPI(
        title="Secure MCP Multi-Agent Research Platform",
        version="0.1.0",
    )
    service = ResearchJobService(InMemoryJobRepository())
    app.state.job_service = service
    app.include_router(create_router(service))

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8")

    return app


app = create_app()
