"""The Temporal worker process: runs the section 9 pipeline for real.

Section 15 lists "separate Temporal/CrewAI workers" as its own deployment unit,
independent of the FastAPI ingress - this module is that unit's entrypoint. It registers
``ResearchJobWorkflow`` plus the activities that back it
(``research_platform.workflow.activities``), assembles the same governed capability
gateway the rest of the platform uses, and polls one task queue until stopped.

A backend a deployment has not configured is left out rather than mounted against a
placeholder (the same rule ``mcp.servers.deployment.build_servers`` already applies), so
a researcher agent calling an unconfigured capability fails with an unreachable server
rather than a placeholder answer it could mistake for a real one. Only the web research
server has a backend wired here today, and even that one (``StaticWebBackend``) is a
fixed-document stand-in, not a real HTTP client - the Postgres, GitHub and sandbox
backends have no production implementation yet, so this worker only ever registers the
capabilities it can actually back.
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from research_platform.agents.crew import build_agent
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.composition import build_gateway, describe_identity
from research_platform.domain.invocations import ErrorClass
from research_platform.mcp.catalogue import DEFAULT_CAPABILITIES
from research_platform.mcp.fastmcp_executor import FastMCPExecutor
from research_platform.mcp.gateway import ExecutionRequest, UpstreamError
from research_platform.mcp.registry import CapabilityRegistry
from research_platform.mcp.servers.backends import StaticWebBackend
from research_platform.mcp.servers.deployment import build_servers
from research_platform.mcp.servers.web_boundary import DomainPolicy
from research_platform.observability.metrics import configure_metrics
from research_platform.observability.tracing import configure_tracing
from research_platform.settings import Settings, load_settings
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.research_workflow import ResearchJobWorkflow

logger = logging.getLogger(__name__)

TASK_QUEUE = "research-jobs"


class UnconfiguredExecutor:
    """Refuses every call rather than silently defaulting to a working transport.

    Used only when a deployment has configured no MCP backend at all - the registry that
    would name a real capability was never populated either, so this exists only to give
    a clear, typed error if something still tries to reach it.
    """

    def execute(self, request: ExecutionRequest) -> str:
        raise UpstreamError(
            f"no MCP backend is configured for {request.capability.server}",
            ErrorClass.UPSTREAM_UNAVAILABLE,
        )


def build_research_activities(settings: Settings) -> ResearchActivities:
    """Build the researcher's tools and agent factory from what this deployment configured."""
    servers = build_servers(
        web_backend=StaticWebBackend(),
        web_policy=(
            DomainPolicy(domains=settings.allowed_domains) if settings.allowed_domains else None
        ),
        web_requests_per_minute=settings.web_requests_per_minute,
    )

    registry = CapabilityRegistry(
        [capability for capability in DEFAULT_CAPABILITIES if capability.server in servers]
    )
    executor = FastMCPExecutor(servers) if servers else UnconfiguredExecutor()
    if not servers:
        logger.warning("no MCP backend is configured; agents will have no tools to call")

    gateway = build_gateway(executor=executor, settings=settings, registry=registry)
    return ResearchActivities(
        gateway=gateway,
        registry=registry,
        build_agent=lambda role, tools: build_agent(role, llm=settings.agent_llm, tools=tools),
    )


def build_job_activities() -> JobActivities:
    """The in-process job store this worker persists status transitions through.

    Matches ``main.py``'s current job service exactly: an in-memory system of record,
    not yet the durable PostgreSQL store section 10 describes. A worker and the API
    process each hold their own instance until that lands, so job state a worker writes
    here is not yet visible to an API process reading it back.
    """
    return JobActivities(jobs=ResearchJobService(InMemoryJobRepository()))


async def run(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = settings or load_settings()
    tracing = configure_tracing(settings)
    metrics = configure_metrics(settings)
    logger.info("identity: %s", describe_identity(settings))
    logger.info("tracing: %s", tracing.description)
    logger.info("metrics: %s", metrics.description)

    client = await Client.connect(
        settings.temporal_target_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )

    research_activities = build_research_activities(settings)
    job_activities = build_job_activities()

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ResearchJobWorkflow],
        activities=[
            research_activities.plan,
            research_activities.research,
            research_activities.analyze,
            research_activities.critique,
            research_activities.report,
            job_activities.transition,
            job_activities.add_evidence,
        ],
    )
    logger.info("worker polling task queue %r on %s", TASK_QUEUE, settings.temporal_target_host)
    await worker.run()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
