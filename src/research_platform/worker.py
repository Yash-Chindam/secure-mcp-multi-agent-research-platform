"""The Temporal worker process: runs the section 9 pipeline for real.

Section 15 lists "separate Temporal/CrewAI workers" as its own deployment unit,
independent of the FastAPI ingress - this module is that unit's entrypoint. It registers
``ResearchJobWorkflow`` plus the activities that back it
(``research_platform.workflow.activities``), assembles the same governed capability
gateway the rest of the platform uses, and polls one task queue until stopped.

A backend a deployment has not configured is left out rather than mounted against a
placeholder (``mcp.servers.configured`` applies that rule), so an agent is never offered
a capability this deployment cannot back. Each server runs in this process unless
``RESEARCH_MCP_SERVER_URLS`` names it, in which case it is called over Streamable HTTP
with a short-lived service token.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from research_platform.agents.crew import build_agent
from research_platform.application.jobs import ResearchJobService
from research_platform.composition import (
    build_artifact_store,
    build_gateway,
    build_job_repository,
    describe_artifacts,
    describe_identity,
    describe_persistence,
)
from research_platform.domain.invocations import ErrorClass
from research_platform.mcp.catalogue import DEFAULT_CAPABILITIES
from research_platform.mcp.fastmcp_executor import FastMCPExecutor
from research_platform.mcp.gateway import CapabilityExecutor, ExecutionRequest, UpstreamError
from research_platform.mcp.registry import CapabilityRegistry
from research_platform.mcp.servers.configured import configure_servers
from research_platform.observability.metrics import configure_metrics
from research_platform.observability.tracing import configure_tracing
from research_platform.settings import Settings, load_settings
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.publishing import GatewaySourceChecker, PublicationActivities
from research_platform.workflow.research_workflow import TASK_QUEUE, ResearchJobWorkflow

logger = logging.getLogger(__name__)


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


def build_job_service(settings: Settings) -> ResearchJobService:
    """The system of record this worker writes job state and its audit trail to.

    With ``RESEARCH_DATABASE_URL`` set this is the same PostgreSQL store the API process
    reads, which is what makes a status this worker writes visible to a requester polling
    the API. Without it the worker falls back to in-process state that no other process
    can see - usable for a single-process demo, not for a deployment.
    """
    return ResearchJobService(build_job_repository(settings))


def build_research_activities(
    settings: Settings, jobs: ResearchJobService | None = None
) -> ResearchActivities:
    """Build the researcher's tools and agent factory from what this deployment configured.

    When a job service is supplied, every MCP call the gateway decides - allowed, denied
    or failed - is written to it as an audit record, which is what makes section 10's
    ``ToolInvocation`` trail durable rather than something only the calling agent saw.
    """
    configured = configure_servers(settings, evidence_source=jobs if jobs is not None else None)
    registry = CapabilityRegistry(
        [
            capability
            for capability in DEFAULT_CAPABILITIES
            if capability.server in configured.targets
        ]
    )
    executor: CapabilityExecutor
    if configured.targets:
        executor = FastMCPExecutor(configured.targets, token_provider=configured.token_provider)
        logger.info("MCP servers available to agents: %s", ", ".join(configured.names))
    else:
        executor = UnconfiguredExecutor()
        logger.warning("no MCP backend is configured; agents will have no tools to call")

    gateway = build_gateway(
        executor=executor,
        settings=settings,
        registry=registry,
        audit=jobs.record_invocation if jobs is not None else None,
    )
    return ResearchActivities(
        gateway=gateway,
        registry=registry,
        build_agent=lambda role, tools: build_agent(role, llm=settings.agent_llm, tools=tools),
    )


def build_job_activities(
    settings: Settings, jobs: ResearchJobService | None = None
) -> JobActivities:
    """The activities that persist status transitions and evidence."""
    return JobActivities(jobs=jobs or build_job_service(settings))


def build_publication_activities(
    settings: Settings, jobs: ResearchJobService, research: ResearchActivities
) -> PublicationActivities:
    """The activity that exports a finished report and its provenance manifest.

    It re-reads cited sources through the same gateway the researchers used, so a
    re-read is governed and audited exactly like the call that captured the evidence.
    """
    return PublicationActivities(
        jobs=jobs,
        artifacts=build_artifact_store(settings),
        sources=GatewaySourceChecker(research.gateway),
    )


def registered_activities(
    research: ResearchActivities, persistence: JobActivities, publication: PublicationActivities
) -> dict[str, Callable[..., Any]]:
    """Every activity ``ResearchJobWorkflow`` calls, by the step it performs.

    Kept in one place so a worker - and a test that stands one up - cannot register a
    different set from the one the workflow expects.
    """
    return {
        "plan": research.plan,
        "research": research.research,
        "analyze": research.analyze,
        "critique": research.critique,
        "report": research.report,
        "transition": persistence.transition,
        "add_evidence": persistence.add_evidence,
        "record_findings": persistence.record_findings,
        "publish": publication.publish,
    }


async def run(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = settings or load_settings()
    tracing = configure_tracing(settings)
    metrics = configure_metrics(settings)
    logger.info("identity: %s", describe_identity(settings))
    logger.info("tracing: %s", tracing.description)
    logger.info("metrics: %s", metrics.description)
    logger.info("persistence: %s", describe_persistence(settings))
    logger.info("artifacts: %s", describe_artifacts(settings))

    client = await Client.connect(
        settings.temporal_target_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )

    jobs = build_job_service(settings)
    research_activities = build_research_activities(settings, jobs)
    job_activities = build_job_activities(settings, jobs)
    publication_activities = build_publication_activities(settings, jobs, research_activities)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ResearchJobWorkflow],
        activities=list(
            registered_activities(
                research_activities, job_activities, publication_activities
            ).values()
        ),
    )
    logger.info("worker polling task queue %r on %s", TASK_QUEUE, settings.temporal_target_host)
    await worker.run()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
