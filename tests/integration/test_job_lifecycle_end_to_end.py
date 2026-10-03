"""Section 9 from the outside: an HTTP request becomes a finished, audited research job.

Everything real that can be real is: the FastAPI application, ``TemporalWorkflowStarter``,
a Temporal test server, a worker running ``ResearchJobWorkflow`` with its registered
activities, the capability gateway and its audit sink. Only the LLM is scripted. The API
and the worker share one job service, the way a deployment shares one database, so what
the worker writes is what the requester reads back.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from support.scripted import StubExecutor, honest_crew
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from research_platform.application.jobs import ResearchJobService
from research_platform.application.workflows import WorkflowNotRunning, WorkflowUnavailable
from research_platform.domain.models import ResearchJob
from research_platform.main import create_app
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway
from research_platform.settings import Settings
from research_platform.workflow.activities import JobActivities, ResearchActivities
from research_platform.workflow.research_workflow import TASK_QUEUE, ResearchJobWorkflow
from research_platform.workflow.starter import TemporalWorkflowStarter, workflow_id_for

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

REQUESTER = {"X-Tenant-ID": "acme", "X-Requester-ID": "requester-1", "X-Roles": "requester"}
REVIEWER = {"X-Tenant-ID": "acme", "X-Requester-ID": "reviewer-1", "X-Roles": "reviewer"}


@dataclass
class Platform:
    http: httpx.AsyncClient
    temporal: Client
    starter: TemporalWorkflowStarter

    async def status_of(self, job_id: str) -> str:
        response = await self.http.get(f"/api/v1/jobs/{job_id}", headers=REQUESTER)
        status: str = response.json()["status"]
        return status

    async def wait_for(self, job_id: str, status: str) -> None:
        for _ in range(400):
            if await self.status_of(job_id) == status:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"job never reached {status}; it is {await self.status_of(job_id)}")


@asynccontextmanager
async def running_platform(*, requires_reviewer: bool) -> AsyncIterator[Platform]:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:

        async def connect() -> Client:
            return env.client

        starter = TemporalWorkflowStarter(connect)
        app: FastAPI = create_app(Settings(), workflows=starter)
        jobs: ResearchJobService = app.state.job_service
        registry = default_registry()

        research = ResearchActivities(
            gateway=CapabilityGateway(
                registry=registry, executor=StubExecutor(), audit=jobs.record_invocation
            ),
            registry=registry,
            build_agent=honest_crew(requires_reviewer=requires_reviewer),
        )
        persistence = JobActivities(jobs=jobs)
        async with (
            Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[ResearchJobWorkflow],
                activities=[
                    research.plan,
                    research.research,
                    research.analyze,
                    research.critique,
                    research.report,
                    persistence.transition,
                    persistence.add_evidence,
                ],
            ),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://platform.test"
            ) as http,
        ):
            yield Platform(http=http, temporal=env.client, starter=starter)


async def submit(platform: Platform) -> dict[str, str]:
    response = await platform.http.post(
        "/api/v1/jobs", headers=REQUESTER, json={"question": "What does the vendor charge?"}
    )
    assert response.status_code == 201
    created: dict[str, str] = response.json()
    return created


async def test_a_submitted_question_runs_to_a_completed_job() -> None:
    async with running_platform(requires_reviewer=False) as platform:
        job = await submit(platform)
        await platform.wait_for(job["id"], "completed")

        evidence = await platform.http.get(f"/api/v1/jobs/{job['id']}/evidence", headers=REQUESTER)

    assert job["workflow_id"] == f"research-job-{job['id']}"
    assert job["workflow_run_id"]
    assert [record["excerpt"] for record in evidence.json()] == [
        "Vendor pricing is 20 USD per seat."
    ]


async def test_every_tool_call_the_agents_made_is_in_the_jobs_audit_trail() -> None:
    async with running_platform(requires_reviewer=False) as platform:
        job = await submit(platform)
        await platform.wait_for(job["id"], "completed")

        trail = await platform.http.get(f"/api/v1/jobs/{job['id']}/invocations", headers=REQUESTER)

    [call] = trail.json()
    assert call["mcp_server"] == "web-research"
    assert call["capability"] == "fetch"
    assert call["outcome"] == "succeeded"
    assert call["authorization_decision"] == "allow"
    assert call["job_id"] == job["id"]


async def test_a_flagged_job_waits_for_a_reviewer_and_resumes_on_their_decision() -> None:
    async with running_platform(requires_reviewer=True) as platform:
        job = await submit(platform)
        await platform.wait_for(job["id"], "review_required")

        own = await platform.http.post(
            f"/api/v1/jobs/{job['id']}/review", headers=REQUESTER, json={"decision": "approve"}
        )
        decided = await platform.http.post(
            f"/api/v1/jobs/{job['id']}/review", headers=REVIEWER, json={"decision": "approve"}
        )
        await platform.wait_for(job["id"], "completed")

    assert own.status_code == 403
    assert decided.status_code == 202


async def test_a_rejected_review_ends_the_job_as_failed() -> None:
    async with running_platform(requires_reviewer=True) as platform:
        job = await submit(platform)
        await platform.wait_for(job["id"], "review_required")

        await platform.http.post(
            f"/api/v1/jobs/{job['id']}/review", headers=REVIEWER, json={"decision": "reject"}
        )
        await platform.wait_for(job["id"], "failed")


async def test_starting_the_same_job_twice_reports_the_execution_that_already_owns_it() -> None:
    async with running_platform(requires_reviewer=True) as platform:
        job = await submit(platform)
        await platform.wait_for(job["id"], "review_required")
        stored = ResearchJob.model_validate(
            (await platform.http.get(f"/api/v1/jobs/{job['id']}", headers=REQUESTER)).json()
        )

        again = await platform.starter.start(stored)

    assert again.workflow_id == job["workflow_id"]
    assert again.workflow_run_id == job["workflow_run_id"]


async def test_a_decision_for_a_job_with_no_workflow_is_reported_as_not_running() -> None:
    async with running_platform(requires_reviewer=False) as platform:
        never_started = ResearchJob(
            tenant_id="acme", requester_id="requester-1", question="Never started"
        )

        with pytest.raises(WorkflowNotRunning):
            await platform.starter.submit_reviewer_decision(never_started, "approve")  # type: ignore[arg-type]


async def test_an_unreachable_workflow_service_is_reported_as_unavailable() -> None:
    async def connect() -> Client:
        raise OSError("connection refused")

    starter = TemporalWorkflowStarter(connect)
    job = ResearchJob(tenant_id="acme", requester_id="requester-1", question="Unreachable")

    with pytest.raises(WorkflowUnavailable, match="could not be reached"):
        await starter.start(job)


async def test_a_jobs_workflow_identifier_is_derived_from_the_job_alone() -> None:
    job_id = uuid4()

    assert workflow_id_for(job_id) == f"research-job-{job_id}"
    assert workflow_id_for(job_id) == workflow_id_for(job_id)
