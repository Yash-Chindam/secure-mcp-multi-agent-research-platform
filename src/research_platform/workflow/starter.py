"""Starts and signals ``ResearchJobWorkflow`` on a Temporal service.

The connection is opened on first use rather than at construction, so an API process
starts - and answers its health check, and serves reads - whether or not Temporal is up
yet. A failure to reach it surfaces as ``WorkflowUnavailable`` on the request that needed
it, not as a crash at boot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from research_platform.application.workflows import (
    WorkflowCheckpoint,
    WorkflowNotRunning,
    WorkflowUnavailable,
)
from research_platform.domain.models import ResearchJob
from research_platform.workflow.orchestration import ReviewerDecision
from research_platform.workflow.research_workflow import TASK_QUEUE, ResearchJobWorkflow


def workflow_id_for(job_id: UUID) -> str:
    """The one workflow identifier a job can ever run under.

    Deriving it from the job makes starting a job idempotent: Temporal refuses a second
    workflow with an identifier that is already running, so a retried request cannot
    launch the same research twice.
    """
    return f"research-job-{job_id}"


class TemporalWorkflowStarter:
    def __init__(self, connect: Callable[[], Awaitable[Client]]) -> None:
        self._connect = connect
        self._client: Client | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def for_service(cls, target_host: str, *, namespace: str) -> TemporalWorkflowStarter:
        async def connect() -> Client:
            return await Client.connect(
                target_host, namespace=namespace, data_converter=pydantic_data_converter
            )

        return cls(connect)

    async def _connected(self) -> Client:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                try:
                    self._client = await self._connect()
                except (RPCError, RuntimeError, OSError) as error:
                    raise WorkflowUnavailable(
                        f"the workflow service could not be reached: {error}"
                    ) from error
        return self._client

    async def start(self, job: ResearchJob) -> WorkflowCheckpoint:
        client = await self._connected()
        workflow_id = workflow_id_for(job.id)
        try:
            handle = await client.start_workflow(
                ResearchJobWorkflow.run,
                job,
                id=workflow_id,
                task_queue=TASK_QUEUE,
            )
        except WorkflowAlreadyStartedError:
            # Already running under this job's identifier: report the execution that
            # owns it rather than failing a request that asked for what already holds.
            description = await client.get_workflow_handle(workflow_id).describe()
            return WorkflowCheckpoint(
                workflow_id=workflow_id, workflow_run_id=description.run_id or ""
            )
        except RPCError as error:
            raise WorkflowUnavailable(f"the workflow could not be started: {error}") from error
        return WorkflowCheckpoint(
            workflow_id=workflow_id, workflow_run_id=handle.result_run_id or ""
        )

    async def submit_reviewer_decision(self, job: ResearchJob, decision: ReviewerDecision) -> None:
        client = await self._connected()
        handle = client.get_workflow_handle(job.workflow_id or workflow_id_for(job.id))
        try:
            await handle.signal(ResearchJobWorkflow.submit_reviewer_decision, decision)
        except RPCError as error:
            if error.status is RPCStatusCode.NOT_FOUND:
                raise WorkflowNotRunning(
                    f"job {job.id} has no running workflow to review"
                ) from error
            raise WorkflowUnavailable(
                f"the reviewer decision could not be delivered: {error}"
            ) from error
