"""The port the API uses to hand a research job to a durable execution.

The API never imports Temporal. It depends on ``WorkflowStarter``, and
``research_platform.workflow.starter.TemporalWorkflowStarter`` is the implementation a
deployment wires in - which keeps the HTTP boundary testable with a substitute and makes
"no workflow engine is configured" an ordinary, explicit state rather than an import
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research_platform.domain.models import ResearchJob
from research_platform.workflow.orchestration import ReviewerDecision


class WorkflowUnavailable(RuntimeError):
    """The workflow engine could not be reached, so nothing was started or signalled."""


class WorkflowNotRunning(LookupError):
    """The job has no running workflow left to receive a signal."""


@dataclass(frozen=True)
class WorkflowCheckpoint:
    """Which durable execution owns a job (section 10's workflow checkpoint)."""

    workflow_id: str
    workflow_run_id: str


class WorkflowStarter(Protocol):
    async def start(self, job: ResearchJob) -> WorkflowCheckpoint:
        """Begin the section 9 pipeline for this job and report what now owns it."""
        ...

    async def submit_reviewer_decision(self, job: ResearchJob, decision: ReviewerDecision) -> None:
        """Deliver a reviewer's decision to the job's suspended workflow."""
        ...
