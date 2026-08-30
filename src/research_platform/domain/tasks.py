"""Task decomposition rules for a research assignment."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from research_platform.domain.models import NonEmptyText, utc_now


class AgentRole(StrEnum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    REPORTER = "reporter"


class TaskState(StrEnum):
    PENDING = "pending"
    BLOCKED = "blocked"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


ALLOWED_TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.BLOCKED, TaskState.RUNNING, TaskState.FAILED}),
    TaskState.BLOCKED: frozenset({TaskState.PENDING, TaskState.FAILED}),
    TaskState.RUNNING: frozenset({TaskState.COMPLETED, TaskState.PENDING, TaskState.FAILED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
}


class InvalidTaskTransition(ValueError):
    def __init__(self, current: TaskState, target: TaskState) -> None:
        super().__init__(f"cannot transition research task from {current} to {target}")
        self.current = current
        self.target = target


class RetryBudgetExhausted(RuntimeError):
    def __init__(self, task_id: UUID, max_retries: int) -> None:
        super().__init__(f"task {task_id} exhausted its {max_retries} retry budget")
        self.task_id = task_id
        self.max_retries = max_retries


class ResearchTask(BaseModel):
    """A single unit of work the planner assigns to one specialist agent."""

    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    tenant_id: str = Field(min_length=1, max_length=100)
    objective: NonEmptyText
    assigned_agent: AgentRole
    dependencies: list[UUID] = Field(default_factory=list, max_length=50)
    evidence_requirements: list[NonEmptyText] = Field(default_factory=list, max_length=50)
    source_restrictions: list[NonEmptyText] = Field(default_factory=list, max_length=50)
    state: TaskState = TaskState.PENDING
    retry_count: Annotated[int, Field(ge=0, le=100)] = 0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def task_must_not_depend_on_itself(self) -> ResearchTask:
        if self.id in self.dependencies:
            raise ValueError("a research task cannot depend on itself")
        return self

    def transition_to(self, target: TaskState) -> ResearchTask:
        if target not in ALLOWED_TASK_TRANSITIONS[self.state]:
            raise InvalidTaskTransition(self.state, target)
        return self.model_copy(update={"state": target, "updated_at": utc_now()})

    def retry(self) -> ResearchTask:
        """Return the task to the queue, refusing to exceed its retry budget."""
        if self.retry_count >= self.max_retries:
            raise RetryBudgetExhausted(self.id, self.max_retries)
        retried = self.transition_to(TaskState.PENDING)
        return retried.model_copy(update={"retry_count": self.retry_count + 1})


def resolve_execution_order(tasks: list[ResearchTask]) -> list[ResearchTask]:
    """Order tasks so every dependency runs first, rejecting cycles and dangling edges."""
    by_id = {task.id: task for task in tasks}
    for task in tasks:
        missing = [str(dep) for dep in task.dependencies if dep not in by_id]
        if missing:
            raise ValueError(f"task {task.id} depends on unknown tasks: {', '.join(missing)}")

    ordered: list[ResearchTask] = []
    placed: set[UUID] = set()
    remaining = list(tasks)
    while remaining:
        ready = [task for task in remaining if placed.issuperset(task.dependencies)]
        if not ready:
            raise ValueError("research task dependencies contain a cycle")
        ready.sort(key=lambda task: task.created_at)
        ordered.extend(ready)
        placed.update(task.id for task in ready)
        remaining = [task for task in remaining if task.id not in placed]
    return ordered
