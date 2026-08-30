from datetime import timedelta
from uuid import uuid4

import pytest

from research_platform.domain.models import utc_now
from research_platform.domain.tasks import (
    AgentRole,
    InvalidTaskTransition,
    ResearchTask,
    RetryBudgetExhausted,
    TaskState,
    resolve_execution_order,
)

JOB_ID = uuid4()


def make_task(**overrides: object) -> ResearchTask:
    defaults: dict[str, object] = {
        "job_id": JOB_ID,
        "tenant_id": "acme",
        "objective": "Collect vendor pricing pages",
        "assigned_agent": AgentRole.RESEARCHER,
    }
    return ResearchTask(**(defaults | overrides))  # type: ignore[arg-type]


def test_task_starts_pending_with_an_unspent_retry_budget() -> None:
    task = make_task()

    assert task.state is TaskState.PENDING
    assert task.retry_count == 0


def test_task_follows_the_declared_transition_table() -> None:
    running = make_task().transition_to(TaskState.RUNNING)
    completed = running.transition_to(TaskState.COMPLETED)

    assert completed.state is TaskState.COMPLETED
    assert completed.updated_at >= running.updated_at


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (TaskState.PENDING, TaskState.COMPLETED),
        (TaskState.BLOCKED, TaskState.RUNNING),
        (TaskState.COMPLETED, TaskState.RUNNING),
        (TaskState.FAILED, TaskState.PENDING),
    ],
)
def test_task_rejects_transitions_outside_the_table(start: TaskState, target: TaskState) -> None:
    task = make_task(state=start)

    with pytest.raises(InvalidTaskTransition) as error:
        task.transition_to(target)

    assert error.value.current is start
    assert error.value.target is target


def test_retry_returns_the_task_to_the_queue_and_counts_the_attempt() -> None:
    running = make_task().transition_to(TaskState.RUNNING)

    retried = running.retry()

    assert retried.state is TaskState.PENDING
    assert retried.retry_count == 1


def test_retry_refuses_to_exceed_the_budget() -> None:
    exhausted = make_task(state=TaskState.RUNNING, retry_count=3, max_retries=3)

    with pytest.raises(RetryBudgetExhausted) as error:
        exhausted.retry()

    assert error.value.max_retries == 3


def test_task_cannot_depend_on_itself() -> None:
    task_id = uuid4()

    with pytest.raises(ValueError, match="cannot depend on itself"):
        make_task(id=task_id, dependencies=[task_id])


def test_execution_order_places_every_dependency_first() -> None:
    now = utc_now()
    collect = make_task(created_at=now)
    analyze = make_task(
        assigned_agent=AgentRole.ANALYST,
        dependencies=[collect.id],
        created_at=now + timedelta(seconds=1),
    )
    report = make_task(
        assigned_agent=AgentRole.REPORTER,
        dependencies=[analyze.id],
        created_at=now + timedelta(seconds=2),
    )

    ordered = resolve_execution_order([report, analyze, collect])

    assert [task.id for task in ordered] == [collect.id, analyze.id, report.id]


def test_execution_order_rejects_a_dependency_cycle() -> None:
    first_id, second_id = uuid4(), uuid4()
    first = make_task(id=first_id, dependencies=[second_id])
    second = make_task(id=second_id, dependencies=[first_id])

    with pytest.raises(ValueError, match="cycle"):
        resolve_execution_order([first, second])


def test_execution_order_rejects_an_edge_to_an_unknown_task() -> None:
    orphan = make_task(dependencies=[uuid4()])

    with pytest.raises(ValueError, match="unknown tasks"):
        resolve_execution_order([orphan])
