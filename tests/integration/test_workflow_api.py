"""The HTTP boundary's side of section 9: a created job starts work, a reviewer decides.

A substitute ``WorkflowStarter`` stands in for Temporal here, so these tests are about
the rules the API enforces - what is recorded, who may review, what happens when the
engine is down. ``test_job_lifecycle_end_to_end.py`` runs the same boundary against a
real Temporal test server.
"""

from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from research_platform.application.jobs import ResearchJobService
from research_platform.application.workflows import (
    WorkflowCheckpoint,
    WorkflowNotRunning,
    WorkflowUnavailable,
)
from research_platform.domain.models import JobStatus, ResearchJob
from research_platform.main import create_app
from research_platform.settings import Settings
from research_platform.workflow.orchestration import ReviewerDecision

pytestmark = pytest.mark.integration


class RecordingStarter:
    def __init__(self) -> None:
        self.started: list[ResearchJob] = []
        self.decisions: list[tuple[UUID, ReviewerDecision]] = []
        self.start_error: Exception | None = None
        self.signal_error: Exception | None = None

    async def start(self, job: ResearchJob) -> WorkflowCheckpoint:
        if self.start_error is not None:
            raise self.start_error
        self.started.append(job)
        return WorkflowCheckpoint(workflow_id=f"research-job-{job.id}", workflow_run_id="run-1")

    async def submit_reviewer_decision(self, job: ResearchJob, decision: ReviewerDecision) -> None:
        if self.signal_error is not None:
            raise self.signal_error
        self.decisions.append((job.id, decision))


@pytest.fixture
def starter() -> RecordingStarter:
    return RecordingStarter()


@pytest.fixture
def client(starter: RecordingStarter) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(), workflows=starter)) as test_client:
        yield test_client


def as_user(requester: str = "requester-1", roles: str = "requester") -> dict[str, str]:
    return {"X-Tenant-ID": "tenant-a", "X-Requester-ID": requester, "X-Roles": roles}


REVIEWER = as_user("reviewer-1", "reviewer")


def create_job(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/jobs", headers=as_user(), json={"question": "Which is cheaper?"}
    )
    assert response.status_code == 201
    created: dict[str, str] = response.json()
    return created


def awaiting_review(client: TestClient) -> str:
    """Create a job and walk it to the status a suspended workflow would leave it in."""
    job = create_job(client)
    service: ResearchJobService = client.app.state.job_service  # type: ignore[attr-defined]
    for status in (
        JobStatus.PLANNING,
        JobStatus.RESEARCHING,
        JobStatus.ANALYZING,
        JobStatus.REVIEW_REQUIRED,
    ):
        service.transition("tenant-a", UUID(job["id"]), status)
    return job["id"]


def review(
    client: TestClient, job_id: str, headers: dict[str, str], decision: str = "approve"
) -> Response:
    return client.post(
        f"/api/v1/jobs/{job_id}/review", headers=headers, json={"decision": decision}
    )


def test_creating_a_job_starts_its_workflow_and_records_the_checkpoint(
    client: TestClient, starter: RecordingStarter
) -> None:
    job = create_job(client)

    assert [str(started.id) for started in starter.started] == [job["id"]]
    assert job["workflow_id"] == f"research-job-{job['id']}"
    assert job["workflow_run_id"] == "run-1"
    stored = client.get(f"/api/v1/jobs/{job['id']}", headers=as_user()).json()
    assert stored["workflow_id"] == job["workflow_id"]


def test_a_job_that_cannot_be_started_is_kept_on_record_as_failed(
    client: TestClient, starter: RecordingStarter
) -> None:
    starter.start_error = WorkflowUnavailable("connection refused")

    response = client.post(
        "/api/v1/jobs", headers=as_user(), json={"question": "Which is cheaper?"}
    )

    assert response.status_code == 503
    assert "could not be started" in response.json()["detail"]
    [recorded] = client.get("/api/v1/jobs", headers=as_user()).json()
    assert recorded["status"] == "failed"


def test_with_workflows_disabled_a_job_is_recorded_and_left_unstarted() -> None:
    with TestClient(create_app(Settings())) as client:
        job = create_job(client)
        health = client.get("/health").json()

    assert job["workflow_id"] is None
    assert job["status"] == "created"
    assert health["workflows"].startswith("disabled")


def test_health_reports_that_new_jobs_start_a_workflow(client: TestClient) -> None:
    assert client.get("/health").json()["workflows"] == "each new job starts a durable workflow"


def test_a_reviewers_decision_is_delivered_to_the_waiting_workflow(
    client: TestClient, starter: RecordingStarter
) -> None:
    job_id = awaiting_review(client)

    response = review(client, job_id, REVIEWER, "request_more_research")

    assert response.status_code == 202
    assert starter.decisions == [(UUID(job_id), ReviewerDecision.REQUEST_MORE_RESEARCH)]


def test_a_requester_without_the_reviewer_role_cannot_decide(
    client: TestClient, starter: RecordingStarter
) -> None:
    job_id = awaiting_review(client)

    response = review(client, job_id, as_user("someone-else", "requester"))

    assert response.status_code == 403
    assert starter.decisions == []


def test_a_reviewer_cannot_approve_a_job_they_requested_themselves(
    client: TestClient, starter: RecordingStarter
) -> None:
    job_id = awaiting_review(client)

    response = review(client, job_id, as_user("requester-1", "requester,reviewer"))

    assert response.status_code == 403
    assert "their own" in response.json()["detail"]
    assert starter.decisions == []


def test_an_administrator_may_decide_their_own_job(
    client: TestClient, starter: RecordingStarter
) -> None:
    job_id = awaiting_review(client)

    response = review(client, job_id, as_user("requester-1", "administrator"))

    assert response.status_code == 202
    assert len(starter.decisions) == 1


def test_a_job_that_is_not_awaiting_review_cannot_be_decided(
    client: TestClient, starter: RecordingStarter
) -> None:
    job = create_job(client)

    response = review(client, job["id"], REVIEWER)

    assert response.status_code == 409
    assert starter.decisions == []


def test_a_reviewer_from_another_tenant_cannot_see_the_job(client: TestClient) -> None:
    job_id = awaiting_review(client)
    outsider = {"X-Tenant-ID": "tenant-b", "X-Requester-ID": "reviewer-9", "X-Roles": "reviewer"}

    assert review(client, job_id, outsider).status_code == 404


def test_an_unknown_decision_is_rejected_at_the_boundary(client: TestClient) -> None:
    job_id = awaiting_review(client)

    assert review(client, job_id, REVIEWER, "maybe").status_code == 422


def test_a_decision_for_a_workflow_that_already_ended_is_a_conflict(
    client: TestClient, starter: RecordingStarter
) -> None:
    job_id = awaiting_review(client)
    starter.signal_error = WorkflowNotRunning("no running workflow")

    assert review(client, job_id, REVIEWER).status_code == 409


def test_a_decision_that_cannot_reach_the_engine_is_reported_unavailable(
    client: TestClient, starter: RecordingStarter
) -> None:
    job_id = awaiting_review(client)
    starter.signal_error = WorkflowUnavailable("connection refused")

    assert review(client, job_id, REVIEWER).status_code == 503


def test_a_review_cannot_be_decided_while_workflows_are_disabled() -> None:
    with TestClient(create_app(Settings())) as client:
        job_id = awaiting_review(client)

        response = review(client, job_id, REVIEWER)

    assert response.status_code == 503


def test_enabling_workflows_does_not_require_the_engine_to_be_up_at_boot() -> None:
    """The connection is opened on first use, so the API starts and serves reads alone."""
    settings = Settings(workflows_enabled=True, temporal_target_host="127.0.0.1:1")

    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        listed = client.get("/api/v1/jobs", headers=as_user())

    assert health.json()["workflows"] == "each new job starts a durable workflow"
    assert listed.status_code == 200
