"""Reading a job's findings, report and provenance manifest back through the API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from research_platform.agents.contracts import ReportSection, ResearchReport
from research_platform.application.artifacts import InMemoryArtifactStore
from research_platform.application.jobs import ResearchJobService
from research_platform.domain.models import (
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    Finding,
    ResearchJob,
)
from research_platform.main import create_app
from research_platform.settings import Settings
from research_platform.workflow.publishing import PublicationActivities

pytestmark = pytest.mark.integration

CLAIM = "The vendor charges 20 USD per seat."


def headers(tenant: str = "tenant-a") -> dict[str, str]:
    return {"X-Tenant-ID": tenant, "X-Requester-ID": "requester-1"}


@dataclass
class Published:
    client: TestClient
    jobs: ResearchJobService
    artifacts: InMemoryArtifactStore
    job: ResearchJob
    record: EvidenceRecord

    def url(self, part: str, job_id: UUID | None = None) -> str:
        return f"/api/v1/jobs/{job_id or self.job.id}/{part}"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings()))


def a_job(client: TestClient) -> ResearchJob:
    jobs: ResearchJobService = client.app.state.job_service  # type: ignore[attr-defined]
    created = client.post("/api/v1/jobs", headers=headers(), json={"question": "What is it?"})
    return jobs.get("tenant-a", UUID(created.json()["id"]))


@pytest.fixture
def published(client: TestClient) -> Published:
    """A job whose report was published the way a worker publishes it."""
    jobs: ResearchJobService = client.app.state.job_service  # type: ignore[attr-defined]
    artifacts: InMemoryArtifactStore = client.app.state.artifacts  # type: ignore[attr-defined]
    job = a_job(client)
    record = jobs.add_evidence(
        job.tenant_id,
        job.id,
        EvidenceRecordCreate(
            excerpt="Vendor pricing is 20 USD per seat.",
            source_uri="https://vendor.test/pricing",
            content_hash=f"sha256:{'a' * 64}",
            producing_task_id=uuid4(),
            tool_invocation_id=uuid4(),
        ),
    )
    jobs.record_findings(
        job.tenant_id,
        job.id,
        [
            Finding(
                claim=CLAIM,
                supporting_evidence_ids=[record.id],
                confidence=0.9,
                critic_verdict=CriticVerdict.SUPPORTED,
            )
        ],
    )
    report = ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body=f"{CLAIM[:-1]} [{record.id}].")],
    )
    asyncio.run(PublicationActivities(jobs=jobs, artifacts=artifacts).publish(job, report, []))
    return Published(client=client, jobs=jobs, artifacts=artifacts, job=job, record=record)


def test_the_published_report_is_returned_as_structured_json(published: Published) -> None:
    response = published.client.get(published.url("report"), headers=headers())

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json()["title"] == "Vendor pricing"
    assert str(published.record.id) in response.json()["sections"][0]["body"]
    assert response.headers["X-Report-SHA256"].startswith("sha256:")
    assert response.headers["X-Report-Partial"] == "false"


def test_the_report_can_be_read_as_markdown_with_numbered_sources(published: Published) -> None:
    response = published.client.get(
        published.url("report"), headers=headers(), params={"format": "markdown"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert response.text.startswith("# Vendor pricing")
    assert "The vendor charges 20 USD per seat [1]." in response.text
    assert "1. https://vendor.test/pricing" in response.text


def test_an_unknown_report_format_is_refused(published: Published) -> None:
    response = published.client.get(
        published.url("report"), headers=headers(), params={"format": "pdf"}
    )

    assert response.status_code == 422


def test_the_manifest_hash_matches_the_report_that_is_served(published: Published) -> None:
    report = published.client.get(published.url("report"), headers=headers())
    manifest = published.client.get(published.url("manifest"), headers=headers())

    assert manifest.status_code == 200
    body = manifest.json()
    assert body["report_sha256"] == report.headers["X-Report-SHA256"]
    assert body["job_id"] == str(published.job.id)
    assert body["evidence"][0]["id"] == str(published.record.id)
    assert body["evidence"][0]["cited"] is True
    assert body["findings"][0]["claim"] == CLAIM


def test_findings_are_listed_with_their_verdict_and_reviewer_status(published: Published) -> None:
    response = published.client.get(published.url("findings"), headers=headers())

    assert response.status_code == 200
    [finding] = response.json()
    assert finding["claim"] == CLAIM
    assert finding["critic_verdict"] == "supported"
    assert finding["reviewer_status"] == "not_required"
    assert finding["supporting_evidence_ids"] == [str(published.record.id)]


@pytest.mark.parametrize("part", ["report", "manifest", "findings"])
def test_another_tenant_cannot_tell_that_the_job_exists(published: Published, part: str) -> None:
    response = published.client.get(published.url(part), headers=headers("tenant-b"))

    assert response.status_code == 404
    assert response.json()["detail"] == "research job not found"


@pytest.mark.parametrize("part", ["report", "manifest", "findings"])
def test_an_unknown_job_has_nothing_to_read(published: Published, part: str) -> None:
    response = published.client.get(published.url(part, uuid4()), headers=headers())

    assert response.status_code == 404


@pytest.mark.parametrize("part", ["report", "manifest"])
def test_a_job_that_has_not_published_says_so(client: TestClient, part: str) -> None:
    job = a_job(client)

    response = client.get(f"/api/v1/jobs/{job.id}/{part}", headers=headers())

    assert response.status_code == 404
    assert response.json()["detail"] == "this research job has not published a report"


def test_a_job_with_no_findings_yet_lists_none(client: TestClient) -> None:
    job = a_job(client)

    response = client.get(f"/api/v1/jobs/{job.id}/findings", headers=headers())

    assert response.status_code == 200
    assert response.json() == []


def test_an_artifact_that_was_lost_is_reported_rather_than_served_empty(
    published: Published,
) -> None:
    published.artifacts._objects.clear()

    response = published.client.get(published.url("report"), headers=headers())

    assert response.status_code == 404
    assert response.json()["detail"] == "the published artifact is no longer stored"


def test_health_reports_where_artifacts_are_kept(client: TestClient) -> None:
    assert "in-process artifacts" in client.get("/health").json()["artifacts"]
