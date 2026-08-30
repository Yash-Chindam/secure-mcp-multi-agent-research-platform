from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from research_platform.main import create_app


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def headers(tenant: str = "tenant-a") -> dict[str, str]:
    return {"X-Tenant-ID": tenant, "X-Requester-ID": "requester-1"}


@pytest.mark.integration
def test_create_list_and_transition_job(client: TestClient) -> None:
    created = client.post(
        "/api/v1/jobs",
        headers=headers(),
        json={"question": "Compare the release claims."},
    )

    assert created.status_code == 201
    job = created.json()
    assert job["status"] == "created"

    listed = client.get("/api/v1/jobs", headers=headers())
    assert [item["id"] for item in listed.json()] == [job["id"]]

    transitioned = client.post(
        f"/api/v1/jobs/{job['id']}/transitions?target=planning", headers=headers()
    )
    assert transitioned.status_code == 200
    assert transitioned.json()["status"] == "planning"


@pytest.mark.integration
def test_cross_tenant_job_access_looks_like_not_found(client: TestClient) -> None:
    created = client.post("/api/v1/jobs", headers=headers(), json={"question": "Question"})

    response = client.get(f"/api/v1/jobs/{created.json()['id']}", headers=headers("tenant-b"))

    assert response.status_code == 404


@pytest.mark.integration
def test_request_identity_headers_are_required(client: TestClient) -> None:
    response = client.post("/api/v1/jobs", json={"question": "Question"})

    assert response.status_code == 422


@pytest.mark.integration
def test_evidence_hash_is_validated_at_api_boundary(client: TestClient) -> None:
    created = client.post("/api/v1/jobs", headers=headers(), json={"question": "Question"})

    response = client.post(
        f"/api/v1/jobs/{created.json()['id']}/evidence",
        headers=headers(),
        json={
            "excerpt": "Source excerpt",
            "source_uri": "https://example.com/source",
            "content_hash": "not-a-hash",
            "producing_task_id": str(uuid4()),
            "tool_invocation_id": str(uuid4()),
        },
    )

    assert response.status_code == 422
