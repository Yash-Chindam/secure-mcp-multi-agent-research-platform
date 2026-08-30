import pytest
from fastapi.testclient import TestClient

from research_platform.main import create_app


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def headers(
    *,
    tenant: str = "tenant-a",
    roles: str | None = None,
    clearance: str | None = None,
) -> dict[str, str]:
    sent = {"X-Tenant-ID": tenant, "X-Requester-ID": "requester-1"}
    if roles is not None:
        sent["X-Roles"] = roles
    if clearance is not None:
        sent["X-Clearance"] = clearance
    return sent


def names(response: object) -> set[str]:
    return {f"{item['server']}.{item['name']}" for item in response}  # type: ignore[union-attr]


@pytest.mark.integration
def test_a_requester_discovers_only_public_capabilities(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers=headers())

    assert response.status_code == 200
    discovered = names(response.json())
    assert "web-research.fetch" in discovered
    assert "filesystem.read_document" not in discovered
    assert "registry.reload_policy" not in discovered


@pytest.mark.integration
def test_raising_clearance_reveals_internal_capabilities(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers=headers(clearance="internal"))

    discovered = names(response.json())
    assert "filesystem.read_document" in discovered
    assert "postgres.run_analytical_query" not in discovered


@pytest.mark.integration
def test_administrative_capabilities_need_both_role_and_clearance(client: TestClient) -> None:
    without_role = client.get("/api/v1/capabilities", headers=headers(clearance="restricted"))
    with_role = client.get(
        "/api/v1/capabilities",
        headers=headers(roles="administrator", clearance="restricted"),
    )

    assert "registry.reload_policy" not in names(without_role.json())
    assert "registry.reload_policy" in names(with_role.json())


@pytest.mark.integration
def test_an_acting_agent_sees_only_its_own_tools(client: TestClient) -> None:
    researcher = client.get(
        "/api/v1/capabilities",
        params={"acting_agent": "researcher"},
        headers=headers(clearance="internal"),
    )
    reporter = client.get(
        "/api/v1/capabilities",
        params={"acting_agent": "reporter"},
        headers=headers(clearance="internal"),
    )

    assert "web-research.fetch" in names(researcher.json())
    assert "web-research.fetch" not in names(reporter.json())
    assert "evidence.retrieve" in names(reporter.json())


@pytest.mark.integration
def test_the_planner_sees_metadata_for_tools_it_cannot_run(client: TestClient) -> None:
    response = client.get(
        "/api/v1/capabilities",
        params={"acting_agent": "planner"},
        headers=headers(clearance="internal"),
    )

    assert "web-research.fetch" in names(response.json())


@pytest.mark.integration
def test_an_unknown_role_is_rejected_at_the_boundary(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers=headers(roles="superuser"))

    assert response.status_code == 400
    assert "unknown role" in response.json()["detail"]


@pytest.mark.integration
def test_an_unknown_clearance_is_rejected_at_the_boundary(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers=headers(clearance="top-secret"))

    assert response.status_code == 400
    assert "unknown clearance" in response.json()["detail"]


@pytest.mark.integration
def test_an_unknown_acting_agent_is_rejected(client: TestClient) -> None:
    response = client.get(
        "/api/v1/capabilities",
        params={"acting_agent": "saboteur"},
        headers=headers(),
    )

    assert response.status_code == 422


@pytest.mark.integration
def test_discovery_requires_a_tenant_header(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers={"X-Requester-ID": "requester-1"})

    assert response.status_code == 422
