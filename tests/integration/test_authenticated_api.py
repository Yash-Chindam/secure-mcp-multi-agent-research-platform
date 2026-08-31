"""The API boundary when a token issuer is configured."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from research_platform.auth import TokenVerifier
from research_platform.main import create_app
from research_platform.settings import Settings

ISSUER = "https://keycloak.test/realms/research"
AUDIENCE = "research-platform"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StaticKeyResolver:
    def key_for(self, token: str) -> Any:
        return SIGNING_KEY.public_key()


def access_token(**overrides: object) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-1",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "tenant_id": "acme",
        "realm_access": {"roles": ["requester"]},
    }
    payload.update(overrides)
    return jwt.encode(payload, SIGNING_KEY, algorithm="RS256")


def bearer(**overrides: object) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token(**overrides)}"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    app.state.token_verifier = TokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        keys=StaticKeyResolver(),  # type: ignore[arg-type]
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.integration
def test_a_verified_token_identifies_the_caller(client: TestClient) -> None:
    created = client.post(
        "/api/v1/jobs",
        headers=bearer(),
        json={"question": "Compare the vendor release claims."},
    )

    assert created.status_code == 201
    job = created.json()
    assert job["tenant_id"] == "acme"
    assert job["requester_id"] == "user-1"


@pytest.mark.integration
def test_a_request_without_a_token_is_refused(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.integration
def test_an_expired_token_is_refused(client: TestClient) -> None:
    past = datetime.now(UTC) - timedelta(hours=2)

    response = client.get(
        "/api/v1/capabilities",
        headers=bearer(iat=past, exp=past + timedelta(minutes=5)),
    )

    assert response.status_code == 401
    assert "expired" in response.json()["detail"]


@pytest.mark.integration
def test_a_token_for_another_audience_is_refused(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers=bearer(aud="another-service"))

    assert response.status_code == 401
    assert "another audience" in response.json()["detail"]


@pytest.mark.integration
def test_the_development_headers_are_ignored_once_tokens_are_verified(
    client: TestClient,
) -> None:
    """A configured deployment cannot be downgraded to header identity."""
    response = client.get(
        "/api/v1/capabilities",
        headers={
            "X-Tenant-ID": "globex",
            "X-Requester-ID": "attacker",
            "X-Roles": "administrator",
            "X-Clearance": "restricted",
        },
    )

    assert response.status_code == 401


@pytest.mark.integration
def test_headers_cannot_escalate_a_verified_token(client: TestClient) -> None:
    response = client.get(
        "/api/v1/capabilities",
        headers={
            **bearer(),
            "X-Roles": "administrator",
            "X-Clearance": "restricted",
        },
    )

    assert response.status_code == 200
    discovered = {f"{item['server']}.{item['name']}" for item in response.json()}
    assert "registry.reload_policy" not in discovered
    assert "postgres.run_analytical_query" not in discovered


@pytest.mark.integration
def test_the_token_clearance_governs_discovery(client: TestClient) -> None:
    response = client.get("/api/v1/capabilities", headers=bearer(clearance="internal"))

    discovered = {f"{item['server']}.{item['name']}" for item in response.json()}
    assert "filesystem.read_document" in discovered


@pytest.mark.integration
def test_a_token_role_the_platform_does_not_know_grants_nothing(client: TestClient) -> None:
    response = client.get(
        "/api/v1/capabilities",
        headers=bearer(realm_access={"roles": ["superuser"]}),
    )

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.integration
def test_one_tenant_cannot_read_another_tenant_jobs(client: TestClient) -> None:
    created = client.post(
        "/api/v1/jobs",
        headers=bearer(),
        json={"question": "Acme question."},
    )
    job_id = created.json()["id"]

    other = client.get(
        f"/api/v1/jobs/{job_id}",
        headers=bearer(tenant_id="globex", sub="user-2"),
    )

    assert other.status_code == 404


@pytest.mark.integration
def test_health_reports_that_tokens_are_verified(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert "registry boundary" in body["authorization"]
