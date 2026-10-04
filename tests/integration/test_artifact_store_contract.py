"""One contract, two artifact stores: in-memory and a real S3-compatible object store.

Section 11 asks for isolation "at the artifact boundary". Both stores must give the same
answers, including the ones that matter most: one tenant can never read, overwrite or
name its way into another tenant's artifacts.

The object-store parameter is skipped unless ``RESEARCH_TEST_ARTIFACT_ENDPOINT`` points
at a MinIO (or other S3-compatible) server; CI provides one.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from research_platform.application.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    InMemoryArtifactStore,
    InvalidArtifactName,
    artifact_key,
    job_artifact,
)
from research_platform.composition import build_artifact_store, describe_artifacts
from research_platform.persistence.object_store import (
    build_artifact_store as connect_artifact_store,
)
from research_platform.settings import Settings

pytestmark = pytest.mark.integration

ENDPOINT = os.getenv("RESEARCH_TEST_ARTIFACT_ENDPOINT")
ACCESS_KEY = os.getenv("RESEARCH_TEST_ARTIFACT_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("RESEARCH_TEST_ARTIFACT_SECRET_KEY", "minioadmin")
SKIP_REASON = "set RESEARCH_TEST_ARTIFACT_ENDPOINT to run the object store tests"

requires_object_store = pytest.mark.skipif(not ENDPOINT, reason=SKIP_REASON)


@pytest.fixture(params=["in-memory", "object-store"])
def store(request: pytest.FixtureRequest) -> ArtifactStore:
    if request.param == "in-memory":
        return InMemoryArtifactStore()
    if not ENDPOINT:
        pytest.skip(SKIP_REASON)
    return connect_artifact_store(
        ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        bucket=f"contract-{uuid4().hex[:12]}",
    )


def test_a_stored_artifact_comes_back_byte_for_byte(store: ArtifactStore) -> None:
    name = job_artifact(uuid4(), "report.json")

    stored = store.put("acme", name, b'{"title": "Vendor pricing"}', "application/json")

    assert store.get("acme", name) == b'{"title": "Vendor pricing"}'
    assert stored.key == f"tenants/acme/{name}"
    assert stored.size == 27
    assert stored.content_type == "application/json"


def test_storing_under_the_same_name_replaces_the_artifact(store: ArtifactStore) -> None:
    """A redelivered publication lands on the same object rather than beside it."""
    name = job_artifact(uuid4(), "report.md")

    store.put("acme", name, b"first", "text/markdown")
    store.put("acme", name, b"second", "text/markdown")

    assert store.get("acme", name) == b"second"


def test_an_artifact_that_was_never_stored_is_reported_as_not_found(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFound):
        store.get("acme", job_artifact(uuid4(), "report.json"))


def test_one_tenant_cannot_read_another_tenants_artifact(store: ArtifactStore) -> None:
    name = job_artifact(uuid4(), "report.json")
    store.put("tenant-a", name, b"tenant a's report", "application/json")

    with pytest.raises(ArtifactNotFound):
        store.get("tenant-b", name)


def test_one_tenant_cannot_overwrite_another_tenants_artifact(store: ArtifactStore) -> None:
    name = job_artifact(uuid4(), "report.json")
    store.put("tenant-a", name, b"tenant a's report", "application/json")

    store.put("tenant-b", name, b"tenant b's report", "application/json")

    assert store.get("tenant-a", name) == b"tenant a's report"


@pytest.mark.parametrize(
    "name",
    [
        "../tenant-b/jobs/report.json",
        "jobs/../../tenant-b/report.json",
        "/jobs/report.json",
        "jobs//report.json",
        "jobs/report.json/",
        "jobs/..",
        "",
        "jobs/re port.json",
        "jobs\\report.json",
    ],
)
def test_a_name_that_could_leave_the_tenants_prefix_is_refused(
    store: ArtifactStore, name: str
) -> None:
    with pytest.raises(InvalidArtifactName):
        store.put("tenant-a", name, b"data", "text/plain")
    with pytest.raises(InvalidArtifactName):
        store.get("tenant-a", name)


@pytest.mark.parametrize(
    ("tenant", "prefix"),
    [
        ("acme", "tenants/acme/"),
        ("tenant/../other", "tenants/tenant%2F%2E%2E%2Fother/"),
        ("..", "tenants/%2E%2E/"),
        ("a b", "tenants/a%20b/"),
        ("a%2Fb", "tenants/a%252Fb/"),
    ],
)
def test_any_tenant_identifier_becomes_exactly_one_path_segment(tenant: str, prefix: str) -> None:
    key = artifact_key(tenant, "jobs/report.json")

    assert key == f"{prefix}jobs/report.json"
    assert key.count("/") == 3


def test_tenants_whose_identifiers_differ_only_by_encoding_never_share_a_prefix() -> None:
    assert artifact_key("a/b", "report.json") != artifact_key("a%2Fb", "report.json")


def test_an_artifact_must_belong_to_a_tenant() -> None:
    with pytest.raises(InvalidArtifactName, match="must belong to a tenant"):
        artifact_key("", "report.json")


def test_a_deployment_with_no_object_store_keeps_artifacts_in_process() -> None:
    settings = Settings(artifact_endpoint=None)

    assert isinstance(build_artifact_store(settings), InMemoryArtifactStore)
    assert "not shared between processes" in describe_artifacts(settings)


def test_an_object_store_without_credentials_is_a_startup_error() -> None:
    settings = Settings(artifact_endpoint="http://minio:9000")

    with pytest.raises(ValueError, match="RESEARCH_ARTIFACT_ACCESS_KEY"):
        build_artifact_store(settings)


def test_an_endpoint_that_is_not_a_url_is_refused() -> None:
    with pytest.raises(ValueError, match="must be an http"):
        connect_artifact_store("minio:9000", access_key="key", secret_key="secret", bucket="b")


@requires_object_store
def test_a_configured_deployment_stores_its_artifacts_in_the_object_store() -> None:
    bucket = f"configured-{uuid4().hex[:12]}"
    settings = Settings(
        artifact_endpoint=ENDPOINT,
        artifact_access_key=ACCESS_KEY,
        artifact_secret_key=SECRET_KEY,
        artifact_bucket=bucket,
    )
    name = job_artifact(uuid4(), "report.json")

    build_artifact_store(settings).put("acme", name, b"{}", "application/json")

    # A second process connecting to the same bucket reads what the first one wrote,
    # and finding the bucket already there is not an error.
    assert build_artifact_store(settings).get("acme", name) == b"{}"
    assert describe_artifacts(settings) == f"object store bucket '{bucket}'"
