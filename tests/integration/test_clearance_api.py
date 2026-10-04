"""A job's agents act with its requester's clearance, and readers see what theirs covers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from research_platform.agents.contracts import ReportSection, ResearchReport
from research_platform.agents.provenance import hash_content
from research_platform.application.jobs import ResearchJobService
from research_platform.application.workflows import WorkflowCheckpoint
from research_platform.domain.invocations import (
    AuthorizationDecision,
    InvocationOutcome,
    ToolInvocation,
)
from research_platform.domain.models import (
    AccessClass,
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

TENANT = "tenant-a"
PUBLIC_CLAIM = "The vendor charges 20 USD per seat."
INTERNAL_CLAIM = "The negotiated price is 14 USD per seat."


def headers(clearance: str | None = None, requester: str = "requester-1") -> dict[str, str]:
    sent = {"X-Tenant-ID": TENANT, "X-Requester-ID": requester}
    if clearance is not None:
        sent["X-Clearance"] = clearance
    return sent


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings()))


def jobs_of(client: TestClient) -> ResearchJobService:
    service: ResearchJobService = client.app.state.job_service  # type: ignore[attr-defined]
    return service


def create_job(client: TestClient, clearance: str | None = None) -> dict[str, Any]:
    response = client.post(
        "/api/v1/jobs", headers=headers(clearance), json={"question": "What does it cost?"}
    )
    assert response.status_code == 201
    created: dict[str, Any] = response.json()
    return created


# -- the job carries its requester's clearance ------------------------------------------------


def test_a_job_is_given_the_clearance_of_the_requester_who_created_it(client: TestClient) -> None:
    assert create_job(client, "internal")["clearance"] == "internal"


def test_a_requester_who_states_no_clearance_creates_a_public_job(client: TestClient) -> None:
    assert create_job(client)["clearance"] == "public"


def test_a_job_keeps_its_clearance_whoever_reads_it_later(client: TestClient) -> None:
    created = create_job(client, "internal")

    read = client.get(f"/api/v1/jobs/{created['id']}", headers=headers("restricted", "admin-1"))

    assert read.json()["clearance"] == "internal"


def test_a_requester_cannot_name_a_clearance_in_the_job_itself(client: TestClient) -> None:
    """The clearance comes from the caller's identity, never from the request body."""
    response = client.post(
        "/api/v1/jobs",
        headers=headers("public"),
        json={"question": "What does it cost?", "clearance": "restricted"},
    )

    assert response.json()["clearance"] == "public"


# -- reading evidence, findings and the audit trail ---------------------------------------------


@dataclass
class Researched:
    client: TestClient
    job: ResearchJob
    public: EvidenceRecord
    internal: EvidenceRecord
    restricted: EvidenceRecord

    def get(self, part: str, clearance: str | None = None, **params: str) -> Any:
        return self.client.get(
            f"/api/v1/jobs/{self.job.id}/{part}", headers=headers(clearance), params=params
        )


def evidence(excerpt: str, access_class: AccessClass) -> EvidenceRecordCreate:
    return EvidenceRecordCreate(
        excerpt=excerpt,
        source_uri="https://vendor.test/pricing",
        access_class=access_class,
        content_hash=hash_content(excerpt),
        producing_task_id=uuid4(),
        tool_invocation_id=uuid4(),
    )


def invocation(job: ResearchJob, server: str, capability: str) -> ToolInvocation:
    return ToolInvocation(
        job_id=job.id,
        task_id=uuid4(),
        tenant_id=job.tenant_id,
        mcp_server=server,
        capability=capability,
        sanitized_arguments={"path": "contracts/vendor.md"},
        argument_digest=f"sha256:{'b' * 64}",
        policy_version="registry-boundary/1",
        authorization_decision=AuthorizationDecision.ALLOW,
        outcome=InvocationOutcome.SUCCEEDED,
    )


@pytest.fixture
def researched(client: TestClient) -> Researched:
    """A job created by a restricted-clearance requester that collected all three classes."""
    jobs = jobs_of(client)
    job = jobs.get(TENANT, UUID(create_job(client, "restricted")["id"]))
    public = jobs.add_evidence(
        TENANT, job.id, evidence("List price is 20 USD.", AccessClass.PUBLIC)
    )
    internal = jobs.add_evidence(
        TENANT, job.id, evidence("Negotiated price is 14 USD.", AccessClass.INTERNAL)
    )
    restricted = jobs.add_evidence(
        TENANT, job.id, evidence("Margin is 62 percent.", AccessClass.RESTRICTED)
    )
    jobs.record_findings(
        TENANT,
        job.id,
        [
            Finding(
                claim=PUBLIC_CLAIM,
                supporting_evidence_ids=[public.id],
                confidence=0.9,
                critic_verdict=CriticVerdict.SUPPORTED,
            ),
            Finding(
                claim=INTERNAL_CLAIM,
                supporting_evidence_ids=[public.id, internal.id],
                confidence=0.8,
                critic_verdict=CriticVerdict.SUPPORTED,
            ),
            Finding(
                claim="The list price is contradicted by the margin data.",
                supporting_evidence_ids=[public.id],
                contradicting_evidence_ids=[restricted.id],
                confidence=0.4,
                critic_verdict=CriticVerdict.CONTRADICTED,
            ),
        ],
    )
    jobs.record_invocation(invocation(job, "web-research", "fetch"))
    jobs.record_invocation(invocation(job, "filesystem", "read_document"))
    jobs.record_invocation(invocation(job, "postgres", "run_analytical_query"))
    jobs.record_invocation(invocation(job, "retired-server", "old_tool"))
    return Researched(
        client=client, job=job, public=public, internal=internal, restricted=restricted
    )


@pytest.mark.parametrize(
    ("clearance", "visible", "withheld"),
    [(None, 1, 2), ("public", 1, 2), ("internal", 2, 1), ("restricted", 3, 0)],
)
def test_evidence_above_the_readers_clearance_is_withheld_and_counted(
    researched: Researched, clearance: str | None, visible: int, withheld: int
) -> None:
    response = researched.get("evidence", clearance)

    assert response.status_code == 200
    assert len(response.json()) == visible
    assert response.headers["X-Withheld-Count"] == str(withheld)
    classes = {record["access_class"] for record in response.json()}
    assert classes == set(["public", "internal", "restricted"][:visible])


def test_withheld_evidence_text_appears_nowhere_in_the_response(researched: Researched) -> None:
    response = researched.get("evidence", "public")

    assert "14 USD" not in response.text
    assert "62 percent" not in response.text


@pytest.mark.parametrize(
    ("clearance", "claims"),
    [("public", 1), ("internal", 2), ("restricted", 3)],
)
def test_a_finding_resting_on_evidence_the_reader_cannot_see_is_withheld(
    researched: Researched, clearance: str, claims: int
) -> None:
    """Including one contradicted by evidence the reader cannot see."""
    response = researched.get("findings", clearance)

    assert len(response.json()) == claims
    assert response.headers["X-Withheld-Count"] == str(3 - claims)
    if clearance == "public":
        assert [finding["claim"] for finding in response.json()] == [PUBLIC_CLAIM]


@pytest.mark.parametrize(
    ("clearance", "servers"),
    [
        ("public", ["web-research"]),
        ("internal", ["web-research", "filesystem"]),
        ("restricted", ["web-research", "filesystem", "postgres", "retired-server"]),
    ],
)
def test_the_audit_trail_withholds_calls_into_data_the_reader_is_not_cleared_for(
    researched: Researched, clearance: str, servers: list[str]
) -> None:
    """A call to a capability the registry no longer knows is treated as restricted."""
    response = researched.get("invocations", clearance)

    assert [call["mcp_server"] for call in response.json()] == servers
    assert response.headers["X-Withheld-Count"] == str(4 - len(servers))


# -- reading the published report ---------------------------------------------------------------


def publish(researched: Researched, *cited: EvidenceRecord) -> None:
    citations = " ".join(f"[{record.id}]" for record in cited)
    report = ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body=f"Seats are priced {citations}.")],
    )
    publishing = PublicationActivities(
        jobs=jobs_of(researched.client),
        artifacts=researched.client.app.state.artifacts,  # type: ignore[attr-defined]
    )
    asyncio.run(publishing.publish(researched.job, report, []))


def test_a_report_citing_only_public_evidence_is_readable_by_anyone_in_the_tenant(
    researched: Researched,
) -> None:
    publish(researched, researched.public)

    assert researched.get("report").status_code == 200
    assert researched.get("report", format="markdown").status_code == 200


def test_a_report_citing_internal_evidence_needs_internal_clearance(
    researched: Researched,
) -> None:
    publish(researched, researched.public, researched.internal)

    refused = researched.get("report", "public")
    refused_markdown = researched.get("report", "public", format="markdown")
    allowed = researched.get("report", "internal")

    assert refused.status_code == 403
    assert refused.json()["detail"] == (
        "this report draws on internal sources, which your public clearance does not cover"
    )
    assert refused_markdown.status_code == 403
    assert allowed.status_code == 200


def test_the_manifest_needs_the_clearance_of_every_source_the_job_read(
    researched: Researched,
) -> None:
    """It lists uncited sources too, so a public report can have a restricted manifest."""
    publish(researched, researched.public)

    assert researched.get("report", "public").status_code == 200
    assert researched.get("manifest", "public").status_code == 403
    assert researched.get("manifest", "internal").status_code == 403
    assert researched.get("manifest", "restricted").status_code == 200


# -- evidence attached by hand --------------------------------------------------------------------


def manual(excerpt: str = "Source excerpt", **changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "excerpt": excerpt,
        "source_uri": "https://example.com/source",
        "content_hash": hash_content(excerpt),
        "producing_task_id": str(uuid4()),
        "tool_invocation_id": str(uuid4()),
    }
    return body | changes


def test_evidence_attached_by_hand_is_always_recorded_as_unverified(client: TestClient) -> None:
    job = create_job(client)

    response = client.post(
        f"/api/v1/jobs/{job['id']}/evidence",
        headers=headers(),
        json=manual(trust_level="primary"),
    )

    assert response.status_code == 201
    assert response.json()["trust_level"] == "unverified"


def test_a_hash_that_is_not_the_hash_of_the_excerpt_is_refused(client: TestClient) -> None:
    job = create_job(client)

    response = client.post(
        f"/api/v1/jobs/{job['id']}/evidence",
        headers=headers(),
        json=manual(content_hash=f"sha256:{'0' * 64}"),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "content_hash is not the hash of the excerpt"
    assert jobs_of(client).list_evidence(TENANT, UUID(job["id"])) == []


def test_evidence_cannot_be_classified_above_the_callers_own_clearance(
    client: TestClient,
) -> None:
    job = create_job(client)

    refused = client.post(
        f"/api/v1/jobs/{job['id']}/evidence",
        headers=headers("public"),
        json=manual(access_class="internal"),
    )
    allowed = client.post(
        f"/api/v1/jobs/{job['id']}/evidence",
        headers=headers("internal"),
        json=manual(access_class="internal"),
    )

    assert refused.status_code == 403
    assert allowed.status_code == 201


# -- moving a job by hand -------------------------------------------------------------------------


def test_a_job_no_workflow_runs_can_still_be_moved_by_hand(client: TestClient) -> None:
    job = create_job(client)

    response = client.post(
        f"/api/v1/jobs/{job['id']}/transitions", headers=headers(), params={"target": "planning"}
    )

    assert response.status_code == 200


def test_a_job_a_workflow_runs_cannot_be_moved_by_hand() -> None:
    class Started:
        async def start(self, job: ResearchJob) -> WorkflowCheckpoint:
            return WorkflowCheckpoint(workflow_id=f"research-job-{job.id}", workflow_run_id="run-1")

        async def submit_reviewer_decision(self, *_args: object) -> None:
            raise AssertionError("no decision is submitted in this test")

    client = TestClient(create_app(Settings(), workflows=Started()))
    job = create_job(client)

    response = client.post(
        f"/api/v1/jobs/{job['id']}/transitions", headers=headers(), params={"target": "completed"}
    )

    assert response.status_code == 409
    assert "run by a workflow" in response.json()["detail"]
    assert jobs_of(client).get(TENANT, UUID(job["id"])).status.value == "created"
