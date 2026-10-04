"""Publication: the report, its manifest, and a second look at every cited source."""

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from research_platform.agents.contracts import ReportSection, ResearchReport
from research_platform.agents.provenance import hash_content
from research_platform.application.artifacts import InMemoryArtifactStore
from research_platform.application.jobs import InMemoryJobRepository, ResearchJobService
from research_platform.application.publication import DriftStatus, SourceCheck
from research_platform.domain.invocations import ErrorClass, ToolInvocation
from research_platform.domain.models import (
    AccessClass,
    CriticVerdict,
    EvidenceRecord,
    EvidenceRecordCreate,
    Finding,
    ResearchBudget,
    ResearchJob,
    ResearchJobCreate,
    TrustLevel,
    most_restrictive,
)
from research_platform.domain.tasks import AgentRole
from research_platform.mcp.catalogue import default_registry
from research_platform.mcp.gateway import CapabilityGateway, ExecutionRequest, UpstreamError
from research_platform.mcp.registry import CapabilityRegistry
from research_platform.workflow.activities import principal_for
from research_platform.workflow.publishing import (
    EVIDENCE_BUNDLE,
    MANIFEST,
    REPORT_JSON,
    REPORT_MARKDOWN,
    GatewaySourceChecker,
    PublicationActivities,
    refetch_arguments,
)

TENANT = "acme"
URL = "https://vendor.test/pricing"
CAPTURED = "Vendor pricing is 20 USD per seat."
CLAIM = "The vendor charges 20 USD per seat."


@dataclass
class Source:
    """Stands in for the MCP servers: returns whatever the source currently says."""

    text: str = CAPTURED
    error: Exception | None = None
    requests: list[ExecutionRequest] = field(default_factory=list)

    def execute(self, request: ExecutionRequest) -> str:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.text


@dataclass
class Platform:
    jobs: ResearchJobService
    gateway: CapabilityGateway
    source: Source
    artifacts: InMemoryArtifactStore
    job: ResearchJob

    @property
    def publishing(self) -> PublicationActivities:
        return PublicationActivities(
            jobs=self.jobs, artifacts=self.artifacts, sources=GatewaySourceChecker(self.gateway)
        )

    def capture(self, excerpt: str = CAPTURED) -> EvidenceRecord:
        """Fetch the page through the gateway and record what it returned as evidence."""
        task_id = uuid4()
        result = self.gateway.invoke(
            principal=principal_for(self.job, AgentRole.RESEARCHER),
            job_id=self.job.id,
            task_id=task_id,
            server="web-research",
            capability_name="fetch",
            arguments={"url": URL},
            budget=self.job.budget,
        )
        return self.record(excerpt, tool_invocation_id=result.invocation.id, task_id=task_id)

    def record(self, excerpt: str, **changes: Any) -> EvidenceRecord:
        fields: dict[str, Any] = {
            "excerpt": excerpt,
            "source_uri": URL,
            "trust_level": TrustLevel.PRIMARY,
            "content_hash": hash_content(excerpt),
            "producing_task_id": changes.pop("task_id", uuid4()),
            "tool_invocation_id": uuid4(),
        }
        return self.jobs.add_evidence(
            TENANT, self.job.id, EvidenceRecordCreate.model_validate(fields | changes)
        )

    def stored(self, filename: str) -> bytes:
        return self.artifacts.get(TENANT, f"jobs/{self.job.id}/{filename}")


@pytest.fixture
def platform() -> Platform:
    jobs = ResearchJobService(InMemoryJobRepository())
    job = jobs.create(
        TENANT,
        "requester-1",
        ResearchJobCreate(
            question="What does the vendor charge?", budget=ResearchBudget(max_tool_calls=5)
        ),
    )
    source = Source()
    gateway = CapabilityGateway(
        registry=default_registry(), executor=source, audit=jobs.record_invocation
    )
    return Platform(
        jobs=jobs, gateway=gateway, source=source, artifacts=InMemoryArtifactStore(), job=job
    )


def report_citing(*identifiers: UUID) -> ResearchReport:
    cited = " ".join(f"[{identifier}]" for identifier in identifiers)
    return ResearchReport(
        title="Vendor pricing",
        sections=[ReportSection(heading="Pricing", body=f"{CLAIM[:-1]} {cited}.")],
    )


def invocation_of(platform: Platform, record: EvidenceRecord) -> ToolInvocation | None:
    trail = platform.jobs.list_invocations(TENANT, platform.job.id)
    return next((call for call in trail if call.id == record.tool_invocation_id), None)


def check(platform: Platform, record: EvidenceRecord) -> SourceCheck:
    return GatewaySourceChecker(platform.gateway).check(
        platform.job, record, invocation_of(platform, record)
    )


# -- looking at a source again -------------------------------------------------------------


def test_a_source_that_still_says_what_was_captured_is_unchanged(platform: Platform) -> None:
    record = platform.capture()

    result = check(platform, record)

    assert result.status is DriftStatus.UNCHANGED
    assert result.evidence_id == record.id
    assert result.detail is None


def test_reflowed_whitespace_at_the_source_is_not_drift(platform: Platform) -> None:
    record = platform.capture()
    platform.source.text = "Header\n\nVendor   pricing is 20 USD\nper seat.\n\nFooter"

    assert check(platform, record).status is DriftStatus.UNCHANGED


def test_a_source_that_no_longer_contains_the_excerpt_has_drifted(platform: Platform) -> None:
    record = platform.capture()
    platform.source.text = "Vendor pricing is 25 USD per seat."

    result = check(platform, record)

    assert result.status is DriftStatus.DRIFTED
    assert result.detail == "the captured excerpt is no longer present at the source"


def test_drift_never_changes_the_evidence_that_was_captured(platform: Platform) -> None:
    """Section 12: preserve the original evidence and flag the drift."""
    record = platform.capture()
    platform.source.text = "Vendor pricing is 25 USD per seat."

    check(platform, record)

    [kept] = platform.jobs.list_evidence(TENANT, platform.job.id)
    assert kept.excerpt == CAPTURED
    assert kept.content_hash == hash_content(CAPTURED)


def test_a_source_that_cannot_be_reached_is_unavailable_not_guessed_at(platform: Platform) -> None:
    record = platform.capture()
    platform.source.error = UpstreamError("vendor.test timed out", ErrorClass.TIMEOUT)

    result = check(platform, record)

    assert result.status is DriftStatus.UNAVAILABLE
    assert result.detail is not None


def test_a_re_read_the_budget_refuses_leaves_the_source_unavailable(platform: Platform) -> None:
    record = platform.capture()
    exhausted = platform.job.model_copy(update={"budget": ResearchBudget(max_tool_calls=1)})

    result = GatewaySourceChecker(platform.gateway).check(
        exhausted, record, invocation_of(platform, record)
    )

    assert result.status is DriftStatus.UNAVAILABLE
    assert len(platform.source.requests) == 1


def test_a_re_read_is_a_governed_audited_call_like_any_other(platform: Platform) -> None:
    record = platform.capture()

    check(platform, record)

    first, second = platform.jobs.list_invocations(TENANT, platform.job.id)
    assert (second.mcp_server, second.capability) == ("web-research", "fetch")
    assert second.argument_digest == first.argument_digest
    assert second.task_id == record.producing_task_id
    assert platform.source.requests[1].principal.agent_role is AgentRole.RESEARCHER


def test_evidence_with_no_recorded_call_behind_it_is_not_re_read(platform: Platform) -> None:
    """Evidence attached by hand names a call the audit trail never saw."""
    record = platform.record(CAPTURED)

    result = check(platform, record)

    assert result.status is DriftStatus.NOT_CHECKED
    assert platform.source.requests == []


def test_a_source_whose_server_is_no_longer_deployed_is_unavailable(platform: Platform) -> None:
    record = platform.capture()
    call = invocation_of(platform, record)
    bare = CapabilityGateway(registry=CapabilityRegistry([]), executor=platform.source)

    result = GatewaySourceChecker(bare).check(platform.job, record, call)

    assert result.status is DriftStatus.UNAVAILABLE
    assert result.detail == "capability web-research.fetch is not registered"


def invocation(server: str, capability: str) -> ToolInvocation:
    return ToolInvocation(
        job_id=uuid4(),
        task_id=uuid4(),
        tenant_id=TENANT,
        mcp_server=server,
        capability=capability,
        argument_digest=f"sha256:{'b' * 64}",
        policy_version="registry-boundary/1",
        authorization_decision="allow",
    )


def evidence_from(source_uri: str) -> EvidenceRecord:
    return EvidenceRecord(
        tenant_id=TENANT,
        job_id=uuid4(),
        excerpt=CAPTURED,
        source_uri=source_uri,
        content_hash=hash_content(CAPTURED),
        producing_task_id=uuid4(),
        tool_invocation_id=uuid4(),
    )


def test_a_fetched_page_is_re_read_by_its_url() -> None:
    assert refetch_arguments(evidence_from(URL), invocation("web-research", "fetch")) == {
        "url": URL
    }


def test_a_workspace_document_is_re_read_by_its_path_within_the_tenant() -> None:
    record = evidence_from("workspace://acme/reports/q3 pricing.md")

    arguments = refetch_arguments(record, invocation("filesystem", "read_document"))

    assert arguments == {"path": "reports/q3 pricing.md"}


@pytest.mark.parametrize(
    ("server", "capability"),
    [
        ("web-research", "search"),
        ("postgres", "run_analytical_query"),
        ("github", "read_repository"),
        ("python-analysis", "run_calculation"),
    ],
)
def test_a_source_with_no_address_of_its_own_is_not_re_read(server: str, capability: str) -> None:
    """Running a search or a query again would be new research, not a re-read."""
    record = evidence_from("web-research://search/0123456789abcdef")

    assert refetch_arguments(record, invocation(server, capability)) is None


# -- publishing ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publishing_writes_the_report_the_manifest_and_the_evidence_bundle(
    platform: Platform,
) -> None:
    record = platform.capture()
    platform.jobs.record_findings(
        TENANT,
        platform.job.id,
        [
            Finding(
                claim=CLAIM,
                supporting_evidence_ids=[record.id],
                confidence=0.9,
                critic_verdict=CriticVerdict.SUPPORTED,
            )
        ],
    )
    report = report_citing(record.id)

    publication = await platform.publishing.publish(platform.job, report, [])

    assert json.loads(platform.stored(REPORT_JSON)) == report.model_dump(mode="json")
    assert "20 USD per seat [1]." in platform.stored(REPORT_MARKDOWN).decode()
    manifest = json.loads(platform.stored(MANIFEST))
    assert manifest["report_sha256"] == publication.report_sha256
    assert manifest["findings"][0]["claim"] == CLAIM
    assert manifest["evidence"][0]["drift"] == "unchanged"
    [bundled] = json.loads(platform.stored(EVIDENCE_BUNDLE))
    assert bundled["excerpt"] == CAPTURED
    assert publication.is_partial is False
    assert publication.drifted_evidence_ids == []
    assert platform.jobs.get_publication(TENANT, platform.job.id) == publication


@pytest.mark.asyncio
async def test_a_drifted_source_is_flagged_in_the_manifest_the_report_and_the_record(
    platform: Platform,
) -> None:
    record = platform.capture()
    platform.source.text = "Vendor pricing is 25 USD per seat."

    publication = await platform.publishing.publish(platform.job, report_citing(record.id), [])

    assert publication.drifted_evidence_ids == [record.id]
    manifest = json.loads(platform.stored(MANIFEST))
    assert manifest["evidence"][0]["drift"] == "drifted"
    assert manifest["evidence"][0]["content_hash"] == hash_content(CAPTURED)
    assert (
        "**source content has changed since capture**" in platform.stored(REPORT_MARKDOWN).decode()
    )
    # The bundle still carries the excerpt exactly as it was captured.
    assert json.loads(platform.stored(EVIDENCE_BUNDLE))[0]["excerpt"] == CAPTURED


@pytest.mark.asyncio
async def test_only_cited_sources_are_re_read(platform: Platform) -> None:
    cited = platform.capture()
    platform.capture("Vendor pricing")

    await platform.publishing.publish(platform.job, report_citing(cited.id), [])

    manifest = json.loads(platform.stored(MANIFEST))
    assert [entry["drift"] for entry in manifest["evidence"]] == ["unchanged", "not_checked"]
    assert len(platform.source.requests) == 3


@pytest.mark.asyncio
async def test_the_manifest_lists_the_researchs_calls_not_publications_own_re_reads(
    platform: Platform,
) -> None:
    record = platform.capture()

    await platform.publishing.publish(platform.job, report_citing(record.id), [])

    manifest = json.loads(platform.stored(MANIFEST))
    assert [call["id"] for call in manifest["tool_invocations"]] == [str(record.tool_invocation_id)]


@pytest.mark.asyncio
async def test_restricted_evidence_never_leaves_in_the_evidence_bundle(platform: Platform) -> None:
    public = platform.capture()
    platform.record("Internal margin is 62 percent.", access_class=AccessClass.RESTRICTED)

    await platform.publishing.publish(platform.job, report_citing(public.id), [])

    bundle = platform.stored(EVIDENCE_BUNDLE).decode()
    assert "62 percent" not in bundle
    assert "62 percent" not in platform.stored(MANIFEST).decode()
    assert len(json.loads(platform.stored(MANIFEST))["evidence"]) == 2


@pytest.mark.asyncio
async def test_a_report_with_shortfalls_is_published_as_partial(platform: Platform) -> None:
    record = platform.capture()
    shortfalls = ["unmet requirement: enterprise pricing"]

    publication = await platform.publishing.publish(
        platform.job, report_citing(record.id), shortfalls
    )

    assert publication.is_partial is True
    assert json.loads(platform.stored(MANIFEST))["shortfalls"] == shortfalls
    assert "**Partial result.**" in platform.stored(REPORT_MARKDOWN).decode()


@pytest.mark.asyncio
async def test_publishing_twice_lands_on_the_same_artifacts(platform: Platform) -> None:
    """Temporal may deliver the activity again; the job still has one published report."""
    record = platform.capture()
    report = report_citing(record.id)

    first = await platform.publishing.publish(platform.job, report, [])
    second = await platform.publishing.publish(platform.job, report, [])

    assert second.report_key == first.report_key
    assert second.report_sha256 == first.report_sha256
    assert platform.jobs.get_publication(TENANT, platform.job.id) == second


@pytest.mark.asyncio
async def test_publishing_without_a_source_checker_leaves_sources_unchecked(
    platform: Platform,
) -> None:
    record = platform.capture()
    publishing = PublicationActivities(jobs=platform.jobs, artifacts=platform.artifacts)

    await publishing.publish(platform.job, report_citing(record.id), [])

    assert json.loads(platform.stored(MANIFEST))["evidence"][0]["drift"] == "not_checked"
    assert len(platform.source.requests) == 1


@pytest.mark.asyncio
async def test_a_job_is_not_recorded_as_published_when_an_artifact_could_not_be_written(
    platform: Platform,
) -> None:
    class FullStore(InMemoryArtifactStore):
        def put(self, tenant_id: str, name: str, data: bytes, content_type: str) -> Any:
            if name.endswith(MANIFEST):
                raise OSError("object store is full")
            return super().put(tenant_id, name, data, content_type)

    record = platform.capture()
    publishing = PublicationActivities(jobs=platform.jobs, artifacts=FullStore())

    with pytest.raises(OSError, match="object store is full"):
        await publishing.publish(platform.job, report_citing(record.id), [])

    assert platform.jobs.get_publication(TENANT, platform.job.id) is None


@pytest.mark.asyncio
async def test_a_publication_is_classified_by_the_evidence_it_draws_on(
    platform: Platform,
) -> None:
    """The report by what it cites; the manifest by everything the job read."""
    public = platform.capture()
    internal = platform.record("Negotiated price is 14 USD.", access_class=AccessClass.INTERNAL)
    platform.record("Margin is 62 percent.", access_class=AccessClass.RESTRICTED)

    cites_public = await platform.publishing.publish(platform.job, report_citing(public.id), [])
    cites_internal = await platform.publishing.publish(
        platform.job, report_citing(public.id, internal.id), []
    )

    assert cites_public.report_access_class is AccessClass.PUBLIC
    assert cites_internal.report_access_class is AccessClass.INTERNAL
    assert cites_public.manifest_access_class is AccessClass.RESTRICTED


def test_the_most_restrictive_class_wins_and_nothing_is_public() -> None:
    assert most_restrictive([]) is AccessClass.PUBLIC
    assert most_restrictive([AccessClass.PUBLIC, AccessClass.INTERNAL]) is AccessClass.INTERNAL
    assert (
        most_restrictive([AccessClass.RESTRICTED, AccessClass.PUBLIC, AccessClass.INTERNAL])
        is AccessClass.RESTRICTED
    )
