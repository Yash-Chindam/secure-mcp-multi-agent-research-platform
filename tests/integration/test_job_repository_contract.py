"""One behavioural contract, run against both ``JobRepository`` implementations.

The in-process repository is what the unit tests and a single-process demo use; the
PostgreSQL one is the system of record a deployment runs on. They are only
interchangeable if they behave identically, so the rules that matter - tenant scoping,
a missing job, the audit trail outliving the job it names - are asserted once here and
executed twice.

The PostgreSQL parameter is skipped unless ``RESEARCH_TEST_DATABASE_URL`` points at a
database the test may create tables in and truncate.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from psycopg import Connection
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool

from research_platform.application.jobs import (
    InMemoryJobRepository,
    JobNotFoundError,
    JobRepository,
)
from research_platform.application.publication import ReportPublication
from research_platform.domain.invocations import (
    AuthorizationDecision,
    ErrorClass,
    InvocationOutcome,
    ToolInvocation,
)
from research_platform.domain.models import (
    AccessClass,
    CriticVerdict,
    EvidenceRecord,
    FindingRecord,
    JobStatus,
    JobUsage,
    ResearchBudget,
    ResearchJob,
    ReviewerStatus,
    TrustLevel,
)
from research_platform.persistence.postgres import (
    PostgresJobRepository,
    create_schema,
    database_isolation_is_enforced,
)

pytestmark = pytest.mark.integration

DATABASE_URL = os.getenv("RESEARCH_TEST_DATABASE_URL")

requires_postgres = pytest.mark.skipif(
    not DATABASE_URL,
    reason="set RESEARCH_TEST_DATABASE_URL to run the PostgreSQL repository tests",
)


APPLICATION_ROLE = "research_app_test"

_GRANT_APPLICATION_ROLE = (
    f"""
    DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APPLICATION_ROLE}') THEN
            CREATE ROLE {APPLICATION_ROLE} LOGIN PASSWORD '{APPLICATION_ROLE}';
        END IF;
    END $$
    """,
    f"GRANT USAGE ON SCHEMA public TO {APPLICATION_ROLE}",
    "GRANT SELECT, INSERT, UPDATE ON research_jobs, evidence_records, tool_invocations, "
    f"report_publications TO {APPLICATION_ROLE}",
    f"GRANT SELECT, INSERT, DELETE ON findings TO {APPLICATION_ROLE}",
)


@pytest.fixture
def pool() -> Iterator[ConnectionPool[Connection[Any]]]:
    """A pool connected as the owner, used to apply the schema and reset between tests."""
    assert DATABASE_URL is not None
    opened: ConnectionPool[Connection[Any]] = ConnectionPool(DATABASE_URL, min_size=1, open=True)
    opened.wait(timeout=30)
    create_schema(opened)
    with opened.connection() as connection:
        connection.execute(
            "TRUNCATE research_jobs, evidence_records, tool_invocations, findings, "
            "report_publications"
        )
        for statement in _GRANT_APPLICATION_ROLE:
            connection.execute(statement)
    yield opened
    opened.close()


@pytest.fixture
def restricted_pool(
    pool: ConnectionPool[Connection[Any]],
) -> Iterator[ConnectionPool[Connection[Any]]]:
    """A pool connected the way a deployment must: a role that cannot bypass policy.

    The row-level security tests below are only meaningful through a role like this. The
    superuser the test database is usually created with is exempt from every policy, so
    running them as that role would prove nothing - which is the footgun
    ``database_isolation_is_enforced`` exists to report.
    """
    assert DATABASE_URL is not None
    settings = conninfo_to_dict(DATABASE_URL)
    settings["user"] = APPLICATION_ROLE
    settings["password"] = APPLICATION_ROLE
    opened: ConnectionPool[Connection[Any]] = ConnectionPool(
        make_conninfo(**settings), min_size=1, open=True
    )
    opened.wait(timeout=30)
    yield opened
    opened.close()


@pytest.fixture(params=["in-memory", "postgresql"])
def repository(request: pytest.FixtureRequest) -> JobRepository:
    if request.param == "in-memory":
        return InMemoryJobRepository()
    if not DATABASE_URL:
        pytest.skip("set RESEARCH_TEST_DATABASE_URL to run the PostgreSQL repository tests")
    return PostgresJobRepository(request.getfixturevalue("pool"))


def a_job(
    tenant_id: str = "tenant-a", question: str = "Which vendor priced lowest?"
) -> ResearchJob:
    return ResearchJob(
        tenant_id=tenant_id,
        requester_id="requester@example.com",
        question=question,
        constraints=["Published after 2024"],
        source_requirements=["At least two primary sources"],
        budget=ResearchBudget(max_tool_calls=12, max_runtime_seconds=600, max_cost_usd=2.5),
    )


def an_evidence_record(job: ResearchJob) -> EvidenceRecord:
    return EvidenceRecord(
        tenant_id=job.tenant_id,
        job_id=job.id,
        excerpt="The list price fell to 4 USD per million tokens.",
        source_uri="https://example.com/pricing",
        title="Pricing update",
        author="Example Research",
        trust_level=TrustLevel.PRIMARY,
        access_class=AccessClass.PUBLIC,
        content_hash=f"sha256:{'a' * 64}",
        producing_task_id=uuid4(),
        tool_invocation_id=uuid4(),
    )


def an_invocation(tenant_id: str, job_id: Any) -> ToolInvocation:
    return ToolInvocation(
        job_id=job_id,
        task_id=uuid4(),
        tenant_id=tenant_id,
        mcp_server="web-research",
        capability="fetch",
        sanitized_arguments={"url": "https://example.com/pricing"},
        argument_digest=f"sha256:{'b' * 64}",
        policy_version="registry-boundary/1",
        authorization_decision=AuthorizationDecision.ALLOW,
        outcome=InvocationOutcome.SUCCEEDED,
        error_class=ErrorClass.NONE,
        duration_ms=118,
    )


def test_a_stored_job_comes_back_with_every_field_it_was_given(repository: JobRepository) -> None:
    job = repository.add(a_job())

    loaded = repository.get(job.tenant_id, job.id)

    assert loaded.question == job.question
    assert loaded.constraints == job.constraints
    assert loaded.source_requirements == job.source_requirements
    assert loaded.budget == job.budget
    assert loaded.status is JobStatus.CREATED
    assert loaded.created_at == job.created_at


@pytest.mark.parametrize("clearance", list(AccessClass))
def test_a_job_keeps_the_clearance_it_was_created_with(
    repository: JobRepository, clearance: AccessClass
) -> None:
    job = repository.add(a_job().model_copy(update={"clearance": clearance}))

    assert repository.get(job.tenant_id, job.id).clearance is clearance
    assert repository.list(job.tenant_id)[0].clearance is clearance


def test_a_status_change_never_alters_a_jobs_clearance(repository: JobRepository) -> None:
    job = repository.add(a_job().model_copy(update={"clearance": AccessClass.INTERNAL}))

    repository.update(job.transition_to(JobStatus.PLANNING))

    assert repository.get(job.tenant_id, job.id).clearance is AccessClass.INTERNAL


def test_a_job_starts_with_nothing_spent_and_keeps_what_a_transition_records(
    repository: JobRepository,
) -> None:
    job = repository.add(a_job())
    spent = JobUsage(
        tool_calls=3,
        agent_calls=4,
        schema_corrections=1,
        prompt_tokens=1_200,
        completion_tokens=300,
        cost_usd=0.0125,
        active_seconds=41.5,
    )

    repository.update(job.transition_to(JobStatus.PLANNING, usage=spent))

    assert job.usage == JobUsage()
    assert repository.get(job.tenant_id, job.id).usage == spent
    assert repository.list(job.tenant_id)[0].usage == spent


def test_a_token_budget_is_stored_with_the_job(repository: JobRepository) -> None:
    job = repository.add(a_job().model_copy(update={"budget": ResearchBudget(max_tokens=250_000)}))

    assert repository.get(job.tenant_id, job.id).budget.max_tokens == 250_000


def test_a_job_is_never_readable_by_another_tenant(repository: JobRepository) -> None:
    job = repository.add(a_job(tenant_id="tenant-a"))

    with pytest.raises(JobNotFoundError):
        repository.get("tenant-b", job.id)


def test_listing_returns_only_the_tenants_own_jobs_newest_first(
    repository: JobRepository,
) -> None:
    first = repository.add(a_job(tenant_id="tenant-a", question="Asked first"))
    second = repository.add(a_job(tenant_id="tenant-a", question="Asked second"))
    repository.add(a_job(tenant_id="tenant-b", question="Another tenant"))

    listed = repository.list("tenant-a")

    assert [job.id for job in listed] == [second.id, first.id]


def test_a_status_transition_and_a_workflow_checkpoint_both_persist(
    repository: JobRepository,
) -> None:
    job = repository.add(a_job())

    repository.update(job.transition_to(JobStatus.PLANNING))
    repository.update(
        repository.get(job.tenant_id, job.id).with_checkpoint(
            workflow_id="research-job-1", workflow_run_id="run-1"
        )
    )

    reloaded = repository.get(job.tenant_id, job.id)
    assert reloaded.status is JobStatus.PLANNING
    assert reloaded.workflow_id == "research-job-1"
    assert reloaded.workflow_run_id == "run-1"


def test_updating_a_job_that_was_never_stored_is_refused(repository: JobRepository) -> None:
    with pytest.raises(JobNotFoundError):
        repository.update(a_job())


def test_evidence_round_trips_with_its_provenance_intact(repository: JobRepository) -> None:
    job = repository.add(a_job())
    evidence = repository.add_evidence(an_evidence_record(job))

    stored = repository.list_evidence(job.tenant_id, job.id)

    assert len(stored) == 1
    assert stored[0].id == evidence.id
    assert stored[0].content_hash == evidence.content_hash
    assert stored[0].producing_task_id == evidence.producing_task_id
    assert stored[0].tool_invocation_id == evidence.tool_invocation_id
    assert stored[0].trust_level is TrustLevel.PRIMARY
    assert str(stored[0].source_uri) == str(evidence.source_uri)


def test_evidence_cannot_be_attached_to_a_job_that_does_not_exist(
    repository: JobRepository,
) -> None:
    orphan = an_evidence_record(a_job())

    with pytest.raises(JobNotFoundError):
        repository.add_evidence(orphan)


def test_evidence_is_never_listed_for_another_tenant(repository: JobRepository) -> None:
    job = repository.add(a_job(tenant_id="tenant-a"))
    repository.add_evidence(an_evidence_record(job))

    with pytest.raises(JobNotFoundError):
        repository.list_evidence("tenant-b", job.id)


def test_the_audit_trail_records_every_attempted_call_in_order(
    repository: JobRepository,
) -> None:
    job = repository.add(a_job())
    first = an_invocation(job.tenant_id, job.id)
    second = an_invocation(job.tenant_id, job.id)

    repository.record_invocation(first)
    repository.record_invocation(second)

    trail = repository.list_invocations(job.tenant_id, job.id)
    assert {record.id for record in trail} == {first.id, second.id}
    assert [record.started_at for record in trail] == sorted(record.started_at for record in trail)
    assert trail[0].sanitized_arguments == {"url": "https://example.com/pricing"}
    assert trail[0].argument_digest == first.argument_digest


def test_an_audit_record_is_kept_even_for_a_job_that_was_never_stored(
    repository: JobRepository,
) -> None:
    """A denial for an unknown job is exactly what an investigation needs to see."""
    unknown_job = uuid4()

    repository.record_invocation(an_invocation("tenant-a", unknown_job))

    assert len(repository.list_invocations("tenant-a", unknown_job)) == 1


def test_recording_the_same_audit_record_twice_leaves_one_row(
    repository: JobRepository,
) -> None:
    """A retried activity re-reports its audit record; the trail must not double-count."""
    job = repository.add(a_job())
    invocation = an_invocation(job.tenant_id, job.id)

    repository.record_invocation(invocation)
    repository.record_invocation(invocation)

    assert len(repository.list_invocations(job.tenant_id, job.id)) == 1


def test_the_audit_trail_is_never_readable_by_another_tenant(repository: JobRepository) -> None:
    job = repository.add(a_job(tenant_id="tenant-a"))
    repository.record_invocation(an_invocation("tenant-a", job.id))

    assert repository.list_invocations("tenant-b", job.id) == []


def a_finding(job: ResearchJob, claim: str = "The list price fell.") -> FindingRecord:
    return FindingRecord(
        tenant_id=job.tenant_id,
        job_id=job.id,
        claim=claim,
        supporting_evidence_ids=[uuid4(), uuid4()],
        calculation_ids=[uuid4()],
        confidence=0.75,
        critic_verdict=CriticVerdict.SUPPORTED,
        reviewer_status=ReviewerStatus.APPROVED,
    )


def a_publication(job: ResearchJob, **changes: Any) -> ReportPublication:
    fields: dict[str, Any] = {
        "job_id": job.id,
        "tenant_id": job.tenant_id,
        "report_key": f"jobs/{job.id}/report.json",
        "markdown_key": f"jobs/{job.id}/report.md",
        "manifest_key": f"jobs/{job.id}/provenance-manifest.json",
        "evidence_key": f"jobs/{job.id}/evidence.json",
        "report_sha256": f"sha256:{'c' * 64}",
        "is_partial": False,
    }
    return ReportPublication.model_validate(fields | changes)


def test_findings_round_trip_in_order_with_every_field(repository: JobRepository) -> None:
    job = repository.add(a_job())
    first, second = a_finding(job, "First claim."), a_finding(job, "Second claim.")

    repository.replace_findings(job.tenant_id, job.id, [first, second])

    stored = repository.list_findings(job.tenant_id, job.id)
    assert [finding.claim for finding in stored] == ["First claim.", "Second claim."]
    assert stored[0].id == first.id
    assert stored[0].supporting_evidence_ids == first.supporting_evidence_ids
    assert stored[0].calculation_ids == first.calculation_ids
    assert stored[0].confidence == 0.75
    assert stored[0].critic_verdict is CriticVerdict.SUPPORTED
    assert stored[0].reviewer_status is ReviewerStatus.APPROVED


def test_storing_findings_again_replaces_the_earlier_set(repository: JobRepository) -> None:
    job = repository.add(a_job())
    repository.replace_findings(job.tenant_id, job.id, [a_finding(job), a_finding(job)])

    repository.replace_findings(job.tenant_id, job.id, [a_finding(job, "The only claim.")])

    assert [finding.claim for finding in repository.list_findings(job.tenant_id, job.id)] == [
        "The only claim."
    ]


def test_a_job_can_be_left_with_no_findings(repository: JobRepository) -> None:
    job = repository.add(a_job())
    repository.replace_findings(job.tenant_id, job.id, [a_finding(job)])

    repository.replace_findings(job.tenant_id, job.id, [])

    assert repository.list_findings(job.tenant_id, job.id) == []


def test_findings_cannot_be_stored_for_a_job_that_does_not_exist(
    repository: JobRepository,
) -> None:
    orphan = a_job()

    with pytest.raises(JobNotFoundError):
        repository.replace_findings(orphan.tenant_id, orphan.id, [a_finding(orphan)])


def test_findings_are_never_listed_for_or_replaced_by_another_tenant(
    repository: JobRepository,
) -> None:
    job = repository.add(a_job(tenant_id="tenant-a"))
    repository.replace_findings(job.tenant_id, job.id, [a_finding(job)])

    with pytest.raises(JobNotFoundError):
        repository.list_findings("tenant-b", job.id)
    with pytest.raises(JobNotFoundError):
        repository.replace_findings("tenant-b", job.id, [])
    assert len(repository.list_findings("tenant-a", job.id)) == 1


def test_a_publication_round_trips_with_its_keys_and_drifted_sources(
    repository: JobRepository,
) -> None:
    job = repository.add(a_job())
    drifted = uuid4()

    repository.record_publication(
        a_publication(job, is_partial=True, drifted_evidence_ids=[drifted])
    )

    stored = repository.get_publication(job.tenant_id, job.id)
    assert stored is not None
    assert stored.report_key == f"jobs/{job.id}/report.json"
    assert stored.manifest_key == f"jobs/{job.id}/provenance-manifest.json"
    assert stored.report_sha256 == f"sha256:{'c' * 64}"
    assert stored.is_partial is True
    assert stored.drifted_evidence_ids == [drifted]


def test_a_publication_keeps_the_classification_of_its_report_and_manifest(
    repository: JobRepository,
) -> None:
    job = repository.add(a_job())

    repository.record_publication(
        a_publication(
            job,
            report_access_class=AccessClass.INTERNAL,
            manifest_access_class=AccessClass.RESTRICTED,
        )
    )

    stored = repository.get_publication(job.tenant_id, job.id)
    assert stored is not None
    assert stored.report_access_class is AccessClass.INTERNAL
    assert stored.manifest_access_class is AccessClass.RESTRICTED


def test_a_job_that_has_not_published_has_no_publication(repository: JobRepository) -> None:
    job = repository.add(a_job())

    assert repository.get_publication(job.tenant_id, job.id) is None


def test_publishing_again_replaces_the_record_rather_than_failing(
    repository: JobRepository,
) -> None:
    """A redelivered publication activity must land on the same record."""
    job = repository.add(a_job())
    repository.record_publication(a_publication(job))

    repository.record_publication(a_publication(job, report_sha256=f"sha256:{'d' * 64}"))

    stored = repository.get_publication(job.tenant_id, job.id)
    assert stored is not None
    assert stored.report_sha256 == f"sha256:{'d' * 64}"


def test_a_publication_cannot_be_recorded_for_a_job_that_does_not_exist(
    repository: JobRepository,
) -> None:
    with pytest.raises(JobNotFoundError):
        repository.record_publication(a_publication(a_job()))


def test_a_publication_is_never_readable_by_another_tenant(repository: JobRepository) -> None:
    job = repository.add(a_job(tenant_id="tenant-a"))
    repository.record_publication(a_publication(job))

    with pytest.raises(JobNotFoundError):
        repository.get_publication("tenant-b", job.id)


@requires_postgres
@pytest.mark.parametrize("table", ["findings", "report_publications"])
def test_row_level_security_covers_findings_and_publications(
    restricted_pool: ConnectionPool[Connection[Any]], table: str
) -> None:
    """The same database boundary as jobs: no predicate, and still only one tenant."""
    repository = PostgresJobRepository(restricted_pool)
    for tenant in ("tenant-a", "tenant-b"):
        job = repository.add(a_job(tenant_id=tenant))
        repository.replace_findings(tenant, job.id, [a_finding(job)])
        repository.record_publication(a_publication(job))

    with restricted_pool.connection() as connection:
        undeclared = connection.execute(f"SELECT tenant_id FROM {table}").fetchall()
        connection.execute("SELECT set_config('research.tenant_id', 'tenant-a', true)")
        visible = connection.execute(f"SELECT tenant_id FROM {table}").fetchall()

    assert undeclared == []
    assert [row[0] for row in visible] == ["tenant-a"]


@requires_postgres
def test_the_application_role_is_one_the_policies_actually_bind(
    restricted_pool: ConnectionPool[Connection[Any]],
) -> None:
    """The precondition every test below depends on, asserted rather than assumed."""
    assert database_isolation_is_enforced(restricted_pool)


@requires_postgres
def test_row_level_security_hides_other_tenants_even_without_a_predicate(
    restricted_pool: ConnectionPool[Connection[Any]],
) -> None:
    """The database boundary, not the query, is what prevents cross-tenant reads.

    This is the guarantee section 14 asks to be unbreakable: a statement with no
    ``tenant_id`` predicate at all - a future caller, a reporting query, a mistake -
    still sees only the tenant the transaction declared.
    """
    repository = PostgresJobRepository(restricted_pool)
    repository.add(a_job(tenant_id="tenant-a"))
    repository.add(a_job(tenant_id="tenant-b"))

    with restricted_pool.connection() as connection:
        connection.execute("SELECT set_config('research.tenant_id', 'tenant-a', true)")
        visible = connection.execute("SELECT tenant_id FROM research_jobs").fetchall()

    assert [row[0] for row in visible] == ["tenant-a"]


@requires_postgres
def test_a_transaction_that_declares_no_tenant_can_read_nothing(
    restricted_pool: ConnectionPool[Connection[Any]],
) -> None:
    """The policy fails closed: an undeclared tenant matches no row rather than every row."""
    repository = PostgresJobRepository(restricted_pool)
    repository.add(a_job(tenant_id="tenant-a"))

    with restricted_pool.connection() as connection:
        visible = connection.execute("SELECT tenant_id FROM research_jobs").fetchall()

    assert visible == []


@requires_postgres
def test_a_pooled_connection_never_carries_a_previous_tenant(
    restricted_pool: ConnectionPool[Connection[Any]],
) -> None:
    """The tenant setting is transaction-local, so the next borrower starts undeclared."""
    repository = PostgresJobRepository(restricted_pool)
    repository.add(a_job(tenant_id="tenant-a"))

    repository.list("tenant-a")
    with restricted_pool.connection() as connection:
        leaked = connection.execute("SELECT tenant_id FROM research_jobs").fetchall()

    assert leaked == []
