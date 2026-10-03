"""The durable ``JobRepository`` section 10 asks for, on PostgreSQL.

Every method opens one transaction, declares the tenant it is acting for, and only then
issues its statement. Declaring the tenant is not an optimisation: the row-level security
policies in ``schema.py`` are keyed on that setting, so a statement that skipped it would
see an empty table rather than another tenant's rows. The ``tenant_id`` predicates in the
SQL below are therefore the second line of defence, not the first.
"""

from __future__ import annotations

import builtins
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from psycopg import Connection, Cursor
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from research_platform.application.jobs import JobNotFoundError
from research_platform.domain.invocations import ToolInvocation
from research_platform.domain.models import EvidenceRecord, FindingRecord, ResearchJob
from research_platform.persistence.schema import SCHEMA_STATEMENTS, TENANT_SETTING

logger = logging.getLogger(__name__)

_JOB_COLUMNS = """
    id, tenant_id, requester_id, question, constraints, source_requirements,
    budget, status, status_detail, workflow_id, workflow_run_id, created_at, updated_at
"""

_EVIDENCE_COLUMNS = """
    id, tenant_id, job_id, excerpt, source_uri, title, author, published_at,
    trust_level, access_class, content_hash, producing_task_id, tool_invocation_id,
    retrieved_at
"""

_INVOCATION_COLUMNS = """
    id, tenant_id, job_id, task_id, mcp_server, capability, sanitized_arguments,
    argument_digest, policy_version, authorization_decision, outcome, error_class,
    started_at, duration_ms
"""


def create_schema(pool: ConnectionPool[Connection[Any]]) -> None:
    """Bring the database up to the current schema, idempotently."""
    with pool.connection() as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)


_FINDING_COLUMNS = """
    id, tenant_id, job_id, claim, supporting_evidence_ids, contradicting_evidence_ids,
    calculation_ids, confidence, critic_verdict, reviewer_status, recorded_at
"""


def _rows(connection: Connection[Any]) -> Cursor[Any]:
    """A cursor that yields mappings, without changing the pooled connection itself.

    Setting ``row_factory`` on the connection would outlive this call: the pool hands the
    same connection to the next borrower, who would then get a shape it never asked for.
    """
    return connection.cursor(row_factory=dict_row)


def database_isolation_is_enforced(pool: ConnectionPool[Connection[Any]]) -> bool:
    """Report whether the row-level security policies actually bind this connection.

    A superuser, and any role with BYPASSRLS, is exempt from every policy in the
    database - ``FORCE ROW LEVEL SECURITY`` does not change that. A deployment that
    connects as one keeps only the ``tenant_id`` predicates in this module, which is one
    layer rather than the two section 11 asks for, so this is reported at startup instead
    of being assumed.
    """
    with pool.connection() as connection:
        row = connection.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    return row is not None and not row[0]


class PostgresJobRepository:
    """Jobs, evidence and the MCP audit trail, in the platform's system of record."""

    def __init__(self, pool: ConnectionPool[Connection[Any]]) -> None:
        self._pool = pool

    @contextmanager
    def _acting_for(self, tenant_id: str) -> Iterator[Connection[Any]]:
        """Open a transaction that may only see one tenant's rows.

        ``set_config`` is called with its is_local flag set, scoping the setting to this
        transaction, so a pooled connection handed to the next caller never carries a
        stale tenant.
        """
        with self._pool.connection() as connection:
            connection.execute(
                "SELECT set_config(%s, %s, true)",
                (TENANT_SETTING, tenant_id),
            )
            yield connection

    def add(self, job: ResearchJob) -> ResearchJob:
        with self._acting_for(job.tenant_id) as connection:
            connection.execute(
                f"""
                INSERT INTO research_jobs ({_JOB_COLUMNS})
                VALUES (
                    %(id)s, %(tenant_id)s, %(requester_id)s, %(question)s, %(constraints)s,
                    %(source_requirements)s, %(budget)s, %(status)s, %(status_detail)s,
                    %(workflow_id)s, %(workflow_run_id)s, %(created_at)s, %(updated_at)s
                )
                """,
                _job_parameters(job),
            )
        return job

    def get(self, tenant_id: str, job_id: UUID) -> ResearchJob:
        with self._acting_for(tenant_id) as connection:
            row = (
                _rows(connection)
                .execute(
                    f"SELECT {_JOB_COLUMNS} FROM research_jobs WHERE tenant_id = %s AND id = %s",
                    (tenant_id, job_id),
                )
                .fetchone()
            )
        if row is None:
            raise JobNotFoundError(str(job_id))
        return ResearchJob.model_validate(row)

    def list(self, tenant_id: str) -> builtins.list[ResearchJob]:
        with self._acting_for(tenant_id) as connection:
            rows = (
                _rows(connection)
                .execute(
                    f"""
                SELECT {_JOB_COLUMNS} FROM research_jobs
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                """,
                    (tenant_id,),
                )
                .fetchall()
            )
        return [ResearchJob.model_validate(row) for row in rows]

    def update(self, job: ResearchJob) -> ResearchJob:
        with self._acting_for(job.tenant_id) as connection:
            updated = (
                _rows(connection)
                .execute(
                    """
                UPDATE research_jobs
                SET status = %(status)s,
                    status_detail = %(status_detail)s,
                    workflow_id = %(workflow_id)s,
                    workflow_run_id = %(workflow_run_id)s,
                    updated_at = %(updated_at)s
                WHERE tenant_id = %(tenant_id)s AND id = %(id)s
                """,
                    _job_parameters(job),
                )
                .rowcount
            )
        if not updated:
            raise JobNotFoundError(str(job.id))
        return job

    def add_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord:
        # Checked here as well as by the foreign key so both repository implementations
        # answer a missing job with the same JobNotFoundError rather than one of them
        # surfacing a driver-level integrity error.
        self.get(evidence.tenant_id, evidence.job_id)
        with self._acting_for(evidence.tenant_id) as connection:
            connection.execute(
                f"""
                INSERT INTO evidence_records ({_EVIDENCE_COLUMNS})
                VALUES (
                    %(id)s, %(tenant_id)s, %(job_id)s, %(excerpt)s, %(source_uri)s, %(title)s,
                    %(author)s, %(published_at)s, %(trust_level)s, %(access_class)s,
                    %(content_hash)s, %(producing_task_id)s, %(tool_invocation_id)s,
                    %(retrieved_at)s
                )
                """,
                _evidence_parameters(evidence),
            )
        return evidence

    def list_evidence(self, tenant_id: str, job_id: UUID) -> builtins.list[EvidenceRecord]:
        self.get(tenant_id, job_id)
        with self._acting_for(tenant_id) as connection:
            rows = (
                _rows(connection)
                .execute(
                    f"""
                SELECT {_EVIDENCE_COLUMNS} FROM evidence_records
                WHERE tenant_id = %s AND job_id = %s
                ORDER BY retrieved_at
                """,
                    (tenant_id, job_id),
                )
                .fetchall()
            )
        return [EvidenceRecord.model_validate(row) for row in rows]

    def record_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        """Append one audit record, without requiring the job it names to exist.

        ``tool_invocations`` carries no foreign key onto ``research_jobs`` for this
        reason: a denial recorded against an unknown job is itself worth keeping, and a
        missing parent row must never be able to reject an audit write. The write is
        idempotent on the record's own identifier, so a retried activity that already
        persisted its audit record does not fail on the second attempt.
        """
        with self._acting_for(invocation.tenant_id) as connection:
            connection.execute(
                f"""
                INSERT INTO tool_invocations ({_INVOCATION_COLUMNS})
                VALUES (
                    %(id)s, %(tenant_id)s, %(job_id)s, %(task_id)s, %(mcp_server)s,
                    %(capability)s, %(sanitized_arguments)s, %(argument_digest)s,
                    %(policy_version)s, %(authorization_decision)s, %(outcome)s,
                    %(error_class)s, %(started_at)s, %(duration_ms)s
                )
                ON CONFLICT (tenant_id, id) DO NOTHING
                """,
                _invocation_parameters(invocation),
            )
        return invocation

    def list_invocations(self, tenant_id: str, job_id: UUID) -> builtins.list[ToolInvocation]:
        with self._acting_for(tenant_id) as connection:
            rows = (
                _rows(connection)
                .execute(
                    f"""
                SELECT {_INVOCATION_COLUMNS} FROM tool_invocations
                WHERE tenant_id = %s AND job_id = %s
                ORDER BY started_at
                """,
                    (tenant_id, job_id),
                )
                .fetchall()
            )
        return [ToolInvocation.model_validate(row) for row in rows]

    def replace_findings(
        self, tenant_id: str, job_id: UUID, findings: builtins.list[FindingRecord]
    ) -> builtins.list[FindingRecord]:
        """Swap the job's findings for the latest set, in one transaction.

        Delete and insert share a transaction, so a reader sees the old set or the new
        one and never an empty or half-written one.
        """
        self.get(tenant_id, job_id)
        with self._acting_for(tenant_id) as connection:
            connection.execute(
                "DELETE FROM findings WHERE tenant_id = %s AND job_id = %s", (tenant_id, job_id)
            )
            for position, finding in enumerate(findings):
                connection.execute(
                    f"""
                    INSERT INTO findings ({_FINDING_COLUMNS}, position)
                    VALUES (
                        %(id)s, %(tenant_id)s, %(job_id)s, %(claim)s, %(supporting)s,
                        %(contradicting)s, %(calculations)s, %(confidence)s,
                        %(critic_verdict)s, %(reviewer_status)s, %(recorded_at)s, %(position)s
                    )
                    """,
                    _finding_parameters(finding, position),
                )
        return findings

    def list_findings(self, tenant_id: str, job_id: UUID) -> builtins.list[FindingRecord]:
        self.get(tenant_id, job_id)
        with self._acting_for(tenant_id) as connection:
            rows = (
                _rows(connection)
                .execute(
                    f"""
                SELECT {_FINDING_COLUMNS} FROM findings
                WHERE tenant_id = %s AND job_id = %s
                ORDER BY position
                """,
                    (tenant_id, job_id),
                )
                .fetchall()
            )
        return [FindingRecord.model_validate(row) for row in rows]


def _finding_parameters(finding: FindingRecord, position: int) -> dict[str, Any]:
    return {
        "id": finding.id,
        "tenant_id": finding.tenant_id,
        "job_id": finding.job_id,
        "claim": finding.claim,
        "supporting": Jsonb([str(item) for item in finding.supporting_evidence_ids]),
        "contradicting": Jsonb([str(item) for item in finding.contradicting_evidence_ids]),
        "calculations": Jsonb([str(item) for item in finding.calculation_ids]),
        "confidence": finding.confidence,
        "critic_verdict": finding.critic_verdict.value,
        "reviewer_status": finding.reviewer_status.value,
        "recorded_at": finding.recorded_at,
        "position": position,
    }


def build_repository(
    database_url: str, *, min_size: int = 1, max_size: int = 10
) -> PostgresJobRepository:
    """Open a pool, apply the schema and return a repository ready to serve requests."""
    pool: ConnectionPool[Connection[Any]] = ConnectionPool(
        database_url, min_size=min_size, max_size=max_size, open=True
    )
    pool.wait()
    create_schema(pool)
    if not database_isolation_is_enforced(pool):
        logger.warning(
            "connected as a role that bypasses row-level security; tenant isolation "
            "rests on query predicates alone - connect as a non-superuser role without "
            "BYPASSRLS to get the database-level boundary as well"
        )
    return PostgresJobRepository(pool)


def _job_parameters(job: ResearchJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "tenant_id": job.tenant_id,
        "requester_id": job.requester_id,
        "question": job.question,
        "constraints": Jsonb(job.constraints),
        "source_requirements": Jsonb(job.source_requirements),
        "budget": Jsonb(job.budget.model_dump()),
        "status": job.status.value,
        "status_detail": job.status_detail,
        "workflow_id": job.workflow_id,
        "workflow_run_id": job.workflow_run_id,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _evidence_parameters(evidence: EvidenceRecord) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "tenant_id": evidence.tenant_id,
        "job_id": evidence.job_id,
        "excerpt": evidence.excerpt,
        "source_uri": str(evidence.source_uri),
        "title": evidence.title,
        "author": evidence.author,
        "published_at": evidence.published_at,
        "trust_level": evidence.trust_level.value,
        "access_class": evidence.access_class.value,
        "content_hash": evidence.content_hash,
        "producing_task_id": evidence.producing_task_id,
        "tool_invocation_id": evidence.tool_invocation_id,
        "retrieved_at": evidence.retrieved_at,
    }


def _invocation_parameters(invocation: ToolInvocation) -> dict[str, Any]:
    return {
        "id": invocation.id,
        "tenant_id": invocation.tenant_id,
        "job_id": invocation.job_id,
        "task_id": invocation.task_id,
        "mcp_server": invocation.mcp_server,
        "capability": invocation.capability,
        "sanitized_arguments": Jsonb(invocation.sanitized_arguments),
        "argument_digest": invocation.argument_digest,
        "policy_version": invocation.policy_version,
        "authorization_decision": invocation.authorization_decision.value,
        "outcome": invocation.outcome.value,
        "error_class": invocation.error_class.value,
        "started_at": invocation.started_at,
        "duration_ms": invocation.duration_ms,
    }
