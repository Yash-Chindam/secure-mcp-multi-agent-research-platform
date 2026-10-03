"""The PostgreSQL system of record for section 10's information model.

Three tables, one per entity the platform must still be able to produce after a crash: a
research job and its durable workflow checkpoint, the evidence that was retrieved for it,
and the audit trail of every MCP call that was attempted on its behalf.

Tenant isolation is enforced twice, deliberately. Every statement this package issues
already carries its own ``tenant_id`` predicate, and every table additionally runs under
a forced row-level security policy keyed on a per-transaction setting. The second layer
is what section 11 means by enforcing isolation *at the database boundary*: a query that
forgets its predicate, or a future caller that reaches these tables by another route,
returns nothing rather than another tenant's rows. ``current_setting`` is read in its
missing-ok form, so a connection that never set the tenant sees no rows at all rather
than failing open.
"""

from __future__ import annotations

TENANT_SETTING = "research.tenant_id"
"""The per-transaction setting every row-level security policy below is keyed on."""


_TENANT_PREDICATE = f"tenant_id = current_setting('{TENANT_SETTING}', true)"


def _tenant_isolation(table: str) -> tuple[str, ...]:
    """Force row-level security on one table, for its owner as well as everyone else.

    ``ENABLE`` alone exempts the table owner, which is the very role an application
    usually connects as - ``FORCE`` is what makes the policy apply to it too.
    """
    return (
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
        f"""
        DROP POLICY IF EXISTS tenant_isolation ON {table}
        """,
        f"""
        CREATE POLICY tenant_isolation ON {table}
            USING ({_TENANT_PREDICATE})
            WITH CHECK ({_TENANT_PREDICATE})
        """,
    )


_JOBS = (
    """
    CREATE TABLE IF NOT EXISTS research_jobs (
        id                  UUID        NOT NULL,
        tenant_id           TEXT        NOT NULL,
        requester_id        TEXT        NOT NULL,
        question            TEXT        NOT NULL,
        constraints         JSONB       NOT NULL DEFAULT '[]'::jsonb,
        source_requirements JSONB       NOT NULL DEFAULT '[]'::jsonb,
        budget              JSONB       NOT NULL,
        status              TEXT        NOT NULL,
        workflow_id         TEXT,
        workflow_run_id     TEXT,
        created_at          TIMESTAMPTZ NOT NULL,
        updated_at          TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (tenant_id, id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS research_jobs_by_tenant
        ON research_jobs (tenant_id, created_at DESC)
    """,
    # Added after the table first shipped; applied in place so an existing database is
    # brought forward by the same idempotent statements a new one is created with.
    "ALTER TABLE research_jobs ADD COLUMN IF NOT EXISTS status_detail TEXT",
)

_EVIDENCE = (
    """
    CREATE TABLE IF NOT EXISTS evidence_records (
        id                  UUID        NOT NULL,
        tenant_id           TEXT        NOT NULL,
        job_id              UUID        NOT NULL,
        excerpt             TEXT        NOT NULL,
        source_uri          TEXT        NOT NULL,
        title               TEXT,
        author              TEXT,
        published_at        TIMESTAMPTZ,
        trust_level         TEXT        NOT NULL,
        access_class        TEXT        NOT NULL,
        content_hash        TEXT        NOT NULL,
        producing_task_id   UUID        NOT NULL,
        tool_invocation_id  UUID        NOT NULL,
        retrieved_at        TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (tenant_id, id),
        FOREIGN KEY (tenant_id, job_id)
            REFERENCES research_jobs (tenant_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_records_by_job
        ON evidence_records (tenant_id, job_id, retrieved_at)
    """,
)

_INVOCATIONS = (
    # Deliberately *not* a foreign key onto research_jobs. An audit record is written
    # whether or not the job it names is still present - a denial for an unknown job, or
    # a trail outliving the job row it describes, is exactly the evidence an
    # investigation needs, so a missing parent must not be able to reject the write.
    """
    CREATE TABLE IF NOT EXISTS tool_invocations (
        id                     UUID        NOT NULL,
        tenant_id              TEXT        NOT NULL,
        job_id                 UUID        NOT NULL,
        task_id                UUID        NOT NULL,
        mcp_server             TEXT        NOT NULL,
        capability             TEXT        NOT NULL,
        sanitized_arguments    JSONB       NOT NULL DEFAULT '{}'::jsonb,
        argument_digest        TEXT        NOT NULL,
        policy_version         TEXT        NOT NULL,
        authorization_decision TEXT        NOT NULL,
        outcome                TEXT        NOT NULL,
        error_class            TEXT        NOT NULL,
        started_at             TIMESTAMPTZ NOT NULL,
        duration_ms            INTEGER,
        PRIMARY KEY (tenant_id, id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS tool_invocations_by_job
        ON tool_invocations (tenant_id, job_id, started_at)
    """,
)


SCHEMA_STATEMENTS: tuple[str, ...] = (
    *_JOBS,
    *_tenant_isolation("research_jobs"),
    *_EVIDENCE,
    *_tenant_isolation("evidence_records"),
    *_INVOCATIONS,
    *_tenant_isolation("tool_invocations"),
)
"""Every statement needed to bring an empty database up to the current schema.

Idempotent, so applying it to an already-migrated database is a no-op rather than an
error. This is deliberately not a migration framework: there is one version of this
schema, and a change to a released one would need a real migration tool.
"""
