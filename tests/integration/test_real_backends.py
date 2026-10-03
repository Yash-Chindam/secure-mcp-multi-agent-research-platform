"""The SQL and sandbox backends against the real things they wrap.

What makes these backends safe is behaviour of PostgreSQL and of the container runtime,
not of this codebase, so neither is substituted. Each half is skipped where its
dependency is missing: the SQL tests need ``RESEARCH_TEST_DATABASE_URL``, the sandbox
tests need a reachable ``docker``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg import Connection
from psycopg_pool import ConnectionPool

from research_platform.mcp.servers.docker_sandbox import CalculationFailed, DockerSandboxBackend
from research_platform.mcp.servers.postgres_backend import PostgresSqlBackend
from research_platform.mcp.servers.postgres_server import PostgresService
from research_platform.mcp.servers.sandbox_boundary import SandboxLimits
from research_platform.mcp.servers.sandbox_server import SandboxService
from research_platform.mcp.servers.sql_boundary import QueryNotAllowed

pytestmark = pytest.mark.integration

DATABASE_URL = os.getenv("RESEARCH_TEST_DATABASE_URL")
SANDBOX_IMAGE = "python:3.12-alpine"

requires_postgres = pytest.mark.skipif(
    not DATABASE_URL, reason="set RESEARCH_TEST_DATABASE_URL to run the SQL backend tests"
)


def _docker_is_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(  # noqa: S603
            ["docker", "info"], capture_output=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_is_reachable(), reason="a reachable docker daemon is needed for sandbox tests"
)


# -- PostgreSQL -------------------------------------------------------------------------


@pytest.fixture
def analytics() -> Iterator[PostgresSqlBackend]:
    assert DATABASE_URL is not None
    pool: ConnectionPool[Connection[Any]] = ConnectionPool(DATABASE_URL, min_size=1, open=True)
    pool.wait(timeout=30)
    with pool.connection() as connection:
        for statement in (
            "DROP SCHEMA IF EXISTS acme_analytics CASCADE",
            "DROP SCHEMA IF EXISTS globex_analytics CASCADE",
            "CREATE SCHEMA acme_analytics",
            "CREATE SCHEMA globex_analytics",
            "CREATE TABLE acme_analytics.prices (vendor TEXT, usd NUMERIC)",
            "CREATE TABLE globex_analytics.prices (vendor TEXT, usd NUMERIC)",
            "INSERT INTO acme_analytics.prices VALUES ('alpha', 4), ('beta', 6)",
            "INSERT INTO globex_analytics.prices VALUES ('secret', 99)",
        ):
            connection.execute(statement)
    yield PostgresSqlBackend(pool, statement_timeout_ms=500)
    pool.close()


@requires_postgres
def test_an_analytical_query_reads_the_tenants_own_schema(analytics: PostgresSqlBackend) -> None:
    service = PostgresService(backend=analytics, tenant_schemas={"acme": "acme_analytics"})

    rows = service.run_analytical_query(
        "acme", "SELECT vendor, usd FROM acme_analytics.prices ORDER BY usd"
    )

    assert [row["vendor"] for row in rows] == ["alpha", "beta"]


@requires_postgres
def test_the_schema_is_described_as_table_column_and_type(analytics: PostgresSqlBackend) -> None:
    assert analytics.describe_schema("acme_analytics") == [
        "prices.vendor text",
        "prices.usd numeric",
    ]


@requires_postgres
def test_the_database_itself_refuses_a_write_the_parser_never_saw(
    analytics: PostgresSqlBackend,
) -> None:
    """The read-only transaction is a second boundary, not a restatement of the first."""
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        analytics.run_query("DELETE FROM prices", schema="acme_analytics")

    assert len(analytics.run_query("SELECT * FROM prices", schema="acme_analytics")) == 2


@requires_postgres
def test_an_unqualified_table_resolves_only_inside_the_tenants_schema(
    analytics: PostgresSqlBackend,
) -> None:
    rows = analytics.run_query("SELECT vendor FROM prices", schema="acme_analytics")

    assert "secret" not in {row["vendor"] for row in rows}


@requires_postgres
def test_another_tenants_schema_cannot_be_named_through_the_service(
    analytics: PostgresSqlBackend,
) -> None:
    service = PostgresService(backend=analytics, tenant_schemas={"acme": "acme_analytics"})

    with pytest.raises(QueryNotAllowed):
        service.run_analytical_query("acme", "SELECT vendor FROM globex_analytics.prices")


@requires_postgres
def test_a_query_that_runs_too_long_is_cancelled_by_the_database(
    analytics: PostgresSqlBackend,
) -> None:
    with pytest.raises(psycopg.errors.QueryCanceled):
        analytics.run_query("SELECT pg_sleep(5)", schema="acme_analytics")


@requires_postgres
def test_a_percent_sign_in_a_query_is_not_mistaken_for_a_placeholder(
    analytics: PostgresSqlBackend,
) -> None:
    rows = analytics.run_query(
        "SELECT vendor FROM prices WHERE vendor LIKE 'al%'", schema="acme_analytics"
    )

    assert [row["vendor"] for row in rows] == ["alpha"]


# -- sandbox ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sandbox() -> DockerSandboxBackend:
    subprocess.run(  # noqa: S603
        ["docker", "pull", "--quiet", SANDBOX_IMAGE], capture_output=True, timeout=300, check=True
    )
    return DockerSandboxBackend(image=SANDBOX_IMAGE, wall_clock_seconds=20)


@requires_docker
def test_a_calculation_really_runs_and_returns_its_output(sandbox: DockerSandboxBackend) -> None:
    service = SandboxService(backend=sandbox, limits=SandboxLimits())

    assert service.run("acme", "print(sum(range(101)))").strip() == "5050"


@requires_docker
def test_the_container_has_no_network(sandbox: DockerSandboxBackend) -> None:
    """Run the backend directly: the screen would refuse this code before it got here."""
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('reached')\n"
        "except OSError as error:\n"
        "    print('blocked')\n"
    )

    assert sandbox.run(code, cpu_seconds=5, memory_mib=128).strip() == "blocked"


@requires_docker
def test_the_container_cannot_write_to_its_filesystem(sandbox: DockerSandboxBackend) -> None:
    code = (
        "try:\n"
        "    open('/home/escape.txt', 'w').write('x')\n"
        "    print('written')\n"
        "except OSError:\n"
        "    print('read-only')\n"
    )

    assert sandbox.run(code, cpu_seconds=5, memory_mib=128).strip() == "read-only"


@requires_docker
def test_the_calculation_runs_as_an_unprivileged_user(sandbox: DockerSandboxBackend) -> None:
    assert sandbox.run("import os; print(os.getuid())", cpu_seconds=5, memory_mib=128).strip() == (
        "65534"
    )


@requires_docker
def test_a_calculation_that_exceeds_its_memory_ceiling_is_stopped(
    sandbox: DockerSandboxBackend,
) -> None:
    with pytest.raises(CalculationFailed):
        sandbox.run("x = bytearray(512 * 1024 * 1024)", cpu_seconds=5, memory_mib=64)


@requires_docker
def test_a_calculation_that_never_finishes_is_killed() -> None:
    impatient = DockerSandboxBackend(image=SANDBOX_IMAGE, wall_clock_seconds=3)

    with pytest.raises(CalculationFailed, match="was stopped"):
        impatient.run("while True: pass", cpu_seconds=60, memory_mib=64)
