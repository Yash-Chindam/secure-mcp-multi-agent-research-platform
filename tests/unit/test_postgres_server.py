from dataclasses import dataclass, field

import pytest

from research_platform.mcp.servers.postgres_server import PostgresService
from research_platform.mcp.servers.sql_boundary import QueryNotAllowed


@dataclass
class RecordingSqlBackend:
    described: list[str] = field(default_factory=list)
    queries: list[tuple[str, str]] = field(default_factory=list)

    def describe_schema(self, schema: str) -> list[str]:
        self.described.append(schema)
        return [f"{schema}.invoices(id, amount)"]

    def run_query(self, sql: str, *, schema: str) -> list[dict[str, object]]:
        self.queries.append((sql, schema))
        return [{"total": 42}]


def service() -> tuple[PostgresService, RecordingSqlBackend]:
    backend = RecordingSqlBackend()
    return PostgresService(backend, {"acme": "tenant_acme"}, row_limit=25), backend


def test_service_describes_only_the_tenants_schema() -> None:
    postgres, backend = service()

    assert postgres.describe_schema("acme") == ["tenant_acme.invoices(id, amount)"]
    assert backend.described == ["tenant_acme"]


def test_service_parses_and_bounds_a_query_before_execution() -> None:
    postgres, backend = service()

    assert postgres.run_analytical_query("acme", "SELECT amount FROM tenant_acme.invoices") == [
        {"total": 42}
    ]
    assert backend.queries == [("SELECT amount FROM tenant_acme.invoices LIMIT 25", "tenant_acme")]


def test_service_refuses_an_unknown_tenant() -> None:
    postgres, _ = service()

    with pytest.raises(QueryNotAllowed, match="has no analytical schema"):
        postgres.describe_schema("globex")


def test_service_requires_a_tenant() -> None:
    postgres, _ = service()

    with pytest.raises(QueryNotAllowed, match="identify its tenant"):
        postgres.describe_schema(" ")


def test_at_least_one_schema_is_required() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PostgresService(RecordingSqlBackend(), {})


def test_schema_configuration_is_validated() -> None:
    with pytest.raises(QueryNotAllowed, match="not a valid tenant schema"):
        PostgresService(RecordingSqlBackend(), {"acme": "tenant-acme"})


def test_a_schema_must_belong_to_a_named_tenant() -> None:
    with pytest.raises(ValueError, match="named tenant"):
        PostgresService(RecordingSqlBackend(), {"": "tenant_acme"})


def test_row_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PostgresService(RecordingSqlBackend(), {"acme": "tenant_acme"}, row_limit=0)
