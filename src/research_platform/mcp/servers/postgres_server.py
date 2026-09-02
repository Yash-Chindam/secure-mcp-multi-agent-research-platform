"""FastMCP tools for tenant-scoped analytical PostgreSQL access."""

from __future__ import annotations

import json
from dataclasses import dataclass

from fastmcp import FastMCP

from research_platform.mcp.servers.backends import SqlBackend
from research_platform.mcp.servers.sql_boundary import (
    DEFAULT_ROW_LIMIT,
    QueryNotAllowed,
    parse_analytical_query,
)


@dataclass(frozen=True)
class PostgresService:
    """Apply tenant-schema and read-only query rules before touching the database."""

    backend: SqlBackend
    tenant_schemas: dict[str, str]
    row_limit: int = DEFAULT_ROW_LIMIT

    def __post_init__(self) -> None:
        if not self.tenant_schemas:
            raise ValueError("at least one tenant analytical schema must be configured")
        if self.row_limit < 1:
            raise ValueError("row_limit must be positive")
        for tenant_id, schema in self.tenant_schemas.items():
            if not tenant_id:
                raise ValueError("an analytical schema must belong to a named tenant")
            parse_analytical_query("SELECT 1", schema=schema, row_limit=1)

    def describe_schema(self, tenant_id: str) -> list[str]:
        schema = self._schema_for(tenant_id)
        return self.backend.describe_schema(schema)

    def run_analytical_query(self, tenant_id: str, sql: str) -> list[dict[str, object]]:
        schema = self._schema_for(tenant_id)
        query = parse_analytical_query(sql, schema=schema, row_limit=self.row_limit)
        return self.backend.run_query(query.sql, schema=query.schema)

    def _schema_for(self, tenant_id: str) -> str:
        if not tenant_id or not tenant_id.strip():
            raise QueryNotAllowed("a PostgreSQL call must identify its tenant")
        try:
            return self.tenant_schemas[tenant_id]
        except KeyError as error:
            raise QueryNotAllowed(f"tenant {tenant_id} has no analytical schema") from error


def build_postgres_server(service: PostgresService) -> FastMCP:
    """Expose schema inspection and bounded analytical reads over MCP."""
    server: FastMCP = FastMCP(name="postgres")

    @server.tool
    def describe_schema(tenant_id: str) -> str:
        """Describe tables and columns exposed to this tenant's read-only role."""
        try:
            description = service.describe_schema(tenant_id)
        except (QueryNotAllowed, ValueError) as error:
            raise ValueError(f"validation error: {error}") from error
        return json.dumps(description, separators=(",", ":"))

    @server.tool
    def run_analytical_query(tenant_id: str, sql: str) -> str:
        """Run one parsed, schema-confined read-only analytical query."""
        try:
            rows = service.run_analytical_query(tenant_id, sql)
        except (QueryNotAllowed, ValueError) as error:
            raise ValueError(f"validation error: {error}") from error
        return json.dumps(rows, default=str, separators=(",", ":"))

    return server
