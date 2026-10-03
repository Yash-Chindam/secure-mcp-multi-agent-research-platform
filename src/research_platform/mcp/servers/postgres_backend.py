"""The production SQL backend: a read-only transaction against the analytical database.

``parse_analytical_query`` has already refused anything but a single schema-confined
``SELECT`` by the time a statement arrives here. This backend does not rely on that. It
runs every statement in a transaction the database itself has been told is read-only,
with a statement timeout and the tenant's schema as the only search path, so a query the
parser misjudged still cannot write, cannot run unbounded and cannot resolve an
unqualified name into another schema. Section 8 also asks for a read-only *role*: connect
this backend as one, and the database refuses a write a third time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


@dataclass
class PostgresSqlBackend:
    pool: ConnectionPool[Connection[Any]]
    statement_timeout_ms: int = 30_000

    def describe_schema(self, schema: str) -> list[str]:
        with self.pool.connection() as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            rows = connection.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position
                """,
                (schema,),
            ).fetchall()
        return [f"{table}.{column} {data_type}" for table, column, data_type in rows]

    def run_query(self, sql: str, *, schema: str) -> list[dict[str, object]]:
        with self.pool.connection() as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            connection.execute(
                "SELECT set_config('statement_timeout', %s, true), "
                "set_config('search_path', %s, true)",
                (str(self.statement_timeout_ms), schema),
            )
            # No parameters are passed, so the driver sends the statement as written and
            # a literal percent sign in it is not mistaken for a placeholder.
            rows = connection.cursor(row_factory=dict_row).execute(sql).fetchall()
        return [dict(row) for row in rows]
