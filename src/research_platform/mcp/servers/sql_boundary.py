"""The boundary the PostgreSQL server must be used within.

Section 8 requires a read-only role, a query parser and row-level security. This module
is the parser: it accepts a single analytical read and refuses everything else. It is a
defence in depth measure, not the only one — the database role must also be read-only and
row-level security must scope the tenant, because a parser alone cannot be trusted with
that guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_QUERY_LENGTH = 8_000
DEFAULT_ROW_LIMIT = 1_000

READ_ONLY_PREFIXES = ("select", "with")

FORBIDDEN_KEYWORDS = frozenset(
    {
        "alter",
        "analyze",
        "call",
        "comment",
        "copy",
        "create",
        "delete",
        "do",
        "drop",
        "execute",
        "explain",
        "grant",
        "insert",
        "listen",
        "lock",
        "merge",
        "notify",
        "prepare",
        "reassign",
        "refresh",
        "reindex",
        "reset",
        "revoke",
        "security",
        "set",
        "truncate",
        "update",
        "vacuum",
    }
)

FORBIDDEN_FUNCTIONS = frozenset(
    {
        "dblink",
        "lo_export",
        "lo_import",
        "pg_read_binary_file",
        "pg_read_file",
        "pg_reload_conf",
        "pg_sleep",
        "pg_terminate_backend",
    }
)

LINE_COMMENT = re.compile(r"--[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
WORD = re.compile(r"[a-z_][a-z0-9_]*")
STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
QUALIFIED_NAME = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\s*\.", re.IGNORECASE)
UNQUALIFIED_TABLE = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b(?!\s*\.)", re.IGNORECASE)
CTE_NAME = re.compile(r"(?:\bwith\s+|,\s*)([a-z_][a-z0-9_]*)\s+as\s*\(", re.IGNORECASE)
LIMIT_CLAUSE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)


class QueryNotAllowed(ValueError):
    """Raised when a query is not a permitted analytical read."""


@dataclass(frozen=True)
class AnalyticalQuery:
    """A query that passed the parser, with the row limit that will be applied."""

    sql: str
    schema: str
    row_limit: int


def _strip_comments_and_literals(sql: str) -> str:
    """Remove comments and string bodies so keyword checks cannot be hidden inside them."""
    without_comments = BLOCK_COMMENT.sub(" ", LINE_COMMENT.sub(" ", sql))
    return STRING_LITERAL.sub("''", without_comments)


def _split_statements(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def parse_analytical_query(
    sql: str,
    *,
    schema: str,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> AnalyticalQuery:
    """Accept one read-only query against the tenant schema, or refuse it.

    Comments and string literals are removed before any keyword check, so a forbidden
    statement cannot be smuggled past the parser inside them.
    """
    if not schema or not WORD.fullmatch(schema):
        raise QueryNotAllowed(f"{schema!r} is not a valid tenant schema name")
    if row_limit < 1:
        raise QueryNotAllowed("a row limit must admit at least one row")

    raw = sql.strip()
    if not raw:
        raise QueryNotAllowed("a query must not be empty")
    if len(raw) > MAX_QUERY_LENGTH:
        raise QueryNotAllowed(f"a query must not exceed {MAX_QUERY_LENGTH} characters")

    inspectable = _strip_comments_and_literals(raw)
    statements = _split_statements(inspectable)
    if len(statements) > 1:
        raise QueryNotAllowed("only a single statement may be executed")
    if not statements:
        raise QueryNotAllowed("a query must contain a statement")

    statement = statements[0]
    lowered = statement.lower()
    if not lowered.startswith(READ_ONLY_PREFIXES):
        raise QueryNotAllowed("only SELECT and WITH queries are permitted")

    words = set(WORD.findall(lowered))
    forbidden = sorted(words & FORBIDDEN_KEYWORDS)
    if forbidden:
        raise QueryNotAllowed(f"query uses forbidden keywords: {', '.join(forbidden)}")
    dangerous = sorted(words & FORBIDDEN_FUNCTIONS)
    if dangerous:
        raise QueryNotAllowed(f"query uses forbidden functions: {', '.join(dangerous)}")

    _reject_foreign_schemas(statement, schema)

    return AnalyticalQuery(
        sql=_apply_row_limit(raw, inspectable, row_limit),
        schema=schema,
        row_limit=row_limit,
    )


def _reject_foreign_schemas(statement: str, schema: str) -> None:
    """Refuse a query that names any schema other than the caller's own."""
    referenced = {match.lower() for match in QUALIFIED_NAME.findall(statement)}
    foreign = sorted(referenced - {schema.lower()})
    if foreign:
        raise QueryNotAllowed(f"query reads outside the tenant schema: {', '.join(foreign)}")

    common_table_names = {match.lower() for match in CTE_NAME.findall(statement)}
    unqualified = {
        match.lower()
        for match in UNQUALIFIED_TABLE.findall(statement)
        if match.lower() not in {"select", "lateral"} | common_table_names
    }
    if unqualified:
        raise QueryNotAllowed(
            "every table must be schema-qualified: " + ", ".join(sorted(unqualified))
        )


def _apply_row_limit(raw: str, inspectable: str, row_limit: int) -> str:
    """Bound the result set, tightening a limit the caller set too high."""
    trimmed = raw.rstrip().rstrip(";").rstrip()
    existing = LIMIT_CLAUSE.search(inspectable)
    if existing is None:
        return f"{trimmed} LIMIT {row_limit}"
    if int(existing.group(1)) <= row_limit:
        return trimmed
    return f"{LIMIT_CLAUSE.sub('', trimmed).rstrip()} LIMIT {row_limit}"
