import pytest

from research_platform.mcp.servers.sql_boundary import (
    MAX_QUERY_LENGTH,
    QueryNotAllowed,
    parse_analytical_query,
)

SCHEMA = "tenant_acme"


def parse(sql: str, *, row_limit: int = 1_000, schema: str = SCHEMA) -> str:
    return parse_analytical_query(sql, schema=schema, row_limit=row_limit).sql


def test_a_simple_select_is_accepted_and_bounded() -> None:
    assert parse("SELECT id FROM tenant_acme.invoices") == (
        "SELECT id FROM tenant_acme.invoices LIMIT 1000"
    )


def test_a_common_table_expression_is_accepted() -> None:
    sql = "WITH totals AS (SELECT sum(amount) AS t FROM tenant_acme.invoices) SELECT t FROM totals"

    assert parse(sql).endswith("LIMIT 1000")


def test_a_trailing_semicolon_is_accepted() -> None:
    assert parse("SELECT id FROM tenant_acme.invoices;") == (
        "SELECT id FROM tenant_acme.invoices LIMIT 1000"
    )


def test_an_empty_query_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="must not be empty"):
        parse("   ")


def test_a_query_of_only_a_semicolon_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="must contain a statement"):
        parse(";")


def test_an_oversized_query_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="must not exceed"):
        parse("SELECT " + "a" * MAX_QUERY_LENGTH)


@pytest.mark.parametrize("schema", ["", "tenant acme", "tenant-acme", "1tenant"])
def test_an_invalid_schema_name_is_refused(schema: str) -> None:
    with pytest.raises(QueryNotAllowed, match="not a valid tenant schema"):
        parse("SELECT 1 FROM tenant_acme.invoices", schema=schema)


def test_a_non_positive_row_limit_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="at least one row"):
        parse("SELECT id FROM tenant_acme.invoices", row_limit=0)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM tenant_acme.invoices",
        "UPDATE tenant_acme.invoices SET amount = 0",
        "INSERT INTO tenant_acme.invoices VALUES (1)",
        "DROP TABLE tenant_acme.invoices",
        "TRUNCATE tenant_acme.invoices",
        "GRANT ALL ON tenant_acme.invoices TO PUBLIC",
        "CREATE TABLE tenant_acme.t (id int)",
    ],
)
def test_write_and_schema_statements_are_refused(sql: str) -> None:
    with pytest.raises(QueryNotAllowed, match="only SELECT and WITH"):
        parse(sql)


def test_a_second_statement_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="single statement"):
        parse("SELECT id FROM tenant_acme.invoices; DROP TABLE tenant_acme.invoices")


def test_a_write_hidden_after_a_line_comment_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="single statement"):
        parse("SELECT id FROM tenant_acme.invoices -- harmless\n; DELETE FROM tenant_acme.invoices")


def test_a_forbidden_keyword_inside_the_statement_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="forbidden keywords: update"):
        parse("SELECT id FROM tenant_acme.invoices FOR UPDATE")


def test_a_block_comment_cannot_hide_a_forbidden_keyword() -> None:
    """A keyword inside a comment is stripped, so the query is judged on what runs."""
    assert parse("SELECT id /* update */ FROM tenant_acme.invoices").startswith("SELECT id")


def test_a_string_literal_cannot_trigger_a_false_refusal() -> None:
    sql = "SELECT id FROM tenant_acme.invoices WHERE note = 'please drop this'"

    assert parse(sql).endswith("LIMIT 1000")


@pytest.mark.parametrize("function", ["pg_read_file", "pg_sleep", "lo_export", "dblink"])
def test_dangerous_functions_are_refused(function: str) -> None:
    with pytest.raises(QueryNotAllowed, match="forbidden functions"):
        parse(f"SELECT {function}('x') FROM tenant_acme.invoices")


def test_another_tenant_schema_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="outside the tenant schema: tenant_globex"):
        parse("SELECT id FROM tenant_globex.invoices")


def test_a_join_into_another_schema_is_refused() -> None:
    sql = "SELECT a.id FROM tenant_acme.invoices a JOIN pg_catalog.pg_user u ON true"

    with pytest.raises(QueryNotAllowed, match="outside the tenant schema: pg_catalog"):
        parse(sql)


def test_an_unqualified_table_is_refused() -> None:
    with pytest.raises(QueryNotAllowed, match="must be schema-qualified: invoices"):
        parse("SELECT id FROM invoices")


def test_a_subquery_in_the_from_clause_is_accepted() -> None:
    sql = "SELECT x FROM (SELECT id AS x FROM tenant_acme.invoices) s"

    assert parse(sql).endswith("LIMIT 1000")


def test_a_limit_within_the_cap_is_preserved() -> None:
    assert parse("SELECT id FROM tenant_acme.invoices LIMIT 10", row_limit=1_000) == (
        "SELECT id FROM tenant_acme.invoices LIMIT 10"
    )


def test_a_limit_above_the_cap_is_tightened() -> None:
    assert parse("SELECT id FROM tenant_acme.invoices LIMIT 50000", row_limit=100) == (
        "SELECT id FROM tenant_acme.invoices LIMIT 100"
    )


def test_the_parsed_query_reports_its_schema_and_limit() -> None:
    parsed = parse_analytical_query(
        "SELECT id FROM tenant_acme.invoices", schema=SCHEMA, row_limit=25
    )

    assert parsed.schema == SCHEMA
    assert parsed.row_limit == 25
    assert parsed.sql.endswith("LIMIT 25")
