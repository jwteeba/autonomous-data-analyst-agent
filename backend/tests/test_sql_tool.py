"""Tests for app/tools/sql_tool.py."""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.sql_tool import SQLTool, SQLValidationError
from tests.conftest import StubDataSource, make_sample_df


# Helpers
def make_tool(df: pd.DataFrame | None = None) -> SQLTool:
    return SQLTool(StubDataSource(df))


def test_sql_validation_error_is_exception():
    err = SQLValidationError("bad query")
    assert isinstance(err, Exception)
    assert "bad query" in str(err)


class TestSQLToolValidate:
    def test_empty_query_raises(self):
        tool = make_tool()
        with pytest.raises(SQLValidationError, match="Empty"):
            tool.validate("")

    def test_whitespace_only_raises(self):
        tool = make_tool()
        with pytest.raises(SQLValidationError, match="Empty"):
            tool.validate("   ")

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM dataset",
            "UPDATE dataset SET x=1",
            "DROP TABLE dataset",
            "ALTER TABLE dataset ADD COLUMN c INT",
            "INSERT INTO dataset VALUES (1)",
            "TRUNCATE TABLE dataset",
            "ATTACH 'other.db'",
        ],
    )
    def test_destructive_statements_blocked(self, sql):
        tool = make_tool()
        with pytest.raises(SQLValidationError, match="blocked"):
            tool.validate(sql)

    def test_non_select_start_blocked(self):
        tool = make_tool()
        with pytest.raises(SQLValidationError, match="SELECT / WITH"):
            tool.validate("EXPLAIN SELECT 1")

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM dataset",
            "select count(*) from dataset",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "  SELECT revenue FROM dataset LIMIT 10",
        ],
    )
    def test_valid_queries_pass(self, sql):
        tool = make_tool()
        tool.validate(sql)  # must not raise


class TestSQLToolRun:
    def test_run_returns_expected_keys(self):
        tool = make_tool()
        result = tool.run("SELECT * FROM dataset LIMIT 5")
        assert set(result.keys()) == {
            "columns",
            "rows",
            "row_count",
            "execution_ms",
            "truncated",
        }

    def test_run_returns_correct_columns(self):
        tool = make_tool()
        result = tool.run("SELECT revenue, region FROM dataset LIMIT 1")
        assert "revenue" in result["columns"]
        assert "region" in result["columns"]

    def test_run_row_count_matches(self):
        df = make_sample_df().head(10)
        tool = make_tool(df)
        result = tool.run("SELECT * FROM dataset")
        assert result["row_count"] == 10

    def test_run_truncates_at_500(self):
        import numpy as np

        rng = np.random.default_rng(0)
        big_df = pd.DataFrame({"x": rng.integers(0, 100, size=600)})
        tool = make_tool(big_df)
        result = tool.run("SELECT * FROM dataset")
        assert result["row_count"] == 600
        assert len(result["rows"]) == 500
        assert result["truncated"] is True

    def test_run_not_truncated_flag_when_small(self):
        tool = make_tool(make_sample_df().head(5))
        result = tool.run("SELECT * FROM dataset")
        assert result["truncated"] is False

    def test_run_invalid_sql_raises_validation_error(self):
        tool = make_tool()
        with pytest.raises(SQLValidationError, match="execution failed"):
            tool.run("SELECT nonexistent_col FROM dataset")

    def test_run_blocked_query_raises(self):
        tool = make_tool()
        with pytest.raises(SQLValidationError):
            tool.run("DELETE FROM dataset")

    def test_run_aggregation(self):
        tool = make_tool()
        result = tool.run(
            "SELECT region, SUM(revenue) AS total FROM dataset GROUP BY region"
        )
        assert result["row_count"] > 0
        assert "total" in result["columns"]

    def test_run_cte(self):
        tool = make_tool()
        result = tool.run(
            "WITH top AS (SELECT * FROM dataset LIMIT 3) SELECT * FROM top"
        )
        assert result["row_count"] == 3

    def test_execution_ms_is_non_negative(self):
        tool = make_tool()
        result = tool.run("SELECT 1 AS n")
        assert result["execution_ms"] >= 0


class TestSQLToolSchema:
    def test_schema_has_required_keys(self):
        tool = make_tool()
        s = tool.schema()
        assert "columns" in s
        assert "row_count" in s

    def test_schema_row_count_correct(self):
        df = make_sample_df().head(7)
        tool = make_tool(df)
        assert tool.schema()["row_count"] == 7

    def test_schema_columns_have_name_and_type(self):
        tool = make_tool()
        for col in tool.schema()["columns"]:
            assert "name" in col
            assert "type" in col

    def test_schema_lists_all_columns(self):
        tool = make_tool()
        names = {c["name"] for c in tool.schema()["columns"]}
        assert {"revenue", "region", "category", "channel", "quantity"}.issubset(names)


class TestSQLToolAsDataframe:
    def test_returns_dataframe(self):
        tool = make_tool()
        df = tool.as_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_source(self):
        source_df = make_sample_df()
        tool = make_tool(source_df)
        assert len(tool.as_dataframe()) == len(source_df)

    def test_columns_match_source(self):
        source_df = make_sample_df()
        tool = make_tool(source_df)
        assert set(tool.as_dataframe().columns) == set(source_df.columns)


class TestNormalizeDateColumns:
    def test_no_op_when_dates_already_valid(self):
        tool = make_tool()  # order_date is ISO strings — all parseable
        repaired = tool.normalize_date_columns(["order_date"])
        assert repaired == {}

    def test_repairs_mixed_format_dates(self):
        df = pd.DataFrame(
            {
                "order_date": ["2022-01-01", "02/03/2022", "2022-04-01"],
                "revenue": [100.0, 200.0, 300.0],
            }
        )
        tool = make_tool(df)
        repaired = tool.normalize_date_columns(["order_date"])
        # At least one row had a non-ISO format
        assert "order_date" in repaired
        assert repaired["order_date"] >= 1

    def test_skips_columns_not_in_list(self):
        tool = make_tool()
        repaired = tool.normalize_date_columns([])
        assert repaired == {}

    def test_ignores_nonexistent_column_gracefully(self):
        # DuckDB will raise; the method should propagate — document this behaviour
        tool = make_tool()
        with pytest.raises(Exception):
            tool.normalize_date_columns(["no_such_col"])
