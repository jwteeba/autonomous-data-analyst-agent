"""Tests for app/tools/datasource.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.tools.datasource import (
    FileDataSource,
    PostgresDataSource,
    _assert_readonly_sql,
    build_data_source,
)


class TestAssertReadonlySql:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM t",
            "select id from users",
            "  SELECT 1",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "with cte as (select 1) select * from cte",
        ],
    )
    def test_allows_select_and_cte(self, sql):
        _assert_readonly_sql(sql)  # must not raise

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM users",
            "UPDATE users SET name='x'",
            "DROP TABLE users",
            "ALTER TABLE t ADD COLUMN c INT",
            "INSERT INTO t VALUES (1)",
            "TRUNCATE TABLE t",
            "GRANT SELECT ON t TO u",
            "REVOKE SELECT ON t FROM u",
            "ATTACH 'db.duckdb'",
        ],
    )
    def test_blocks_write_keywords(self, sql):
        with pytest.raises(ValueError, match="non-read-only"):
            _assert_readonly_sql(sql)

    def test_blocks_non_select_start(self):
        with pytest.raises(ValueError, match="SELECT / WITH"):
            _assert_readonly_sql("EXPLAIN SELECT 1")

    def test_blocks_copy_to(self):
        with pytest.raises(ValueError):
            _assert_readonly_sql("COPY t TO '/tmp/out.csv'")


class TestFileDataSource:
    def test_load_csv(self, tmp_csv):
        src = FileDataSource(str(tmp_csv))
        df = src.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "revenue" in df.columns

    def test_load_json(self, tmp_json):
        src = FileDataSource(str(tmp_json))
        df = src.load()
        assert isinstance(df, pd.DataFrame)
        assert "revenue" in df.columns

    def test_load_parquet(self, tmp_parquet):
        src = FileDataSource(str(tmp_parquet))
        df = src.load()
        assert isinstance(df, pd.DataFrame)
        assert "revenue" in df.columns

    def test_load_unsupported_raises(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        src = FileDataSource(str(p))
        with pytest.raises(ValueError, match="Unsupported"):
            src.load()

    def test_describe_returns_type_and_path(self, tmp_csv):
        src = FileDataSource(str(tmp_csv))
        d = src.describe()
        assert d["type"] == "file"
        assert d["path"] == str(tmp_csv)


class TestPostgresDataSourceInit:
    def test_requires_table_or_query(self):
        with pytest.raises(ValueError, match="Provide either"):
            PostgresDataSource(
                host="h", port=5432, database="db", user="u", password="p"
            )

    def test_rejects_both_table_and_query(self):
        with pytest.raises(ValueError, match="only one"):
            PostgresDataSource(
                host="h",
                port=5432,
                database="db",
                user="u",
                password="p",
                table="t",
                query="SELECT 1",
            )

    def test_engine_url_with_table(self):
        src = PostgresDataSource(
            host="localhost",
            port=5432,
            database="db",
            user="usr",
            password="pw",
            table="t",
        )
        url = src._engine_url()
        assert "postgresql+psycopg2" in url
        assert "localhost" in url
        assert "db" in url

    def test_engine_url_encodes_special_chars(self):
        src = PostgresDataSource(
            host="h", port=5432, database="db", user="u@me", password="p@ss!", table="t"
        )
        url = src._engine_url()
        assert "@" not in url.split("@")[0].split("://")[1]  # user part is encoded

    def test_select_sql_uses_table(self):
        src = PostgresDataSource(
            host="h", port=5432, database="db", user="u", password="p", table="sales"
        )
        assert 'SELECT * FROM "public"."sales"' == src._select_sql()

    def test_select_sql_uses_custom_schema(self):
        src = PostgresDataSource(
            host="h",
            port=5432,
            database="db",
            user="u",
            password="p",
            table="t",
            db_schema="myschema",
        )
        assert '"myschema"."t"' in src._select_sql()

    def test_select_sql_uses_query(self):
        q = "SELECT id FROM t WHERE active = true"
        src = PostgresDataSource(
            host="h", port=5432, database="db", user="u", password="p", query=q
        )
        assert src._select_sql() == q

    def test_connect_args_has_sslmode(self):
        src = PostgresDataSource(
            host="h", port=5432, database="db", user="u", password="p", table="t"
        )
        args = src._connect_args()
        assert args["sslmode"] == "require"
        assert "connect_timeout" in args

    def test_describe_omits_password(self):
        src = PostgresDataSource(
            host="h", port=5432, database="db", user="u", password="secret", table="t"
        )
        d = src.describe()
        assert "password" not in d
        assert d["type"] == "postgres"
        assert d["host"] == "h"
        assert d["table"] == "t"

    def test_load_blocks_write_query(self):
        src = PostgresDataSource(
            host="h",
            port=5432,
            database="db",
            user="u",
            password="p",
            query="DELETE FROM t",
        )
        with pytest.raises(ValueError):
            src.load()

    def test_test_connection_propagates_error(self):
        src = PostgresDataSource(
            host="bad-host", port=9999, database="db", user="u", password="p", table="t"
        )
        with pytest.raises(Exception):
            src.test_connection()


class TestBuildDataSource:
    def test_builds_file_source(self, tmp_csv):
        src = build_data_source({"type": "file", "path": str(tmp_csv)})
        assert isinstance(src, FileDataSource)

    def test_builds_postgres_source(self):
        src = build_data_source(
            {
                "type": "postgres",
                "host": "h",
                "port": 5432,
                "database": "db",
                "user": "u",
                "password": "p",
                "table": "t",
            }
        )
        assert isinstance(src, PostgresDataSource)

    def test_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown data source type"):
            build_data_source({"type": "s3"})

    def test_postgres_with_query(self):
        src = build_data_source(
            {
                "type": "postgres",
                "host": "h",
                "port": 5432,
                "database": "db",
                "user": "u",
                "password": "p",
                "query": "SELECT 1",
            }
        )
        assert isinstance(src, PostgresDataSource)
        assert src.query == "SELECT 1"
