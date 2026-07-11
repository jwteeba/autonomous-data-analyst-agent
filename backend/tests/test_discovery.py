"""Tests for app/nodes/discovery.py."""

from __future__ import annotations

import pytest

from app.nodes.discovery import _cache_key, get_sql_tool, invalidate_cache
from app.tools.sql_tool import SQLTool
from tests.conftest import StubDataSource, make_sample_df


class TestCacheKey:
    def test_file_key(self):
        key = _cache_key({"type": "file", "path": "/data/sales.csv"})
        assert key == "file:/data/sales.csv"

    def test_postgres_key_with_table(self):
        key = _cache_key(
            {
                "type": "postgres",
                "host": "localhost",
                "port": 5432,
                "database": "db",
                "table": "sales",
            }
        )
        assert "postgres:" in key
        assert "sales" in key

    def test_postgres_key_with_query(self):
        key = _cache_key(
            {
                "type": "postgres",
                "host": "h",
                "port": 5432,
                "database": "db",
                "query": "SELECT 1",
            }
        )
        assert "SELECT 1" in key

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown data source type"):
            _cache_key({"type": "s3"})


class TestGetSqlToolCache:
    def test_returns_sql_tool(self, tmp_csv):
        descriptor = {"type": "file", "path": str(tmp_csv)}
        invalidate_cache(descriptor)  # ensure clean state
        tool = get_sql_tool(descriptor)
        assert isinstance(tool, SQLTool)

    def test_returns_same_instance_on_second_call(self, tmp_csv):
        descriptor = {"type": "file", "path": str(tmp_csv)}
        invalidate_cache(descriptor)
        t1 = get_sql_tool(descriptor)
        t2 = get_sql_tool(descriptor)
        assert t1 is t2

    def test_invalidate_removes_from_cache(self, tmp_csv):
        descriptor = {"type": "file", "path": str(tmp_csv)}
        t1 = get_sql_tool(descriptor)
        invalidate_cache(descriptor)
        t2 = get_sql_tool(descriptor)
        assert t1 is not t2

    def test_invalidate_nonexistent_key_is_noop(self):
        descriptor = {"type": "file", "path": "/nonexistent/path.csv"}
        invalidate_cache(descriptor)  # must not raise


class TestDiscoveryNode:
    @pytest.mark.asyncio
    async def test_adds_schema_to_state(self, tmp_csv):
        from app.nodes.discovery import discovery_node

        descriptor = {"type": "file", "path": str(tmp_csv)}
        invalidate_cache(descriptor)
        state = {
            "question": "test",
            "dataset_source": descriptor,
            "dataset_name": "test",
            "trace": [],
            "retries": 0,
        }
        result = await discovery_node(state)
        assert "schema" in result
        assert "columns" in result["schema"]
        assert "row_count" in result["schema"]

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_csv):
        from app.nodes.discovery import discovery_node

        descriptor = {"type": "file", "path": str(tmp_csv)}
        invalidate_cache(descriptor)
        state = {
            "question": "test",
            "dataset_source": descriptor,
            "dataset_name": "test",
            "trace": [],
            "retries": 0,
        }
        result = await discovery_node(state)
        assert len(result["trace"]) == 1
        assert result["trace"][0]["node"] == "dataset_discovery_and_schema"
        assert result["trace"][0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_schema_row_count_matches_file(self, tmp_csv, sample_df):
        from app.nodes.discovery import discovery_node

        descriptor = {"type": "file", "path": str(tmp_csv)}
        invalidate_cache(descriptor)
        state = {
            "question": "test",
            "dataset_source": descriptor,
            "dataset_name": "test",
            "trace": [],
            "retries": 0,
        }
        result = await discovery_node(state)
        assert result["schema"]["row_count"] == len(sample_df)
