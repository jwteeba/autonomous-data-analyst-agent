"""Tests for app/nodes/planner.py and app/nodes/sql_agent.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.nodes.discovery import invalidate_cache
from app.nodes.sql_agent import _heuristic_query
from tests.conftest import make_sample_df


def _make_state(tmp_path, question="What is our revenue trend?", plan=None):
    df = make_sample_df()
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    descriptor = {"type": "file", "path": str(p)}
    invalidate_cache(descriptor)
    columns = list(df.columns)
    return {
        "question": question,
        "dataset_source": descriptor,
        "dataset_name": "test",
        "schema": {
            "columns": [{"name": c, "type": "VARCHAR"} for c in columns],
            "row_count": len(df),
        },
        "plan": plan
        or {"needs_sql": True, "needs_stats": True, "needs_forecast": False},
        "trace": [],
        "retries": 0,
    }


class TestPlannerNode:
    @pytest.mark.asyncio
    async def test_returns_plan_in_state(self, tmp_path):
        from app.nodes.planner import planner_node

        state = _make_state(tmp_path)
        result = await planner_node(state)
        assert "plan" in result
        plan = result["plan"]
        assert "needs_sql" in plan
        assert "needs_forecast" in plan

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.planner import planner_node

        state = _make_state(tmp_path)
        result = await planner_node(state)
        assert any(t["node"] == "planner" for t in result["trace"])

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json_from_llm(self, tmp_path):
        from app.nodes.planner import planner_node
        import app.llm as llm_mod

        state = _make_state(tmp_path, question="forecast next quarter")
        with patch.object(
            llm_mod.llm_client, "complete", new=AsyncMock(return_value="not json")
        ):
            result = await planner_node(state)
        plan = result["plan"]
        assert plan["needs_sql"] is True
        assert "forecast" in plan["reasoning"].lower() or plan["needs_forecast"] is True

    @pytest.mark.asyncio
    async def test_forecast_keyword_sets_needs_forecast(self, tmp_path):
        from app.nodes.planner import planner_node

        state = _make_state(tmp_path, question="forecast next quarter revenue")
        result = await planner_node(state)
        assert result["plan"]["needs_forecast"] is True


class TestHeuristicQuery:
    COLS = [
        "order_date",
        "region",
        "category",
        "channel",
        "quantity",
        "unit_price",
        "revenue",
    ]

    def test_trend_question_returns_monthly_aggregation(self):
        sql, reason = _heuristic_query(
            "What is the revenue trend over time?", self.COLS
        )
        assert "date_trunc" in sql.lower() or "month" in sql.lower()
        assert "heuristic" in reason

    def test_region_question_groups_by_region(self):
        sql, reason = _heuristic_query("Revenue by region", self.COLS)
        assert "region" in sql.lower()
        assert "GROUP BY" in sql

    def test_category_question_groups_by_category(self):
        sql, reason = _heuristic_query("revenue by category", self.COLS)
        assert "category" in sql.lower()
        assert "GROUP BY" in sql

    def test_channel_question_groups_by_channel(self):
        sql, reason = _heuristic_query("Sales by channel", self.COLS)
        assert "channel" in sql.lower()
        assert "GROUP BY" in sql

    def test_generic_question_returns_top_rows(self):
        sql, reason = _heuristic_query("Show me the data", self.COLS)
        assert "SELECT" in sql.upper()
        assert "LIMIT" in sql.upper()

    def test_no_value_col_returns_generic_sample(self):
        sql, reason = _heuristic_query("Show me the data", ["id", "name"])
        assert "SELECT * FROM dataset LIMIT" in sql

    def test_all_queries_are_select(self):
        questions = [
            "trend over time",
            "by region",
            "by category",
            "by channel",
            "generic question",
            "forecast next quarter",
        ]
        for q in questions:
            sql, _ = _heuristic_query(q, self.COLS)
            assert sql.strip().upper().startswith(
                "SELECT"
            ) or sql.strip().upper().startswith("WITH")


class TestSqlAgentNode:
    @pytest.mark.asyncio
    async def test_returns_sql_query_and_result(self, tmp_path):
        from app.nodes.sql_agent import sql_agent_node

        state = _make_state(tmp_path)
        result = await sql_agent_node(state)
        assert "sql_query" in result
        assert "sql_result" in result
        assert isinstance(result["sql_query"], str)

    @pytest.mark.asyncio
    async def test_result_has_expected_keys(self, tmp_path):
        from app.nodes.sql_agent import sql_agent_node

        state = _make_state(tmp_path)
        result = await sql_agent_node(state)
        sql_result = result["sql_result"]
        assert "columns" in sql_result
        assert "rows" in sql_result
        assert "row_count" in sql_result

    @pytest.mark.asyncio
    async def test_skips_when_needs_sql_false(self, tmp_path):
        from app.nodes.sql_agent import sql_agent_node

        state = _make_state(tmp_path, plan={"needs_sql": False})
        result = await sql_agent_node(state)
        assert any(t["status"] == "skipped" for t in result["trace"])
        assert "sql_query" not in result

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.sql_agent import sql_agent_node

        state = _make_state(tmp_path)
        result = await sql_agent_node(state)
        assert any(t["node"] == "sql_agent" for t in result["trace"])

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_sql_gracefully(self, tmp_path):
        from app.nodes.sql_agent import sql_agent_node
        import app.llm as llm_mod

        state = _make_state(tmp_path)
        with patch.object(llm_mod.llm_client, "live", True), patch.object(
            llm_mod.llm_client,
            "complete",
            new=AsyncMock(return_value="DELETE FROM dataset"),
        ):
            result = await sql_agent_node(state)
        # Should record an error, not raise
        assert result["sql_result"].get("error") is not None
        assert any(t["status"] == "error" for t in result["trace"])
