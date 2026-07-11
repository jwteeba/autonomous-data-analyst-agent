"""Tests for app/nodes/insights.py, report_writer.py, and reviewer.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.nodes.insights import _fallback_insights
from app.nodes.discovery import invalidate_cache
from tests.conftest import make_sample_df


# Shared complete state builder
def _complete_state(tmp_path):
    df = make_sample_df()
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    descriptor = {"type": "file", "path": str(p)}
    invalidate_cache(descriptor)
    return {
        "question": "What is our revenue trend?",
        "dataset_source": descriptor,
        "dataset_name": "Test Dataset",
        "schema": {
            "columns": [{"name": c, "type": "VARCHAR"} for c in df.columns],
            "row_count": len(df),
        },
        "plan": {"needs_sql": True, "needs_stats": True, "needs_forecast": True},
        "sql_query": "SELECT * FROM dataset LIMIT 10",
        "sql_result": {
            "columns": list(df.columns),
            "rows": df.head(5).to_dict(orient="records"),
            "row_count": len(df),
        },
        "python_result": {
            "descriptive_stats": {
                "revenue": {
                    "mean": 2500.0,
                    "median": 2400.0,
                    "std": 800.0,
                    "min": 500.0,
                    "max": 5000.0,
                    "p25": 1800.0,
                    "p75": 3200.0,
                }
            },
            "correlation": {"matrix": {}, "notable_pairs": []},
        },
        "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        "charts": [],
        "insights": {
            "executive_summary": "Revenue is growing.",
            "key_findings": ["Revenue increased 10% MoM."],
            "risks": ["No major risks."],
            "opportunities": ["Expand to new regions."],
            "recommendations": ["Invest in top channel."],
            "confidence_level": "medium",
            "confidence_reason": "Based on R².",
        },
        "trace": [
            {"node": "discovery", "duration_ms": 10, "status": "ok", "detail": "ok"}
        ],
        "retries": 0,
    }


class TestFallbackInsights:
    def test_returns_required_keys(self):
        state = {
            "question": "test",
            "sql_result": {"rows": [{"revenue": 100}]},
            "python_result": {},
            "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        }
        result = _fallback_insights(state)
        required = {
            "executive_summary",
            "key_findings",
            "risks",
            "opportunities",
            "recommendations",
            "confidence_level",
            "confidence_reason",
        }
        assert required.issubset(result.keys())

    def test_includes_top_sql_row_in_findings(self):
        state = {
            "question": "test",
            "sql_result": {"rows": [{"region": "North", "total": 5000}]},
            "python_result": {},
            "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        }
        result = _fallback_insights(state)
        assert any("North" in f for f in result["key_findings"])

    def test_flags_duplicate_rows_as_risk(self):
        state = {
            "question": "test",
            "sql_result": {},
            "python_result": {},
            "cleaning_report": {"duplicate_rows": 5, "columns": {}},
        }
        result = _fallback_insights(state)
        assert any("duplicate" in r.lower() for r in result["risks"])

    def test_flags_negative_values_as_risk(self):
        state = {
            "question": "test",
            "sql_result": {},
            "python_result": {},
            "cleaning_report": {
                "duplicate_rows": 0,
                "columns": {"revenue": {"impossible_negative_values": 3}},
            },
        }
        result = _fallback_insights(state)
        assert any("negative" in r.lower() for r in result["risks"])

    def test_positive_trend_generates_opportunity(self):
        state = {
            "question": "test",
            "sql_result": {},
            "python_result": {
                "trend_and_forecast": {
                    "slope_per_month": 500.0,
                    "r_squared": 0.8,
                    "statistically_significant_trend": True,
                    "forecast_next_3_months": {},
                }
            },
            "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        }
        result = _fallback_insights(state)
        assert any("positive" in o.lower() for o in result["opportunities"])

    def test_confidence_medium_for_high_r_squared(self):
        state = {
            "question": "test",
            "sql_result": {},
            "python_result": {
                "trend_and_forecast": {
                    "slope_per_month": 100.0,
                    "r_squared": 0.9,
                    "statistically_significant_trend": True,
                    "forecast_next_3_months": {},
                }
            },
            "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        }
        result = _fallback_insights(state)
        assert result["confidence_level"] == "medium"

    def test_confidence_low_without_trend(self):
        state = {
            "question": "test",
            "sql_result": {},
            "python_result": {},
            "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        }
        result = _fallback_insights(state)
        assert result["confidence_level"] == "low"


class TestInsightsNode:
    @pytest.mark.asyncio
    async def test_returns_insights_in_state(self, tmp_path):
        from app.nodes.insights import insights_node

        state = _complete_state(tmp_path)
        result = await insights_node(state)
        assert "insights" in result
        assert "executive_summary" in result["insights"]

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.insights import insights_node

        state = _complete_state(tmp_path)
        result = await insights_node(state)
        assert any(t["node"] == "insight_generation" for t in result["trace"])

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_llm_json(self, tmp_path):
        from app.nodes.insights import insights_node
        import app.llm as llm_mod

        state = _complete_state(tmp_path)
        with patch.object(llm_mod.llm_client, "live", True), patch.object(
            llm_mod.llm_client, "complete", new=AsyncMock(return_value="not valid json")
        ):
            result = await insights_node(state)
        assert "executive_summary" in result["insights"]
        assert (
            "grounded fallback" in result["insights"]["executive_summary"].lower()
            or "fallback" in result["insights"]["executive_summary"].lower()
        )


class TestRenderMarkdown:
    def test_contains_question(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert state["question"] in md

    def test_contains_dataset_name(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert "Test Dataset" in md

    def test_contains_executive_summary(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert "Revenue is growing." in md

    def test_contains_sql_query(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert "SELECT * FROM dataset LIMIT 10" in md

    def test_contains_methodology_section(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert "## Methodology" in md

    def test_contains_data_quality_section(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert "## Data Quality" in md

    def test_no_duplicate_rows_shows_zero(self, tmp_path):
        from app.nodes.report_writer import _render_markdown

        state = _complete_state(tmp_path)
        md = _render_markdown(state)
        assert "Duplicate rows: 0" in md


class TestReportWriterNode:
    @pytest.mark.asyncio
    async def test_writes_file_to_disk(self, tmp_path):
        from app.nodes.report_writer import report_writer_node

        state = _complete_state(tmp_path)
        result = await report_writer_node(state)
        assert "report_path" in result
        assert Path(result["report_path"]).exists()

    @pytest.mark.asyncio
    async def test_report_markdown_in_state(self, tmp_path):
        from app.nodes.report_writer import report_writer_node

        state = _complete_state(tmp_path)
        result = await report_writer_node(state)
        assert "report_markdown" in result
        assert len(result["report_markdown"]) > 100

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.report_writer import report_writer_node

        state = _complete_state(tmp_path)
        result = await report_writer_node(state)
        assert any(t["node"] == "report_writer" for t in result["trace"])

    @pytest.mark.asyncio
    async def test_report_file_is_markdown(self, tmp_path):
        from app.nodes.report_writer import report_writer_node

        state = _complete_state(tmp_path)
        result = await report_writer_node(state)
        assert result["report_path"].endswith(".md")


class TestReviewerNode:
    @pytest.mark.asyncio
    async def test_no_issues_on_clean_state(self, tmp_path):
        from app.nodes.reviewer import reviewer_node

        state = _complete_state(tmp_path)
        # Add a real report path so reviewer doesn't flag it
        p = tmp_path / "report.md"
        p.write_text("# Report")
        state["report_path"] = str(p)
        result = await reviewer_node(state)
        trace = result["trace"][-1]
        assert trace["node"] == "reviewer"
        assert trace["status"] == "ok"

    @pytest.mark.asyncio
    async def test_flags_missing_report(self, tmp_path):
        from app.nodes.reviewer import reviewer_node

        state = _complete_state(tmp_path)
        state.pop("report_path", None)
        result = await reviewer_node(state)
        trace = result["trace"][-1]
        assert trace["status"] == "flagged"
        assert "Report was not generated" in trace["detail"]

    @pytest.mark.asyncio
    async def test_flags_sql_error(self, tmp_path):
        from app.nodes.reviewer import reviewer_node

        state = _complete_state(tmp_path)
        state["sql_result"] = {"error": "syntax error"}
        state["report_path"] = str(tmp_path / "r.md")
        (tmp_path / "r.md").write_text("x")
        result = await reviewer_node(state)
        trace = result["trace"][-1]
        assert trace["status"] == "flagged"
        assert "SQL execution error" in trace["detail"]

    @pytest.mark.asyncio
    async def test_flags_low_confidence(self, tmp_path):
        from app.nodes.reviewer import reviewer_node

        state = _complete_state(tmp_path)
        state["insights"]["confidence_level"] = "low"
        p = tmp_path / "r.md"
        p.write_text("x")
        state["report_path"] = str(p)
        result = await reviewer_node(state)
        trace = result["trace"][-1]
        assert trace["status"] == "flagged"
        assert "LOW" in trace["detail"]

    @pytest.mark.asyncio
    async def test_flags_missing_key_findings(self, tmp_path):
        from app.nodes.reviewer import reviewer_node

        state = _complete_state(tmp_path)
        state["insights"]["key_findings"] = []
        p = tmp_path / "r.md"
        p.write_text("x")
        state["report_path"] = str(p)
        result = await reviewer_node(state)
        trace = result["trace"][-1]
        assert trace["status"] == "flagged"

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.reviewer import reviewer_node

        state = _complete_state(tmp_path)
        p = tmp_path / "r.md"
        p.write_text("x")
        state["report_path"] = str(p)
        result = await reviewer_node(state)
        assert any(t["node"] == "reviewer" for t in result["trace"])
