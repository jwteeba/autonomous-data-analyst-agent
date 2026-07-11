"""Tests for app/nodes/visualization.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.nodes.visualization import (
    _chart_bar,
    _chart_correlation_heatmap,
    _chart_trend_with_forecast,
    _save,
)
from app.nodes.discovery import invalidate_cache
from tests.conftest import make_sample_df


class TestSave:
    def test_returns_string_path(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        path = _save(fig, "test_chart")
        assert isinstance(path, str)
        assert path.endswith(".png")

    def test_file_exists_on_disk(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        path = _save(fig, "test_exists")
        assert Path(path).exists()

    def test_name_hint_appears_in_filename(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        path = _save(fig, "my_hint")
        assert "my_hint" in Path(path).name


class TestChartBar:
    def test_returns_none_when_dim_missing(self):
        df = make_sample_df()
        result = _chart_bar(df, "nonexistent_dim", "revenue")
        assert result is None

    def test_returns_none_when_value_col_missing(self):
        df = make_sample_df()
        result = _chart_bar(df, "region", "nonexistent_value")
        assert result is None

    def test_returns_none_for_empty_aggregation(self):
        df = pd.DataFrame({"region": [], "revenue": []})
        result = _chart_bar(df, "region", "revenue")
        assert result is None

    def test_returns_chart_descriptor(self):
        df = make_sample_df()
        result = _chart_bar(df, "region", "revenue")
        assert result is not None
        assert set(result.keys()) == {
            "title",
            "type",
            "path",
            "caption",
            "business_explanation",
        }

    def test_type_is_bar_chart(self):
        df = make_sample_df()
        result = _chart_bar(df, "region", "revenue")
        assert result["type"] == "bar_chart"

    def test_file_written_to_disk(self):
        df = make_sample_df()
        result = _chart_bar(df, "region", "revenue")
        assert result is not None
        assert Path(result["path"]).exists()

    def test_title_contains_dim_and_value(self):
        df = make_sample_df()
        result = _chart_bar(df, "category", "revenue")
        assert result is not None
        assert "Category" in result["title"]
        assert "Revenue" in result["title"]


class TestChartCorrelationHeatmap:
    def test_returns_none_for_empty_matrix(self):
        assert _chart_correlation_heatmap({}) is None

    def test_returns_none_for_single_variable(self):
        matrix = {"x": {"x": 1.0}}
        assert _chart_correlation_heatmap(matrix) is None

    def test_returns_chart_descriptor(self):
        df = make_sample_df()[["revenue", "quantity", "unit_price"]]
        matrix = df.corr().to_dict()
        result = _chart_correlation_heatmap(matrix)
        assert result is not None
        assert result["type"] == "heatmap"

    def test_file_written_to_disk(self):
        df = make_sample_df()[["revenue", "quantity", "unit_price"]]
        matrix = df.corr().to_dict()
        result = _chart_correlation_heatmap(matrix)
        assert result is not None
        assert Path(result["path"]).exists()


class TestChartTrendWithForecast:
    def _make_trend(self):
        months = pd.date_range("2022-01-01", periods=12, freq="MS")
        return {
            "monthly_series": {
                str(m.date()): float(i * 1000) for i, m in enumerate(months, 1)
            },
            "forecast_next_3_months": {
                "2023-01-01": 13000.0,
                "2023-02-01": 14000.0,
                "2023-03-01": 15000.0,
            },
            "r_squared": 0.95,
            "slope_per_month": 1000.0,
            "statistically_significant_trend": True,
        }

    def test_returns_chart_descriptor(self):
        result = _chart_trend_with_forecast(self._make_trend())
        assert set(result.keys()) == {
            "title",
            "type",
            "path",
            "caption",
            "business_explanation",
        }

    def test_type_is_line_chart(self):
        result = _chart_trend_with_forecast(self._make_trend())
        assert result["type"] == "line_chart"

    def test_file_written_to_disk(self):
        result = _chart_trend_with_forecast(self._make_trend())
        assert Path(result["path"]).exists()

    def test_caption_contains_r_squared(self):
        result = _chart_trend_with_forecast(self._make_trend())
        assert (
            "R²" in result["caption"]
            or "R2" in result["caption"]
            or "0.95" in result["caption"]
        )

    def test_significant_trend_in_caption(self):
        result = _chart_trend_with_forecast(self._make_trend())
        assert "significant" in result["caption"].lower()


class TestVisualizationNode:
    def _make_state(self, tmp_path, python_result=None):
        df = make_sample_df()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        descriptor = {"type": "file", "path": str(p)}
        invalidate_cache(descriptor)
        return {
            "question": "revenue trend",
            "dataset_source": descriptor,
            "dataset_name": "test",
            "python_result": python_result or {},
            "trace": [],
            "retries": 0,
        }

    @pytest.mark.asyncio
    async def test_returns_charts_list(self, tmp_path):
        from app.nodes.visualization import visualization_node

        state = self._make_state(tmp_path)
        result = await visualization_node(state)
        assert "charts" in result
        assert isinstance(result["charts"], list)

    @pytest.mark.asyncio
    async def test_generates_bar_charts_for_revenue_data(self, tmp_path):
        from app.nodes.visualization import visualization_node

        state = self._make_state(tmp_path)
        result = await visualization_node(state)
        types = [c["type"] for c in result["charts"]]
        assert "bar_chart" in types

    @pytest.mark.asyncio
    async def test_generates_trend_chart_when_trend_present(self, tmp_path):
        from app.nodes.visualization import visualization_node
        from app.nodes.python_analyst import _trend_regression

        df = make_sample_df()
        trend = _trend_regression(df, "order_date", "revenue")
        state = self._make_state(tmp_path, python_result={"trend_and_forecast": trend})
        result = await visualization_node(state)
        types = [c["type"] for c in result["charts"]]
        assert "line_chart" in types

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.visualization import visualization_node

        state = self._make_state(tmp_path)
        result = await visualization_node(state)
        assert any(t["node"] == "visualization" for t in result["trace"])

    @pytest.mark.asyncio
    async def test_all_chart_files_exist(self, tmp_path):
        from app.nodes.visualization import visualization_node

        state = self._make_state(tmp_path)
        result = await visualization_node(state)
        for chart in result["charts"]:
            assert Path(chart["path"]).exists(), f"Missing chart: {chart['path']}"
