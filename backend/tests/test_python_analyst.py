"""Tests for app/nodes/python_analyst.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.nodes.python_analyst import _correlation, _descriptive_stats, _trend_regression
from app.nodes.discovery import invalidate_cache
from tests.conftest import make_sample_df


class TestDescriptiveStats:
    def test_returns_dict_for_numeric_columns(self):
        df = make_sample_df()
        stats = _descriptive_stats(df)
        assert "revenue" in stats
        assert "quantity" in stats

    def test_stat_keys_present(self):
        df = make_sample_df()
        stats = _descriptive_stats(df)
        for col_stats in stats.values():
            assert set(col_stats.keys()) == {
                "mean",
                "median",
                "std",
                "min",
                "max",
                "p25",
                "p75",
            }

    def test_skips_non_numeric_columns(self):
        df = make_sample_df()
        stats = _descriptive_stats(df)
        assert "region" not in stats
        assert "category" not in stats

    def test_skips_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
        stats = _descriptive_stats(df)
        assert "a" not in stats
        assert "b" in stats

    def test_values_are_rounded(self):
        df = pd.DataFrame({"x": [1.123456789, 2.987654321]})
        stats = _descriptive_stats(df)
        assert stats["x"]["mean"] == round(stats["x"]["mean"], 2)


class TestCorrelation:
    def test_returns_empty_for_single_numeric_column(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x"] * 5})
        assert _correlation(df) == {}

    def test_returns_empty_for_fewer_than_5_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert _correlation(df) == {}

    def test_returns_matrix_and_notable_pairs(self):
        df = make_sample_df()
        result = _correlation(df)
        assert "matrix" in result
        assert "notable_pairs" in result

    def test_notable_pairs_have_required_keys(self):
        df = make_sample_df()
        result = _correlation(df)
        for pair in result.get("notable_pairs", []):
            assert {"a", "b", "r"} == set(pair.keys())

    def test_notable_pairs_sorted_by_abs_r(self):
        df = make_sample_df()
        pairs = _correlation(df).get("notable_pairs", [])
        rs = [abs(p["r"]) for p in pairs]
        assert rs == sorted(rs, reverse=True)

    def test_perfect_correlation_detected(self):
        df = pd.DataFrame(
            {
                "x": [float(i) for i in range(20)],
                "y": [float(i) * 2 for i in range(20)],
            }
        )
        result = _correlation(df)
        assert any(abs(p["r"]) > 0.99 for p in result.get("notable_pairs", []))


class TestTrendRegression:
    def test_returns_none_for_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _trend_regression(df, "order_date", "revenue") is None

    def test_returns_none_for_too_few_rows(self):
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2022-01-01", periods=5, freq="MS").astype(
                    str
                ),
                "revenue": [100.0] * 5,
            }
        )
        assert _trend_regression(df, "order_date", "revenue") is None

    def test_returns_none_for_too_few_months(self):
        # 10 rows but all in the same month → only 1 monthly bucket
        df = pd.DataFrame(
            {
                "order_date": ["2022-01-15"] * 10,
                "revenue": [100.0] * 10,
            }
        )
        assert _trend_regression(df, "order_date", "revenue") is None

    def test_returns_dict_with_required_keys(self):
        df = make_sample_df()
        result = _trend_regression(df, "order_date", "revenue")
        assert result is not None
        required = {
            "monthly_series",
            "slope_per_month",
            "slope_95ci",
            "r_squared",
            "p_value",
            "statistically_significant_trend",
            "forecast_next_3_months",
            "method",
        }
        assert required.issubset(result.keys())

    def test_forecast_has_3_entries(self):
        df = make_sample_df()
        result = _trend_regression(df, "order_date", "revenue")
        assert result is not None
        assert len(result["forecast_next_3_months"]) == 3

    def test_r_squared_between_0_and_1(self):
        df = make_sample_df()
        result = _trend_regression(df, "order_date", "revenue")
        assert result is not None
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_significance_flag_is_bool(self):
        df = make_sample_df()
        result = _trend_regression(df, "order_date", "revenue")
        assert result is not None
        assert isinstance(result["statistically_significant_trend"], bool)

    def test_positive_trend_detected(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        revenues = [float(i * 1000) for i in range(1, 25)]  # strictly increasing
        df = pd.DataFrame({"order_date": dates.astype(str), "revenue": revenues})
        result = _trend_regression(df, "order_date", "revenue")
        assert result is not None
        assert result["slope_per_month"] > 0


class TestPythonAnalystNode:
    def _make_state(self, df, tmp_path, question="What is our revenue trend?"):
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        descriptor = {"type": "file", "path": str(p)}
        invalidate_cache(descriptor)
        return {
            "question": question,
            "dataset_source": descriptor,
            "dataset_name": "test",
            "plan": {"needs_sql": True, "needs_stats": True, "needs_forecast": True},
            "trace": [],
            "retries": 0,
        }

    @pytest.mark.asyncio
    async def test_returns_python_result(self, tmp_path):
        from app.nodes.python_analyst import python_analyst_node

        state = self._make_state(make_sample_df(), tmp_path)
        result = await python_analyst_node(state)
        assert "python_result" in result
        assert "descriptive_stats" in result["python_result"]

    @pytest.mark.asyncio
    async def test_includes_correlation_when_needs_stats(self, tmp_path):
        from app.nodes.python_analyst import python_analyst_node

        state = self._make_state(make_sample_df(), tmp_path)
        result = await python_analyst_node(state)
        assert "correlation" in result["python_result"]

    @pytest.mark.asyncio
    async def test_includes_trend_for_revenue_question(self, tmp_path):
        from app.nodes.python_analyst import python_analyst_node

        state = self._make_state(make_sample_df(), tmp_path, question="revenue trend")
        result = await python_analyst_node(state)
        assert "trend_and_forecast" in result["python_result"]

    @pytest.mark.asyncio
    async def test_appends_trace_entry(self, tmp_path):
        from app.nodes.python_analyst import python_analyst_node

        state = self._make_state(make_sample_df(), tmp_path)
        result = await python_analyst_node(state)
        nodes = [t["node"] for t in result["trace"]]
        assert "python_analyst" in nodes
