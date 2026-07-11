from __future__ import annotations
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.nodes.discovery import get_sql_tool
from app.state import AgentState


def _descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute per-column descriptive statistics for all numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        A dict mapping column name to a dict of mean, median, std, min, max, p25, p75.
    """
    numeric = df.select_dtypes(include=[np.number])
    out = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        out[col] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "p25": round(float(s.quantile(0.25)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
        }
    return out


def _correlation(df: pd.DataFrame) -> dict:
    """Compute a Pearson correlation matrix and flag notable pairs.

    Args:
        df: Input DataFrame.

    Returns:
        A dict with ``matrix`` (full correlation dict) and ``notable_pairs``
        (pairs with |r| >= 0.3, sorted by strength), or an empty dict if
        fewer than two numeric columns or five rows are available.
    """
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric.shape[1] < 2 or len(numeric) < 5:
        return {}
    corr = numeric.corr(numeric_only=True).round(3)
    # Flag the strongest non-trivial correlations for the insight generator
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.3:
                pairs.append({"a": a, "b": b, "r": float(r)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"matrix": corr.to_dict(), "notable_pairs": pairs[:8]}


def _trend_regression(df: pd.DataFrame, date_col: str, value_col: str) -> dict | None:
    """Fit an OLS linear trend on monthly aggregates and produce a 3-month forecast.

    Args:
        df: Input DataFrame containing the date and value columns.
        date_col: Name of the datetime column to use as the time axis.
        value_col: Name of the numeric column to aggregate and regress.

    Returns:
        A dict with monthly series, slope, R², p-value, significance flag, and
        3-month forecast, or None if there is insufficient data.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return None
    d = df[[date_col, value_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", format="mixed")
    d = d.dropna()
    if len(d) < 10:
        return None
    monthly = d.set_index(date_col)[value_col].resample("MS").sum()
    if len(monthly) < 4:
        return None

    x = np.arange(len(monthly))
    y = monthly.values
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

    # 95% CI on the slope
    t_crit = scipy_stats.t.ppf(0.975, df=len(x) - 2)
    slope_ci = (slope - t_crit * std_err, slope + t_crit * std_err)

    # simple 3-month forecast by extrapolating the linear trend
    future_x = np.arange(len(monthly), len(monthly) + 3)
    forecast_values = intercept + slope * future_x
    future_index = pd.date_range(
        monthly.index[-1] + pd.DateOffset(months=1), periods=3, freq="MS"
    )

    return {
        "monthly_series": {
            str(k.date()): round(float(v), 2) for k, v in monthly.items()
        },
        "slope_per_month": round(float(slope), 2),
        "slope_95ci": [round(float(slope_ci[0]), 2), round(float(slope_ci[1]), 2)],
        "r_squared": round(float(r_value**2), 3),
        "p_value": float(p_value),
        "statistically_significant_trend": bool(p_value < 0.05),
        "forecast_next_3_months": {
            str(k.date()): round(float(v), 2)
            for k, v in zip(future_index, forecast_values)
        },
        "method": "OLS linear trend on monthly aggregates (statsmodels/scipy linregress); "
        "adequate for a short horizon, not a substitute for seasonal models on longer series.",
    }


async def python_analyst_node(state: AgentState) -> AgentState:
    """Run statistical analysis (descriptive stats, correlation, trend/forecast) on the dataset.

    Args:
        state: Current agent state with ``dataset_source``, ``plan``, and ``question``.

    Returns:
        Updated agent state with ``python_result`` and an appended ``trace`` entry.
    """
    t0 = time.time()
    trace = state.get("trace", [])
    plan = state.get("plan", {})

    if not (plan.get("needs_stats") or plan.get("needs_forecast") or True):
        # descriptive stats are cheap and almost always useful; only skip
        # the heavier trend/forecast work if the planner said it's not needed
        trace.append({"node": "python_analyst", "duration_ms": 0, "status": "skipped"})
        return {**state, "trace": trace}

    tool = get_sql_tool(state["dataset_source"])
    df = tool.as_dataframe()

    result: dict = {"descriptive_stats": _descriptive_stats(df)}

    if plan.get("needs_stats", True):
        result["correlation"] = _correlation(df)

    if (
        plan.get("needs_forecast")
        or "revenue" in state["question"].lower()
        or "trend" in state["question"].lower()
    ):
        date_col = "order_date" if "order_date" in df.columns else None
        value_col = "revenue" if "revenue" in df.columns else None
        if date_col and value_col:
            trend = _trend_regression(df, date_col, value_col)
            if trend:
                result["trend_and_forecast"] = trend

    trace.append(
        {
            "node": "python_analyst",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok",
            "detail": f"computed stats over {len(df)} rows; "
            f"keys={list(result.keys())}",
        }
    )
    return {**state, "python_result": result, "trace": trace}
