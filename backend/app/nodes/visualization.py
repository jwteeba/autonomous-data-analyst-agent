from __future__ import annotations
import time
import uuid
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import pandas as pd

from app.state import AgentState
from app.nodes.discovery import get_sql_tool

matplotlib.use("Agg")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name_hint: str) -> str:
    """Save a matplotlib figure to the charts output directory.

    Args:
        fig: The matplotlib Figure to save.
        name_hint: A short prefix used in the output filename.

    Returns:
        The absolute path of the saved PNG file as a string.
    """
    fname = f"{name_hint}_{uuid.uuid4().hex[:8]}.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_trend_with_forecast(trend: dict) -> dict:
    """Render a line chart of monthly revenue with a 3-month linear forecast overlay.

    Args:
        trend: The ``trend_and_forecast`` dict produced by ``_trend_regression``.

    Returns:
        A chart descriptor dict with keys ``title``, ``type``, ``path``, ``caption``,
        and ``business_explanation``.
    """
    series = trend["monthly_series"]
    forecast = trend.get("forecast_next_3_months", {})
    x_hist = list(series.keys())
    y_hist = list(series.values())
    x_fc = list(forecast.keys())
    y_fc = list(forecast.values())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x_hist, y_hist, marker="o", color="#2563eb", label="Actual monthly revenue")
    if x_fc:
        ax.plot(
            [x_hist[-1]] + x_fc,
            [y_hist[-1]] + y_fc,
            marker="o",
            linestyle="--",
            color="#f97316",
            label="Linear trend forecast",
        )
    ax.set_title("Monthly Revenue: Actual vs. Trend Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.legend()
    step = max(1, len(x_hist + x_fc) // 10)
    ax.set_xticks(range(0, len(x_hist + x_fc), step))
    ax.set_xticklabels((x_hist + x_fc)[::step], rotation=45, ha="right")
    fig.tight_layout()

    sig = trend.get("statistically_significant_trend")
    caption = (
        f"R²={trend['r_squared']}, slope={trend['slope_per_month']}/month "
        f"({'statistically significant, p<0.05' if sig else 'not statistically significant at p<0.05'})."
    )
    return {
        "title": "Monthly Revenue Trend & 3-Month Forecast",
        "type": "line_chart",
        "path": _save(fig, "trend_forecast"),
        "caption": caption,
        "business_explanation": "Shows how revenue has moved month over month and "
        "extrapolates the linear trend three months forward.",
    }


def _chart_bar(df: pd.DataFrame, dim: str, value_col: str) -> dict | None:
    """Render a bar chart of total value grouped by a categorical dimension.

    Args:
        df: Input DataFrame.
        dim: Name of the categorical column to group by.
        value_col: Name of the numeric column to sum.

    Returns:
        A chart descriptor dict, or None if the required columns are absent or the
        aggregation is empty.
    """
    if dim not in df.columns or value_col not in df.columns:
        return None
    agg = df.groupby(dim)[value_col].sum().sort_values(ascending=False)
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(agg.index.astype(str), agg.values, color="#2563eb")
    ax.set_title(f"Total {value_col.title()} by {dim.title()}")
    ax.set_xlabel(dim.title())
    ax.set_ylabel(f"Total {value_col.title()}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    top = agg.index[0]
    return {
        "title": f"Total {value_col.title()} by {dim.title()}",
        "type": "bar_chart",
        "path": _save(fig, f"bar_{dim}"),
        "caption": f"'{top}' leads with {agg.iloc[0]:,.0f} in total {value_col}.",
        "business_explanation": f"Compares total {value_col} across each {dim} to surface where value concentrates.",
    }


def _chart_correlation_heatmap(corr_matrix: dict) -> dict | None:
    """Render a correlation matrix as a colour-mapped heatmap.

    Args:
        corr_matrix: A dict-of-dicts correlation matrix (as returned by
            ``pd.DataFrame.corr().to_dict()``).

    Returns:
        A chart descriptor dict, or None if the matrix is empty or has fewer
        than two variables.
    """
    if not corr_matrix:
        return None
    corr_df = pd.DataFrame(corr_matrix)
    if corr_df.shape[0] < 2:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.columns)
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_df.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix (numeric columns)")
    fig.tight_layout()
    return {
        "title": "Correlation Matrix",
        "type": "heatmap",
        "path": _save(fig, "correlation_heatmap"),
        "caption": "Pearson correlation coefficients between numeric fields.",
        "business_explanation": "Identifies which numeric metrics move together, "
        "useful for spotting redundant metrics or leading indicators.",
    }


async def visualization_node(state: AgentState) -> AgentState:
    """Generate charts from statistical and SQL results and attach them to state.

    Args:
        state: Current agent state with ``python_result`` and ``dataset_source``.

    Returns:
        Updated agent state with ``charts`` and an appended ``trace`` entry.
    """
    t0 = time.time()
    trace = state.get("trace", [])
    charts: list[dict] = []

    py = state.get("python_result", {}) or {}

    df = get_sql_tool(state["dataset_source"]).as_dataframe()

    if "trend_and_forecast" in py:
        charts.append(_chart_trend_with_forecast(py["trend_and_forecast"]))

    value_col = "revenue" if "revenue" in df.columns else None
    if value_col:
        for dim in ("region", "category", "channel"):
            c = _chart_bar(df, dim, value_col)
            if c:
                charts.append(c)

    corr = py.get("correlation", {}).get("matrix")
    heatmap = _chart_correlation_heatmap(corr) if corr else None
    if heatmap:
        charts.append(heatmap)

    trace.append(
        {
            "node": "visualization",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok",
            "detail": f"generated {len(charts)} charts",
        }
    )
    return {**state, "charts": charts, "trace": trace}
