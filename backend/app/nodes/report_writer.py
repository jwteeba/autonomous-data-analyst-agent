from __future__ import annotations
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.state import AgentState

REPORT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _render_markdown(state: AgentState) -> str:
    """Render a full Markdown analysis report from the completed agent state.

    Args:
        state: Completed agent state containing schema, cleaning report, insights,
            charts, SQL result, and execution trace.

    Returns:
        A Markdown string representing the full analysis report.
    """
    schema = state.get("schema", {})
    cleaning = state.get("cleaning_report", {})
    insights = state.get("insights", {}) or {}
    charts = state.get("charts", []) or []
    sql_result = state.get("sql_result", {}) or {}
    py = state.get("python_result", {}) or {}

    lines = []
    lines.append(f"# Data Analysis Report\n")
    lines.append(f"**Question:** {state['question']}\n")
    lines.append(f"**Dataset:** {state.get('dataset_name', 'unnamed dataset')}  ")
    lines.append(
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z\n"
    )

    lines.append("## Dataset Summary")
    lines.append(f"- Rows: {schema.get('row_count', 'n/a')}")
    lines.append(f"- Columns: {len(schema.get('columns', []))}")
    col_list = ", ".join(
        f"`{c['name']}` ({c['type']})" for c in schema.get("columns", [])
    )
    lines.append(f"- Schema: {col_list}\n")

    lines.append("## Methodology")
    lines.append(
        "1. Dataset loaded read-only into an in-memory DuckDB table (source file never modified).\n"
        "2. Automated data validation scanned for missing values, duplicates, outliers (IQR method), "
        "impossible values, and categorical inconsistencies.\n"
        "3. A SQL query was generated and validated (destructive statements blocked) to answer the question.\n"
        "4. Python statistical analysis (pandas/NumPy/SciPy/statsmodels) computed descriptive stats, "
        "correlations, and — where relevant — an OLS trend regression with a 3-month linear forecast.\n"
        "5. Charts were rendered from the computed results, not from raw guesses.\n"
    )

    lines.append("## Data Quality Findings")
    lines.append(f"- Duplicate rows: {cleaning.get('duplicate_rows', 0)}")
    if cleaning.get("columns"):
        for col, issues in cleaning["columns"].items():
            issue_str = "; ".join(f"{k}={v}" for k, v in issues.items())
            lines.append(f"- `{col}`: {issue_str}")
    else:
        lines.append("- No significant issues detected.")
    lines.append("")

    lines.append("## Charts")
    for c in charts:
        lines.append(f"### {c['title']}")
        lines.append(f"![{c['title']}]({Path(c['path']).name})")
        lines.append(f"*{c['caption']}*")
        lines.append(f"> {c['business_explanation']}\n")

    lines.append("## Executive Summary")
    lines.append(insights.get("executive_summary", "n/a") + "\n")

    lines.append("## Key Findings")
    for f in insights.get("key_findings", []):
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## Risks")
    for r in insights.get("risks", []):
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Opportunities")
    for o in insights.get("opportunities", []):
        lines.append(f"- {o}")
    lines.append("")

    lines.append("## Recommendations")
    for r in insights.get("recommendations", []):
        lines.append(f"- {r}")
    lines.append("")

    lines.append(
        f"**Confidence level:** {insights.get('confidence_level', 'n/a')} "
        f"— {insights.get('confidence_reason', '')}\n"
    )

    lines.append("## Appendix")
    lines.append("### SQL Query Used")
    lines.append("```sql")
    lines.append(state.get("sql_query", "-- none"))
    lines.append("```\n")

    if sql_result.get("rows"):
        lines.append("### SQL Result Sample")
        lines.append(f"Columns: {sql_result.get('columns')}")
        lines.append(
            f"Row count: {sql_result.get('row_count')} "
            f"(showing up to {min(10, len(sql_result['rows']))})"
        )
        for row in sql_result["rows"][:10]:
            lines.append(f"- {row}")
        lines.append("")

    if py.get("descriptive_stats"):
        lines.append("### Descriptive Statistics")
        for col, stats in py["descriptive_stats"].items():
            lines.append(f"- **{col}**: {stats}")
        lines.append("")

    lines.append("### Execution Trace")
    for t in state.get("trace", []):
        lines.append(
            f"- `{t['node']}` — {t['status']} — {t['duration_ms']}ms — {t['detail']}"
        )

    return "\n".join(lines)


async def report_writer_node(state: AgentState) -> AgentState:
    """Render and persist the Markdown analysis report to disk.

    Args:
        state: Completed agent state passed from the insights node.

    Returns:
        Updated agent state with ``report_path``, ``report_markdown``, and an
        appended ``trace`` entry.
    """
    t0 = time.time()
    markdown = _render_markdown(state)
    report_id = uuid.uuid4().hex[:10]
    path = REPORT_DIR / f"report_{report_id}.md"
    path.write_text(markdown, encoding="utf-8")

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "report_writer",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok",
            "detail": f"wrote {path}",
        }
    )
    return {
        **state,
        "report_path": str(path),
        "report_markdown": markdown,
        "trace": trace,
    }
