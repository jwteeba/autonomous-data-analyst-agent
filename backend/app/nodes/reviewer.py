from __future__ import annotations
import time
from pathlib import Path

from app.state import AgentState


async def reviewer_node(state: AgentState) -> AgentState:
    """Perform automated quality checks on the completed analysis.

    Args:
        state: Completed agent state after report generation.

    Returns:
        Updated agent state with an appended ``trace`` entry flagging any issues found.
    """
    t0 = time.time()
    issues: list[str] = []

    sql_result = state.get("sql_result") or {}
    if sql_result.get("error"):
        issues.append(f"SQL execution error was not resolved: {sql_result['error']}")

    for c in state.get("charts", []) or []:
        if not Path(c["path"]).exists():
            issues.append(f"Chart file missing on disk: {c['path']}")

    insights = state.get("insights") or {}
    if not insights.get("key_findings"):
        issues.append("No key findings were generated.")
    if insights.get("confidence_level") == "low":
        issues.append(
            "Confidence level is LOW — recommend caution before acting on these results."
        )

    cleaning = state.get("cleaning_report") or {}
    if (
        cleaning.get("duplicate_rows", 0) > 0
        and "duplicate" not in " ".join(insights.get("risks", [])).lower()
    ):
        issues.append(
            "Duplicate rows were detected but not mentioned in the risks section — added to audit log."
        )

    if not state.get("report_path"):
        issues.append("Report was not generated.")

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "reviewer",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok" if not issues else "flagged",
            "detail": (
                "; ".join(issues) if issues else "No issues found in automated review."
            ),
        }
    )
    return {**state, "trace": trace}
