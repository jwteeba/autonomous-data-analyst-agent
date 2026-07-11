from __future__ import annotations
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """Typed state dictionary passed between all nodes in the agent graph."""

    # input
    question: str
    dataset_source: dict[
        str, Any
    ]  # {"type": "file", "path": ...} or {"type": "postgres", ...}
    dataset_name: str

    # planning
    plan: dict[
        str, Any
    ]  # {"needs_sql": bool, "needs_stats": bool, "needs_forecast": bool, "intent": str}
    error: Optional[str]
    retries: int

    # schema / cleaning
    schema: dict[str, Any]
    cleaning_report: dict[str, Any]

    # execution results
    sql_query: Optional[str]
    sql_result: Optional[
        dict[str, Any]
    ]  # {"columns": [...], "rows": [...], "row_count": int}
    python_result: Optional[dict[str, Any]]  # stats output

    # visualization
    charts: list[
        dict[str, Any]
    ]  # [{"title":..., "path":..., "type":..., "caption":...}]

    # synthesis
    insights: Optional[dict[str, Any]]
    report_path: Optional[str]
    report_markdown: Optional[str]

    # observability
    trace: list[dict[str, Any]]  # ordered log of {node, duration_ms, status, detail}
