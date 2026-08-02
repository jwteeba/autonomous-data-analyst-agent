from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    dataset_source: dict[
        str, Any
    ]  # {"type": "file", "path": ...} or {"type": "postgres", ...}
    dataset_name: str

    # --- per-request, credential-bearing objects (never persisted beyond
    # this one graph run; constructed fresh by main.py for each /analyze
    # call and discarded when it returns) ---
    llm_client: Any  # app.llm.LLMClient, built from this request's own credentials
    sql_tool: Any  # app.tools.sql_tool.SQLTool, built fresh from this request's dataset_source

    # --- planning ---
    plan: dict[
        str, Any
    ]  # {"needs_sql": bool, "needs_stats": bool, "needs_forecast": bool, "intent": str}
    error: Optional[str]
    retries: int

    # --- schema / cleaning ---
    schema: dict[str, Any]
    cleaning_report: dict[str, Any]

    # --- execution results ---
    sql_query: Optional[str]
    sql_result: Optional[
        dict[str, Any]
    ]  # {"columns": [...], "rows": [...], "row_count": int}
    python_result: Optional[dict[str, Any]]  # stats output

    # --- visualization ---
    charts: list[
        dict[str, Any]
    ]  # [{"title":..., "path":..., "type":..., "caption":...}]

    # --- synthesis ---
    insights: Optional[dict[str, Any]]
    report_path: Optional[str]
    report_markdown: Optional[str]

    # --- observability ---
    trace: list[dict[str, Any]]  # ordered log of {node, duration_ms, status, detail}
