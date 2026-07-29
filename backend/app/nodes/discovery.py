from __future__ import annotations
import time

from app.state import AgentState
from app.tools.datasource import build_data_source
from app.tools.sql_tool import SQLTool


async def discovery_node(state: AgentState) -> AgentState:
    """
    Builds a fresh SQLTool for this request only, from the dataset_source
    descriptor the request supplied (which may include Postgres credentials
    sent directly by the frontend). The tool — and any credentials it
    holds — lives only in this request's `state` dict and is garbage
    collected once the graph run returns. Nothing is cached across
    requests, so no dataset connection or credential outlives a single
    /analyze call, and no data from one user's request is ever reachable
    from another user's request.
    """
    t0 = time.time()
    source = build_data_source(state["dataset_source"])
    tool = SQLTool(source)
    schema = tool.schema()

    trace = state.get("trace", [])
    trace.append({
        "node": "dataset_discovery_and_schema",
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "status": "ok",
        "detail": f"source={state['dataset_source'].get('type')}, "
                  f"{schema['row_count']} rows, {len(schema['columns'])} columns",
    })
    return {**state, "sql_tool": tool, "schema": schema, "trace": trace}
