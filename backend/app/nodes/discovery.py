from __future__ import annotations

import time
from typing import Any

from app.state import AgentState
from app.tools.datasource import build_data_source
from app.tools.sql_tool import SQLTool

_CACHE: dict[str, SQLTool] = {}


def _cache_key(descriptor: dict[str, Any]) -> str:
    dtype = descriptor.get("type")
    if dtype == "file":
        return f"file:{descriptor['path']}"
    if dtype == "postgres":
        table_or_query = descriptor.get("table") or descriptor.get("query", "")
        return f"postgres:{descriptor.get('host')}:{descriptor.get('port')}:{descriptor.get('database')}:{table_or_query}"
    raise ValueError(f"Unknown data source type: {dtype}")


def get_sql_tool(descriptor: dict[str, Any]) -> SQLTool:
    key = _cache_key(descriptor)
    if key not in _CACHE:
        _CACHE[key] = SQLTool(build_data_source(descriptor))
    return _CACHE[key]


def invalidate_cache(descriptor: dict[str, Any]) -> None:
    key = _cache_key(descriptor)
    _CACHE.pop(key, None)


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
    trace.append(
        {
            "node": "dataset_discovery_and_schema",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok",
            "detail": f"source={state['dataset_source'].get('type')}, "
            f"{schema['row_count']} rows, {len(schema['columns'])} columns",
        }
    )
    return {**state, "sql_tool": tool, "schema": schema, "trace": trace}
