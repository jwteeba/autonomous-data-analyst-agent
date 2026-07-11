from __future__ import annotations
import time
from typing import Any

from app.state import AgentState
from app.tools.datasource import build_data_source
from app.tools.sql_tool import SQLTool

# One SQLTool instance per resolved data source, kept warm across requests.
_TOOL_CACHE: dict[str, SQLTool] = {}


def _cache_key(descriptor: dict[str, Any]) -> str:
    """Derive a stable cache key string from a data source descriptor.

    Args:
        descriptor: A source descriptor dict with at least a ``type`` key.

    Returns:
        A unique string key identifying the data source.

    Raises:
        ValueError: If the descriptor type is not ``file`` or ``postgres``.
    """
    kind = descriptor.get("type")
    if kind == "file":
        return f"file:{descriptor['path']}"
    if kind == "postgres":
        locator = descriptor.get("table") or descriptor.get("query")
        return f"postgres:{descriptor['host']}:{descriptor['port']}/{descriptor['database']}/{locator}"
    raise ValueError(f"Unknown data source type: {kind!r}")


def get_sql_tool(descriptor: dict[str, Any]) -> SQLTool:
    """Return a cached SQLTool for the given data source descriptor, creating one if needed.

    Args:
        descriptor: A source descriptor dict passed through agent state.

    Returns:
        A SQLTool instance backed by the described data source.
    """
    key = _cache_key(descriptor)
    if key not in _TOOL_CACHE:
        source = build_data_source(descriptor)
        _TOOL_CACHE[key] = SQLTool(source)
    return _TOOL_CACHE[key]


def invalidate_cache(descriptor: dict[str, Any]) -> None:
    """Drop a cached table, e.g. to pick up fresh rows from Postgres on the next call.

    Args:
        descriptor: A source descriptor dict passed through agent state.

    Returns:
        None
    """
    _TOOL_CACHE.pop(_cache_key(descriptor), None)


async def discovery_node(state: AgentState) -> AgentState:
    """Discover the dataset schema and record it in agent state.

    Args:
        state: Current agent state containing ``dataset_source``.

    Returns:
        Updated agent state with ``schema`` and an appended ``trace`` entry.
    """
    t0 = time.time()
    tool = get_sql_tool(state["dataset_source"])
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
    return {**state, "schema": schema, "trace": trace}
