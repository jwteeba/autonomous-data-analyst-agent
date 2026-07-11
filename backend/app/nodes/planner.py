from __future__ import annotations
import json
import time

from app.llm import llm_client
from app.state import AgentState

SYSTEM = """You are the Planner for a data analyst agent.
Classify the analytical intent of the user's business question and decide
which capabilities are required. Respond ONLY with JSON:
{"needs_sql": bool, "needs_stats": bool, "needs_forecast": bool,
 "needs_segmentation": bool, "intent": string, "reasoning": string}
"""


async def planner_node(state: AgentState) -> AgentState:
    """Classify the analytical intent of the question and produce an execution plan.

    Args:
        state: Current agent state containing ``question`` and ``schema``.

    Returns:
        Updated agent state with ``plan`` and an appended ``trace`` entry.
    """
    t0 = time.time()
    question = state["question"]
    schema = state.get("schema", {})

    user_msg = (
        f"Question: {question}\n"
        f"Available columns: {[c['name'] for c in schema.get('columns', [])]}\n"
        "Classify the analytical intent."
    )
    raw = await llm_client.complete(SYSTEM, user_msg, json_mode=True, max_tokens=300)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {
            "needs_sql": True,
            "needs_stats": True,
            "needs_forecast": "forecast" in question.lower(),
            "needs_segmentation": False,
            "intent": "general_analysis",
            "reasoning": "planner output was not valid JSON; defaulted to full pipeline",
        }

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "planner",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok",
            "detail": f"provider={llm_client.provider}, plan={plan}",
        }
    )
    return {**state, "plan": plan, "trace": trace}
