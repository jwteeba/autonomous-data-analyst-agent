from __future__ import annotations
import json
import time

from app.llm import llm_client
from app.state import AgentState

SYSTEM = """You are the Business Analyst. You are given real computed SQL/statistical
results (JSON) for a business question. Write insights grounded ONLY in the
numbers provided — never invent a statistic that isn't in the data. Respond
with ONLY JSON: {"executive_summary": string, "key_findings": [string],
"risks": [string], "opportunities": [string], "recommendations": [string],
"confidence_level": "low"|"medium"|"high", "confidence_reason": string}"""


def _fallback_insights(state: AgentState) -> dict:
    """Templated, numbers-grounded insight generation used when no live LLM is
    configured. Pulls directly from computed sql_result / python_result so
    every claim traces back to a real number, never an invented one.

    Args:
        state: Current agent state with ``sql_result``, ``python_result``, and ``cleaning_report``.

    Returns:
        A dict with structured insights based on the computed results.
    """

    sql_result = state.get("sql_result") or {}
    py = state.get("python_result") or {}
    trend = py.get("trend_and_forecast")
    cleaning = state.get("cleaning_report", {})

    findings = []
    if sql_result.get("rows"):
        top_row = sql_result["rows"][0]
        findings.append(f"Top result from query: {top_row}")

    if trend:
        direction = "increasing" if trend["slope_per_month"] > 0 else "decreasing"
        sig = (
            "a statistically significant"
            if trend["statistically_significant_trend"]
            else "a not statistically significant (p>=0.05)"
        )
        findings.append(
            f"Revenue shows {sig} monthly trend of {trend['slope_per_month']:+.2f} "
            f"per month (R²={trend['r_squared']}), i.e. revenue is {direction}."
        )
        findings.append(
            f"3-month linear extrapolation: {trend['forecast_next_3_months']}."
        )

    notable = py.get("correlation", {}).get("notable_pairs", [])
    for pair in notable[:3]:
        findings.append(
            f"Correlation between {pair['a']} and {pair['b']}: r={pair['r']}"
        )

    risks = []
    if cleaning.get("duplicate_rows"):
        risks.append(
            f"{cleaning['duplicate_rows']} duplicate rows detected in source data; "
            "confirm this isn't inflating totals."
        )
    for col, issues in cleaning.get("columns", {}).items():
        if "impossible_negative_values" in issues:
            risks.append(
                f"Column '{col}' has {issues['impossible_negative_values']} impossible negative values."
            )
        if "outliers_iqr_method" in issues:
            risks.append(
                f"Column '{col}' has {issues['outliers_iqr_method']} statistical outliers (IQR method) "
                "that may distort aggregates."
            )

    opportunities = []
    if trend and trend["slope_per_month"] > 0:
        opportunities.append(
            "Positive revenue trend suggests current strategy is working; "
            "consider reallocating budget toward the top-performing segment shown in the chart."
        )
    elif trend:
        opportunities.append(
            "Declining/flat trend warrants investigating the top vs. bottom "
            "performing segments to identify what changed."
        )

    recommendations = [
        "Validate the flagged data-quality issues (duplicates/outliers) before using these "
        "numbers in a board-level report.",
        "Re-run this analysis after the data quality issues above are resolved to confirm findings hold.",
    ]
    if trend:
        recommendations.append(
            "Treat the 3-month forecast as directional only — it's a simple linear extrapolation, "
            "not a seasonal model; validate against a longer historical window before committing budget."
        )

    confidence = "medium" if trend and trend["r_squared"] > 0.3 else "low"

    return {
        "executive_summary": "[rule-based fallback — no live LLM configured] "
        "Summary generated directly from computed SQL and statistical results below; "
        "set ANTHROPIC_API_KEY for narrative-quality prose.",
        "key_findings": findings
        or ["No specific numeric findings were computed for this question."],
        "risks": risks
        or ["No material data-quality risks detected in the automated checks."],
        "opportunities": opportunities
        or ["Insufficient trend data to identify a specific opportunity."],
        "recommendations": recommendations,
        "confidence_level": confidence,
        "confidence_reason": "Based on R² of the trend fit and presence/absence of data-quality flags.",
    }


async def insights_node(state: AgentState) -> AgentState:
    """Generate business insights from computed SQL and statistical results.

    Args:
        state: Current agent state with ``sql_result``, ``python_result``, and ``cleaning_report``.

    Returns:
        Updated agent state with ``insights`` and an appended ``trace`` entry.
    """
    t0 = time.time()
    trace = state.get("trace", [])

    payload = {
        "question": state["question"],
        "sql_result_sample": (state.get("sql_result") or {}).get("rows", [])[:10],
        "python_result": state.get("python_result"),
        "cleaning_report": state.get("cleaning_report"),
    }

    if llm_client.live:
        raw = await llm_client.complete(
            SYSTEM, json.dumps(payload, default=str), json_mode=True, max_tokens=900
        )
        try:
            insights = json.loads(raw)
        except json.JSONDecodeError:
            insights = _fallback_insights(state)
            insights["executive_summary"] = (
                "[LLM returned invalid JSON, used grounded fallback] "
                + insights["executive_summary"]
            )
    else:
        insights = _fallback_insights(state)

    trace.append(
        {
            "node": "insight_generation",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "status": "ok",
            "detail": f"provider={llm_client.provider}, confidence={insights.get('confidence_level')}",
        }
    )
    return {**state, "insights": insights, "trace": trace}
