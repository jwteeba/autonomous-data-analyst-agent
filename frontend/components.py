from __future__ import annotations
from io import BytesIO

import streamlit as st

from api import api_get_raw, fetch_image_bytes

CSS = """
<style>
    .stApp { font-feature-settings: "tnum"; }
    .agent-card {
        background: #FFFFFF;
        border: 1px solid #E4E6EB;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .agent-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 600;
        color: #6B7280;
        margin-bottom: 0.3rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .confidence-high { background: #DCFCE7; color: #166534; }
    .confidence-medium { background: #FEF3C7; color: #92400E; }
    .confidence-low { background: #FEE2E2; color: #991B1B; }
    .trace-ok { color: #166534; }
    .trace-flagged { color: #92400E; }
    .trace-error { color: #991B1B; }
    .trace-skipped { color: #9CA3AF; }
    code, pre { font-size: 0.85rem !important; }
</style>
"""

_BADGE = {
    "high": "confidence-high",
    "medium": "confidence-medium",
    "low": "confidence-low",
}
_TRACE = {
    "ok": "trace-ok",
    "flagged": "trace-flagged",
    "error": "trace-error",
    "skipped": "trace-skipped",
}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _card(label: str, items: list[str]):
    st.markdown(
        f'<div class="agent-card"><div class="agent-eyebrow">{label}</div>',
        unsafe_allow_html=True,
    )
    for item in items:
        st.markdown(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis(record: dict, key_prefix: str = "analyze"):
    insights = record.get("insights") or {}
    confidence = (insights.get("confidence_level") or "unknown").lower()
    badge_class = _BADGE.get(confidence, "confidence-medium")

    st.markdown(
        f'<span class="confidence-badge {badge_class}">Confidence: {confidence.upper()}</span>'
        f'&nbsp;&nbsp;<span style="color:#6B7280;font-size:0.85rem;">'
        f'{record.get("elapsed_seconds", "?")}s · LLM: {record.get("llm_provider", "?")}</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="agent-eyebrow">Executive Summary</div>', unsafe_allow_html=True
    )
    st.write(insights.get("executive_summary", "No summary generated."))

    col1, col2 = st.columns(2)
    with col1:
        _card("Key Findings", insights.get("key_findings", []))
        _card("Opportunities", insights.get("opportunities", []))
    with col2:
        _card("Risks", insights.get("risks", []))
        _card("Recommendations", insights.get("recommendations", []))

    charts = record.get("charts") or []
    if charts:
        st.markdown(
            '<div class="agent-eyebrow" style="margin-top:0.5rem;">Charts</div>',
            unsafe_allow_html=True,
        )
        chart_cols = st.columns(2)
        for i, c in enumerate(charts):
            img_bytes = fetch_image_bytes(c["url"])
            with chart_cols[i % 2]:
                if img_bytes:
                    st.image(
                        BytesIO(img_bytes),
                        caption=c.get("caption", ""),
                        width="stretch",
                    )
                else:
                    st.warning(f"Could not load chart: {c.get('title')}")

    with st.expander("SQL query used"):
        st.code(record.get("sql_query") or "-- none", language="sql")

    cleaning = record.get("cleaning_report") or {}
    with st.expander("Data quality report"):
        st.write(f"Duplicate rows: {cleaning.get('duplicate_rows', 0)}")
        if cleaning.get("auto_repaired"):
            ar = cleaning["auto_repaired"]
            st.info(f"{ar['action']} — {ar['rows_repaired_by_column']}")
        if cleaning.get("columns"):
            st.json(cleaning["columns"])
        else:
            st.write("No significant issues detected.")

    with st.expander("Execution trace"):
        for t in record.get("trace", []):
            css = _TRACE.get(t["status"], "")
            st.markdown(
                f'`{t["node"]}` — <span class="{css}">{t["status"]}</span> — '
                f'{t["duration_ms"]}ms — {t["detail"]}',
                unsafe_allow_html=True,
            )

    if record.get("report_url"):
        report_text = api_get_raw(record["report_url"])
        if report_text:
            st.download_button(
                "Download full report (Markdown)",
                data=report_text,
                file_name=f"report_{record['id']}.md",
                mime="text/markdown",
                key=f"dl_{key_prefix}_{record['id']}",
            )
