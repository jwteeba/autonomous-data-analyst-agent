from __future__ import annotations

import uuid
from io import BytesIO

import requests
import streamlit as st

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-only state. Nothing here is ever written to disk by this app:
# closing the browser tab discards the API base URL, the Anthropic
# credentials, and every connected dataset's connection details (including
# Postgres passwords). Each browser session gets its own independent
# st.session_state, so credentials never leak between users of the same
# deployed app.
# ---------------------------------------------------------------------------
st.session_state.setdefault("api_base_url", "http://localhost:8000")
st.session_state.setdefault("anthropic_api_key", "")
st.session_state.setdefault("anthropic_model", "claude-sonnet-4-6")
st.session_state.setdefault("datasets", {})  # local_id -> {"name": str, "source": dict}


def _api_base_url() -> str:
    return st.session_state.get("api_base_url", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Deliberate visual polish beyond the theme file: a console-like monospace
# treatment for anything code/data (SQL, trace, stats) so it reads as an
# analytical instrument rather than a generic chat wrapper.
# ---------------------------------------------------------------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API client — thin wrappers around the FastAPI backend, with clear errors
# surfaced in the UI rather than raw stack traces. All read the backend URL
# from session state at call time, so changing it in the sidebar takes
# effect immediately without restarting the app.
# ---------------------------------------------------------------------------


def api_get(path: str, timeout: int = 10):
    try:
        r = requests.get(f"{_api_base_url()}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the backend at {_api_base_url()}. Is it running?"
    except requests.exceptions.HTTPError as e:
        return None, _error_detail(e.response)
    except requests.exceptions.RequestException as e:
        return None, str(e)


def api_post(path: str, json: dict | None = None, timeout: int = 120):
    try:
        r = requests.post(f"{_api_base_url()}{path}", json=json, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the backend at {_api_base_url()}. Is it running?"
    except requests.exceptions.HTTPError as e:
        return None, _error_detail(e.response)
    except requests.exceptions.RequestException as e:
        return None, str(e)


def _error_detail(response) -> str:
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


def fetch_image_bytes(url_path: str) -> bytes | None:
    try:
        r = requests.get(f"{_api_base_url()}{url_path}", timeout=15)
        r.raise_for_status()
        return r.content
    except requests.exceptions.RequestException:
        return None


def api_get_raw(path: str) -> str | None:
    try:
        r = requests.get(f"{_api_base_url()}{path}", timeout=15)
        r.raise_for_status()
        return r.text
    except requests.exceptions.RequestException:
        return None


# ---------------------------------------------------------------------------
# Shared rendering: an analysis record (from /analyze or /analysis/{id})
# renders identically whether it's fresh or pulled from history.
# ---------------------------------------------------------------------------


def render_analysis(record: dict, key_prefix: str = "analyze"):
    insights = record.get("insights") or {}
    confidence = (insights.get("confidence_level") or "unknown").lower()
    badge_class = {
        "high": "confidence-high",
        "medium": "confidence-medium",
        "low": "confidence-low",
    }.get(confidence, "confidence-medium")

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
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="agent-eyebrow">Key Findings</div>', unsafe_allow_html=True
        )
        for f in insights.get("key_findings", []):
            st.markdown(f"- {f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="agent-eyebrow">Opportunities</div>', unsafe_allow_html=True
        )
        for o in insights.get("opportunities", []):
            st.markdown(f"- {o}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown('<div class="agent-eyebrow">Risks</div>', unsafe_allow_html=True)
        for r in insights.get("risks", []):
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="agent-eyebrow">Recommendations</div>', unsafe_allow_html=True
        )
        for r in insights.get("recommendations", []):
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

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
            st.info(
                cleaning["auto_repaired"]["action"]
                + f" — {cleaning['auto_repaired']['rows_repaired_by_column']}"
            )
        if cleaning.get("columns"):
            st.json(cleaning["columns"])
        else:
            st.write("No significant issues detected.")

    with st.expander("Execution trace"):
        for t in record.get("trace", []):
            css = {
                "ok": "trace-ok",
                "flagged": "trace-flagged",
                "error": "trace-error",
                "skipped": "trace-skipped",
            }.get(t["status"], "")
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


# ---------------------------------------------------------------------------
# Sidebar: connection status + all credentials, session-scoped only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ◆ Data Analyst Agent")
    st.caption("Frontend for the LangGraph backend")

    st.text_input("API base URL", key="api_base_url")

    health, health_err = api_get("/health")
    if health:
        st.success("Backend reachable")
    else:
        st.error(health_err or "Backend unreachable")

    st.divider()
    st.markdown("**LLM credentials**")
    st.text_input(
        "Anthropic API key",
        key="anthropic_api_key",
        type="password",
        placeholder="sk-ant-...",
    )
    st.text_input("Anthropic model", key="anthropic_model")
    if st.session_state["anthropic_api_key"]:
        st.caption("Live Anthropic calls enabled for this session.")
    else:
        st.caption(
            "No key set — analyses will use the rule-based fallback "
            "(still fully functional, just not LLM-written prose)."
        )

    st.divider()
    st.caption(
        "🔒 Everything above, and every database password entered in "
        "**Connect Data**, is held only in this browser tab's session "
        "state. Nothing is written to disk or shared between sessions — "
        "closing this tab discards it all."
    )


st.title("Autonomous Data Analyst Agent")
tab_analyze, tab_connect, tab_history = st.tabs(["Analyze", "Connect Data", "History"])


# ---------------------------------------------------------------------------
# Tab: Analyze
# ---------------------------------------------------------------------------

with tab_analyze:
    datasets = st.session_state["datasets"]

    if not datasets:
        st.info(
            "No datasets connected yet. Go to the **Connect Data** tab to "
            "upload a file or connect a Postgres database — there's no "
            "default dataset in this app."
        )
    else:
        col_q, col_d = st.columns([3, 1])
        with col_d:
            local_ids = list(datasets.keys())
            selected_local_id = st.selectbox(
                "Dataset",
                options=local_ids,
                format_func=lambda lid: datasets[lid]["name"],
            )
            selected = datasets[selected_local_id]
            src = selected["source"]
            if src["type"] == "postgres":
                st.caption(
                    f"Source: postgres · {src['host']}/{src['database']}."
                    f"{src.get('table') or '(custom query)'}"
                )
            else:
                st.caption("Source: uploaded file")

        with col_q:
            question = st.text_area(
                "Business question",
                placeholder="e.g. What is our revenue trend and forecast next quarter?",
                height=100,
            )
            run = st.button("Run Analysis", type="primary", width="stretch")

        if run:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner(
                    "Planning, cleaning, querying, analyzing, charting, and writing the report..."
                ):
                    payload = {
                        "question": question,
                        "dataset_name": selected["name"],
                        "dataset_source": src,
                        "anthropic_api_key": st.session_state["anthropic_api_key"]
                        or None,
                        "anthropic_model": st.session_state["anthropic_model"] or None,
                    }
                    record, err = api_post("/analyze", json=payload)
                if err:
                    st.error(err)
                else:
                    st.session_state["last_result"] = record

        if st.session_state.get("last_result"):
            st.divider()
            render_analysis(st.session_state["last_result"])


# ---------------------------------------------------------------------------
# Tab: Connect Data
# ---------------------------------------------------------------------------

with tab_connect:
    col_file, col_db = st.columns(2)

    with col_file:
        st.subheader("Upload a file")
        st.caption("CSV, Excel, JSON, or Parquet")
        uploaded = st.file_uploader(
            "Choose a file", type=["csv", "xlsx", "xls", "json", "parquet"]
        )
        if uploaded is not None and st.button("Upload", key="upload_btn"):
            with st.spinner("Uploading..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    r = requests.post(
                        f"{_api_base_url()}/upload", files=files, timeout=60
                    )
                    r.raise_for_status()
                    resp = r.json()
                    local_id = uuid.uuid4().hex[:8]
                    st.session_state["datasets"][local_id] = {
                        "name": resp["name"],
                        "source": resp["source"],
                    }
                    st.success(
                        f"Uploaded and added to your dataset list: {resp['name']}"
                    )
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: {e}")

    with col_db:
        st.subheader("Connect a Postgres database")
        st.caption(
            "Tested immediately, never stored server-side — the "
            "connection details (including the password) are kept "
            "only in this browser session and sent again with each "
            "question you ask."
        )
        with st.form("connect_db_form"):
            name = st.text_input("Display name", placeholder="e.g. Production Orders")
            db_col1, db_col2 = st.columns(2)
            with db_col1:
                host = st.text_input("Host", value="localhost")
                database = st.text_input("Database")
                user = st.text_input("User")
            with db_col2:
                port = st.number_input("Port", value=5432, step=1)
                table_or_query = st.radio(
                    "Source", ["Table", "Custom query"], horizontal=True
                )
                password = st.text_input("Password", type="password")

            if table_or_query == "Table":
                table = st.text_input("Table name")
                query = None
            else:
                table = None
                query = st.text_area(
                    "Read-only SQL query (SELECT/WITH only)", height=80
                )

            submitted = st.form_submit_button(
                "Test & Connect", type="primary", width="stretch"
            )

            if submitted:
                test_payload = {
                    "host": host,
                    "port": int(port),
                    "database": database,
                    "user": user,
                    "password": password,
                }
                if table:
                    test_payload["table"] = table
                if query:
                    test_payload["query"] = query

                resp, err = api_post("/test-connection", json=test_payload)
                if err:
                    st.error(err)
                else:
                    local_id = uuid.uuid4().hex[:8]
                    source = {
                        "type": "postgres",
                        "host": host,
                        "port": int(port),
                        "database": database,
                        "user": user,
                        "password": password,
                        "table": table,
                        "query": query,
                        "db_schema": "public",
                    }
                    st.session_state["datasets"][local_id] = {
                        "name": name or f"{database}.{table or 'query'}",
                        "source": source,
                    }
                    st.success(
                        f"Connected and added to your dataset list: "
                        f"{st.session_state['datasets'][local_id]['name']}"
                    )
                    st.rerun()

    st.divider()
    st.subheader("Your connected datasets (this session)")
    if not st.session_state["datasets"]:
        st.caption("None yet.")
    else:
        for local_id, d in list(st.session_state["datasets"].items()):
            src = d["source"]
            detail = (
                "uploaded file"
                if src["type"] == "file"
                else (
                    f"postgres · {src['host']}/{src['database']}.{src.get('table') or '(custom query)'}"
                )
            )
            row_col1, row_col2 = st.columns([5, 1])
            with row_col1:
                st.markdown(f"**{d['name']}** &nbsp;·&nbsp; {detail}")
            with row_col2:
                if st.button("Remove", key=f"remove_{local_id}"):
                    del st.session_state["datasets"][local_id]
                    st.rerun()


# ---------------------------------------------------------------------------
# Tab: History
# ---------------------------------------------------------------------------

with tab_history:
    history_resp, err = api_get("/history")
    if err:
        st.error(err)
    else:
        analyses = history_resp.get("analyses", [])
        if not analyses:
            st.caption("No analyses run yet.")
        else:
            for a in reversed(analyses):
                with st.expander(
                    f"{a['question']}  ·  {a['dataset_name']}  ·  {a['elapsed_seconds']}s"
                ):
                    if st.button("Load full result", key=f"load_{a['id']}"):
                        record, rec_err = api_get(f"/analysis/{a['id']}")
                        if rec_err:
                            st.error(rec_err)
                        else:
                            st.session_state["history_result"] = record

                    if st.session_state.get("history_result", {}).get("id") == a["id"]:
                        render_analysis(
                            st.session_state["history_result"], key_prefix="history"
                        )
