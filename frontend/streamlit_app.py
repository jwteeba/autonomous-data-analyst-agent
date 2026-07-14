from __future__ import annotations
import requests
import streamlit as st

from api import API_BASE_URL, api_get, api_post
from components import inject_css, render_analysis

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# Sidebar
with st.sidebar:
    st.markdown("### ◆ Data Analyst Agent")
    st.caption("Frontend for the LangGraph backend")

    health, health_err = api_get("/health")
    if health:
        st.success(f"Connected · LLM: {health['llm_provider']}")
    else:
        st.error(health_err or "Backend unreachable")

    with st.expander("Backend settings"):
        st.text_input(
            "API base URL",
            value=API_BASE_URL,
            key="api_base_url_display",
            disabled=True,
        )
        st.caption(
            "Set the API_BASE_URL environment variable before launching to change this."
        )

    st.divider()
    st.caption(
        "Ask a business question, point it at a dataset, and it plans, "
        "cleans, queries, analyzes, charts, and reports — grounded in the "
        "actual computed numbers."
    )


st.title("Autonomous Data Analyst Agent")
tab_analyze, tab_connect, tab_history = st.tabs(["Analyze", "Connect Data", "History"])


# Tab: Analyze
with tab_analyze:
    datasets_resp, datasets_err = api_get("/datasets")
    datasets = datasets_resp.get("datasets", []) if datasets_resp else []

    if datasets_err:
        st.error(datasets_err)
    elif not datasets:
        st.info(
            "No datasets connected yet. Head to the **Connect Data** tab to upload a file or connect a database."
        )
    else:
        col_q, col_d = st.columns([3, 1])
        with col_d:
            dataset_labels = {d["id"]: d["name"] for d in datasets}
            selected_id = st.selectbox(
                "Dataset",
                options=list(dataset_labels.keys()),
                format_func=lambda x: dataset_labels[x],
            )
            selected = next(d for d in datasets if d["id"] == selected_id)
            st.caption(
                f"Source: {selected['source_type']}"
                + (
                    f" · {selected.get('database')}.{selected.get('table')}"
                    if selected["source_type"] == "postgres"
                    else ""
                )
            )

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
                    record, err = api_post(
                        "/analyze",
                        json={"question": question, "dataset_id": selected_id},
                    )
                if err:
                    st.error(err)
                else:
                    st.session_state["last_result"] = record

        if st.session_state.get("last_result"):
            st.divider()
            render_analysis(st.session_state["last_result"])


# Tab: Connect Data
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
                    r = requests.post(
                        f"{API_BASE_URL}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        timeout=60,
                    )
                    r.raise_for_status()
                    st.success(f"Uploaded as dataset_id: {r.json()['dataset_id']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Upload failed: {e}")

    with col_db:
        st.subheader("Connect a Postgres database")
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

            if st.form_submit_button("Connect", type="primary", width="stretch"):
                payload = {
                    "name": name or "Untitled dataset",
                    "host": host,
                    "port": int(port),
                    "database": database,
                    "user": user,
                    "password": password,
                }
                if table:
                    payload["table"] = table
                if query:
                    payload["query"] = query

                resp, err = api_post("/connect-database", json=payload)
                if err:
                    st.error(err)
                else:
                    st.success(f"Connected as dataset_id: {resp['dataset_id']}")

    st.divider()
    st.subheader("Connected datasets")
    datasets_resp, err = api_get("/datasets")
    if err:
        st.error(err)
    else:
        rows = datasets_resp.get("datasets", [])
        if not rows:
            st.caption("No datasets yet.")
        else:
            for d in rows:
                detail = d["source_type"]
                if d["source_type"] == "postgres":
                    detail += f" · {d.get('host')}/{d.get('database')}.{d.get('table')}"
                st.markdown(
                    f"**{d['name']}** &nbsp;·&nbsp; `{d['id']}` &nbsp;·&nbsp; {detail}"
                )


# Tab: History
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
                    f"{a['question']}  ·  {a['elapsed_seconds']}s  ·  {a['id']}"
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
