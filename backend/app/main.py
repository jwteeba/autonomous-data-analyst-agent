from __future__ import annotations
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph import agent_graph
from app.llm import LLMClient
from app.tools.datasource import PostgresDataSource, _assert_readonly_sql

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Autonomous Data Analyst Agent — Backend Core",
    description="LangGraph-based agent: plans, cleans, queries, analyzes, "
                "visualizes, and reports on tabular datasets. Stateless with "
                "respect to credentials — every request supplies its own "
                "database connection details and LLM API key; nothing is "
                "stored server-side beyond the lifetime of that one request.",
    version="0.3.0",
)
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

# --- in-memory store of past ANALYSIS RESULTS only (no credentials) ---
# Kept so /history and report re-download work across a session. Contains
# questions, SQL used, insights, chart paths, and traces — never database
# passwords or API keys, and nothing here is written to disk.
ANALYSES: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    question: str
    dataset_name: str = "Dataset"
    # Full source descriptor supplied by the caller for THIS request only:
    #   {"type": "file", "path": "..."}  (from a prior /upload call), or
    #   {"type": "postgres", "host":..., "port":..., "database":...,
    #    "user":..., "password":..., "table" or "query":...}
    # Postgres credentials here are used to build a fresh, request-scoped
    # connection and are never persisted after this call returns.
    dataset_source: dict[str, Any]
    # Optional per-request LLM credentials. If omitted, the agent runs on
    # the rule-based fallback instead of a live model. Never read from an
    # environment variable or any server-side config.
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None


class TestConnectionRequest(BaseModel):
    host: str
    port: int = 5432
    database: str
    user: str
    password: str
    table: Optional[str] = None
    query: Optional[str] = None
    db_schema: str = "public"


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accepts a file and writes it to a transient location on the backend so
    it can be read for analysis. Returns a `source` descriptor the caller
    should hang onto (e.g. in their own session state) and pass back with
    each /analyze call — there is no server-side dataset registry to look
    it up by id.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls", ".json", ".parquet"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}{suffix}"
    dest.write_bytes(await file.read())
    return {"name": file.filename, "source": {"type": "file", "path": str(dest)}}


@app.post("/test-connection")
def test_connection(req: TestConnectionRequest):
    """
    Validates that a Postgres connection works and (for custom queries)
    that it's read-only. Nothing is stored: the connection is opened only
    long enough to run `SELECT 1`, then closed. Credentials never touch
    disk and are discarded the moment this function returns. The caller is
    expected to hold onto the connection details itself (e.g. in its own
    session state) and pass them again with each /analyze call.
    """
    if not req.table and not req.query:
        raise HTTPException(400, "Provide either 'table' or 'query'.")
    if req.table and req.query:
        raise HTTPException(400, "Provide only one of 'table' or 'query', not both.")
    if req.query:
        try:
            _assert_readonly_sql(req.query)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    source = PostgresDataSource(
        host=req.host, port=req.port, database=req.database,
        user=req.user, password=req.password,
        table=req.table, query=req.query, db_schema=req.db_schema,
    )
    try:
        source.test_connection()
    except Exception as e:
        raise HTTPException(400, f"Could not connect: {e}") from e
    return {"ok": True}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    analysis_id = uuid.uuid4().hex[:10]
    t0 = time.time()

    # Constructed fresh for this request only — never a shared/global
    # client, so one caller's API key can't leak into another's request.
    llm_client = LLMClient(api_key=req.anthropic_api_key, model=req.anthropic_model)

    initial_state = {
        "question": req.question,
        "dataset_source": req.dataset_source,
        "dataset_name": req.dataset_name,
        "llm_client": llm_client,
        "trace": [],
        "retries": 0,
    }

    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(400, f"Analysis failed: {e}") from e
    elapsed = round(time.time() - t0, 2)

    record = {
        "id": analysis_id,
        "question": req.question,
        "dataset_name": req.dataset_name,
        "elapsed_seconds": elapsed,
        "llm_provider": llm_client.provider,
        "sql_query": final_state.get("sql_query"),
        "insights": final_state.get("insights"),
        "charts": [
            {**c, "url": f"/files/charts/{Path(c['path']).name}"}
            for c in final_state.get("charts", [])
        ],
        "cleaning_report": final_state.get("cleaning_report"),
        "report_path": final_state.get("report_path"),
        "report_url": f"/analysis/{analysis_id}/report" if final_state.get("report_path") else None,
        "trace": final_state.get("trace"),
    }
    # Note: dataset_source (which may contain a Postgres password) and
    # anthropic_api_key are intentionally NOT included in the stored
    # record — only the display name and the analysis results are kept.
    ANALYSES[analysis_id] = record
    return record


# Alias endpoint per spec: conversational entry point, same underlying engine.
@app.post("/chat")
async def chat(req: AnalyzeRequest):
    return await analyze(req)


@app.get("/history")
def history():
    return {
        "analyses": [
            {"id": a["id"], "question": a["question"], "dataset_name": a["dataset_name"],
             "elapsed_seconds": a["elapsed_seconds"]}
            for a in ANALYSES.values()
        ]
    }


@app.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    record = ANALYSES.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found")
    return record


@app.get("/analysis/{analysis_id}/report")
def get_report(analysis_id: str):
    record = ANALYSES.get(analysis_id)
    if not record or not record.get("report_path"):
        raise HTTPException(404, "Report not found")
    return FileResponse(record["report_path"], media_type="text/markdown")


@app.delete("/analysis/{analysis_id}")
def delete_analysis(analysis_id: str):
    if analysis_id not in ANALYSES:
        raise HTTPException(404, "Analysis not found")
    del ANALYSES[analysis_id]
    return {"deleted": analysis_id}


@app.get("/health")
def health():
    return {"status": "ok"}
