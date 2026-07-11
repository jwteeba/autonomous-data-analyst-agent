from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph import agent_graph
from app.llm import llm_client
from app.nodes.discovery import invalidate_cache
from app.tools.datasource import PostgresDataSource, _assert_readonly_sql

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Autonomous Data Analyst Agent — Backend Core",
    description="LangGraph-based agent: plans, cleans, queries, analyzes, "
    "visualizes, and reports on tabular datasets.",
    version="0.2.0",
)
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

# --- in-memory stores (swap for Postgres-backed tables in production) ---
# Each dataset entry stores a "source" descriptor consumed by
# app.tools.datasource.build_data_source(). type="file" or type="postgres".
DATASETS: dict[str, dict[str, Any]] = {}
ANALYSES: dict[str, dict[str, Any]] = {}


def _register_default_postgres_dataset() -> None:
    """
    The primary sample dataset now lives in Postgres (see
    app/data/seed_postgres.py) instead of a bundled CSV. Connection details
    come from environment variables so this works the same way in
    docker-compose / k8s as it does locally.

    Return:
        None (modifies global DATASETS in place).
    """
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    database = os.environ.get("POSTGRES_DB", "analytics")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    table = os.environ.get("POSTGRES_SAMPLE_TABLE", "sales")

    DATASETS["sample_sales"] = {
        "id": "sample_sales",
        "name": "Sample Sales Orders (Postgres)",
        "source": {
            "type": "postgres",
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "table": table,
        },
    }


_register_default_postgres_dataset()


def _dataset_public(d: dict[str, Any]) -> dict[str, Any]:
    """Never leak credentials back to the client.

    Args:
        d: A dataset dict as stored in the global DATASETS.

    Return:
        A copy of the input dict with sensitive credentials removed.

    """
    out = {"id": d["id"], "name": d["name"], "source_type": d["source"]["type"]}
    if d["source"]["type"] == "postgres":
        out["host"] = d["source"]["host"]
        out["database"] = d["source"]["database"]
        out["table"] = d["source"].get("table")
    return out


class AnalyzeRequest(BaseModel):
    question: str
    dataset_id: str = "sample_sales"


class ConnectDatabaseRequest(BaseModel):
    name: str
    host: str
    port: int = 5432
    database: str
    user: str
    password: str
    table: Optional[str] = None
    query: Optional[str] = None
    db_schema: str = "public"


@app.get("/datasets")
def list_datasets():
    """Return all registered datasets (credentials redacted)."""
    return {"datasets": [_dataset_public(d) for d in DATASETS.values()]}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a tabular data file and register it as a new dataset.

    Args:
        file: The uploaded file (CSV, XLSX, XLS, JSON, or Parquet).

    Returns:
        A dict with ``dataset_id`` and ``name`` of the registered dataset.

    Raises:
        HTTPException: 400 if the file type is not supported.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls", ".json", ".parquet"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    dataset_id = uuid.uuid4().hex[:10]
    dest = UPLOAD_DIR / f"{dataset_id}{suffix}"
    dest.write_bytes(await file.read())
    DATASETS[dataset_id] = {
        "id": dataset_id,
        "name": file.filename,
        "source": {"type": "file", "path": str(dest)},
    }
    return {"dataset_id": dataset_id, "name": file.filename}


@app.post("/connect-database")
def connect_database(req: ConnectDatabaseRequest):
    """Validate and register a Postgres database connection as a new dataset.

    Args:
        req: Connection parameters including host, port, database, credentials,
            and either a table name or a raw SELECT query.

    Returns:
        A dict with ``dataset_id``, ``name``, and ``status``.

    Raises:
        HTTPException: 400 if validation or connection fails.
    """
    if not req.table and not req.query:
        raise HTTPException(400, "Provide either 'table' or 'query'.")
    if req.table and req.query:
        raise HTTPException(400, "Provide only one of 'table' or 'query', not both.")

    source = PostgresDataSource(
        host=req.host,
        port=req.port,
        database=req.database,
        user=req.user,
        password=req.password,
        table=req.table,
        query=req.query,
        db_schema=req.db_schema,
    )
    if req.query:
        try:
            _assert_readonly_sql(req.query)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    try:
        source.test_connection()
    except Exception as e:
        raise HTTPException(400, f"Could not connect: {e}") from e

    dataset_id = uuid.uuid4().hex[:10]
    DATASETS[dataset_id] = {
        "id": dataset_id,
        "name": req.name,
        "source": {
            "type": "postgres",
            "host": req.host,
            "port": req.port,
            "database": req.database,
            "user": req.user,
            "password": req.password,
            "table": req.table,
            "query": req.query,
            "db_schema": req.db_schema,
        },
    }
    return {"dataset_id": dataset_id, "name": req.name, "status": "connected"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the full agent pipeline on a registered dataset and return the analysis record.

    Args:
        req: Request containing the natural-language ``question`` and ``dataset_id``.

    Returns:
        A dict with the analysis id, insights, charts, SQL query, report path, and trace.

    Raises:
        HTTPException: 404 if the dataset is not found; 400 if the agent pipeline fails.
    """
    dataset = DATASETS.get(req.dataset_id)
    if not dataset:
        raise HTTPException(404, f"Unknown dataset_id: {req.dataset_id}")

    analysis_id = uuid.uuid4().hex[:10]
    t0 = time.time()

    initial_state = {
        "question": req.question,
        "dataset_source": dataset["source"],
        "dataset_name": dataset["name"],
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
        "dataset_id": req.dataset_id,
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
        "report_url": (
            f"/analysis/{analysis_id}/report"
            if final_state.get("report_path")
            else None
        ),
        "trace": final_state.get("trace"),
    }
    ANALYSES[analysis_id] = record
    return record


# Alias endpoint per spec: conversational entry point, same underlying engine.
@app.post("/chat")
async def chat(req: AnalyzeRequest):
    """Alias for /analyze; accepts the same request and returns the same response.

    Args:
        req: Request containing the natural-language ``question`` and ``dataset_id``.

    Returns:
        The analysis record produced by the agent pipeline.
    """
    return await analyze(req)


@app.post("/datasets/{dataset_id}/refresh")
def refresh_dataset(dataset_id: str):
    """Drop the cached in-memory copy so the next /analyze re-reads fresh
    rows from Postgres (or the source file).

    Args:
        dataset_id: The unique identifier of the dataset to refresh.

    Returns:
        A dict with the ``dataset_id`` and ``status``.

    Raises:
        HTTPException: 404 if the dataset is not found.


    """
    dataset = DATASETS.get(dataset_id)
    if not dataset:
        raise HTTPException(404, f"Unknown dataset_id: {dataset_id}")
    invalidate_cache(dataset["source"])
    return {"dataset_id": dataset_id, "status": "cache invalidated"}


@app.get("/history")
def history():
    """Return a summary list of all completed analyses."""
    return {
        "analyses": [
            {
                "id": a["id"],
                "question": a["question"],
                "dataset_id": a["dataset_id"],
                "elapsed_seconds": a["elapsed_seconds"],
            }
            for a in ANALYSES.values()
        ]
    }


@app.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    """Retrieve the full record for a completed analysis.

    Args:
        analysis_id: The unique identifier of the analysis.

    Returns:
        The full analysis record dict.

    Raises:
        HTTPException: 404 if the analysis is not found.
    """
    record = ANALYSES.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found")
    return record


@app.get("/analysis/{analysis_id}/report")
def get_report(analysis_id: str):
    """Download the Markdown report for a completed analysis.

    Args:
        analysis_id: The unique identifier of the analysis.

    Returns:
        A FileResponse streaming the Markdown report.

    Raises:
        HTTPException: 404 if the analysis or its report file is not found.
    """
    record = ANALYSES.get(analysis_id)
    if not record or not record.get("report_path"):
        raise HTTPException(404, "Report not found")
    return FileResponse(record["report_path"], media_type="text/markdown")


@app.delete("/analysis/{analysis_id}")
def delete_analysis(analysis_id: str):
    """Delete a completed analysis record from the in-memory store.

    Args:
        analysis_id: The unique identifier of the analysis to delete.

    Returns:
        A dict with the ``deleted`` analysis id.

    Raises:
        HTTPException: 404 if the analysis is not found.
    """
    if analysis_id not in ANALYSES:
        raise HTTPException(404, "Analysis not found")
    del ANALYSES[analysis_id]
    return {"deleted": analysis_id}


@app.get("/health")
def health():
    """Return service health status and the active LLM provider."""
    return {"status": "ok", "llm_provider": llm_client.provider}
