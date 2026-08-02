from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional

from app.graph import agent_graph
from app.llm import LLMClient
from app.llm import llm_client as _default_llm_client
from app.tools.datasource import PostgresDataSource, _assert_readonly_sql
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Autonomous Data Analyst Agent — Backend Core",
    version="0.3.0",
)
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

# In-memory stores
DATASETS: dict[str, dict[str, Any]] = {}
ANALYSES: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    question: str
    dataset_id: str
    dataset_name: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None


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


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": _default_llm_client.provider}


@app.get("/datasets")
def list_datasets():
    safe = []
    for ds in DATASETS.values():
        entry = {"id": ds["id"], "name": ds["name"]}
        src = ds.get("source", {})
        if src.get("type") == "postgres":
            entry["host"] = src.get("host")
            entry["database"] = src.get("database")
            entry["table"] = src.get("table") or src.get("query")
        safe.append(entry)
    return {"datasets": safe}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls", ".json", ".parquet"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}{suffix}"
    dest.write_bytes(await file.read())
    dataset_id = uuid.uuid4().hex[:10]
    source = {"type": "file", "path": str(dest)}
    DATASETS[dataset_id] = {"id": dataset_id, "name": file.filename, "source": source}
    return {"dataset_id": dataset_id, "name": file.filename, "source": source}


@app.post("/connect-database")
def connect_database(req: ConnectDatabaseRequest):
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
        host=req.host,
        port=req.port,
        database=req.database,
        user=req.user,
        password=req.password,
        table=req.table,
        query=req.query,
        db_schema=req.db_schema,
    )
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


@app.post("/datasets/{dataset_id}/refresh")
def refresh_dataset(dataset_id: str):
    ds = DATASETS.get(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    from app.nodes.discovery import invalidate_cache

    invalidate_cache(ds["source"])
    return {"status": "cache invalidated", "dataset_id": dataset_id}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    ds = DATASETS.get(req.dataset_id)
    if not ds:
        raise HTTPException(404, f"Dataset '{req.dataset_id}' not found")

    analysis_id = uuid.uuid4().hex[:10]
    t0 = time.time()

    llm = LLMClient(api_key=req.anthropic_api_key, model=req.anthropic_model)

    initial_state = {
        "question": req.question,
        "dataset_source": ds["source"],
        "dataset_name": req.dataset_name or ds["name"],
        "llm_client": llm,
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
        "dataset_name": req.dataset_name or ds["name"],
        "elapsed_seconds": elapsed,
        "llm_provider": llm.provider,
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


@app.post("/chat")
async def chat(req: AnalyzeRequest):
    return await analyze(req)


@app.get("/history")
def history():
    return {
        "analyses": [
            {
                "id": a["id"],
                "question": a["question"],
                "dataset_id": a["dataset_id"],
                "dataset_name": a["dataset_name"],
                "elapsed_seconds": a["elapsed_seconds"],
            }
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
