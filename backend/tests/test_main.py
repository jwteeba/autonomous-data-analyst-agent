"""Tests for app/main.py FastAPI endpoints."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tests.conftest import make_sample_df


# Helpers
def _make_final_state(report_path: str) -> dict:
    """Return a minimal agent final state for mocking agent_graph.ainvoke."""
    df = make_sample_df()
    return {
        "question": "What is our revenue trend?",
        "dataset_source": {"type": "file", "path": "/tmp/stub.csv"},
        "dataset_name": "stub",
        "schema": {
            "columns": [{"name": c, "type": "VARCHAR"} for c in df.columns],
            "row_count": len(df),
        },
        "plan": {"needs_sql": True, "needs_stats": True, "needs_forecast": True},
        "sql_query": "SELECT * FROM dataset LIMIT 10",
        "sql_result": {"columns": list(df.columns), "rows": [], "row_count": 0},
        "python_result": {"descriptive_stats": {}},
        "cleaning_report": {"duplicate_rows": 0, "columns": {}},
        "charts": [],
        "insights": {
            "executive_summary": "Test summary.",
            "key_findings": ["Finding 1"],
            "risks": [],
            "opportunities": [],
            "recommendations": [],
            "confidence_level": "medium",
            "confidence_reason": "test",
        },
        "report_path": report_path,
        "report_markdown": "# Report\n",
        "trace": [
            {"node": "discovery", "duration_ms": 1, "status": "ok", "detail": "ok"}
        ],
        "retries": 0,
    }


@pytest_asyncio.fixture
async def client(tmp_path):
    """AsyncClient with a file-backed stub dataset and mocked agent graph."""
    import app.main as main_module
    from app.nodes.discovery import invalidate_cache

    # Write a real CSV so the file source resolves
    csv_path = tmp_path / "stub.csv"
    make_sample_df().to_csv(csv_path, index=False)
    descriptor = {"type": "file", "path": str(csv_path)}

    # Register the stub dataset
    main_module.DATASETS["test_ds"] = {
        "id": "test_ds",
        "name": "Test Dataset",
        "source": descriptor,
    }
    invalidate_cache(descriptor)

    # Write a stub report file
    report_path = str(tmp_path / "report.md")
    Path(report_path).write_text("# Report\n")

    mock_state = _make_final_state(report_path)

    with patch.object(
        main_module.agent_graph, "ainvoke", new=AsyncMock(return_value=mock_state)
    ):
        async with AsyncClient(
            transport=ASGITransport(app=main_module.app), base_url="http://test"
        ) as ac:
            yield ac, "test_ds", report_path

    # Cleanup
    main_module.DATASETS.pop("test_ds", None)


class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_ok(self, client):
        ac, _, _ = client
        resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_returns_llm_provider(self, client):
        ac, _, _ = client
        resp = await ac.get("/health")
        assert "llm_provider" in resp.json()


class TestListDatasets:
    @pytest.mark.asyncio
    async def test_returns_datasets_list(self, client):
        ac, _, _ = client
        resp = await ac.get("/datasets")
        assert resp.status_code == 200
        assert "datasets" in resp.json()

    @pytest.mark.asyncio
    async def test_no_credentials_in_response(self, client):
        ac, _, _ = client
        resp = await ac.get("/datasets")
        body = resp.text
        assert "password" not in body
        assert "secret" not in body


class TestUploadDataset:
    @pytest.mark.asyncio
    async def test_upload_csv_succeeds(self, client):
        ac, _, _ = client
        csv_bytes = make_sample_df().to_csv(index=False).encode()
        resp = await ac.post(
            "/upload",
            files={"file": ("sales.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "dataset_id" in data
        assert data["name"] == "sales.csv"

    @pytest.mark.asyncio
    async def test_upload_unsupported_type_returns_400(self, client):
        ac, _, _ = client
        resp = await ac.post(
            "/upload",
            files={"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_uploaded_dataset_appears_in_list(self, client):
        ac, _, _ = client
        csv_bytes = make_sample_df().to_csv(index=False).encode()
        upload_resp = await ac.post(
            "/upload",
            files={"file": ("new.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        dataset_id = upload_resp.json()["dataset_id"]
        list_resp = await ac.get("/datasets")
        ids = [d["id"] for d in list_resp.json()["datasets"]]
        assert dataset_id in ids


class TestConnectDatabase:
    @pytest.mark.asyncio
    async def test_missing_table_and_query_returns_400(self, client):
        ac, _, _ = client
        resp = await ac.post(
            "/connect-database",
            json={
                "name": "DB",
                "host": "h",
                "port": 5432,
                "database": "db",
                "user": "u",
                "password": "p",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_both_table_and_query_returns_400(self, client):
        ac, _, _ = client
        resp = await ac.post(
            "/connect-database",
            json={
                "name": "DB",
                "host": "h",
                "port": 5432,
                "database": "db",
                "user": "u",
                "password": "p",
                "table": "t",
                "query": "SELECT 1",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_write_query_returns_400(self, client):
        ac, _, _ = client
        resp = await ac.post(
            "/connect-database",
            json={
                "name": "DB",
                "host": "h",
                "port": 5432,
                "database": "db",
                "user": "u",
                "password": "p",
                "query": "DELETE FROM t",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unreachable_host_returns_400(self, client):
        ac, _, _ = client
        resp = await ac.post(
            "/connect-database",
            json={
                "name": "DB",
                "host": "bad-host-xyz",
                "port": 9999,
                "database": "db",
                "user": "u",
                "password": "p",
                "table": "t",
            },
        )
        assert resp.status_code == 400


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_returns_record(self, client):
        ac, dataset_id, _ = client
        resp = await ac.post(
            "/analyze",
            json={
                "question": "What is our revenue trend?",
                "dataset_id": dataset_id,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "insights" in data
        assert "sql_query" in data

    @pytest.mark.asyncio
    async def test_analyze_unknown_dataset_returns_404(self, client):
        ac, _, _ = client
        resp = await ac.post(
            "/analyze",
            json={
                "question": "test",
                "dataset_id": "nonexistent_id",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_analyze_stores_in_history(self, client):
        ac, dataset_id, _ = client
        await ac.post(
            "/analyze",
            json={
                "question": "revenue trend",
                "dataset_id": dataset_id,
            },
        )
        history_resp = await ac.get("/history")
        assert history_resp.status_code == 200
        assert len(history_resp.json()["analyses"]) >= 1

    @pytest.mark.asyncio
    async def test_analyze_agent_failure_returns_400(self, client):
        ac, dataset_id, _ = client
        import app.main as main_module

        with patch.object(
            main_module.agent_graph,
            "ainvoke",
            new=AsyncMock(side_effect=RuntimeError("agent exploded")),
        ):
            resp = await ac.post(
                "/analyze",
                json={
                    "question": "test",
                    "dataset_id": dataset_id,
                },
            )
        assert resp.status_code == 400


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_returns_same_as_analyze(self, client):
        ac, dataset_id, _ = client
        resp = await ac.post(
            "/chat",
            json={
                "question": "revenue trend",
                "dataset_id": dataset_id,
            },
        )
        assert resp.status_code == 200
        assert "insights" in resp.json()


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_returns_list(self, client):
        ac, _, _ = client
        resp = await ac.get("/history")
        assert resp.status_code == 200
        assert "analyses" in resp.json()

    @pytest.mark.asyncio
    async def test_history_entries_have_required_fields(self, client):
        ac, dataset_id, _ = client
        await ac.post("/analyze", json={"question": "test", "dataset_id": dataset_id})
        resp = await ac.get("/history")
        for entry in resp.json()["analyses"]:
            assert "id" in entry
            assert "question" in entry
            assert "dataset_id" in entry
            assert "elapsed_seconds" in entry


class TestGetAnalysis:
    @pytest.mark.asyncio
    async def test_get_existing_analysis(self, client):
        ac, dataset_id, _ = client
        analyze_resp = await ac.post(
            "/analyze", json={"question": "test", "dataset_id": dataset_id}
        )
        analysis_id = analyze_resp.json()["id"]
        resp = await ac.get(f"/analysis/{analysis_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == analysis_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_analysis_returns_404(self, client):
        ac, _, _ = client
        resp = await ac.get("/analysis/doesnotexist")
        assert resp.status_code == 404


class TestGetReport:
    @pytest.mark.asyncio
    async def test_get_report_returns_markdown(self, client):
        ac, dataset_id, _ = client
        analyze_resp = await ac.post(
            "/analyze", json={"question": "test", "dataset_id": dataset_id}
        )
        analysis_id = analyze_resp.json()["id"]
        resp = await ac.get(f"/analysis/{analysis_id}/report")
        assert resp.status_code == 200
        assert "markdown" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_get_report_nonexistent_returns_404(self, client):
        ac, _, _ = client
        resp = await ac.get("/analysis/doesnotexist/report")
        assert resp.status_code == 404


class TestDeleteAnalysis:
    @pytest.mark.asyncio
    async def test_delete_existing_analysis(self, client):
        ac, dataset_id, _ = client
        analyze_resp = await ac.post(
            "/analyze", json={"question": "test", "dataset_id": dataset_id}
        )
        analysis_id = analyze_resp.json()["id"]
        del_resp = await ac.delete(f"/analysis/{analysis_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] == analysis_id

    @pytest.mark.asyncio
    async def test_deleted_analysis_no_longer_accessible(self, client):
        ac, dataset_id, _ = client
        analyze_resp = await ac.post(
            "/analyze", json={"question": "test", "dataset_id": dataset_id}
        )
        analysis_id = analyze_resp.json()["id"]
        await ac.delete(f"/analysis/{analysis_id}")
        resp = await ac.get(f"/analysis/{analysis_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        ac, _, _ = client
        resp = await ac.delete("/analysis/doesnotexist")
        assert resp.status_code == 404


class TestRefreshDataset:
    @pytest.mark.asyncio
    async def test_refresh_existing_dataset(self, client):
        ac, dataset_id, _ = client
        resp = await ac.post(f"/datasets/{dataset_id}/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cache invalidated"

    @pytest.mark.asyncio
    async def test_refresh_nonexistent_dataset_returns_404(self, client):
        ac, _, _ = client
        resp = await ac.post("/datasets/doesnotexist/refresh")
        assert resp.status_code == 404
