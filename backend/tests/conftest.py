"""Shared pytest fixtures for the autonomous data analyst agent test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import pandas as pd
import pytest


# Canonical sample DataFrame used across all tests

SAMPLE_ROWS = 24  # 2 years of monthly data — enough for trend regression


def make_sample_df() -> pd.DataFrame:
    """Return a deterministic sales DataFrame with all columns the agent expects."""
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=SAMPLE_ROWS, freq="MS")
    return pd.DataFrame(
        {
            "order_date": dates.astype(str),
            "region": rng.choice(["North", "South", "East", "West"], size=SAMPLE_ROWS),
            "category": rng.choice(
                ["Electronics", "Clothing", "Food"], size=SAMPLE_ROWS
            ),
            "channel": rng.choice(["Online", "Retail"], size=SAMPLE_ROWS),
            "quantity": rng.integers(1, 50, size=SAMPLE_ROWS),
            "unit_price": rng.uniform(10.0, 200.0, size=SAMPLE_ROWS).round(2),
            "revenue": rng.uniform(500.0, 5000.0, size=SAMPLE_ROWS).round(2),
        }
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_sample_df()


# Minimal DataSource stub backed by the sample DataFrame


class StubDataSource:
    """In-memory DataSource backed by a fixed DataFrame — no I/O."""

    def __init__(self, df: pd.DataFrame | None = None):
        self._df = df if df is not None else make_sample_df()

    def load(self) -> pd.DataFrame:
        return self._df.copy()

    def describe(self) -> dict[str, Any]:
        return {"type": "stub"}


@pytest.fixture
def stub_source(sample_df) -> StubDataSource:
    return StubDataSource(sample_df)


# SQLTool fixture (real DuckDB, stub source)


@pytest.fixture
def sql_tool(stub_source):
    from app.tools.sql_tool import SQLTool

    return SQLTool(stub_source)


# Agent state fixture

FILE_DESCRIPTOR = {"type": "file", "path": "/tmp/stub.csv"}


@pytest.fixture
def base_state(sql_tool) -> dict[str, Any]:
    """Minimal AgentState with a pre-warmed cache entry so nodes don't hit disk."""
    from app.nodes import discovery as disc

    disc._TOOL_CACHE["/tmp/stub.csv"] = sql_tool  # type: ignore[attr-defined]
    # Use the real cache key format
    disc._TOOL_CACHE["file:/tmp/stub.csv"] = sql_tool
    return {
        "question": "What is our revenue trend?",
        "dataset_source": FILE_DESCRIPTOR,
        "dataset_name": "stub",
        "trace": [],
        "retries": 0,
    }


# Temporary CSV file fixture
@pytest.fixture
def tmp_csv(tmp_path, sample_df) -> Path:
    p = tmp_path / "sales.csv"
    sample_df.to_csv(p, index=False)
    return p


@pytest.fixture
def tmp_json(tmp_path, sample_df) -> Path:
    p = tmp_path / "sales.json"
    sample_df.to_json(p, orient="records")
    return p


@pytest.fixture
def tmp_parquet(tmp_path, sample_df) -> Path:
    p = tmp_path / "sales.parquet"
    sample_df.to_parquet(p, index=False)
    return p


# FastAPI test client
@pytest.fixture
def api_client(sql_tool):
    """TestClient with the sample_sales dataset replaced by the stub source."""
    from httpx import ASGITransport, AsyncClient

    import app.main as main_module
    from app.nodes import discovery as disc

    # Register stub in the discovery cache under the key main.py will use
    stub_descriptor = {
        "type": "file",
        "path": str(tmp_path_for_client()),
    }
    return stub_descriptor  # caller uses the fixture differently; see test_main.py


def tmp_path_for_client() -> Path:
    """Return a stable temp path used by the API client fixture."""
    p = Path(tempfile.gettempdir()) / "api_stub.csv"
    if not p.exists():
        make_sample_df().to_csv(p, index=False)
    return p
