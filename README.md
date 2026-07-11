# Autonomous Data Analyst Agent — Backend Core

A working LangGraph agent that takes a natural-language business question and
a dataset — **sourced live from database** — and produces a grounded
analysis: generated SQL, real statistics, real charts, and a Markdown report,
via a FastAPI backend.

## Postgres as the primary data source

The agent connects to a postgresql database.

- `app/tools/datasource.py` defines a pluggable `DataSource` interface with
  two implementations: `FileDataSource` (CSV/Excel/JSON/Parquet, for
  `/upload`) and `PostgresDataSource` (for the default dataset and anything
  connected via `POST /connect-database`).
- The agent still runs all SQL/stats/cleaning against an **in-memory DuckDB
  copy** — Postgres is only ever read once (a single `SELECT`) to pull the
  table or query result into memory. Nothing in the graph talks to Postgres
  again after that initial load.

### Why not DuckDB's native Postgres scanner?

DuckDB has a `postgres_scanner` extension that can query Postgres directly.
It needs to download a binary extension from `extensions.duckdb.org` at
runtime, which isn't reachable in every network environment (it wasn't in
the one this was built in). The psycopg2/pandas approach used here has no
such runtime dependency and works anywhere Postgres itself is reachable — if
your environment *can* reach `extensions.duckdb.org`, swapping in the native
scanner later is a drop-in change inside `PostgresDataSource.load()`.

### Read-only guarantees (defense in depth)

1. **Server-level**: the SQLAlchemy session is opened with
   `default_transaction_read_only=on`, so Postgres itself rejects any write
   even if application code had a bug.
2. **App-level**: `_assert_readonly_sql()` blocks anything that isn't
   `SELECT`/`WITH` before it's ever sent — verified in testing: a
   `DELETE FROM ...` passed as a custom query via `/connect-database` is
   rejected immediately with a 400, and the underlying table is confirmed
   untouched.
3. **Single read**: only one `SELECT` is ever issued per dataset load. All
   further analysis runs against the in-memory DuckDB copy.

## Run it

### 1. Start Postgres and seed the sample data

```bash
# via Docker (recommended)
docker run -d --name analyst-pg -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=analytics -p 5432:5432 postgres:16

# seed the sample_sales table
export POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_DB=analytics \
       POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres
```

### 2. Run the API

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Same env vars (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`, and optionally
`POSTGRES_SAMPLE_TABLE`) control which table `main.py` registers as the
default `sample_sales` dataset.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our revenue trend and forecast next quarter?", "dataset_id": "sample_sales"}'
```

### 3. Connect additional databases at runtime

```bash
curl -X POST http://localhost:8000/connect-database \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Inventory DB", "host": "localhost", "port": 5432,
    "database": "analytics", "user": "postgres", "password": "postgres",
    "table": "inventory"
  }'
# -> {"dataset_id": "...", "name": "Inventory DB", "status": "connected"}
```

Use `"query"` instead of `"table"` to point at an arbitrary read-only view.
The connection is tested (and any custom query is validated as read-only)
before the dataset is registered — bad credentials or non-`SELECT` queries
are rejected immediately with a 400, not discovered later mid-analysis.

`POST /datasets/{id}/refresh` drops the cached in-memory copy so the next
`/analyze` call re-reads current rows from Postgres.

`GET /datasets` never returns credentials — only host/database/table.

File uploads (`POST /upload`) still work unchanged, for CSV/Excel/JSON/
Parquet sources alongside your Postgres connections.

## Pluggable LLM (unchanged, still important)

`app/llm.py` defines one `LLMClient.complete()` call used by the planner and
insight generator. If `ANTHROPIC_API_KEY` is set, it calls the real
Anthropic Messages API. If it isn't, a transparent, clearly-labeled
rule-based fallback keeps the whole pipeline runnable and testable without
any API key — every fallback output says so explicitly.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture notes / deliberate simplifications

- **Graph shape**: Schema Understanding → Data Validation → Data Cleaning
  are combined into one `discovery` + one `cleaning` node — same logical
  steps, fewer boxes.
- **State store**: `DATASETS`/`ANALYSES` are in-memory dicts, not a Postgres
  table of their own — swap for real tables before running more than one
  worker process (the irony of using Postgres for *data* but not yet for
  *app state* is a good next slice to build).
- **Single dataset table per query**: the SQL agent queries one `dataset`
  table per source. `TODO`: Multi-table joins across two connected databases.
- **Forecasting**: a linear OLS trend, not Prophet/statsmodels seasonal
  models — deliberately labeled "directional only" in every report.
- **SQL generation without a live LLM**: falls back to a keyword-driven
  heuristic query builder. Real, tested, working SQL — just not free-form
  NL→SQL. Set `ANTHROPIC_API_KEY` for real NL→SQL generation.

## Files

```
backend/
  app/
    main.py              FastAPI app + endpoints (/analyze, /connect-database, ...)
    graph.py               LangGraph wiring
    state.py                Shared agent state schema
    llm.py                   Pluggable LLM client (live Anthropic / rule-based fallback)
    nodes/
      discovery.py           Dataset load + schema inference (source-agnostic)
      planner.py               Intent classification
      cleaning.py               Data quality report + date repair
      sql_agent.py               NL/heuristic -> SQL -> validated execution
      python_analyst.py          Descriptive stats, correlation, trend regression
      visualization.py            Chart generation (matplotlib)
      insights.py                  Executive summary, findings, risks, recommendations
      report_writer.py              Markdown report assembly
      reviewer.py                    Automated audit of the run
    tools/
      datasource.py                 DataSource abstraction: FileDataSource, PostgresDataSource
      sql_tool.py                    DuckDB wrapper: load from any DataSource, validate, execute, repair
    data/
  requirements.txt
  README.md
```

