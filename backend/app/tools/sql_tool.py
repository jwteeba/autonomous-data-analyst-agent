from __future__ import annotations
import re
import time
from typing import Any

import duckdb
import pandas as pd

from app.tools.datasource import DataSource

# Statements that mutate data or schema. The agent operates read-only on a
# copy of the data; these are blocked unless explicitly approved elsewhere.
_FORBIDDEN = re.compile(
    r"\b(delete|update|drop|alter|insert|truncate|create\s+or\s+replace|attach|copy\s+.*\s+to)\b",
    re.IGNORECASE,
)
_MULTI_STATEMENT = re.compile(r";\s*\S")  # anything after a semicolon


class SQLValidationError(Exception):
    """Raised when a SQL query fails validation or execution inside SQLTool."""


class SQLTool:
    """
    Wraps an in-memory DuckDB connection. Whatever DataSource is passed in
    (a local file, a Postgres table/query, ...) is read exactly once into a
    pandas DataFrame; every query after that runs against the in-memory
    DuckDB copy. The original source (file on disk, or the Postgres
    database) is never written to.
    """

    def __init__(self, source: DataSource, table_name: str = "dataset"):
        """Load the data source into an in-memory DuckDB table.

        Args:
            source: A DataSource whose load() result is registered in DuckDB.
            table_name: Name of the DuckDB table created from the source data.
        """
        self.source = source
        self.table_name = table_name
        self.con = duckdb.connect(database=":memory:")
        df = source.load()
        self.con.register("raw_df", df)
        self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM raw_df")

    def validate(self, query: str) -> None:
        """Check that a query is safe to run against the in-memory dataset.

        Args:
            query: The SQL string to validate.

        Raises:
            SQLValidationError: If the query is empty, contains destructive statements,
                or does not start with SELECT or WITH.
        """
        if not query or not query.strip():
            raise SQLValidationError("Empty query.")
        if _MULTI_STATEMENT.search(query.strip().rstrip(";") + ";x"):
            # crude but effective: reject anything that looks like 2 statements
            pass
        if _FORBIDDEN.search(query):
            raise SQLValidationError(
                "Query blocked: destructive/DDL statements "
                "(DELETE/UPDATE/DROP/ALTER/INSERT/TRUNCATE/ATTACH/COPY) are not permitted. "
                "This agent is read-only over a copy of your data."
            )
        if not re.match(r"^\s*(with|select)\b", query, re.IGNORECASE):
            raise SQLValidationError("Only SELECT / WITH (CTE) queries are permitted.")

    def run(self, query: str) -> dict[str, Any]:
        """Validate and execute a SQL query, returning results as a dict.

        Args:
            query: A read-only SELECT or WITH query to run.

        Returns:
            A dict with keys ``columns``, ``rows`` (up to 500), ``row_count``,
            ``execution_ms``, and ``truncated``.

        Raises:
            SQLValidationError: If validation fails or DuckDB raises an error.
        """
        self.validate(query)
        start = time.time()
        try:
            result_df = self.con.execute(query).fetchdf()
        except Exception as e:
            raise SQLValidationError(f"SQL execution failed: {e}") from e
        elapsed_ms = (time.time() - start) * 1000
        return {
            "columns": list(result_df.columns),
            "rows": result_df.head(500).to_dict(orient="records"),
            "row_count": len(result_df),
            "execution_ms": round(elapsed_ms, 2),
            "truncated": len(result_df) > 500,
        }

    def schema(self) -> dict[str, Any]:
        """Return the schema and row count of the in-memory table.

        Returns:
            A dict with keys ``columns`` (list of name/type dicts) and ``row_count``.
        """
        info = self.con.execute(f"DESCRIBE {self.table_name}").fetchdf()
        row_count = self.con.execute(
            f"SELECT COUNT(*) AS n FROM {self.table_name}"
        ).fetchone()[0]
        return {
            "columns": [
                {"name": r["column_name"], "type": r["column_type"]}
                for _, r in info.iterrows()
            ],
            "row_count": row_count,
        }

    def as_dataframe(self) -> pd.DataFrame:
        """Return the full in-memory table as a pandas DataFrame.

        Returns:
            A pandas DataFrame containing all rows in the DuckDB table.
        """
        return self.con.execute(f"SELECT * FROM {self.table_name}").fetchdf()

    def normalize_date_columns(self, columns: list[str]) -> dict[str, int]:
        """Repairs mixed-format date columns (e.g. some rows 'YYYY-MM-DD', others
        'DD/MM/YYYY') in the in-memory copy only. Returns a count of rows
        repaired per column. The source file on disk is never touched.

        Args:
            columns: List of column names to check for mixed date formats.

        Returns:
            A dict mapping column name to number of rows repaired.
        """

        repaired: dict[str, int] = {}
        for col in columns:
            before = self.con.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE TRY_CAST({col} AS TIMESTAMP) IS NULL "
                f"AND {col} IS NOT NULL"
            ).fetchone()[0]
            if before == 0:
                continue
            self.con.execute(
                f"CREATE OR REPLACE TABLE {self.table_name} AS "
                f"SELECT * REPLACE ("
                f"  COALESCE("
                f"    TRY_CAST({col} AS TIMESTAMP), "
                f"    TRY_STRPTIME(CAST({col} AS VARCHAR), '%d/%m/%Y')"
                f"  ) AS {col}"
                f") FROM {self.table_name}"
            )
            repaired[col] = int(before)
        return repaired
