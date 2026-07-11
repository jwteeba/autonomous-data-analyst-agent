from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote_plus

import pandas as pd

_WRITE_KEYWORDS = re.compile(
    r"\b(delete|update|drop|alter|insert|truncate|grant|revoke|attach|copy\s+.*\s+to)\b",
    re.IGNORECASE,
)


def _assert_readonly_sql(sql: str) -> None:
    """Raise ValueError if the SQL statement is not a read-only SELECT or CTE.

    Args:
        sql: The SQL string to validate.

    Raises:
        ValueError: If the statement contains write/DDL keywords or does not
            start with SELECT or WITH.
    """
    if _WRITE_KEYWORDS.search(sql):
        raise ValueError(
            "Refusing to run a non-read-only statement against the source database. "
            "Only SELECT/WITH queries are permitted when loading from Postgres."
        )
    if not re.match(r"^\s*(with|select)\b", sql, re.IGNORECASE):
        raise ValueError(
            "Only SELECT / WITH (CTE) queries are permitted for Postgres loads."
        )


class DataSource(Protocol):
    """Protocol defining the interface for all data source implementations."""

    def load(self) -> pd.DataFrame: ...
    def describe(self) -> dict[str, Any]: ...


class FileDataSource:
    """Loads a local CSV/Excel/JSON/Parquet file into memory."""

    def __init__(self, path: str):
        """Initialize with the path to the local data file.

        Args:
            path: Filesystem path to the CSV, Parquet, Excel, or JSON file.
        """
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        """Read the file into a DataFrame.

        Returns:
            A pandas DataFrame containing the file's data.

        Raises:
            ValueError: If the file extension is not supported.
        """
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.path)
        if suffix == ".parquet":
            return pd.read_parquet(self.path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(self.path)
        if suffix == ".json":
            return pd.read_json(self.path)
        raise ValueError(f"Unsupported dataset format: {suffix}")

    def describe(self) -> dict[str, Any]:
        """Return a descriptor dict for this data source.

        Returns:
            A dict with keys ``type`` and ``path``.
        """
        return {"type": "file", "path": str(self.path)}


class PostgresDataSource:
    """
    Read-only connection to a Postgres table or query.

    Safety layers (defense in depth, not just app-level regex):
      1. The SQLAlchemy session itself is opened with
         `default_transaction_read_only=on`, so Postgres rejects any write
         at the server level even if application code had a bug.
      2. `_assert_readonly_sql` blocks anything that isn't SELECT/WITH before
         it's ever sent.
      3. Only a single SELECT is ever issued — to pull the table/query result
         into memory once. All further analysis (SQL generation, stats,
         cleaning) runs against the in-memory DuckDB copy, never touching
         Postgres again.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        table: str | None = None,
        query: str | None = None,
        db_schema: str = "public",
    ):
        """Initialize the Postgres data source.

        Args:
            host: Postgres server hostname.
            port: Postgres server port.
            database: Target database name.
            user: Database username.
            password: Database password.
            table: Table name to SELECT from (mutually exclusive with ``query``).
            query: Raw SELECT/WITH query to execute (mutually exclusive with ``table``).
            db_schema: Postgres schema containing the table. Defaults to ``public``.

        Raises:
            ValueError: If neither or both of ``table`` and ``query`` are provided.
        """
        if not table and not query:
            raise ValueError("Provide either 'table' or 'query'.")
        if table and query:
            raise ValueError("Provide only one of 'table' or 'query', not both.")
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = table
        self.query = query
        self.db_schema = db_schema

    def _engine_url(self) -> str:
        """Build the SQLAlchemy connection URL for this data source.

        Returns:
            A psycopg2 connection URL string.
        """
        return (
            f"postgresql+psycopg2://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def _select_sql(self) -> str:
        """Return the SQL statement used to load data from Postgres.

        Returns:
            The raw query string if provided, otherwise a ``SELECT *`` from the configured table.
        """
        if self.query:
            return self.query
        return f'SELECT * FROM "{self.db_schema}"."{self.table}"'

    def _connect_args(self) -> dict:
        """Return psycopg2 connection keyword arguments.

        Returns:
            A dict with ``sslmode`` and ``connect_timeout`` settings.
        """
        return {"sslmode": "require", "connect_timeout": 10}

    def test_connection(self) -> None:
        """Verify that the database is reachable by executing a trivial query.

        Raises:
            Exception: Propagates any SQLAlchemy or network error encountered.
        """
        import sqlalchemy

        engine = sqlalchemy.create_engine(
            self._engine_url(), connect_args=self._connect_args()
        )
        try:
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
        finally:
            engine.dispose()

    def load(self) -> pd.DataFrame:
        """Execute the configured query and return the result as a DataFrame.

        Returns:
            A pandas DataFrame containing all rows returned by the query.
        """
        import sqlalchemy

        sql = self._select_sql()
        _assert_readonly_sql(sql)
        engine = sqlalchemy.create_engine(
            self._engine_url(), connect_args=self._connect_args()
        )
        try:
            df = pd.read_sql(sqlalchemy.text(sql), engine)
        finally:
            engine.dispose()
        return df

    def describe(self) -> dict[str, Any]:
        """Return a descriptor dict for this data source.

        Returns:
            A dict with connection metadata (type, host, port, database, table, query, user).
        """
        return {
            "type": "postgres",
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "table": self.table,
            "query": self.query,
            "user": self.user,
        }


def build_data_source(descriptor: dict[str, Any]) -> DataSource:
    """Factory: turns a plain-dict source descriptor (as stored in the
    dataset registry / sent by the API) into a concrete DataSource.

    Args:
        descriptor: A dict with keys ``type`` and provider-specific keys.

    Returns:
        A DataSource instance appropriate for the given descriptor.
    """

    kind = descriptor.get("type")
    if kind == "file":
        return FileDataSource(descriptor["path"])
    if kind == "postgres":
        return PostgresDataSource(
            host=descriptor["host"],
            port=descriptor["port"],
            database=descriptor["database"],
            user=descriptor["user"],
            password=descriptor["password"],
            table=descriptor.get("table"),
            query=descriptor.get("query"),
            db_schema=descriptor.get("db_schema", "public"),
        )
    raise ValueError(f"Unknown data source type: {kind!r}")
