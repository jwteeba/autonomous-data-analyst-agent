from __future__ import annotations
import time

import numpy as np
import pandas as pd

from app.state import AgentState


def _detect_outliers_iqr(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 5:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


async def cleaning_node(state: AgentState) -> AgentState:
    """Validates and reports data quality issues WITHOUT modifying the source file."""
    t0 = time.time()
    tool = state["sql_tool"]
    df = tool.as_dataframe().copy()  # operate on a copy, never the source

    report: dict = {"columns": {}, "duplicate_rows": int(df.duplicated().sum())}

    for col in df.columns:
        col_report: dict = {}
        series = df[col]
        missing = int(series.isna().sum())
        if missing:
            col_report["missing_values"] = missing

        if series.dtype == object:
            # inconsistent casing/whitespace in categorical-looking columns
            non_null = series.dropna().astype(str)
            if non_null.nunique() < max(50, len(non_null) * 0.5):
                normalized = non_null.str.strip().str.lower()
                if normalized.nunique() < non_null.nunique():
                    col_report["categorical_inconsistencies"] = int(
                        non_null.nunique() - normalized.nunique()
                    )
            # try to detect columns that are "really" dates but stored as strings
            if "date" in col.lower():
                parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
                bad = int(parsed.isna().sum())
                if bad:
                    col_report["unparseable_or_mixed_format_dates"] = bad

        # pandas' is_numeric_dtype() returns True for boolean columns too, but
        # numpy can't do the subtraction quantile/IQR math needs on booleans
        # (raises "numpy boolean subtract..."), and "outliers" aren't a
        # meaningful concept for a True/False column anyway — skip them.
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            outliers = _detect_outliers_iqr(series)
            if outliers:
                col_report["outliers_iqr_method"] = outliers
            if (series.dropna() < 0).any() and col.lower() in ("quantity", "unit_price", "revenue", "price", "amount"):
                col_report["impossible_negative_values"] = int((series.dropna() < 0).sum())

        if col_report:
            report["columns"][col] = col_report

    # Repair mixed-format date columns in the in-memory copy so downstream
    # SQL/Python nodes don't fail on rows like "02/09/2023" mixed in with
    # ISO timestamps. Source file on disk is never touched.
    date_cols = [c for c in df.columns if "date" in c.lower()]
    repaired = tool.normalize_date_columns(date_cols)
    if repaired:
        report["auto_repaired"] = {
            "action": "normalized mixed-format date strings to timestamps (in-memory copy only)",
            "rows_repaired_by_column": repaired,
        }

    trace = state.get("trace", [])
    trace.append({
        "node": "data_validation_and_cleaning",
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "status": "ok",
        "detail": f"{report['duplicate_rows']} duplicate rows, "
                   f"{len(report['columns'])} columns flagged with issues. "
                   "Source file was not modified.",
    })
    return {**state, "cleaning_report": report, "trace": trace}
