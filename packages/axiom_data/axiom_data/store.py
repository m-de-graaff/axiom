"""Parquet corpus layout + DuckDB query helpers (P1-05, P1-07).

Layout: `{root}/{venue}/{symbol}/{tf}/year=YYYY/month=MM.parquet`
Columns: `ts, open, high, low, close, volume, amount` (`ts` = bar close, see
`axiom_data.resample`). One file per month keeps re-ingest of a single month cheap
and makes the tree diffable for `modal volume put`.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_ROOT = Path(os.environ.get("AXIOM_DATA_ROOT", "data/parquet"))
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "amount"]


def month_path(root: Path, venue: str, symbol: str, tf: str, year: int, month: int) -> Path:
    return Path(root) / venue / symbol / tf / f"year={year:04d}" / f"month={month:02d}.parquet"


def write_months(
    df: pd.DataFrame, root: Path, venue: str, symbol: str, tf: str, merge: bool = True
) -> list[Path]:
    """Split `df` by calendar month and write one parquet per month.

    `merge=True` folds new rows into an existing month file (new rows win on a `ts`
    collision). That matters because a monthly source zip spills one bar into the
    next month under close-labeling: without merging, re-ingesting a single month
    would silently drop its first bar.
    """
    if df.empty:
        return []
    df = df.sort_values("ts").drop_duplicates(subset="ts")
    cols = ["ts", *[c for c in df.columns if c != "ts"]]
    written = []
    for (year, month), part in df.groupby([df.ts.dt.year, df.ts.dt.month], sort=True):
        path = month_path(root, venue, symbol, tf, int(year), int(month))
        path.parent.mkdir(parents=True, exist_ok=True)
        part = part[cols]
        if merge and path.exists():
            part = (
                pd.concat([pd.read_parquet(path), part])
                .drop_duplicates(subset="ts", keep="last")
                .sort_values("ts")
            )
        part.to_parquet(path, index=False)
        written.append(path)
    return written


def glob_pattern(root: Path, venue: str, symbol: str, tf: str) -> str:
    return str(Path(root) / venue / symbol / tf / "year=*" / "month=*.parquet")


def available_symbols(root: Path, venue: str, tf: str) -> list[str]:
    base = Path(root) / venue
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / tf).is_dir())


def read(
    symbol: str,
    tf: str,
    root: Path = DEFAULT_ROOT,
    venue: str = "binance",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one symbol/timeframe as a ts-sorted DataFrame. `start`/`end` inclusive."""
    where = []
    params: list[object] = []
    if start is not None:
        where.append("ts >= ?")
        params.append(pd.Timestamp(start))
    if end is not None:
        where.append("ts <= ?")
        params.append(pd.Timestamp(end))
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT {', '.join(columns or COLUMNS)} FROM read_parquet(?) {clause} ORDER BY ts"
    with duckdb.connect() as con:
        try:
            return con.execute(sql, [glob_pattern(root, venue, symbol, tf), *params]).df()
        except duckdb.IOException:  # nothing ingested for this symbol/timeframe yet
            return pd.DataFrame(columns=columns or COLUMNS)


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Escape hatch for ad-hoc DuckDB over the tree; pass globs from `glob_pattern`."""
    with duckdb.connect() as con:
        return con.execute(sql, params or []).df()


__all__ = [
    "COLUMNS",
    "DEFAULT_ROOT",
    "available_symbols",
    "glob_pattern",
    "month_path",
    "query",
    "read",
    "write_months",
]
