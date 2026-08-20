"""Parse layer: Binance Vision kline archives into canonical bar tables (ADR-0010).

Three things here are not obvious from the file format, and each of them has bitten somebody:

* The archives ship **both with and without a header row**, in the same directory tree. The
  format changed partway through the bucket's life and old files were not rewritten. Deciding by
  date would need a cutoff nobody has published, so the parser sniffs the first field instead.
* `open_time` has appeared in **milliseconds and in microseconds**. The unit is detected by
  magnitude rather than assumed (ADR-0010).
* The last complete monthly archive and the first daily archive **overlap**. The same bar is
  published twice, and it must be published identically; a disagreement means the two files
  disagree about the market, which is a failure rather than something to average.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from axiom.schema.bars import BARS_SCHEMA_V1, normalize_ts_ms

#: The 12 columns Binance writes, in order. Positional, because headerless files have no other
#: way to be read and header-bearing ones must agree with them anyway.
SOURCE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "n_trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

#: Source column -> schema column. `close_time` is derivable and `ignore` is a placeholder, so
#: neither is read at all (ADR-0010).
COLUMN_MAP = {
    "open_time": "ts",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "quote_asset_volume": "amount",
    "n_trades": "n_trades",
    "taker_buy_volume": "taker_buy_volume",
    "taker_buy_quote_volume": "taker_buy_quote_volume",
}

#: Everything is read as float64 and cast afterwards. Binance has written integers as `0`, as
#: `0.0`, and in scientific notation across the corpus; one permissive read plus an explicit cast
#: is shorter than a per-column theory of which files are which.
_READ_TYPES = {name: pa.float64() for name in SOURCE_COLUMNS}


def _has_header(first_line: bytes) -> bool:
    """True when the first row is a header rather than data."""
    field = first_line.split(b",", 1)[0].strip().strip(b'"')
    try:
        float(field)
    except ValueError:
        return True
    return False


def read_csv_bytes(raw: bytes) -> pa.Table:
    """Read one kline CSV into its source columns, header or not."""
    if not raw.strip():
        raise ValueError("empty kline CSV")
    return pacsv.read_csv(
        io.BytesIO(raw),
        read_options=pacsv.ReadOptions(
            column_names=list(SOURCE_COLUMNS),
            skip_rows=1 if _has_header(raw.split(b"\n", 1)[0]) else 0,
        ),
        convert_options=pacsv.ConvertOptions(
            column_types=_READ_TYPES,
            include_columns=list(COLUMN_MAP),
        ),
    )


def extract_csv(data: bytes) -> bytes:
    """Pull the single CSV member out of a kline zip."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in the archive, found {members}")
        return archive.read(members[0])


def to_bars(source: pa.Table) -> pa.Table:
    """Map source columns onto schema v1: rename, normalize timestamps, cast."""
    if source.num_rows == 0:
        raise ValueError("kline CSV parsed to zero rows")

    ts = normalize_ts_ms(source["open_time"].to_numpy(zero_copy_only=False))
    columns: dict[str, pa.Array] = {"ts": pa.array(ts, type=pa.int64())}
    for src, dest in COLUMN_MAP.items():
        if dest == "ts":
            continue
        field = BARS_SCHEMA_V1.field(dest)
        columns[dest] = source[src].combine_chunks().cast(field.type, safe=False)

    return pa.table([columns[f.name] for f in BARS_SCHEMA_V1], schema=BARS_SCHEMA_V1)


def parse_archive(data: bytes) -> pa.Table:
    """Zip bytes -> a schema-v1 bar table. Unsorted and undeduplicated; see :func:`merge`."""
    return to_bars(read_csv_bytes(extract_csv(data)))


def _disagreeing_columns(table: pa.Table, rows: np.ndarray) -> list[str]:
    """Columns where row ``i`` and row ``i+1`` differ. NaN matches NaN: both mean 'absent'."""
    disagreeing = []
    for name in BARS_SCHEMA_V1.names:
        a = table[name].take(pa.array(rows)).to_numpy(zero_copy_only=False)
        b = table[name].take(pa.array(rows + 1)).to_numpy(zero_copy_only=False)
        same = a == b
        if a.dtype.kind == "f":
            same = same | (np.isnan(a) & np.isnan(b))
        if not same.all():
            disagreeing.append(name)
    return disagreeing


def merge(tables: list[pa.Table], *, context: str = "series") -> pa.Table:
    """Concatenate archives in period order, sort by ``ts``, and resolve the monthly/daily seam.

    The sort is stable, so when the same bar appears in both a monthly and a daily archive the
    monthly copy — passed first by the caller — is the one kept. Which copy survives should not
    matter, and the equality check below is what turns "should not" into "does not".
    """
    if not tables:
        raise ValueError(f"{context}: nothing to merge")

    combined = pa.concat_tables(tables).combine_chunks()
    ts = combined["ts"].to_numpy(zero_copy_only=False)
    combined = combined.take(pa.array(np.argsort(ts, kind="stable")))

    ts = combined["ts"].to_numpy(zero_copy_only=False)
    duplicated = np.flatnonzero(ts[1:] == ts[:-1])
    if duplicated.size:
        disagreeing = _disagreeing_columns(combined, duplicated)
        if disagreeing:
            first = int(ts[duplicated[0]])
            raise ValueError(
                f"{context}: the monthly and daily archives disagree at ts={first} on "
                f"{disagreeing}; one of the two source files is wrong and neither may be trusted"
            )
        keep = np.ones(len(ts), dtype=bool)
        keep[duplicated + 1] = False
        combined = combined.filter(pa.array(keep))

    return combined
