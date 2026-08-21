"""Canonical bar schema v1 and its invariants (ADR-0010).

Every source in the corpus is translated into this shape at parse time. The module is
deliberately pure: it takes a table in memory and reports on it. Nothing here reads a file,
opens a socket, or knows what Binance is.

Identity -- source, market, symbol, frequency -- is not in the columns. It is constant within a
file, so it lives in the path, the sidecar manifest, and the Parquet key-value metadata block
that :func:`bars_metadata` builds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

#: Bumping this is a re-pull, not a migration. See ADR-0010.
BARS_SCHEMA_VERSION = 1

#: Parquet row-group size. Large enough that per-group overhead is noise, small enough that a
#: reader can skip most of a multi-year series without decompressing it.
ROW_GROUP_SIZE = 131_072

#: Milliseconds per bar, per supported frequency. 1d bars open at 00:00 UTC, which falls out of
#: alignment to an 86 400 000 ms grid because the epoch itself is a UTC midnight.
FREQUENCIES: dict[str, int] = {"1h": 3_600_000, "1d": 86_400_000}

#: Columns downstream consumes. Everything else in the schema is retained raw detail.
OHLCVA = ("open", "high", "low", "close", "volume", "amount")

#: Trading calendars a file can declare, as file-level metadata (ADR-0014). Not a column.
#:
#: - ``24x7``   crypto: every grid slot is tradeable, and a gap is an outage
#: - ``24x5``   FX and CFDs: one continuous session from Sunday evening to Friday evening
#: - ``XNYS-regular``  US equities, regular hours, one bar per exchange calendar date
SESSIONS = frozenset({"24x7", "24x5", "XNYS-regular"})

#: The span in which a 24x5 market is normally shut.
#:
#: Dukascopy's week boundary follows its server clock, which observes European DST: in modern
#: data the week opens 22:00 UTC Sunday in winter and 21:00 UTC in summer. All of Saturday and
#: Sunday before 20:00 UTC is outside that in both regimes.
#:
#: A bar in here is **counted, not rejected**. Measuring the feed across its whole history found
#: two reasons a real bar lands in the window, and neither is a corrupt timestamp: the week
#: opened at 19:00 UTC in 2003, earlier than it does now, and some eras pad the weekend with
#: synthetic flat zero-volume bars carrying the Friday close forward. Rejecting those would throw
#: away a whole instrument's history over a vendor convention, and would be the undocumented
#: cleaning pass ADR-0010 exists to forbid. v0.3 drops the padding; the raw tier records it and
#: says how much there is (ADR-0015).
WEEKEND_CLOSE_UNTIL_HOUR_UTC = 20

#: A ts above this is not a plausible millisecond timestamp (it would be year 5138), so it is
#: microseconds. Binance Vision has shipped both units; the parser detects rather than trusts.
_US_THRESHOLD = 10**14

BARS_SCHEMA_V1 = pa.schema(
    [
        # Bar OPEN time, UTC, milliseconds. Not close time -- that ambiguity is the classic way
        # to shift a whole corpus by one bar without noticing.
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),  # base asset
        pa.field("amount", pa.float64(), nullable=False),  # quote asset
        pa.field("n_trades", pa.int64(), nullable=True),
        pa.field("taker_buy_volume", pa.float64(), nullable=True),
        pa.field("taker_buy_quote_volume", pa.float64(), nullable=True),
    ]
)


def grid_step_ms(frequency: str) -> int:
    """Milliseconds between two consecutive bar opens at ``frequency``."""
    try:
        return FREQUENCIES[frequency]
    except KeyError:
        raise ValueError(
            f"unsupported frequency {frequency!r}; v0.1 carries {sorted(FREQUENCIES)}"
        ) from None


def normalize_ts_ms(ts: np.ndarray | pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Return timestamps as int64 milliseconds, detecting microsecond input by magnitude.

    The rule is a magnitude test rather than a per-source setting because the unit has changed
    inside one source before, and a per-source setting would have been right at the time it was
    written and wrong afterwards. Detection is per-array, so a mixed array is impossible: either
    every value is over the threshold or the array is treated as milliseconds.
    """
    values = np.asarray(ts if isinstance(ts, np.ndarray) else ts.to_numpy(zero_copy_only=False))
    values = values.astype(np.int64, copy=False)
    if values.size and int(values.max()) >= _US_THRESHOLD:
        return values // 1000
    return values


@dataclass(frozen=True)
class Violation:
    """One broken invariant: how many rows broke it, and where the first one is."""

    count: int
    first_row: int

    def __str__(self) -> str:
        return f"{self.count} row(s), first at index {self.first_row}"


@dataclass
class ValidationReport:
    """What :func:`validate_bars` found.

    Two buckets, and the difference matters. A **violation** means the file cannot be true — a
    high below its own open is not something a market did. A **warning** means the file is odd
    but honest, and the raw tier's job is to carry it forward rather than to argue with it.
    """

    frequency: str
    row_count: int
    violations: dict[str, Violation] = field(default_factory=dict)
    warnings: dict[str, Violation] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        parts = []
        if self.violations:
            broken = "; ".join(f"{name}: {v}" for name, v in sorted(self.violations.items()))
            parts.append(f"violations -- {broken}")
        if self.warnings:
            noted = "; ".join(f"{name}: {v}" for name, v in sorted(self.warnings.items()))
            parts.append(f"warnings -- {noted}")
        tail = ", ".join(parts) if parts else "all invariants hold"
        return f"{self.row_count} rows, {self.frequency}, {tail}"

    def raise_for_status(self, context: str = "bars") -> None:
        if not self.ok:
            raise ValueError(f"{context}: {self.summary()}")


def _record(
    report: ValidationReport,
    name: str,
    bad: np.ndarray,
    row_offset: int = 0,
    *,
    warning: bool = False,
) -> None:
    """Record a boolean mask of offending rows, if any are set."""
    offenders = np.flatnonzero(bad)
    if offenders.size:
        bucket = report.warnings if warning else report.violations
        bucket[name] = Violation(int(offenders.size), int(offenders[0]) + row_offset)


def validate_bars(
    table: pa.Table,
    frequency: str,
    *,
    session_id: str = "24x7",
    raise_on_error: bool = False,
) -> ValidationReport:
    """Check every ADR-0010 invariant over ``table``, vectorized.

    Gaps in the timestamp grid are not a violation, under any session. A gap is a fact about the
    market -- a halt, a weekend, a listing that had not happened yet, an outage at the exchange --
    and the raw tier records facts. Filling one would be the cleaning pass that v0.3 owns. That
    holds for crypto too: a 24x7 series with a missing hour is *counted*, never failed, because
    "the exchange was down" is data.

    ``session_id`` adds the two checks that only make sense once a market can be shut:

    - **24x5 intraday** -- bars inside the weekend close are *counted*, as a warning. They are
      real bars the vendor published: an earlier historical reopen, or synthetic flat padding
      carrying the Friday close across the weekend. Which of the two is a question for v0.3, and
      failing the series here would answer it by deleting the instrument.
    - **XNYS-regular daily** -- bars must sit exactly on 00:00 UTC. For crypto that alignment is a
      warning, because Binance really does publish phase-shifted bars after a restart; for a
      vendor's daily equity dump it is a parse guarantee, and a violation of it means the date
      column was misread.
    """
    step = grid_step_ms(frequency)
    if session_id not in SESSIONS:
        raise ValueError(f"unknown session_id {session_id!r}; expected one of {sorted(SESSIONS)}")
    report = ValidationReport(frequency=frequency, row_count=table.num_rows)

    missing = [name for name in BARS_SCHEMA_V1.names if name not in table.column_names]
    if missing:
        raise ValueError(f"table is missing schema columns: {missing}")
    for name in ("ts", *OHLCVA):
        expected = BARS_SCHEMA_V1.field(name).type
        actual = table.schema.field(name).type
        if actual != expected:
            raise ValueError(f"column {name!r} has type {actual}, expected {expected}")

    if table.num_rows == 0:
        report.violations["empty"] = Violation(0, 0)
        if raise_on_error:
            report.raise_for_status()
        return report

    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in ("ts", *OHLCVA)}

    for name in ("ts", *OHLCVA):
        values = columns[name]
        # NaN counts as null for a float column: a NaN close is a missing price wearing a number.
        nulls = np.isnan(values) if values.dtype.kind == "f" else np.zeros(len(values), bool)
        if table[name].null_count:
            nulls = nulls | np.asarray(table[name].is_null())
        _record(report, f"null_{name}", nulls)

    ts = columns["ts"]
    _record(report, "ts_not_increasing", np.diff(ts) <= 0, row_offset=1)
    # Off-grid bars are a warning, not a violation. Binance Vision publishes stretches of
    # phase-shifted bars after an exchange restart -- 43 consecutive hourly bars on spot
    # BTCUSDT from 2018-02-09, all offset by the same 28m14.789s, each still exactly one hour
    # after the last. Those are real bars that really traded. Snapping them to the grid would
    # be imputation, and rejecting them would throw away the most important series in the
    # corpus over 0.05% of its rows. They are counted into the manifest instead (ADR-0010).
    off_grid = ts % step != 0
    strict_grid = session_id == "XNYS-regular" and step >= 86_400_000
    _record(report, "ts_off_grid", off_grid, warning=not strict_grid)
    _record(
        report,
        "bars_in_weekend_close",
        closed_window_bars(ts, session_id, frequency),
        warning=True,
    )

    open_, high, low, close = (columns[k] for k in ("open", "high", "low", "close"))
    _record(report, "high_below_open_or_close", high < np.maximum(open_, close))
    _record(report, "low_above_open_or_close", low > np.minimum(open_, close))
    _record(report, "high_below_low", high < low)
    _record(report, "volume_negative", columns["volume"] < 0)
    _record(report, "amount_negative", columns["amount"] < 0)

    if raise_on_error:
        report.raise_for_status()
    return report


def weekday_utc(ts: np.ndarray) -> np.ndarray:
    """Day of week per timestamp, 0 = Monday.

    1970-01-01 was a Thursday, which is index 3 under a Monday-first convention -- hence the +3.
    """
    return ((ts // 86_400_000) + 3) % 7


def closed_window_bars(ts: np.ndarray, session_id: str, frequency: str) -> np.ndarray:
    """Boolean mask of bars that fall when ``session_id`` says the market was shut.

    Only meaningful for intraday bars. A daily bar is stamped at 00:00 UTC of its calendar date
    by convention (ADR-0014), so a 24x5 daily series legitimately carries a Sunday bar -- the
    two-hour tail of the week's opening evening -- and testing its hour would reject real data.
    """
    if session_id != "24x5" or grid_step_ms(frequency) >= 86_400_000:
        return np.zeros(len(ts), dtype=bool)
    dow = weekday_utc(ts)
    hour = (ts % 86_400_000) // 3_600_000
    return (dow == 5) | ((dow == 6) & (hour < WEEKEND_CLOSE_UNTIL_HOUR_UTC))


def count_off_grid(ts: np.ndarray | pa.ChunkedArray, frequency: str) -> int:
    """Bars whose open time is not a multiple of the frequency step. Recorded, never repaired."""
    step = grid_step_ms(frequency)
    values = np.asarray(ts if isinstance(ts, np.ndarray) else ts.to_numpy(zero_copy_only=False))
    return int(np.count_nonzero(values % step))


def count_closed_window(ts: np.ndarray | pa.ChunkedArray, session_id: str, frequency: str) -> int:
    """Bars falling when the market was normally shut. Recorded, never repaired."""
    values = np.asarray(ts if isinstance(ts, np.ndarray) else ts.to_numpy(zero_copy_only=False))
    return int(np.count_nonzero(closed_window_bars(values, session_id, frequency)))


def count_gaps(ts: np.ndarray | pa.ChunkedArray, frequency: str) -> int:
    """Number of grid slots between the first and last bar that hold no bar.

    Recorded in the manifest so a series with a two-week outage is visibly different from one
    without, long before anybody plots it.
    """
    step = grid_step_ms(frequency)
    values = np.asarray(ts if isinstance(ts, np.ndarray) else ts.to_numpy(zero_copy_only=False))
    if values.size < 2:
        return 0
    expected = (int(values[-1]) - int(values[0])) // step + 1
    return int(expected - values.size)


def bars_metadata(
    *,
    source: str,
    asset_class: str,
    market: str,
    symbol: str,
    frequency: str,
    manifest_sha256: str,
    exchange_tz: str = "UTC",
    session_id: str = "24x7",
) -> dict[bytes, bytes]:
    """The Parquet key-value metadata block for one bar file.

    A file separated from its sidecar manifest still knows what it is. ``manifest_sha256`` is the
    link back: it names exactly one manifest, and a manifest whose hash does not match this file's
    sidecar does not describe this file.
    """
    payload = {
        "axiom_schema_version": str(BARS_SCHEMA_VERSION),
        "source": source,
        "asset_class": asset_class,
        "market": market,
        "symbol": symbol,
        "frequency": frequency,
        "exchange_tz": exchange_tz,
        "session_id": session_id,
        "manifest_sha256": manifest_sha256,
    }
    return {k.encode("utf-8"): v.encode("utf-8") for k, v in payload.items()}
