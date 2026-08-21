"""Synthetic bar series with ground truth about where cleaning should cut.

Library code, not test helpers. v0.4's contract tests and v0.8's leakage tripwires need series
whose pathologies are known exactly, and a generator that lives inside one test module is a
generator the next version reimplements.

**Independence is the point.** Nothing here imports :mod:`axiom.clean`. A toolkit that shared a
calendar helper with the engine it tests would agree with the engine about weekends by
construction, and the test would prove nothing. The session grids below are built from
``exchange_calendars`` and plain weekday arithmetic, on purpose, by a second route.

Annotations are in **timestamp space**, never row indices. Half these generators delete rows, and
an index-based annotation would silently point at the wrong bar the moment two of them compose.

Every generator takes a :class:`SynthSeries` and returns a new one, so pathologies stack::

    series = with_stagnant(with_gap(walk("1h", 600), at=200, n_bars=5), at=400, n=10)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa

from axiom.schema.bars import BARS_SCHEMA_V1, grid_step_ms, weekday_utc

MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000

#: Default first bar: 2015-01-05 00:00 UTC, a Monday. Starting on a Monday means a 24x5 series
#: opens with a full week rather than a two-bar stub, which would make the first weekend gap the
#: series' second event and every off-by-one harder to see.
DEFAULT_START_MS = 1_420_416_000_000

#: The 24x5 week these generators simulate: it closes after Friday `WEEK_CLOSE_HOUR` UTC and
#: reopens at Sunday `WEEK_OPEN_HOUR`. Dukascopy's real boundary moves an hour with European DST,
#: which `with_dst_weekend` reproduces by shifting both edges.
WEEK_CLOSE_HOUR = 21
WEEK_OPEN_HOUR = 22


@dataclass(frozen=True)
class SynthSeries:
    """Bars plus the truth about them.

    ``cut_ts`` names bars that must **start** a new segment: the bar after a jump, the bar after
    an unexplained gap. ``excised_ts`` names closed ``[first, last]`` timestamp ranges that must
    be removed entirely by a run rule. ``notes`` is for humans reading a failure.
    """

    table: pa.Table
    frequency: str
    session_id: str
    cut_ts: tuple[int, ...] = ()
    excised_ts: tuple[tuple[int, int], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ts(self) -> np.ndarray:
        return self.table["ts"].to_numpy(zero_copy_only=False)

    def column(self, name: str) -> np.ndarray:
        return self.table[name].to_numpy(zero_copy_only=False).copy()

    def index_of(self, ts: int) -> int:
        """Row index of a timestamp. Raises if the bar is not there, which is the useful failure."""
        hits = np.flatnonzero(self.ts == ts)
        if hits.size != 1:
            raise KeyError(f"ts {ts} appears {hits.size} time(s) in this series")
        return int(hits[0])


# --- construction -----------------------------------------------------------------------


def _amount(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume):
    """Quote volume the way every source without a native one gets it (ADR-0010)."""
    return volume * (open_ + high + low + close) / 4.0


def _table(
    ts: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> pa.Table:
    columns = {
        "ts": pa.array(ts.astype(np.int64), pa.int64()),
        "open": pa.array(open_.astype(np.float64), pa.float64()),
        "high": pa.array(high.astype(np.float64), pa.float64()),
        "low": pa.array(low.astype(np.float64), pa.float64()),
        "close": pa.array(close.astype(np.float64), pa.float64()),
        "volume": pa.array(volume.astype(np.float64), pa.float64()),
        "amount": pa.array(
            _amount(open_, high, low, close, volume).astype(np.float64), pa.float64()
        ),
        "n_trades": pa.nulls(len(ts), pa.int64()),
        "taker_buy_volume": pa.nulls(len(ts), pa.float64()),
        "taker_buy_quote_volume": pa.nulls(len(ts), pa.float64()),
    }
    return pa.table(columns, schema=BARS_SCHEMA_V1)


def _rebuild(series: SynthSeries, **columns: np.ndarray) -> SynthSeries:
    """A copy of ``series`` with some columns replaced, keeping OHLC self-consistent.

    Accepts ``open_`` for the ``open`` column, because the trailing underscore is what the rest
    of this module writes to stay clear of the builtin.
    """
    if "open_" in columns:
        columns["open"] = columns.pop("open_")
    data = {
        name: columns.get(name, series.column(name)) for name in ("open", "high", "low", "close")
    }
    volume = columns.get("volume", series.column("volume"))
    ts = columns.get("ts", series.ts)
    high = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    low = np.minimum(data["low"], np.minimum(data["open"], data["close"]))
    table = _table(ts, data["open"], high, low, data["close"], volume)
    return replace(series, table=table)


def session_grid(
    frequency: str, n: int, *, session_id: str = "24x7", start_ts: int = DEFAULT_START_MS
) -> np.ndarray:
    """``n`` bar-open timestamps a market with this session would actually publish.

    Built independently of :mod:`axiom.clean.calendars`. That duplication is deliberate: it is
    what makes "the engine agrees the weekend is expected" a fact rather than a tautology.
    """
    step = grid_step_ms(frequency)
    if session_id == "24x7":
        return start_ts + np.arange(n, dtype=np.int64) * step

    if session_id == "24x5":
        # Generate generously and filter, rather than reasoning about how many slots a week holds.
        candidate = start_ts + np.arange(n * 3 + 64, dtype=np.int64) * step
        dow = weekday_utc(candidate)
        if step >= MS_PER_DAY:
            keep = dow < 5
        else:
            hour = (candidate % MS_PER_DAY) // MS_PER_HOUR
            keep = ~(
                (dow == 5)
                | ((dow == 4) & (hour >= WEEK_CLOSE_HOUR))
                | ((dow == 6) & (hour < WEEK_OPEN_HOUR))
            )
        kept = candidate[keep]
        if kept.size < n:
            raise ValueError(f"could not build {n} 24x5 bars at {frequency}")
        return kept[:n]

    if session_id == "XNYS-regular":
        if step < MS_PER_DAY:
            raise NotImplementedError("intraday XNYS series are not needed before v0.5")
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS", start="1990-01-01", end="2035-12-31")
        days = calendar.sessions.view("int64") // (MS_PER_DAY * 1_000_000)
        days = days[days >= start_ts // MS_PER_DAY]
        if days.size < n:
            raise ValueError(f"only {days.size} XNYS sessions after the start date, need {n}")
        return (days[:n] * MS_PER_DAY).astype(np.int64)

    raise ValueError(f"unknown session_id {session_id!r}")


def walk(
    frequency: str = "1h",
    n: int = 512,
    *,
    seed: int = 0,
    session_id: str = "24x7",
    start_ts: int = DEFAULT_START_MS,
    price0: float = 100.0,
    volume: float = 1_000.0,
    sigma: float = 0.004,
) -> SynthSeries:
    """A clean base series: on its session's grid, no gaps, nothing a rule should cut.

    ``sigma`` is small enough that no bar-to-bar move approaches the loosest Table 4 jump
    threshold (0.10 at 1m), so any cut the engine reports on a bare ``walk`` is a bug in the
    engine rather than a tail draw. That is checked, not assumed -- see the toolkit self-tests.
    """
    rng = np.random.default_rng(seed)
    ts = session_grid(frequency, n, session_id=session_id, start_ts=start_ts)

    steps = rng.normal(0.0, sigma, size=n)
    close = price0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate(([price0], close[:-1]))
    wick = np.abs(rng.normal(0.0, sigma / 2, size=(2, n)))
    high = np.maximum(open_, close) * (1.0 + wick[0])
    low = np.minimum(open_, close) * (1.0 - wick[1])
    vol = volume * np.exp(rng.normal(0.0, 0.3, size=n))

    return SynthSeries(
        table=_table(ts, open_, high, low, close, vol),
        frequency=frequency,
        session_id=session_id,
        notes=(f"walk(seed={seed}, n={n}, {frequency}, {session_id})",),
    )


# --- corporate actions and discontinuities ----------------------------------------------


def with_split(series: SynthSeries, ratio: float, at: int) -> SynthSeries:
    """An **unadjusted** forward split at row ``at``: prices divide, volume multiplies.

    This is the discontinuity the jump rule exists for, and it must be cut.
    """
    ts = series.ts
    mask = np.arange(len(ts)) >= at
    scale = np.where(mask, 1.0 / ratio, 1.0)
    volume = series.column("volume") * np.where(mask, ratio, 1.0)
    out = _rebuild(
        series,
        open_=series.column("open") * scale,
        high=series.column("high") * scale,
        low=series.column("low") * scale,
        close=series.column("close") * scale,
        volume=volume,
    )
    return replace(
        out,
        cut_ts=(*out.cut_ts, int(ts[at])),
        notes=(*out.notes, f"unadjusted {ratio}:1 split at ts={int(ts[at])}"),
    )


def with_adjusted_split(series: SynthSeries, ratio: float, at: int) -> SynthSeries:
    """A split the vendor already adjusted: volume steps, prices do not. **No cut.**

    Kept as a separate generator rather than a no-op because the volume step is real, and a
    volume-sensitive rule that fired on it would be a bug this series catches.
    """
    ts = series.ts
    volume = series.column("volume") * np.where(np.arange(len(ts)) >= at, 1.0, ratio)
    out = _rebuild(series, volume=volume)
    return replace(out, notes=(*out.notes, f"adjusted {ratio}:1 split at ts={int(ts[at])}"))


def with_rollover_jump(series: SynthSeries, at: int, size: float = 0.35) -> SynthSeries:
    """A futures-style contract rollover: a one-off gap up at ``at``, then the walk continues."""
    ts = series.ts
    scale = np.where(np.arange(len(ts)) >= at, 1.0 + size, 1.0)
    out = _rebuild(
        series,
        open_=series.column("open") * scale,
        high=series.column("high") * scale,
        low=series.column("low") * scale,
        close=series.column("close") * scale,
    )
    return replace(
        out,
        cut_ts=(*out.cut_ts, int(ts[at])),
        notes=(*out.notes, f"rollover jump +{size:.0%} at ts={int(ts[at])}"),
    )


def with_flash_crash(series: SynthSeries, at: int, *, intrabar: bool, depth: float = 0.4):
    """A violent move at ``at``.

    ``intrabar=True`` puts the whole move inside one bar -- the low spikes down, the close
    recovers. That is a real price path and **must not** be cut: the jump rule compares open to
    the previous close and neither of them moved.

    ``intrabar=False`` puts the move across the bar boundary, which **must** be cut. Kronos cuts
    genuine crashes as well as data errors, and ADR-0018 keeps that behaviour and names the bias.
    """
    ts = series.ts
    low = series.column("low")
    if intrabar:
        low[at] = min(low[at], series.column("open")[at] * (1.0 - depth))
        out = _rebuild(series, low=low)
        note = f"intrabar flash crash -{depth:.0%} at ts={int(ts[at])} (no cut)"
        return replace(out, notes=(*out.notes, note))

    scale = np.where(np.arange(len(ts)) >= at, 1.0 - depth, 1.0)
    out = _rebuild(
        series,
        open_=series.column("open") * scale,
        high=series.column("high") * scale,
        low=series.column("low") * scale,
        close=series.column("close") * scale,
    )
    return replace(
        out,
        cut_ts=(*out.cut_ts, int(ts[at])),
        notes=(*out.notes, f"cross-bar flash crash -{depth:.0%} at ts={int(ts[at])}"),
    )


# --- gaps -------------------------------------------------------------------------------


def _drop_rows(series: SynthSeries, mask_keep: np.ndarray) -> SynthSeries:
    return replace(series, table=series.table.filter(pa.array(mask_keep)))


def with_gap(
    series: SynthSeries, at: int, n_bars: int = 3, kind: str = "unexpected"
) -> SynthSeries:
    """Open a hole in the grid at row ``at``.

    ``unexpected`` deletes ``n_bars`` rows, and the bar that follows must start a new segment.

    ``expected_weekend`` and ``expected_holiday`` delete nothing. The hole they name is already
    in the series -- a 24x5 walk has no Saturday bars, an XNYS walk has no Independence Day -- and
    the assertion worth making is that the engine does **not** cut there. They record a note and
    verify the hole is really there, so a test that thinks it is checking a weekend is not quietly
    checking a Tuesday.
    """
    ts = series.ts
    if kind == "unexpected":
        keep = np.ones(len(ts), dtype=bool)
        keep[at : at + n_bars] = False
        out = _drop_rows(series, keep)
        resumes = int(ts[at + n_bars])
        return replace(
            out,
            cut_ts=(*out.cut_ts, resumes),
            notes=(*out.notes, f"unexpected gap of {n_bars} bar(s), resumes at ts={resumes}"),
        )

    if kind not in ("expected_weekend", "expected_holiday"):
        raise ValueError(f"unknown gap kind {kind!r}")

    step = grid_step_ms(series.frequency)
    holes = np.flatnonzero(np.diff(ts) > step)
    holes = holes[holes >= at]
    if holes.size == 0:
        raise ValueError(
            f"asked for an {kind} at or after row {at}, but this series has no natural gap there; "
            "the base generator's session is probably 24x7, which by definition has none"
        )
    where = int(ts[holes[0]])
    return replace(series, notes=(*series.notes, f"{kind} after ts={where}, no cut expected"))


def with_dst_weekend(series: SynthSeries, shift_hours: int = 1) -> SynthSeries:
    """Move the 24x5 weekend edges by ``shift_hours``, the way a DST changeover does.

    The week closes an hour earlier and opens an hour later, which is the wider of the two real
    regimes: Dukascopy's boundary sits at 21:00 UTC in summer and 22:00 in winter. **No cut** --
    a session rule that only tolerated one of the two would partition every FX series twice a
    year.
    """
    if series.session_id != "24x5":
        raise ValueError("DST weekend edges only mean anything for a 24x5 session")
    if grid_step_ms(series.frequency) >= MS_PER_DAY:
        raise ValueError("a daily bar is stamped 00:00 UTC; its hour carries no DST information")

    ts = series.ts
    dow = weekday_utc(ts)
    hour = (ts % MS_PER_DAY) // MS_PER_HOUR
    # The bars adjacent to the shut window: late Friday and early Sunday-evening.
    edge = ((dow == 4) & (hour >= WEEK_CLOSE_HOUR - shift_hours)) | (
        (dow == 6) & (hour < WEEK_OPEN_HOUR + shift_hours)
    )
    out = _drop_rows(series, ~edge)
    return replace(
        out, notes=(*out.notes, f"DST-shifted weekend edges ({shift_hours}h), no cut expected")
    )


def with_suspension(series: SynthSeries, at: int, n_bars: int = 20) -> SynthSeries:
    """A trading suspension: a hole the session cannot explain, with segments both sides."""
    out = with_gap(series, at, n_bars=n_bars, kind="unexpected")
    return replace(out, notes=(*out.notes[:-1], f"suspension of {n_bars} bar(s) from row {at}"))


def ends_at(series: SynthSeries, ts: int) -> SynthSeries:
    """Delisting: the series simply stops. The last segment ends cleanly, with no cut reason."""
    keep = series.ts <= ts
    out = _drop_rows(series, keep)
    return replace(out, notes=(*out.notes, f"series ends at ts={ts} (delisting)"))


def truncate_tail(series: SynthSeries, n: int) -> SynthSeries:
    """Keep only the first ``n`` bars -- min-length bait.

    Composed after a cut, this leaves a final segment too short to survive ``min_bars``.
    """
    keep = np.arange(len(series.ts)) < n
    out = _drop_rows(series, keep)
    dropped = tuple(t for t in out.cut_ts if t in set(out.ts.tolist()))
    return replace(out, cut_ts=dropped, notes=(*out.notes, f"truncated to {n} bars"))


# --- run pathologies --------------------------------------------------------------------


def with_illiquid(series: SynthSeries, at: int, n: int) -> SynthSeries:
    """``n`` consecutive bars with zero volume. Prices keep moving."""
    volume = series.column("volume")
    volume[at : at + n] = 0.0
    out = _rebuild(series, volume=volume)
    ts = series.ts
    span = (int(ts[at]), int(ts[at + n - 1]))
    return replace(
        out,
        excised_ts=(*out.excised_ts, span),
        notes=(*out.notes, f"illiquid run of {n} bar(s) at ts={span[0]}..{span[1]}"),
    )


def with_stagnant(series: SynthSeries, at: int, n: int) -> SynthSeries:
    """``n`` consecutive bars with an identical close. Volume keeps trading."""
    close = series.column("close")
    open_ = series.column("open")
    frozen = float(close[at - 1]) if at > 0 else float(close[at])
    close[at : at + n] = frozen
    open_[at : at + n] = frozen
    out = _rebuild(series, open_=open_, close=close)
    ts = series.ts
    span = (int(ts[at]), int(ts[at + n - 1]))
    return replace(
        out,
        excised_ts=(*out.excised_ts, span),
        notes=(*out.notes, f"stagnant run of {n} bar(s) at ts={span[0]}..{span[1]}"),
    )


def with_limit_lock(series: SynthSeries, at: int, n: int) -> SynthSeries:
    """A limit-up/limit-down halt: the close freezes **and** volume goes to zero.

    Both run rules apply, which is the point -- the excision must be the same span either way.
    """
    out = with_stagnant(series, at, n)
    volume = out.column("volume")
    volume[at : at + n] = 0.0
    out = _rebuild(out, volume=volume)
    # `with_stagnant` already recorded the excision span, which is the same one either rule
    # produces. Only its note is replaced, so the halt is not described twice.
    return replace(out, notes=(*out.notes[:-1], f"limit lock of {n} bar(s) from row {at}"))


__all__ = [
    "DEFAULT_START_MS",
    "SynthSeries",
    "ends_at",
    "session_grid",
    "truncate_tail",
    "walk",
    "with_adjusted_split",
    "with_dst_weekend",
    "with_flash_crash",
    "with_gap",
    "with_illiquid",
    "with_limit_lock",
    "with_rollover_jump",
    "with_split",
    "with_stagnant",
    "with_suspension",
]
