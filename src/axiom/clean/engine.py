"""Compose the stages into a segment index for one series.

The output is metadata: intervals over bars that already exist, never a copy of the bars. Raw
Parquet is immutable (ADR-0010) and a clean run must be cheap enough to redo whenever a threshold
moves, which it is only if it produces a table of intervals rather than a corpus (ADR-0018).

Pure. Nothing here reads a file or opens a socket -- `axiom.clean.calendars` imports
`exchange_calendars` lazily, and that is the module's only external dependency. A test enforces
it, because the value of a pure engine is that it can be exercised exhaustively offline, and that
value survives exactly as long as nobody adds a download to it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import pyarrow as pa

from axiom.clean.config import CleanConfig
from axiom.clean.stages import Span, run_stages

SEGMENTS_SCHEMA = pa.schema(
    [
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset_class", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("frequency", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("start_ts", pa.int64(), nullable=False),
        pa.field("end_ts", pa.int64(), nullable=False),
        pa.field("n_bars", pa.int64(), nullable=False),
        pa.field("cut_reason_start", pa.string(), nullable=False),
        pa.field("cut_reason_end", pa.string(), nullable=False),
        pa.field("clean_config_hash", pa.string(), nullable=False),
        pa.field("clean_version", pa.int32(), nullable=False),
        pa.field("raw_artifact_sha256", pa.string(), nullable=False),
        pa.field("artifact_path", pa.string(), nullable=False),
    ]
)

DROPSTATS_SCHEMA = pa.schema(
    [
        pa.field("artifact_path", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("asset_class", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("frequency", pa.string(), nullable=False),
        pa.field("rule", pa.string(), nullable=False),
        pa.field("bars_dropped", pa.int64(), nullable=False),
        pa.field("runs_excised", pa.int64(), nullable=False),
        pa.field("segments_created", pa.int64(), nullable=False),
        pa.field("segments_dropped", pa.int64(), nullable=False),
        pa.field("total_bars", pa.int64(), nullable=False),
        pa.field("kept_bars", pa.int64(), nullable=False),
        pa.field("clean_config_hash", pa.string(), nullable=False),
        pa.field("clean_version", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class SeriesIdentity:
    """What a series is. Not in the bar columns -- it is constant within a file (ADR-0010)."""

    source: str
    market: str
    asset_class: str
    symbol: str
    frequency: str
    session_id: str
    artifact_path: str
    raw_artifact_sha256: str


@dataclass
class CleanResult:
    """One series cleaned: the segments, the per-rule accounting, and the totals."""

    identity: SeriesIdentity
    segments: list[dict[str, Any]] = field(default_factory=list)
    dropstats: list[dict[str, Any]] = field(default_factory=list)
    total_bars: int = 0
    kept_bars: int = 0

    @property
    def dropped_bars(self) -> int:
        return self.total_bars - self.kept_bars


def segment_id(identity: SeriesIdentity, start_ts: int) -> str:
    """A segment's unique name.

    The plan called for ``{symbol}:{frequency}:{start_ts}``, and the corpus proved that is not
    unique: Binance lists the same ticker on spot and on USDT-M futures, both at 1d, and the two
    listings begin on the same day. Thirty-two ids collided on the first full run. Source and
    market are what separate them, so both are in the id.
    """
    return ":".join(
        (identity.source, identity.market, identity.symbol, identity.frequency, str(start_ts))
    )


def _check_invariants(spans: list[Span], total: int, kept: int) -> None:
    """The four things that must be true of any segment set, asserted rather than assumed.

    These are cheap -- a handful of comparisons over a list that is almost always shorter than
    ten -- and they are the difference between a corpus-wide bug and a caught one. The corpus run
    checks them again over the assembled table, because an invariant that only holds per series
    is not the invariant anybody cares about.
    """
    last_end = -1
    for span in spans:
        assert span.start <= span.end, f"inverted span {span}"
        assert span.start > last_end, f"span {span} overlaps or precedes the one before it"
        assert span.start >= 0 and span.end < total, f"span {span} runs off a series of {total}"
        last_end = span.end
    assert kept == sum(s.n_bars for s in spans)
    assert 0 <= kept <= total


def clean_series(bars: pa.Table, identity: SeriesIdentity, config: CleanConfig) -> CleanResult:
    """Turn one series of bars into segment rows and drop-stat rows.

    ``bars`` must be sorted by ``ts`` and schema-v1 valid; both are guaranteed by the raw tier
    and neither is re-derived here. A series with no surviving segment is not an error -- a
    ticker that traded for forty days and delisted has no usable window, and saying so is the
    filter working.
    """
    rule = config.rule_for(identity.frequency)
    session = config.session_for(identity.session_id)
    columns = {
        name: bars[name].to_numpy(zero_copy_only=False)
        for name in ("ts", "open", "high", "low", "close", "volume")
    }
    total = bars.num_rows

    spans, stats = run_stages(
        columns,
        config=config,
        rule=rule,
        session=session,
        frequency=identity.frequency,
    )
    kept = sum(s.n_bars for s in spans)
    _check_invariants(spans, total, kept)

    ts = columns["ts"]
    config_hash = config.config_hash
    segments = [
        {
            "segment_id": segment_id(identity, int(ts[s.start])),
            "source": identity.source,
            "market": identity.market,
            "asset_class": identity.asset_class,
            "symbol": identity.symbol,
            "frequency": identity.frequency,
            "session_id": identity.session_id,
            "start_ts": int(ts[s.start]),
            "end_ts": int(ts[s.end]),
            "n_bars": s.n_bars,
            "cut_reason_start": s.reason_start,
            "cut_reason_end": s.reason_end,
            "clean_config_hash": config_hash,
            "clean_version": config.clean_version,
            "raw_artifact_sha256": identity.raw_artifact_sha256,
            "artifact_path": identity.artifact_path,
        }
        for s in spans
    ]

    dropstats = [
        {
            "artifact_path": identity.artifact_path,
            "source": identity.source,
            "market": identity.market,
            "asset_class": identity.asset_class,
            "symbol": identity.symbol,
            "frequency": identity.frequency,
            **{k: int(v) for k, v in asdict(stat).items() if k != "rule"},
            "rule": stat.rule,
            "total_bars": total,
            "kept_bars": kept,
            "clean_config_hash": config_hash,
            "clean_version": config.clean_version,
        }
        for stat in stats
    ]

    return CleanResult(
        identity=identity,
        segments=segments,
        dropstats=dropstats,
        total_bars=total,
        kept_bars=kept,
    )


def segments_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build the segment index, in the canonical order that makes a rerun byte-identical."""
    return _table(
        rows,
        SEGMENTS_SCHEMA,
        ("source", "asset_class", "market", "symbol", "frequency", "start_ts"),
    )


def dropstats_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table(
        rows, DROPSTATS_SCHEMA, ("source", "asset_class", "market", "symbol", "frequency", "rule")
    )


def _table(rows: list[dict[str, Any]], schema: pa.Schema, sort_key: tuple[str, ...]) -> pa.Table:
    ordered = sorted(rows, key=lambda r: tuple(_sortable(r[k]) for k in sort_key))
    columns = {f.name: pa.array([r.get(f.name) for r in ordered], f.type) for f in schema}
    return pa.table(columns, schema=schema)


def _sortable(value: Any) -> tuple[int, Any]:
    """Sort ints as ints and strings as strings, without comparing the two.

    A key that stringified everything would order ``start_ts`` lexicographically, which is fine
    until a series crosses a digit-count boundary and the segment table silently reorders.
    """
    return (0, value) if isinstance(value, int | np.integer) else (1, str(value))


def verify_corpus_invariants(segments: pa.Table) -> list[str]:
    """Check the segment index as a whole. Returns the problems, empty when there are none.

    Per-series asserts fire inside :func:`clean_series`. This is the other half: that the
    assembled table has no duplicate segment ids, no overlaps within a series, and one config
    hash throughout -- none of which any single series can tell you.
    """
    problems: list[str] = []
    if segments.num_rows == 0:
        return problems

    hashes = set(segments["clean_config_hash"].to_pylist())
    if len(hashes) > 1:
        problems.append(f"segments carry {len(hashes)} different config hashes: {sorted(hashes)}")

    counts = Counter(segments["segment_id"].to_pylist())
    dupes = sorted(i for i, c in counts.items() if c > 1)
    if dupes:
        problems.append(f"{len(dupes)} duplicate segment_id(s), e.g. {dupes[:5]}")

    paths = segments["artifact_path"].to_pylist()
    starts = segments["start_ts"].to_pylist()
    ends = segments["end_ts"].to_pylist()
    by_path: dict[str, list[tuple[int, int]]] = {}
    for path, start, end in zip(paths, starts, ends, strict=True):
        by_path.setdefault(path, []).append((start, end))
    for path, spans in by_path.items():
        spans.sort()
        for (a_start, a_end), (b_start, _) in pairwise(spans):
            if b_start <= a_end:
                problems.append(f"{path}: segments overlap at {a_start}..{a_end} and {b_start}")
                break
    return problems


def usable_windows(n_bars: np.ndarray | list[int], context: int = 512) -> int:
    """How many training windows of ``context`` bars a set of segments yields.

    ``sum(max(0, n - context + 1))``. This is the number v0.5 sizes the corpus against, and it is
    not the bar count: a segment of 511 bars contributes fifty million bars' worth of nothing.
    """
    n = np.asarray(n_bars, dtype=np.int64)
    return int(np.maximum(n - context + 1, 0).sum())
