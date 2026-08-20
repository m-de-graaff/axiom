"""Split and dividend events from Yahoo, via yfinance. An adjunct, never a pillar.

Two jobs, neither of which the corpus depends on:

1. Capture corporate actions as **data**, so v0.3's adjustment policy works from recorded events
   rather than from inferring them out of price discontinuities.
2. Supply the comparison that classifies what Stooq's adjustment actually is (ADR-0016).

Everything about this module assumes it may fail. Yahoo has no licence, no stability promise, and
an active habit of refusing datacenter IPs. Partial success is success; total failure is a dated
line in the audit report and the version proceeds, because the split probes stand on Stooq data
and a calendar alone.

**These are not bars**, so they do not go through the bar driver. An event series is
`(ts, event_type, value)` with no OHLC, no grid, and no frequency -- and running it through
`validate_bars` would mean weakening the bar invariants for the one caller that is not bars. The
manifest and the store are shared; the validation is not.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from axiom.config.settings import resolve_config_path
from axiom.provenance.manifest import FileManifest, sha256_bytes
from axiom.raw.store import RawStore
from axiom.sources.base import loader_version, shard_dir, write_parquet

log = logging.getLogger("axiom.yahoo")

SOURCE = "yahoo"
MARKET = "adjustments"
ASSET_CLASS = "equity"

#: A frequency name that is deliberately not one of the bar frequencies. Nothing indexes an event
#: series on a grid, and calling it `1d` would invite something downstream to try.
FREQUENCY = "events"

#: Client-side ceiling, per ADR-0016. Yahoo publishes no rate limit, which is a reason to pick a
#: conservative number rather than a licence to pick none -- and being the reason this endpoint
#: starts refusing everyone would be a poor trade for a non-load-bearing adjunct.
MAX_REQUESTS_PER_HOUR = 300

EVENT_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
)

EVENT_TYPES = ("split", "dividend")

#: `(yahoo_symbol) -> list of (ts_ms, event_type, value)`. Injected so the tests never touch Yahoo.
EventFetcher = Callable[[str], list[tuple[int, str, float]]]


@dataclass(frozen=True)
class Ticker:
    """One entry of the pinned cross-check population."""

    symbol: str
    yahoo_symbol: str


def load_tickers(name_or_path: str | Path = "yahoo_events_v1") -> list[Ticker]:
    """Load the pinned list. Its header says what it is and, more importantly, what it is not."""
    path = resolve_config_path(name_or_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Ticker(symbol=entry["symbol"], yahoo_symbol=entry["yahoo_symbol"])
        for entry in payload["tickers"]
    ]


def artifact_path(symbol: str) -> str:
    return f"raw/{SOURCE}/{MARKET}/{shard_dir(symbol)}/{symbol}.parquet"


def events_table(rows: Iterable[tuple[int, str, float]]) -> pa.Table:
    """Assemble an event series, sorted, with the types checked.

    A ticker with no corporate actions is a real answer and gets an empty table -- which is
    different from a fetch that failed, and the two must not be conflated: one means "nothing
    happened", the other means "we do not know".
    """
    ordered = sorted(rows, key=lambda r: (r[0], r[1]))
    for _, event_type, _ in ordered:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {event_type!r}; expected one of {EVENT_TYPES}")
    return pa.table(
        {
            "ts": pa.array([r[0] for r in ordered], pa.int64()),
            "event_type": pa.array([r[1] for r in ordered], pa.string()),
            "value": pa.array([float(r[2]) for r in ordered], pa.float64()),
        },
        schema=EVENT_SCHEMA,
    )


class RateLimiter:
    """A jittered client-side pacer.

    Not a token bucket: the point is not to maximize throughput up to a limit, it is to be
    unmistakably polite to an endpoint nobody granted us. The jitter keeps five hundred requests
    from arriving on a metronome, which is what a scraper looks like.
    """

    def __init__(
        self,
        per_hour: int = MAX_REQUESTS_PER_HOUR,
        *,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.interval = 3600.0 / per_hour
        self._sleep = sleep
        self._rng = rng or random.Random(0)

    def wait(self) -> None:
        self._sleep(self.interval * (0.75 + 0.5 * self._rng.random()))


def live_fetcher() -> EventFetcher:
    """The real fetcher, bound to yfinance. Imported lazily; CI installs no scraper."""
    import yfinance  # ty: ignore[unresolved-import]

    def fetch(yahoo_symbol: str) -> list[tuple[int, str, float]]:
        actions = yfinance.Ticker(yahoo_symbol).actions
        if actions is None or len(actions) == 0:
            return []
        rows: list[tuple[int, str, float]] = []
        for stamp, row in actions.iterrows():
            ts = (
                int(stamp.tz_localize("UTC").timestamp() * 1000)
                if stamp.tzinfo is None
                else int(stamp.timestamp() * 1000)
            )
            dividend = float(row.get("Dividends", 0.0) or 0.0)
            split = float(row.get("Stock Splits", 0.0) or 0.0)
            if dividend:
                rows.append((ts, "dividend", dividend))
            if split:
                rows.append((ts, "split", split))
        return rows

    return fetch


def live_price_fetcher(*, auto_adjust: bool = True):
    """Adjusted daily closes from Yahoo, for the cross-check.

    `auto_adjust=True` gives split *and* dividend adjusted closes. That is the comparison the
    audit wants: if Stooq tracks it closely, Stooq is dividend-adjusted too, and if it drifts on
    dividend payers only, it is not.
    """
    import yfinance  # ty: ignore[unresolved-import]

    def fetch(symbol: str, start: datetime, end: datetime) -> dict[int, float]:
        frame = yfinance.Ticker(symbol).history(
            start=start.date(), end=end.date(), auto_adjust=auto_adjust
        )
        if frame is None or len(frame) == 0:
            return {}
        out: dict[int, float] = {}
        for stamp, row in frame.iterrows():
            # Normalize to 00:00 UTC of the calendar date, the way every daily bar in the corpus
            # is stamped (ADR-0014), so the two paths can be joined on `ts` at all.
            day = datetime(stamp.year, stamp.month, stamp.day, tzinfo=UTC)
            out[int(day.timestamp() * 1000)] = float(row["Close"])
        return out

    return fetch


@dataclass
class EventResult:
    """What happened to one ticker. ``status`` is one of ok, skipped, failed."""

    symbol: str
    status: str
    events: int = 0
    error: str = ""


@dataclass
class EventRun:
    """The run's tally. Failures are listed rather than counted, because they will be read."""

    ok: int = 0
    skipped: int = 0
    failed: int = 0
    events: int = 0
    results: list[EventResult] = field(default_factory=list)

    def record(self, result: EventResult) -> None:
        self.results.append(result)
        setattr(self, result.status, getattr(self, result.status) + 1)
        self.events += result.events

    @property
    def failures(self) -> list[EventResult]:
        return [r for r in self.results if r.status == "failed"]

    def line(self) -> str:
        return f"ok={self.ok} skipped={self.skipped} failed={self.failed} events={self.events}"


def pull_ticker(
    ticker: Ticker,
    store: RawStore,
    fetch: EventFetcher,
    *,
    pull_run_id: str,
    as_of: str,
    force: bool = False,
) -> EventResult:
    """Fetch one ticker's events and land them, or report why not.

    Resume works the same way it does for bars: the sidecar records what the last run asked for,
    and a run on the same day skips a ticker it already has. Yahoo has nothing to checksum, so
    the as-of date is the whole digest -- events are append-only in practice, and re-asking daily
    is the correct cadence for a source that publishes a split the day it happens.
    """
    path = artifact_path(ticker.symbol)
    digest = sha256_bytes(f"yahoo://{ticker.yahoo_symbol}/actions#{as_of}".encode())

    try:
        return _pull_ticker(ticker, store, fetch, path, digest, pull_run_id, force=force)
    except Exception as exc:  # one bad ticker must not end a five-hundred-ticker run
        log.warning("failed %s: %s: %s", ticker.symbol, type(exc).__name__, exc)
        return EventResult(ticker.symbol, "failed", error=f"{type(exc).__name__}: {exc}")


def _pull_ticker(
    ticker: Ticker,
    store: RawStore,
    fetch: EventFetcher,
    path: str,
    digest: str,
    pull_run_id: str,
    *,
    force: bool,
) -> EventResult:
    """The body of :func:`pull_ticker`, so the blast wall around it can cover all of it.

    Every line here can fail, and two of the ways are not the fetch. `read_sidecar` and `put` are
    both calls to the Hub, and the Hub rate-limits: a 429 on the commit ended a 503-ticker run
    that had already landed 125 of them. `base.pull_item` never had that problem because it wraps
    its whole body; this function exists so this one does too.
    """
    try:
        remote = store.read_sidecar(path)
    except ValueError:  # a tampered sidecar is absent, not fatal -- re-pulling heals it
        remote = None
    if not force and remote is not None and remote.source_sha256s == [digest]:
        return EventResult(ticker.symbol, "skipped")

    table = events_table(fetch(ticker.yahoo_symbol))
    ts = table["ts"].to_pylist()
    manifest = FileManifest(
        schema_version=1,
        source=SOURCE,
        market=MARKET,
        asset_class=ASSET_CLASS,
        symbol=ticker.symbol,
        frequency=FREQUENCY,
        pull_run_id=pull_run_id,
        pulled_at=datetime.now(UTC).isoformat(),
        loader_version=loader_version(),
        source_urls=[f"yahoo://{ticker.yahoo_symbol}/actions"],
        source_sha256s=[digest],
        artifact_path=path,
        row_count=table.num_rows,
        first_ts=ts[0] if ts else 0,
        last_ts=ts[-1] if ts else 0,
        # An event series has no grid, so there is nothing for a gap to be a gap in.
        gap_count=0,
        off_grid_count=0,
        source_symbol=ticker.yahoo_symbol,
        volume_convention="n/a",
        amount_synthesized=False,
        adjustment_policy="none",
        price_side="n/a",
        # Yahoo grants nothing, so this is the strictest class: the loader may be published, the
        # manifests may not (`docs/DATA_LICENSING.md`).
        redistribution_class="loader_only_private",
        universe_hash="",
    )
    data = write_parquet(
        table,
        {
            b"axiom_schema_version": b"1",
            b"source": SOURCE.encode(),
            b"asset_class": ASSET_CLASS.encode(),
            b"market": MARKET.encode(),
            b"symbol": ticker.symbol.encode(),
            b"frequency": FREQUENCY.encode(),
            b"manifest_sha256": manifest.manifest_sha256.encode(),
        },
    )
    manifest = manifest.model_copy(update={"artifact_sha256": sha256_bytes(data)})
    store.put(path, data, manifest)
    return EventResult(ticker.symbol, "ok", events=table.num_rows)


def pull_events(
    tickers: list[Ticker],
    store: RawStore,
    *,
    pull_run_id: str,
    as_of: str,
    fetch: EventFetcher | None = None,
    limiter: RateLimiter | None = None,
    force: bool = False,
    fail_fast_after: int = 25,
) -> EventRun:
    """Walk the pinned list, paced, tolerating failures.

    ``fail_fast_after`` stops the run after that many *consecutive* failures. That is what a
    block looks like, and grinding through four hundred more requests at one every twelve seconds
    to rediscover it would take ninety minutes and annoy a backend that has already said no.
    Consecutive rather than total: a block can start partway through, and it did.
    """
    fetch = fetch or live_fetcher()
    limiter = limiter or RateLimiter()
    run = EventRun()
    consecutive = 0

    for index, ticker in enumerate(tickers, start=1):
        result = pull_ticker(
            ticker, store, fetch, pull_run_id=pull_run_id, as_of=as_of, force=force
        )
        run.record(result)
        log.info("[%d/%d] %s: %s", index, len(tickers), ticker.symbol, result.status)

        consecutive = consecutive + 1 if result.status == "failed" else 0
        if consecutive >= fail_fast_after:
            # Counted consecutively rather than in total, because a backend can start refusing
            # *partway*: the run that prompted this had landed 125 tickers before the Hub began
            # returning 429, and a total-with-nothing-landed rule would have ground through the
            # remaining 380 at twelve seconds each to learn what the first 25 already said.
            log.error(
                "%d consecutive failures; stopping. This is what a block looks like, and it is a "
                "documented outcome rather than a bug (ADR-0016)",
                consecutive,
            )
            break
        if result.status != "skipped":
            limiter.wait()

    store.flush()
    return run


def blocked_report(run: EventRun, *, as_of: str) -> str:
    """The dated line the audit report gets when Yahoo will not answer at all."""
    sample = "; ".join(f"{r.symbol}: {r.error}" for r in run.failures[:3])
    return (
        f"yfinance unavailable from this backend as of {as_of}: {run.failed} ticker(s) attempted, "
        f"none landed. Representative errors -- {sample}. The adjunct is non-load-bearing by "
        "design (ADR-0016); the split probes in the adjustment audit stand on Stooq data and a "
        "calendar alone, and v0.2 proceeds."
    )


def known_split_probes() -> dict[str, tuple[str, float]]:
    """The splits the adjustment audit checks Stooq against (ADR-0016).

    Chosen because each is large, recent, and unambiguous: a 4:1 leaves a 75% price cliff in an
    unadjusted series, which no market move imitates.
    """
    return {
        "AAPL": ("2020-08-31", 4.0),
        "TSLA": ("2022-08-25", 3.0),
        "NVDA": ("2024-06-10", 10.0),
    }


def detect_split_discontinuity(
    ts: list[int], close: list[float], split_date_ms: int, ratio: float, *, tolerance: float = 0.25
) -> dict[str, Any]:
    """Does the close path jump by roughly ``ratio`` across the split date?

    Returns the measured ratio and a verdict. An adjusted series shows a ratio near 1; an
    unadjusted one shows a ratio near the split. The tolerance is wide on purpose -- the question
    is "is there a 4x cliff here", not "what exactly was the close".
    """
    before = [(t, c) for t, c in zip(ts, close, strict=True) if t < split_date_ms]
    after = [(t, c) for t, c in zip(ts, close, strict=True) if t >= split_date_ms]
    if not before or not after:
        return {"measured": None, "adjusted": None, "reason": "no bars on one side of the split"}

    last_before = before[-1][1]
    first_after = after[0][1]
    if first_after <= 0:
        return {"measured": None, "adjusted": None, "reason": "non-positive close after the split"}

    measured = last_before / first_after
    return {
        "measured": round(measured, 4),
        "expected_if_unadjusted": ratio,
        # Near 1 means the vendor already applied the split.
        "adjusted": abs(measured - 1.0) <= tolerance,
        "reason": "",
    }
