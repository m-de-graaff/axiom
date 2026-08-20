"""FX and commodities from Dukascopy (ADR-0015).

The shape of this source is set by one fact: there is nothing upstream to checksum. Binance
publishes a `.CHECKSUM` beside every archive, so "is this series current" is answered by comparing
digests. Dukascopy publishes a query interface. Asking it whether anything changed costs the same
as fetching the data.

So the plan is built out of **calendar years** instead. A year that has ended cannot gain bars, so
its token is a constant; the current year can gain bars every day, so its token carries the run's
as-of date. That makes `is_current` -- one list comparison, shared with every other source -- mean
exactly the right thing here: a re-run on the same day skips a finished instrument, and a re-run
tomorrow re-extends it. No second skip mechanism, no override.

Prior years are then immutable by convention, which is what makes the rewrite cheap: only the
years at the end of the series are re-fetched, the rest are read back out of the artifact that is
already in the raw tier, and the file is written whole. Parquet has no append, and building one
would cost more than rewriting five megabytes.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pyarrow as pa

from axiom.schema.bars import BARS_SCHEMA_V1, normalize_ts_ms
from axiom.sources.base import SourcePlan, WorkItem
from axiom.universe.dukascopy import DukascopyUniverse, load_dukascopy_universe

log = logging.getLogger("axiom.dukascopy")

SOURCE = "dukascopy"

#: Our frequency names to the library's interval constants. Resolved lazily -- CI has no network
#: and no reason to install a broker client to run the offline tests.
INTERVALS = {"1h": "INTERVAL_HOUR_1", "1d": "INTERVAL_DAY_1"}

#: Bar frames, as `(source_symbol, frequency, start, end)` -> a schema-v1 table for that window.
Fetcher = Callable[[str, str, datetime, datetime], pa.Table]


def year_bounds(year: int) -> tuple[datetime, datetime]:
    """The half-open UTC window `[Jan 1 year, Jan 1 year+1)`."""
    return datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)


def year_token(source_symbol: str, frequency: str, year: int) -> str:
    """The identifier for one year-chunk of one series. Not a URL; nothing dereferences it."""
    return f"dukascopy://{source_symbol}/{frequency}/{year}"


def year_digest(token: str, *, sealed: bool, as_of: date) -> str:
    """The digest that decides whether a year-chunk needs fetching.

    A **sealed** year has ended: it cannot gain a bar, so its digest is a constant and a re-run
    skips it forever. The current year gets the as-of date folded in, so a re-run on the same day
    skips and a re-run tomorrow re-fetches. That is the whole resume policy, expressed in the one
    field `is_current` already compares.
    """
    material = token if sealed else f"{token}#{as_of.isoformat()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def to_bars(
    ts_ms: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> pa.Table:
    """Assemble a schema-v1 table, synthesizing `amount` per ADR-0010.

    Dukascopy has no quote volume of any kind, so `amount` is `volume x mean(OHLC)` -- which for
    this source is a *tick count* scaled into price units. It is retained for schema uniformity
    and flagged `amount_synthesized` so nothing downstream mistakes it for money (ADR-0014).
    """
    mean_price = (open_ + high + low + close) / 4.0
    n = len(ts_ms)
    return pa.table(
        {
            "ts": pa.array(ts_ms.astype(np.int64), pa.int64()),
            "open": pa.array(open_.astype(np.float64), pa.float64()),
            "high": pa.array(high.astype(np.float64), pa.float64()),
            "low": pa.array(low.astype(np.float64), pa.float64()),
            "close": pa.array(close.astype(np.float64), pa.float64()),
            "volume": pa.array(volume.astype(np.float64), pa.float64()),
            "amount": pa.array((volume * mean_price).astype(np.float64), pa.float64()),
            "n_trades": pa.array([None] * n, pa.int64()),
            "taker_buy_volume": pa.array([None] * n, pa.float64()),
            "taker_buy_quote_volume": pa.array([None] * n, pa.float64()),
        },
        schema=BARS_SCHEMA_V1,
    )


def frame_to_bars(frame: Any) -> pa.Table:
    """Translate one `dukascopy-python` DataFrame into schema v1.

    The library returns a tz-aware UTC `timestamp` index and `open, high, low, close, volume`.
    Nothing else: no trade count, no quote volume, no ask side.
    """
    if len(frame) == 0:
        return to_bars(*(np.array([], dtype=np.float64) for _ in range(6)))
    ts = normalize_ts_ms(frame.index.view("int64") // 1_000_000)
    columns = (frame[name].to_numpy(dtype=np.float64) for name in BARS_COLUMNS)
    return to_bars(ts, *columns)


BARS_COLUMNS = ("open", "high", "low", "close", "volume")


def live_fetcher(max_retries: int = 7) -> Fetcher:
    """The real fetcher, bound to `dukascopy-python`.

    Imported here rather than at module scope so the offline tests -- and CI, which has no network
    and installs no broker client -- can exercise everything else in this file.

    The library owns the bucket's URL scheme, and that is deliberate: Dukascopy numbers months
    from zero, so January lives under `/00/` and December under `/11/`. An off-by-one there shifts
    a whole series by a month without failing anything. `test_dukascopy.py` round-trips a known
    window through the live path to check the library gets it right, rather than assuming it.
    """
    import dukascopy_python  # ty: ignore[unresolved-import]

    def fetch(source_symbol: str, frequency: str, start: datetime, end: datetime) -> pa.Table:
        frame = dukascopy_python.fetch(
            source_symbol,
            getattr(dukascopy_python, INTERVALS[frequency]),
            dukascopy_python.OFFER_SIDE_BID,
            start,
            end,
            max_retries=max_retries,
        )
        return frame_to_bars(frame)

    return fetch


def clip_to_window(table: pa.Table, start: datetime, end: datetime) -> pa.Table:
    """Keep only the rows inside `[start, end)`.

    The library's window is inclusive at edges in a way that varies with the interval, so the
    year boundary is enforced here rather than trusted. Without it two adjacent year-chunks
    overlap by a bar and the splice produces a duplicate timestamp -- which `validate_bars` would
    catch, but as a confusing failure rather than as a thing that never happened.
    """
    lo, hi = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    ts = table["ts"].to_numpy(zero_copy_only=False)
    return table.filter(pa.array((ts >= lo) & (ts < hi)))


def splice(prior: pa.Table | None, fresh: list[pa.Table]) -> pa.Table:
    """Concatenate prior years with newly fetched ones, then sort and de-duplicate.

    De-duplication keeps the **last** row for a timestamp, so a re-fetched year wins over whatever
    the artifact held for the same bar. That is the right precedence: the fresh copy is the one
    the vendor is publishing now.
    """
    parts = [t for t in ([prior] if prior is not None else []) + fresh if t.num_rows]
    if not parts:
        return to_bars(*(np.array([], dtype=np.float64) for _ in range(6)))

    table = pa.concat_tables(parts).sort_by([("ts", "ascending")])
    ts = table["ts"].to_numpy(zero_copy_only=False)
    # Keep the last of each run of equal timestamps: a row survives if the next one differs.
    keep = np.ones(len(ts), dtype=bool)
    keep[:-1] = ts[:-1] != ts[1:]
    return table.filter(pa.array(keep))


class DukascopySource:
    """Dukascopy as the driver sees it."""

    name = SOURCE

    def __init__(
        self,
        universe: DukascopyUniverse | None = None,
        *,
        as_of: date | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.universe = universe or load_dukascopy_universe()
        #: Pinned once per run rather than read per item, so a pull that crosses midnight does not
        #: seal a year for half its instruments and not the other half.
        self.as_of = as_of or datetime.now(UTC).date()
        self._fetcher = fetcher
        self._by_symbol = self.universe.by_symbol()

    @property
    def fetcher(self) -> Fetcher:
        if self._fetcher is None:
            self._fetcher = live_fetcher()
        return self._fetcher

    # --- work list -------------------------------------------------------------------

    def work_items(
        self,
        frequencies: list[str],
        *,
        symbols: list[str] | None = None,
        limit: int | None = None,
    ) -> list[WorkItem]:
        """The work list, in universe order so a resumed run walks it the same way."""
        chosen = self.universe.instruments
        if symbols:
            wanted = {s.upper() for s in symbols}
            chosen = [i for i in chosen if i.symbol.upper() in wanted]
        if limit is not None:
            chosen = chosen[:limit]

        return [
            WorkItem(
                market=instrument.asset_class,
                symbol=instrument.symbol,
                frequency=frequency,
                asset_class=instrument.asset_class,
                source_symbol=instrument.source_symbol,
                exchange_tz="UTC",
                session_id="24x5",
            )
            for instrument in chosen
            for frequency in frequencies
        ]

    # --- the Source protocol ---------------------------------------------------------

    def artifact_path(self, item: WorkItem) -> str:
        return f"raw/dukascopy/{item.market}/{item.frequency}/{item.symbol}.parquet"

    def years(self, item: WorkItem) -> list[int]:
        instrument = self._by_symbol[item.symbol]
        return list(range(instrument.start_year, self.as_of.year + 1))

    def plan(self, item: WorkItem) -> SourcePlan:
        tokens, digests = [], []
        for year in self.years(item):
            token = year_token(item.vendor_symbol, item.frequency, year)
            tokens.append(token)
            digests.append(year_digest(token, sealed=year < self.as_of.year, as_of=self.as_of))
        return SourcePlan(tokens, digests)

    def build(
        self,
        item: WorkItem,
        plan: SourcePlan,
        load_existing: Callable[[], pa.Table | None],
    ) -> pa.Table:
        """Fetch only what can still change, and read the rest back out of the raw tier."""
        years = self.years(item)
        existing = load_existing()
        first_fetch_year = years[0]

        if existing is not None and existing.num_rows:
            last_ts = int(existing["ts"].to_numpy(zero_copy_only=False).max())
            last_year = datetime.fromtimestamp(last_ts / 1000, UTC).year
            # Re-fetch the year the artifact ends in: it was almost certainly partial when it was
            # written. Anything before that is sealed and is trusted as stored.
            first_fetch_year = max(years[0], min(last_year, self.as_of.year))

        boundary, _ = year_bounds(first_fetch_year)
        prior = (
            clip_to_window(existing, datetime(1970, 1, 1, tzinfo=UTC), boundary)
            if existing is not None
            else None
        )

        fresh = []
        for year in range(first_fetch_year, self.as_of.year + 1):
            start, end = year_bounds(year)
            fetched = clip_to_window(
                self.fetcher(item.vendor_symbol, item.frequency, start, end), start, end
            )
            log.debug("%s %s: %d bars in %d", item.symbol, item.frequency, fetched.num_rows, year)
            fresh.append(fetched)

        table = splice(prior, fresh)
        if not table.num_rows:
            # The library returns an empty frame rather than raising when the feed refuses, so
            # "every year came back empty" is almost never a fact about the instrument -- it is
            # the feed declining to answer this host. Say which, because the two have completely
            # different fixes (ADR-0015's fallback ladder versus a universe correction).
            raise ValueError(
                f"{item}: every year from {first_fetch_year} to {self.as_of.year} came back "
                f"empty for {item.vendor_symbol!r}. An instrument with a pinned start date does "
                "not have zero bars across its whole history, so this is the feed refusing this "
                "host rather than a gap in the data -- see the reachability ladder in ADR-0015"
            )
        return table

    def manifest_extras(self, item: WorkItem) -> dict[str, Any]:
        """Bid quotes, tick volume, and an `amount` that is neither (ADR-0014, ADR-0015)."""
        return {
            "price_side": "bid",
            "volume_convention": "dukascopy_tick_volume",
            "amount_synthesized": True,
            "adjustment_policy": "none",
            "redistribution_class": "loader_manifest_private_cache",
        }
