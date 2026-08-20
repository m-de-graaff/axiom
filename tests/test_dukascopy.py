"""The Dukascopy loader, offline.

Every test here drives the real source class through a synthetic fetcher, so the year-chunk
policy, the splice, and the weekend assertion are exercised against the same code the live pull
runs -- only the transport is replaced. Nothing in this file touches the network.

The live path has exactly one thing these cannot check: whether `dukascopy-python` handles the
feed's 0-indexed months correctly (ADR-0015). `test_live_window_round_trips` covers that, and is
skipped unless the client is installed and a live run is asked for.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from axiom.provenance.manifest import PullRunManifest
from axiom.raw.store import LocalRawStore
from axiom.sources.base import pull_item, run_pull
from axiom.sources.dukascopy import (
    DukascopySource,
    clip_to_window,
    splice,
    to_bars,
    year_bounds,
    year_digest,
    year_token,
)
from axiom.universe.dukascopy import (
    DukascopyCriteria,
    DukascopyInstrument,
    DukascopyUniverse,
)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
UNIVERSE_HASH = "0123456789ab"


def universe(start_date: str = "2020-01-01") -> DukascopyUniverse:
    return DukascopyUniverse(
        criteria=DukascopyCriteria(
            selection="test", frequencies=["1h", "1d"], start_dates_measured_at="2026-08-20"
        ),
        instruments=[
            DukascopyInstrument(
                symbol="EURUSD", source_symbol="EUR/USD", asset_class="fx", start_date=start_date
            ),
            DukascopyInstrument(
                symbol="XAUUSD",
                source_symbol="XAU/USD",
                asset_class="commodity",
                start_date=start_date,
            ),
        ],
    ).with_hash()


def bars_at(ts_ms: list[int], *, price: float = 1.0, volume: float = 10.0) -> pa.Table:
    n = len(ts_ms)
    ones = np.full(n, price)
    return to_bars(np.array(ts_ms, dtype=np.int64), ones, ones, ones, ones, np.full(n, volume))


class RecordingFetcher:
    """A fetcher that emits a clean weekday grid and remembers what it was asked for."""

    def __init__(self, *, gap_hours: int = 0) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.gap_hours = gap_hours

    def __call__(self, source_symbol: str, frequency: str, start: datetime, end: datetime):
        self.calls.append((source_symbol, frequency, start.year))
        step = HOUR_MS if frequency == "1h" else DAY_MS
        lo = int(start.timestamp() * 1000)
        hi = int(end.timestamp() * 1000)
        ts = np.arange(lo, hi, step, dtype=np.int64)

        if frequency == "1h":
            # Drop the weekend, the way a real 24x5 feed does: all of Saturday, and Sunday until
            # the 22:00 UTC reopen.
            dow = ((ts // DAY_MS) + 3) % 7
            hour = (ts % DAY_MS) // HOUR_MS
            ts = ts[~((dow == 5) | ((dow == 6) & (hour < 22)))]
        else:
            dow = ((ts // DAY_MS) + 3) % 7
            ts = ts[dow != 5]  # no Saturday bar; Sunday's opening tail is real

        # A price that differs per year, so a spliced series can be checked for the right rows.
        year = start.year
        n = len(ts)
        prices = np.full(n, float(year))
        return to_bars(ts, prices, prices, prices, prices, np.full(n, 10.0))


def source(fetcher, *, as_of: date, start_date: str = "2020-01-01") -> DukascopySource:
    return DukascopySource(universe(start_date), as_of=as_of, fetcher=fetcher)


# --- year tokens and the resume policy ------------------------------------------------


def test_a_sealed_year_has_a_stable_digest():
    """A year that has ended cannot gain a bar, so re-running never re-fetches it."""
    token = year_token("EUR/USD", "1h", 2020)
    monday = year_digest(token, sealed=True, as_of=date(2026, 8, 20))
    tuesday = year_digest(token, sealed=True, as_of=date(2026, 8, 21))
    assert monday == tuesday


def test_the_current_year_digest_moves_with_the_day():
    """It can gain bars daily, so a re-run tomorrow must re-extend it."""
    token = year_token("EUR/USD", "1h", 2026)
    monday = year_digest(token, sealed=False, as_of=date(2026, 8, 20))
    tuesday = year_digest(token, sealed=False, as_of=date(2026, 8, 21))
    assert monday != tuesday


def test_the_plan_spans_the_instrument_start_to_the_as_of_year():
    src = source(RecordingFetcher(), as_of=date(2023, 6, 1))
    plan = src.plan(src.work_items(["1h"])[0])
    assert plan.source_urls == [
        year_token("EUR/USD", "1h", year) for year in (2020, 2021, 2022, 2023)
    ]


def test_two_instruments_with_different_starts_get_different_spans():
    src = DukascopySource(
        DukascopyUniverse(
            criteria=DukascopyCriteria(
                selection="t", frequencies=["1d"], start_dates_measured_at="2026-08-20"
            ),
            instruments=[
                DukascopyInstrument(
                    symbol="EURUSD",
                    source_symbol="EUR/USD",
                    asset_class="fx",
                    start_date="2003-05-04",
                ),
                DukascopyInstrument(
                    symbol="XAGUSD",
                    source_symbol="XAG/USD",
                    asset_class="commodity",
                    start_date="2014-07-25",
                ),
            ],
        ).with_hash(),
        as_of=date(2020, 1, 1),
        fetcher=RecordingFetcher(),
    )
    items = {i.symbol: i for i in src.work_items(["1d"])}
    assert len(src.plan(items["EURUSD"]).source_urls) == 18
    assert len(src.plan(items["XAGUSD"]).source_urls) == 7


# --- window clipping and splicing -----------------------------------------------------


def test_clip_to_window_is_half_open():
    start, end = year_bounds(2021)
    lo, hi = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    clipped = clip_to_window(bars_at([lo - 1, lo, hi - 1, hi]), start, end)
    assert clipped["ts"].to_pylist() == [lo, hi - 1]


def test_splice_sorts_and_keeps_the_fresh_copy_of_a_duplicate():
    prior = bars_at([2000, 1000], price=1.0)
    fresh = bars_at([1000], price=9.0)
    out = splice(prior, [fresh])
    assert out["ts"].to_pylist() == [1000, 2000]
    assert out["close"].to_pylist() == [9.0, 1.0]


def test_splice_of_nothing_is_an_empty_table_not_a_crash():
    assert splice(None, []).num_rows == 0


# --- amount synthesis -----------------------------------------------------------------


def test_amount_is_volume_times_mean_ohlc():
    table = to_bars(
        np.array([0], dtype=np.int64),
        np.array([2.0]),
        np.array([4.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([10.0]),
    )
    assert table["amount"].to_pylist() == [20.0]  # mean(2,4,1,1) == 2.0


def test_the_manifest_says_the_amount_was_synthesized():
    src = source(RecordingFetcher(), as_of=date(2021, 6, 1))
    extras = src.manifest_extras(src.work_items(["1h"])[0])
    assert extras["amount_synthesized"] is True
    assert extras["price_side"] == "bid"
    assert extras["volume_convention"] == "dukascopy_tick_volume"


# --- the pull, end to end through the shared driver -----------------------------------


def pull(src: DukascopySource, store: LocalRawStore, item, *, run_id: str = "run-1"):
    return pull_item(src, store, item, pull_run_id=run_id, universe_hash=UNIVERSE_HASH, force=False)


def test_a_first_pull_fetches_every_year(tmp_path):
    fetcher = RecordingFetcher()
    src = source(fetcher, as_of=date(2022, 6, 1))
    item = src.work_items(["1d"])[0]

    result = pull(src, LocalRawStore(tmp_path), item)

    assert result.status == "ok", result.error
    assert [year for _, _, year in fetcher.calls] == [2020, 2021, 2022]
    assert result.manifest is not None
    assert result.manifest.source_symbol == "EUR/USD"
    assert result.manifest.artifact_path == "raw/dukascopy/fx/1d/EURUSD.parquet"


def test_a_same_day_rerun_skips(tmp_path):
    store = LocalRawStore(tmp_path)
    fetcher = RecordingFetcher()
    src = source(fetcher, as_of=date(2022, 6, 1))
    item = src.work_items(["1d"])[0]

    assert pull(src, store, item).status == "ok"
    before = len(fetcher.calls)
    assert pull(src, store, item, run_id="run-2").status == "skipped"
    assert len(fetcher.calls) == before, "a skip must not fetch anything"


def test_the_next_day_refetches_only_the_current_year(tmp_path):
    """The kill-drill property, and the one that keeps a full re-pull off the wire every day."""
    store = LocalRawStore(tmp_path)
    fetcher = RecordingFetcher()
    item = source(fetcher, as_of=date(2022, 6, 1)).work_items(["1d"])[0]

    assert pull(source(fetcher, as_of=date(2022, 6, 1)), store, item).status == "ok"
    fetcher.calls.clear()

    later = source(fetcher, as_of=date(2022, 6, 2))
    assert pull(later, store, item, run_id="run-2").status == "ok"
    assert [year for _, _, year in fetcher.calls] == [2022]


def test_prior_years_survive_a_refetch_byte_for_byte(tmp_path):
    """ADR-0015's immutability claim, checked on the rows rather than asserted in prose."""
    store = LocalRawStore(tmp_path)
    fetcher = RecordingFetcher()
    item = source(fetcher, as_of=date(2022, 6, 1)).work_items(["1d"])[0]

    pull(source(fetcher, as_of=date(2022, 6, 1)), store, item)
    path = "raw/dukascopy/fx/1d/EURUSD.parquet"
    import io

    import pyarrow.parquet as pq

    first = pq.read_table(io.BytesIO(store.get(path)))
    boundary = int(datetime(2022, 1, 1, tzinfo=UTC).timestamp() * 1000)
    sealed_before = first.filter(pa.compute.less(first["ts"], boundary))

    pull(source(fetcher, as_of=date(2022, 6, 2)), store, item, run_id="run-2")
    second = pq.read_table(io.BytesIO(store.get(path)))
    sealed_after = second.filter(pa.compute.less(second["ts"], boundary))

    assert sealed_before.equals(sealed_after)
    assert sealed_before.num_rows > 0


def test_a_stale_artifact_refetches_every_year_it_is_missing(tmp_path):
    """A tier that went a year without a pull must not silently skip the gap."""
    store = LocalRawStore(tmp_path)
    fetcher = RecordingFetcher()
    item = source(fetcher, as_of=date(2021, 6, 1)).work_items(["1d"])[0]

    pull(source(fetcher, as_of=date(2021, 6, 1)), store, item)
    fetcher.calls.clear()

    pull(source(fetcher, as_of=date(2024, 6, 1)), store, item, run_id="run-2")
    assert [year for _, _, year in fetcher.calls] == [2021, 2022, 2023, 2024]


def test_a_weekend_bar_fails_the_item_instead_of_landing(tmp_path):
    """Nothing trades on a Saturday, so a bar there means the timestamps are wrong."""

    def saturday_fetcher(source_symbol, frequency, start, end):
        # 2021-01-02 was a Saturday.
        saturday_noon = int(datetime(2021, 1, 2, 12, tzinfo=UTC).timestamp() * 1000)
        return bars_at([saturday_noon, saturday_noon + HOUR_MS])

    src = DukascopySource(universe("2021-01-01"), as_of=date(2021, 6, 1), fetcher=saturday_fetcher)
    result = pull(src, LocalRawStore(tmp_path), src.work_items(["1h"])[0])

    assert result.status == "failed"
    assert "bars_in_weekend_close" in result.error


def test_the_run_walks_both_frequencies_and_both_instruments(tmp_path):
    fetcher = RecordingFetcher()
    src = source(fetcher, as_of=date(2021, 6, 1))
    items = src.work_items(["1h", "1d"])
    manifest = PullRunManifest(
        pull_run_id="run-1",
        started_at="2021-06-01T00:00:00+00:00",
        loader_version="test",
        backend_tag="test",
        universe_hash=UNIVERSE_HASH,
        universe_path="test",
        markets=["fx", "commodity"],
        frequencies=["1h", "1d"],
    )

    run = run_pull(src, LocalRawStore(tmp_path), items, manifest)

    assert run.manifest.ok == 4
    assert run.manifest.failed == 0
    assert {r.task.symbol for r in run.results} == {"EURUSD", "XAUUSD"}


def test_hourly_series_carry_weekend_gaps_and_that_is_fine(tmp_path):
    """A 24x5 gap is a fact about the market. It is counted, never repaired (ADR-0015)."""
    src = source(RecordingFetcher(), as_of=date(2021, 6, 1))
    item = next(i for i in src.work_items(["1h"]) if i.symbol == "EURUSD")

    result = pull(src, LocalRawStore(tmp_path), item)

    assert result.status == "ok", result.error
    assert result.manifest is not None
    assert result.manifest.gap_count > 0, "a 24x5 hourly series must show its weekends as gaps"
    assert result.manifest.off_grid_count == 0


# --- the one thing offline tests cannot answer ----------------------------------------


@pytest.mark.skipif(
    not os.environ.get("AXIOM_LIVE_DUKASCOPY"),
    reason="live feed; set AXIOM_LIVE_DUKASCOPY=1 to run",
)
def test_live_window_round_trips():
    """Dukascopy numbers months from zero. This checks the library gets that right.

    A month off-by-one would return the right *shape* of data for the wrong month, which no
    offline test can detect -- so the assertion is that the bars come back inside the window that
    was asked for, on a window narrow enough that a month's drift cannot hide in it.
    """
    from axiom.sources.dukascopy import live_fetcher

    start = datetime(2024, 6, 3, tzinfo=UTC)
    end = start + timedelta(days=2)
    table = live_fetcher()("EUR/USD", "1h", start, end)

    assert table.num_rows > 0
    ts = table["ts"].to_numpy(zero_copy_only=False)
    # One day of slack at the start: the week's session opens the evening before.
    assert ts.min() >= int((start - timedelta(days=1)).timestamp() * 1000)
    assert ts.max() < int((end + timedelta(days=1)).timestamp() * 1000)
