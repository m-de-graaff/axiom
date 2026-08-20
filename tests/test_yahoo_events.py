"""The yfinance adjunct, offline.

The module's defining property is that it is allowed to fail, so most of these tests are about
failing *correctly*: a blocked run must stop early and say what it was, a ticker with no
corporate actions must be distinguishable from one that could not be fetched, and nothing here
may take the corpus down with it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from axiom.raw.store import LocalRawStore
from axiom.sources.yahoo_events import (
    EVENT_SCHEMA,
    EventRun,
    RateLimiter,
    Ticker,
    artifact_path,
    blocked_report,
    detect_split_discontinuity,
    events_table,
    known_split_probes,
    load_tickers,
    pull_events,
    pull_ticker,
)

AS_OF = "2026-08-20"
DAY_MS = 86_400_000


def ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp() * 1000)


def fetcher(events: dict[str, list[tuple[int, str, float]]]):
    def fetch(yahoo_symbol: str):
        if yahoo_symbol not in events:
            raise RuntimeError(f"HTTP 429 for {yahoo_symbol}")
        return events[yahoo_symbol]

    return fetch


def nowait() -> RateLimiter:
    return RateLimiter(sleep=lambda _: None)


# --- the pinned population ------------------------------------------------------------------


def test_the_pinned_list_loads():
    tickers = load_tickers()
    assert len(tickers) == 503
    assert Ticker("AAPL", "AAPL") in tickers


def test_class_suffixes_are_translated_into_yahoos_spelling():
    """The index writes BRK.B; Yahoo writes BRK-B, and asking for the wrong one 404s."""
    by_symbol = {t.symbol: t for t in load_tickers()}
    dotted = [t for t in by_symbol.values() if "." in t.symbol]
    assert dotted, "expected at least one class-suffixed ticker in the S&P 500"
    for ticker in dotted:
        assert ticker.yahoo_symbol == ticker.symbol.replace(".", "-")


def test_the_file_says_what_it_is_not():
    """A cross-check population read as a universe is the kind of mistake that reaches a paper."""
    from axiom.config.settings import resolve_config_path

    text = resolve_config_path("yahoo_events_v1").read_text(encoding="utf-8")
    assert "NOT A SURVIVORSHIP-SAFE UNIVERSE" in text
    assert "Retrieved: 2026-08-20" in text


# --- the event table ------------------------------------------------------------------------


def test_events_are_sorted_and_typed():
    table = events_table([(ms("2024-06-10"), "split", 10.0), (ms("2023-01-05"), "dividend", 0.24)])
    assert table.schema == EVENT_SCHEMA
    assert table["ts"].to_pylist() == [ms("2023-01-05"), ms("2024-06-10")]
    assert table["event_type"].to_pylist() == ["dividend", "split"]


def test_an_unknown_event_type_is_refused():
    with pytest.raises(ValueError, match="unknown event_type"):
        events_table([(0, "spinoff", 1.0)])


def test_a_ticker_with_no_actions_is_an_empty_table_not_an_error():
    """'Nothing happened' and 'we do not know' must never look the same."""
    assert events_table([]).num_rows == 0


def test_the_layout_is_letter_sharded():
    assert artifact_path("AAPL") == "raw/yahoo/adjustments/A/AAPL.parquet"


# --- pulling --------------------------------------------------------------------------------


def test_a_ticker_lands_with_the_strictest_redistribution_class(tmp_path):
    """Yahoo grants nothing, so not even the manifests may be published."""
    store = LocalRawStore(tmp_path)
    events = {"AAPL": [(ms("2020-08-31"), "split", 4.0), (ms("2024-02-15"), "dividend", 0.24)]}
    result = pull_ticker(
        Ticker("AAPL", "AAPL"),
        store,
        fetcher(events),
        pull_run_id="yahoo-1",
        as_of=AS_OF,
    )

    assert result.status == "ok"
    assert result.events == 2
    landed = store.read_sidecar("raw/yahoo/adjustments/A/AAPL.parquet")
    assert landed is not None
    assert landed.redistribution_class == "loader_only_private"
    assert landed.frequency == "events"
    assert landed.source_symbol == "AAPL"


def test_a_same_day_rerun_skips(tmp_path):
    store = LocalRawStore(tmp_path)
    fetch = fetcher({"AAPL": [(ms("2020-08-31"), "split", 4.0)]})
    assert (
        pull_ticker(Ticker("AAPL", "AAPL"), store, fetch, pull_run_id="a", as_of=AS_OF).status
        == "ok"
    )
    assert (
        pull_ticker(Ticker("AAPL", "AAPL"), store, fetch, pull_run_id="b", as_of=AS_OF).status
        == "skipped"
    )


def test_the_next_day_refetches(tmp_path):
    """A split is published the day it happens, so daily is the right cadence to re-ask."""
    store = LocalRawStore(tmp_path)
    fetch = fetcher({"AAPL": [(ms("2020-08-31"), "split", 4.0)]})
    pull_ticker(Ticker("AAPL", "AAPL"), store, fetch, pull_run_id="a", as_of=AS_OF)
    assert (
        pull_ticker(
            Ticker("AAPL", "AAPL"), store, fetch, pull_run_id="b", as_of="2026-08-21"
        ).status
        == "ok"
    )


def test_one_blocked_ticker_does_not_end_the_run(tmp_path):
    store = LocalRawStore(tmp_path)
    tickers = [Ticker("AAPL", "AAPL"), Ticker("GHOST", "GHOST"), Ticker("MSFT", "MSFT")]
    run = pull_events(
        tickers,
        store,
        pull_run_id="yahoo-1",
        as_of=AS_OF,
        fetch=fetcher({"AAPL": [], "MSFT": [(ms("2024-03-14"), "dividend", 0.75)]}),
        limiter=nowait(),
    )

    assert run.ok == 2
    assert run.failed == 1
    assert run.failures[0].symbol == "GHOST"


def test_a_total_block_stops_early_rather_than_grinding(tmp_path):
    """Four hundred more requests to rediscover a 'no' would take ninety minutes."""
    tickers = [Ticker(f"T{i}", f"T{i}") for i in range(200)]
    run = pull_events(
        tickers,
        LocalRawStore(tmp_path),
        pull_run_id="yahoo-1",
        as_of=AS_OF,
        fetch=fetcher({}),
        limiter=nowait(),
        fail_fast_after=25,
    )

    assert run.ok == 0
    assert run.failed == 25
    assert len(run.results) == 25


def test_interspersed_failures_never_trip_the_early_stop(tmp_path):
    """Partial success is success.

    The counter is consecutive, so a source that fails every other ticker is unreliable rather
    than blocked, and the run walks the whole list. An earlier version of this test asserted the
    run continued through *sixty consecutive* failures — that rule was wrong and the v0.2 Hub
    incident is what showed it: a backend can start refusing partway, and grinding on afterwards
    buys nothing.
    """
    tickers = [Ticker(f"T{i}", f"T{i}") for i in range(60)]
    reachable = {f"T{i}": [] for i in range(0, 60, 2)}

    run = pull_events(
        tickers,
        LocalRawStore(tmp_path),
        pull_run_id="yahoo-1",
        as_of=AS_OF,
        fetch=fetcher(reachable),
        limiter=nowait(),
        fail_fast_after=25,
    )

    assert run.ok == 30
    assert run.failed == 30
    assert len(run.results) == 60, "every ticker was attempted"


def test_the_blocked_report_is_dated_and_names_the_reason():
    run = EventRun()
    from axiom.sources.yahoo_events import EventResult

    for i in range(3):
        run.record(EventResult(f"T{i}", "failed", error="RuntimeError: HTTP 429"))
    text = blocked_report(run, as_of=AS_OF)
    assert AS_OF in text
    assert "HTTP 429" in text
    assert "non-load-bearing" in text


def test_the_rate_limiter_paces_below_the_ceiling():
    slept: list[float] = []
    limiter = RateLimiter(300, sleep=slept.append)
    for _ in range(50):
        limiter.wait()
    # 300/hour is one every 12 s; jitter spans 0.75x to 1.25x of that.
    assert all(9.0 <= s <= 15.0 for s in slept)
    assert len(set(slept)) > 1, "a metronome is what a scraper looks like"


# --- the split probes -----------------------------------------------------------------------


def test_the_probes_are_the_ones_the_adr_names():
    assert set(known_split_probes()) == {"AAPL", "TSLA", "NVDA"}


def test_an_unadjusted_series_shows_the_split_cliff():
    split = ms("2020-08-31")
    ts = [split - DAY_MS, split, split + DAY_MS]
    close = [500.0, 125.0, 126.0]  # a 4:1 that nobody applied
    verdict = detect_split_discontinuity(ts, close, split, 4.0)
    assert verdict["adjusted"] is False
    assert verdict["measured"] == pytest.approx(4.0)


def test_an_adjusted_series_shows_no_cliff():
    split = ms("2020-08-31")
    ts = [split - DAY_MS, split, split + DAY_MS]
    close = [124.0, 125.0, 126.0]
    verdict = detect_split_discontinuity(ts, close, split, 4.0)
    assert verdict["adjusted"] is True


def test_a_probe_with_no_bars_on_one_side_says_so_rather_than_guessing():
    split = ms("2020-08-31")
    verdict = detect_split_discontinuity([split + DAY_MS], [125.0], split, 4.0)
    assert verdict["adjusted"] is None
    assert "no bars" in verdict["reason"]


# --- the blast wall ---------------------------------------------------------------------------


class ExplodingStore(LocalRawStore):
    """A store that fails the way the Hub does: on the commit, mid-run, transiently."""

    def __init__(self, root, *, fail_on: set[str] | None = None, fail_reads: bool = False) -> None:
        super().__init__(root)
        self.fail_on = fail_on or set()
        self.fail_reads = fail_reads

    def read_sidecar(self, artifact_path: str):
        if self.fail_reads:
            raise RuntimeError("429 Client Error: Too Many Requests")
        return super().read_sidecar(artifact_path)

    def put(self, artifact_path: str, data: bytes, manifest) -> None:
        if any(symbol in artifact_path for symbol in self.fail_on):
            raise RuntimeError("429 Client Error: Too Many Requests")
        super().put(artifact_path, data, manifest)


def test_a_hub_error_on_put_fails_one_ticker_not_the_run(tmp_path):
    """The v0.2 incident: a 429 on upload_folder killed a 503-ticker run mid-flight."""
    store = ExplodingStore(tmp_path, fail_on={"MSFT"})
    tickers = [Ticker("AAPL", "AAPL"), Ticker("MSFT", "MSFT"), Ticker("NVDA", "NVDA")]

    run = pull_events(
        tickers,
        store,
        pull_run_id="yahoo-1",
        as_of=AS_OF,
        fetch=fetcher({"AAPL": [], "MSFT": [], "NVDA": []}),
        limiter=nowait(),
    )

    assert run.ok == 2
    assert run.failed == 1
    assert run.failures[0].symbol == "MSFT"
    assert "429" in run.failures[0].error


def test_a_hub_error_on_read_fails_one_ticker_not_the_run(tmp_path):
    """The resume read is a Hub call too, and it was catching only ValueError."""
    store = ExplodingStore(tmp_path, fail_reads=True)
    run = pull_events(
        [Ticker("AAPL", "AAPL")],
        store,
        pull_run_id="yahoo-1",
        as_of=AS_OF,
        fetch=fetcher({"AAPL": []}),
        limiter=nowait(),
    )
    assert run.failed == 1
    assert run.ok == 0


def test_a_mid_run_block_stops_rather_than_grinding(tmp_path):
    """Once the backend starts refusing, the remaining tickers are not worth 12 seconds each."""
    tickers = [Ticker(f"T{i}", f"T{i}") for i in range(200)]
    fetchable = {f"T{i}": [] for i in range(200)}
    store = ExplodingStore(tmp_path, fail_on={f"T{i}" for i in range(10, 200)})

    run = pull_events(
        tickers,
        store,
        pull_run_id="yahoo-1",
        as_of=AS_OF,
        fetch=fetcher(fetchable),
        limiter=nowait(),
        fail_fast_after=25,
    )

    assert run.ok == 10, "the tickers before the block must still land"
    assert run.failed == 25, "and it must stop 25 consecutive failures in, not grind through 190"
