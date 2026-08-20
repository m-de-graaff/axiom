"""The Stooq loader, against a miniature fake archive.

The real dump is a multi-gigabyte zip nobody can commit, so the fixture here is the same zip
built small: the same directory layout, the same header line, the same column order, and one
member per case the parser has to handle. Everything downstream of `zipfile.ZipFile` is the
production code path.
"""

from __future__ import annotations

import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from axiom.provenance.manifest import PullRunManifest
from axiom.raw.store import LocalRawStore
from axiom.sources.base import pull_item, run_pull
from axiom.sources.stooq import (
    MalformedFile,
    StooqArchive,
    StooqSource,
    date_to_ms,
    is_kept_member,
    parse_ticker_file,
    symbol_from_member,
)

UNIVERSE_HASH = "0123456789ab"
HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"


def series(ticker: str, n: int, *, start: str = "2020-01-02", price: float = 10.0) -> str:
    """A clean daily file: real consecutive calendar dates, in Stooq's column order."""
    day = date.fromisoformat(start)
    lines = [HEADER]
    for i in range(n):
        stamp = (day + timedelta(days=i)).strftime("%Y%m%d")
        lines.append(f"{ticker},D,{stamp},000000,{price},{price + 1},{price - 1},{price},1000,0")
    return "\n".join(lines) + "\n"


@pytest.fixture
def archive(tmp_path: Path) -> StooqArchive:
    """A miniature dump with one member per interesting case."""
    path = tmp_path / "d_us_txt.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data/daily/us/nasdaq stocks/1/aapl.us.txt", series("AAPL.US", 60))
        zf.writestr(
            "data/daily/us/nyse stocks/2/ibm.us.txt", series("IBM.US", 40, start="2020-01-05")
        )
        zf.writestr("data/daily/us/nyse etfs/spy.us.txt", series("SPY.US", 80))
        # Too short to be useful -- recorded as skipped, not failed.
        zf.writestr("data/daily/us/nasdaq stocks/1/new.us.txt", series("NEW.US", 5))
        # Not a tradeable instrument; never enumerated.
        zf.writestr("data/daily/us/indices/^spx.txt", series("^SPX", 90))
        zf.writestr("data/daily/us/futures/cl.f.txt", series("CL.F", 90))
        zf.writestr("data/daily/world/xetra stocks/sap.de.txt", series("SAP.DE", 90))
    return StooqArchive(url="https://stooq.com/db/d/?b=d_us_txt", path=path)


# --- member classification --------------------------------------------------------------


def test_stocks_and_etfs_are_kept():
    assert is_kept_member("data/daily/us/nasdaq stocks/1/aapl.us.txt")
    assert is_kept_member("data/daily/us/nyse etfs/spy.us.txt")


def test_indices_and_futures_are_not():
    assert not is_kept_member("data/daily/us/indices/^spx.txt")
    assert not is_kept_member("data/daily/us/futures/cl.f.txt")


def test_non_us_trees_are_not():
    assert not is_kept_member("data/daily/world/xetra stocks/sap.de.txt")


def test_the_market_suffix_is_stripped_into_the_symbol():
    assert symbol_from_member("data/daily/us/nasdaq stocks/1/aapl.us.txt") == "AAPL"
    assert symbol_from_member("data/daily/us/nyse stocks/brk-b.us.txt") == "BRK-B"


def test_a_date_becomes_midnight_utc_of_that_calendar_date():
    """ADR-0014: not the session open, not midnight local."""
    assert date_to_ms("20240610") == 1_717_977_600_000


# --- parsing ------------------------------------------------------------------------------


def test_a_clean_file_parses_to_sorted_bars():
    table, counts = parse_ticker_file(series("AAPL.US", 5))
    assert table.num_rows == 5
    assert counts.malformed == 0
    ts = table["ts"].to_pylist()
    assert ts == sorted(ts)


def test_amount_is_synthesized_from_volume_and_mean_ohlc():
    table, _ = parse_ticker_file(series("AAPL.US", 1, price=10.0))
    # mean(10, 11, 9, 10) == 10.0, volume 1000
    assert table["amount"].to_pylist() == [10_000.0]


def test_openint_is_dropped():
    table, _ = parse_ticker_file(series("AAPL.US", 1))
    assert "openint" not in [name.lower() for name in table.column_names]


def test_a_few_malformed_lines_are_dropped_and_counted():
    text = series("AAPL.US", 2000)
    lines = text.splitlines()
    lines[500] = "AAPL.US,D,notadate,000000,x,y,z,w,v,0"
    table, counts = parse_ticker_file("\n".join(lines))
    assert counts.malformed == 1
    assert table.num_rows == 1999


def test_a_badly_damaged_file_is_failed_outright():
    """Past some density of damage, what survived is not a series -- it is a sample of one."""
    lines = (
        [HEADER]
        + ["AAPL.US,D,notadate,0,x,y,z,w,v,0"] * 50
        + series("AAPL.US", 50).splitlines()[1:]
    )
    with pytest.raises(MalformedFile, match="over the"):
        parse_ticker_file("\n".join(lines), context="AAPL")


def test_non_daily_rows_are_treated_as_malformed():
    """A weekly row in a daily dump means the wrong file, not a bar to keep."""
    text = series("AAPL.US", 2000).replace(",D,", ",5,", 1)
    _, counts = parse_ticker_file(text)
    assert counts.malformed == 1


# --- enumeration ---------------------------------------------------------------------------


def test_only_tradeable_us_series_are_enumerated(archive):
    with StooqSource(archive) as source:
        symbols = {item.symbol for item in source.work_items()}
    assert symbols == {"AAPL", "IBM", "SPY"}


def test_short_series_are_recorded_rather_than_silently_absent(archive):
    with StooqSource(archive) as source:
        source.work_items()
        assert source.skipped_short == ["NEW"]


def test_every_item_carries_the_equity_session_metadata(archive):
    with StooqSource(archive) as source:
        item = next(i for i in source.work_items() if i.symbol == "AAPL")
    assert item.exchange_tz == "America/New_York"
    assert item.session_id == "XNYS-regular"
    assert item.source_symbol == "aapl.us"
    assert item.frequency == "1d"


def test_the_layout_is_letter_sharded(archive):
    with StooqSource(archive) as source:
        item = next(i for i in source.work_items() if i.symbol == "AAPL")
        assert source.artifact_path(item) == "raw/stooq/us/1d/A/AAPL.parquet"


def test_every_series_shares_the_archive_provenance(archive):
    """They all came out of one download, so they all carry one source record."""
    with StooqSource(archive) as source:
        plans = [source.plan(item) for item in source.work_items()]
    assert {tuple(p.source_urls) for p in plans} == {(archive.url,)}
    assert {tuple(p.source_sha256s) for p in plans} == {(archive.sha256,)}


# --- end to end through the shared driver ----------------------------------------------------


def test_the_archive_lands_as_parquet(tmp_path, archive):
    store = LocalRawStore(tmp_path / "raw")
    with StooqSource(archive) as source:
        items = source.work_items()
        manifest = PullRunManifest(
            pull_run_id="stooq-1",
            started_at="2026-08-20T00:00:00+00:00",
            loader_version="test",
            backend_tag="test",
            universe_hash=UNIVERSE_HASH,
            universe_path="n/a",
            markets=["us"],
            frequencies=["1d"],
        )
        run = run_pull(source, store, items, manifest)

    assert run.manifest.ok == 3
    assert run.manifest.failed == 0
    landed = store.read_sidecar("raw/stooq/us/1d/A/AAPL.parquet")
    assert landed is not None
    assert landed.volume_convention == "shares"
    assert landed.amount_synthesized is True
    assert landed.adjustment_policy == "vendor_adjusted_unverified"
    assert landed.source_symbol == "aapl.us"


def test_a_rerun_against_the_same_archive_skips_everything(tmp_path, archive):
    store = LocalRawStore(tmp_path / "raw")
    with StooqSource(archive) as source:
        items = source.work_items()
        for item in items:
            assert (
                pull_item(source, store, item, pull_run_id="a", universe_hash=UNIVERSE_HASH).status
                == "ok"
            )
        for item in items:
            assert (
                pull_item(source, store, item, pull_run_id="b", universe_hash=UNIVERSE_HASH).status
                == "skipped"
            )


def test_a_newer_archive_repulls_everything(tmp_path, archive):
    """A new dump is a new copy of every ticker, not an extension of any one."""
    store = LocalRawStore(tmp_path / "raw")
    with StooqSource(archive) as source:
        items = source.work_items()
        for item in items:
            pull_item(source, store, item, pull_run_id="a", universe_hash=UNIVERSE_HASH)

    newer = StooqArchive(url=archive.url, path=archive.path, sha256="f" * 64)
    with StooqSource(newer) as source:
        for item in source.work_items():
            result = pull_item(source, store, item, pull_run_id="b", universe_hash=UNIVERSE_HASH)
            assert result.status == "ok"


def test_a_duplicate_date_fails_the_ticker_with_no_tolerance(tmp_path):
    """Every other defect here is absence of information. A duplicate date is a contradiction."""
    path = tmp_path / "dupe.zip"
    text = series("DUP.US", 40)
    lines = text.splitlines()
    lines[2] = lines[1]  # the same date twice
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data/daily/us/nasdaq stocks/1/dup.us.txt", "\n".join(lines) + "\n")

    with StooqSource(StooqArchive(url="u", path=path)) as source:
        item = source.work_items()[0]
        result = pull_item(
            source,
            LocalRawStore(tmp_path / "raw"),
            item,
            pull_run_id="a",
            universe_hash=UNIVERSE_HASH,
        )

    assert result.status == "failed"
    assert "ts_not_increasing" in result.error


def test_an_off_grid_date_fails_an_equity_series(tmp_path):
    """Daily equity bars must sit on 00:00 UTC; a shifted one means the date column was misread."""
    from axiom.schema.bars import validate_bars

    table, _ = parse_ticker_file(series("AAPL.US", 3))
    shifted = table.set_column(
        0, "ts", __import__("pyarrow").array([t + 1 for t in table["ts"].to_pylist()], "int64")
    )
    report = validate_bars(shifted, "1d", session_id="XNYS-regular")
    assert "ts_off_grid" in report.violations


def test_archive_digest_is_self_computed(archive):
    """Stooq ships no checksum, so the manifest's digest is ours, not the vendor's."""
    assert len(archive.sha256) == 64
    assert archive.sha256 != ""


def test_downloading_streams_to_disk(tmp_path):
    """The dump is measured in gigabytes; a runner has single-digit gigabytes of RAM."""
    import httpx

    from axiom.sources.stooq import download_archive

    payload = b"x" * (3 << 20)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = download_archive("https://stooq.example/d_us_txt.zip", tmp_path / "a.zip", client=client)
    assert dest.read_bytes() == payload
