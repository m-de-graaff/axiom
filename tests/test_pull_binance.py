"""End-to-end pull: a fake bucket in, a local `axiom-raw` stand-in out, and back again.

The stand-in is `LocalRawStore` pointed at a tmp directory -- the same class the Hub store
implements, laid out identically -- so "second run skips" is tested against the real skip logic
rather than against a mock that was told to return True.
"""

from __future__ import annotations

import json
import random

import pyarrow.parquet as pq
import pytest

from axiom.provenance.manifest import SIDECAR_SUFFIX, FileManifest, PullRunManifest
from axiom.raw.store import LocalRawStore
from axiom.sources.binance import (
    PullTask,
    artifact_path,
    build_tasks,
    enumerate_sources,
    pull_symbol,
    run_pull,
)
from axiom.sources.binance_vision import BinanceVision, period_from_key, zip_key
from tests.fakes import DAY_MS, EPOCH, HOUR_MS, FakeBucket, kline_zip

UNIVERSE_HASH = "0123456789ab"


@pytest.fixture
def bucket() -> FakeBucket:
    bucket = FakeBucket()
    for month, hours in (("2024-01", 744), ("2024-02", 696)):
        start = EPOCH if month == "2024-01" else EPOCH + 744 * HOUR_MS
        bucket.put_month("spot", "BTCUSDT", "1h", month, kline_zip(hours, start=start))
    bucket.put_month("spot", "BTCUSDT", "1d", "2024-01", kline_zip(31, step=DAY_MS))
    bucket.put_month("spot", "ETHUSDT", "1h", "2024-01", kline_zip(744))
    return bucket


@pytest.fixture
def client(bucket):
    with BinanceVision(
        client=bucket.client(),
        concurrency=4,
        backoff_base=0.0,
        sleep=lambda _: None,
        rng=random.Random(3),
    ) as instance:
        yield instance


@pytest.fixture
def store(tmp_path) -> LocalRawStore:
    return LocalRawStore(tmp_path / "axiom-raw")


def pull(client, store, symbol="BTCUSDT", frequency="1h", market="spot", **kwargs):
    return pull_symbol(
        client,
        store,
        PullTask(market, symbol, frequency),
        pull_run_id=kwargs.pop("pull_run_id", "pull-test"),
        universe_hash=UNIVERSE_HASH,
        **kwargs,
    )


# --- enumeration -------------------------------------------------------------------------


def test_sources_are_months_then_the_daily_tail(bucket, client):
    bucket.put_day("spot", "BTCUSDT", "1h", "2024-03-01", kline_zip(24))
    urls = enumerate_sources(client, PullTask("spot", "BTCUSDT", "1h"))
    assert [period_from_key(u) for u in urls[:2]] == ["2024-01", "2024-02"]
    assert urls[-1].endswith("2024-03-01.zip")


def test_days_already_covered_by_a_month_are_not_downloaded_twice(bucket, client):
    # February is a published month, so its daily archives add no bars.
    bucket.put_day("spot", "BTCUSDT", "1h", "2024-02-15", kline_zip(24))
    bucket.put_day("spot", "BTCUSDT", "1h", "2024-03-01", kline_zip(24))
    urls = enumerate_sources(client, PullTask("spot", "BTCUSDT", "1h"))
    assert not any("2024-02-15" in url for url in urls)
    assert any("2024-03-01" in url for url in urls)


def test_a_symbol_with_only_daily_archives_still_enumerates(bucket, client):
    bucket.put_day("spot", "NEWUSDT", "1h", "2024-05-01", kline_zip(24))
    urls = enumerate_sources(client, PullTask("spot", "NEWUSDT", "1h"))
    assert len(urls) == 1


# --- the happy path ----------------------------------------------------------------------


def test_a_pull_writes_parquet_and_a_sidecar(client, store, tmp_path):
    result = pull(client, store)
    assert result.status == "ok"

    path = tmp_path / "axiom-raw" / artifact_path("spot", "1h", "BTCUSDT")
    assert path.exists()
    table = pq.read_table(path)
    assert table.num_rows == 744 + 696
    assert table.schema.metadata[b"symbol"] == b"BTCUSDT"
    assert table.schema.metadata[b"market"] == b"spot"
    assert table.schema.metadata[b"axiom_schema_version"] == b"1"

    sidecar = FileManifest.from_json(path.with_suffix(".parquet" + SIDECAR_SUFFIX).read_text())
    assert sidecar.row_count == table.num_rows
    assert sidecar.universe_hash == UNIVERSE_HASH
    assert sidecar.source_urls and len(sidecar.source_urls) == len(sidecar.source_sha256s)
    assert sidecar.amount_synthesized is False
    assert sidecar.gap_count == 0


def test_the_parquet_metadata_names_the_sidecar(client, store, tmp_path):
    pull(client, store)
    path = tmp_path / "axiom-raw" / artifact_path("spot", "1h", "BTCUSDT")
    table = pq.read_table(path)
    sidecar = store.read_sidecar(artifact_path("spot", "1h", "BTCUSDT"))
    assert table.schema.metadata[b"manifest_sha256"].decode() == sidecar.manifest_sha256


def test_the_artifact_hash_matches_the_bytes_on_disk(client, store, tmp_path):
    import hashlib

    pull(client, store)
    path = tmp_path / "axiom-raw" / artifact_path("spot", "1h", "BTCUSDT")
    sidecar = store.read_sidecar(artifact_path("spot", "1h", "BTCUSDT"))
    assert sidecar.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_off_grid_bars_are_pulled_and_counted(bucket, client, store):
    # An exchange restart leaves a stretch of phase-shifted hourly bars. They are real, so they
    # land, and the manifest says how many there are.
    from tests.fakes import csv_bytes, kline_rows, make_zip

    rows = kline_rows(24, start=EPOCH + 1_694_789)
    bucket.put_month("spot", "SHIFTUSDT", "1h", "2024-01", make_zip(csv_bytes(rows)))
    result = pull(client, store, symbol="SHIFTUSDT")
    assert result.status == "ok"
    assert result.manifest.off_grid_count == 24
    assert result.manifest.row_count == 24


def test_an_on_grid_series_reports_no_off_grid_bars(client, store):
    pull(client, store)
    assert store.read_sidecar(artifact_path("spot", "1h", "BTCUSDT")).off_grid_count == 0


def test_daily_bars_land_on_the_daily_grid(client, store):
    assert pull(client, store, frequency="1d").status == "ok"
    assert store.read_sidecar(artifact_path("spot", "1d", "BTCUSDT")).row_count == 31


# --- idempotence, which is the resume mechanism ------------------------------------------


def test_a_second_run_skips(client, store):
    assert pull(client, store).status == "ok"
    assert pull(client, store).status == "skipped"


def test_a_skip_costs_no_archive_downloads(bucket, client, store):
    pull(client, store)
    before = sum(1 for url in bucket.requests if url.endswith(".zip"))
    pull(client, store)
    assert sum(1 for url in bucket.requests if url.endswith(".zip")) == before


def test_force_re_pulls_a_current_symbol(client, store):
    pull(client, store)
    assert pull(client, store, force=True).status == "ok"


def test_a_new_daily_archive_forces_a_re_pull(bucket, client, store):
    pull(client, store)
    tail = kline_zip(24, start=EPOCH + 1440 * HOUR_MS)
    bucket.put_day("spot", "BTCUSDT", "1h", "2024-03-01", tail)
    result = pull(client, store)
    assert result.status == "ok"
    assert result.manifest.row_count == 744 + 696 + 24


def test_a_tampered_sidecar_forces_a_re_pull(client, store, tmp_path):
    pull(client, store)
    sidecar = tmp_path / "axiom-raw" / (artifact_path("spot", "1h", "BTCUSDT") + SIDECAR_SUFFIX)
    payload = json.loads(sidecar.read_text())
    payload["source_sha256s"] = ["0" * 64] * len(payload["source_sha256s"])
    payload.pop("manifest_sha256")
    sidecar.write_text(json.dumps(payload))
    assert pull(client, store).status == "ok"


def test_an_unparseable_sidecar_heals_by_re_pulling(client, store, tmp_path):
    pull(client, store)
    sidecar = tmp_path / "axiom-raw" / (artifact_path("spot", "1h", "BTCUSDT") + SIDECAR_SUFFIX)
    payload = json.loads(sidecar.read_text())
    payload["row_count"] = 1  # leaves the recorded manifest_sha256 no longer matching
    sidecar.write_text(json.dumps(payload))
    assert pull(client, store).status == "ok"
    assert store.read_sidecar(artifact_path("spot", "1h", "BTCUSDT")).row_count == 744 + 696


def test_a_re_pull_produces_identical_bytes(client, store, tmp_path):
    # The v0.1 exit gate turns on this: a manifest hash that varied per run would go into the
    # Parquet metadata and make byte-identity impossible by construction.
    first = pull(client, store)
    path = tmp_path / "axiom-raw" / artifact_path("spot", "1h", "BTCUSDT")
    original = path.read_bytes()
    second = pull(client, store, force=True, pull_run_id="pull-later")
    assert second.status == "ok"
    assert path.read_bytes() == original
    assert second.manifest.artifact_sha256 == first.manifest.artifact_sha256
    assert second.manifest.pull_run_id != first.manifest.pull_run_id


# --- failure handling --------------------------------------------------------------------


def test_a_missing_series_is_a_recorded_failure_not_an_exception(client, store):
    result = pull(client, store, symbol="GHOSTUSDT")
    assert result.status == "failed"
    assert "no archives published" in result.error


def test_a_corrupt_archive_fails_the_symbol_and_writes_nothing(bucket, client, store):
    bucket.corrupt.add(zip_key("spot", "monthly", "BTCUSDT", "1h", "2024-01"))
    result = pull(client, store)
    assert result.status == "failed"
    assert "ChecksumMismatch" in result.error
    assert store.read_sidecar(artifact_path("spot", "1h", "BTCUSDT")) is None


def test_an_invariant_violation_fails_the_symbol(bucket, client, store):
    from tests.fakes import csv_bytes, kline_rows, make_zip

    rows = kline_rows(24)
    rows[3][2] = "0.00000001"  # a high below the open
    bucket.put_month("spot", "BADUSDT", "1h", "2024-01", make_zip(csv_bytes(rows)))
    result = pull(client, store, symbol="BADUSDT")
    assert result.status == "failed"
    assert "high_below_open_or_close" in result.error


# --- the work list -----------------------------------------------------------------------


def test_tasks_are_market_major_and_stable():
    universe = {"spot": ["BTCUSDT", "ETHUSDT"], "um": ["BTCUSDT"]}
    tasks = build_tasks(universe, ["spot", "um"], ["1h", "1d"])
    assert [str(t) for t in tasks] == [
        "spot/1h/BTCUSDT",
        "spot/1d/BTCUSDT",
        "spot/1h/ETHUSDT",
        "spot/1d/ETHUSDT",
        "um/1h/BTCUSDT",
        "um/1d/BTCUSDT",
    ]
    assert build_tasks(universe, ["spot", "um"], ["1h", "1d"]) == tasks


def test_limit_counts_symbols_not_tasks():
    universe = {"spot": ["A", "B", "C", "D"]}
    tasks = build_tasks(universe, ["spot"], ["1h", "1d"], limit=2)
    assert {t.symbol for t in tasks} == {"A", "B"}
    assert len(tasks) == 4


def test_symbols_filter_is_case_insensitive():
    universe = {"spot": ["BTCUSDT", "ETHUSDT"]}
    tasks = build_tasks(universe, ["spot"], ["1h"], symbols=["btcusdt"])
    assert [t.symbol for t in tasks] == ["BTCUSDT"]


def test_an_unknown_market_yields_no_tasks():
    assert build_tasks({"spot": ["BTCUSDT"]}, ["um"], ["1h"]) == []


# --- the run manifest --------------------------------------------------------------------


def make_run_manifest(**overrides) -> PullRunManifest:
    return PullRunManifest(
        **{
            "pull_run_id": "pull-test",
            "started_at": "2026-08-20T00:00:00+00:00",
            "loader_version": "test",
            "backend_tag": "test",
            "universe_hash": UNIVERSE_HASH,
            "universe_path": "universe_v1.yaml",
            "markets": ["spot"],
            "frequencies": ["1h"],
            **overrides,
        }
    )


def test_a_run_records_ok_skipped_and_failed(client, store):
    tasks = build_tasks({"spot": ["BTCUSDT", "GHOSTUSDT"]}, ["spot"], ["1h"])
    first = run_pull(client, store, tasks, make_run_manifest()).finish()
    assert (first.ok, first.skipped, first.failed) == (1, 0, 1)
    assert first.total_rows == 744 + 696
    assert first.failures[0].symbol == "GHOSTUSDT"

    second = run_pull(client, store, tasks, make_run_manifest()).finish()
    assert (second.ok, second.skipped, second.failed) == (0, 1, 1)


def test_a_killed_run_resumes_where_it_stopped(client, store):
    # The kill drill in miniature: the first half of the work list lands, the process dies, and
    # the relaunch has no state but the sidecars it left behind.
    everything = build_tasks({"spot": ["BTCUSDT", "ETHUSDT"]}, ["spot"], ["1h"])
    run_pull(client, store, everything[:1], make_run_manifest())

    resumed = run_pull(client, store, everything, make_run_manifest()).finish()
    assert (resumed.ok, resumed.skipped, resumed.failed) == (1, 1, 0)


def test_the_run_finishes_with_a_timestamp(client, store):
    run = run_pull(client, store, [], make_run_manifest())
    assert run.finish().finished_at
