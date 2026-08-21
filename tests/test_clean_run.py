"""The corpus clean run, end to end over a local raw tier.

Everything the Modal job does, minus Modal: build a raw tier from synthetic series, build the
registry over it, clean the registry, write the three outputs, and read them back. The staleness
guard and the config-hash refusal are exercised here rather than described, because both of them
only ever fire on a second run and a rule that has never fired is a rule nobody has tested.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from axiom.adjust.derive import TR_MANIFEST_PATH, derive_tr
from axiom.adjust.policy import POLICY_SPLIT_AND_DIVIDEND, POLICY_SPLIT_ONLY
from axiom.clean.config import load_clean_config
from axiom.clean.reports import (
    clean_summary_markdown,
    drop_rates,
    most_cut_series,
    red_flags,
    usable_bars,
)
from axiom.clean.run import (
    ConfigHashChanged,
    bar_artifacts,
    clean_corpus,
    clean_paths,
    session_id_for,
    stale_artifacts,
    write_outputs,
)
from axiom.provenance.manifest import FileManifest, sha256_bytes
from axiom.raw.store import LocalRawStore
from axiom.registry import build_from_manifests
from axiom.schema.bars import ROW_GROUP_SIZE, bars_metadata
from axiom.sources.yahoo_events import events_table
from axiom.testing import synth

DAY = 86_400_000


def config(min_bars: int = 4):
    cfg = load_clean_config("clean_v1")
    payload = cfg.model_dump()
    for frequency in ("1h", "1d"):
        payload["frequencies"][frequency]["min_bars"] = min_bars
    return type(cfg).model_validate(payload)


def write_series(
    store: LocalRawStore,
    series: synth.SynthSeries,
    *,
    source: str,
    market: str,
    asset_class: str,
    symbol: str,
    adjustment_policy: str = "none",
) -> FileManifest:
    """Land one synthetic series as a raw artifact, sidecar and Parquet metadata included."""
    path = f"raw/{source}/{market}/{series.frequency}/{symbol}.parquet"
    manifest = FileManifest(
        schema_version=1,
        source=source,
        market=market,
        asset_class=asset_class,
        symbol=symbol,
        frequency=series.frequency,
        pull_run_id="pull-test",
        pulled_at="2026-08-21T00:00:00+00:00",
        loader_version="0.3.0+test",
        source_urls=["https://example/a.zip"],
        source_sha256s=["a" * 64],
        artifact_path=path,
        row_count=series.table.num_rows,
        first_ts=int(series.ts[0]),
        last_ts=int(series.ts[-1]),
        gap_count=0,
        adjustment_policy=adjustment_policy,
        universe_hash="u" * 12,
    )
    table = series.table.replace_schema_metadata(
        bars_metadata(
            source=source,
            asset_class=asset_class,
            market=market,
            symbol=symbol,
            frequency=series.frequency,
            manifest_sha256=manifest.manifest_sha256,
            session_id=series.session_id,
        )
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    payload = buffer.getvalue()
    manifest = manifest.model_copy(update={"artifact_sha256": sha256_bytes(payload)})
    store.put(path, payload, manifest)
    return manifest


@pytest.fixture
def tier(tmp_path: Path) -> tuple[LocalRawStore, list[FileManifest]]:
    """A four-source raw tier: crypto, FX, equities, plus a Yahoo event series."""
    store = LocalRawStore(tmp_path)
    manifests = [
        write_series(
            store,
            synth.with_split(synth.walk("1h", 400, seed=1), 4.0, at=200),
            source="binance_vision",
            market="spot",
            asset_class="crypto",
            symbol="BTCUSDT",
        ),
        write_series(
            store,
            synth.walk("1h", 400, seed=2, session_id="24x5"),
            source="dukascopy",
            market="fx",
            asset_class="fx",
            symbol="EURUSD",
        ),
        write_series(
            store,
            synth.with_limit_lock(
                synth.walk("1d", 400, seed=3, session_id="XNYS-regular", start_ts=946_684_800_000),
                at=100,
                n=10,
            ),
            source="stooq",
            market="us",
            asset_class="equity",
            symbol="AAPL",
            adjustment_policy=POLICY_SPLIT_AND_DIVIDEND,
        ),
    ]
    events = events_table([(DAY, "dividend", 0.25), (2 * DAY, "split", 4.0)])
    buffer = io.BytesIO()
    pq.write_table(events, buffer, compression="zstd")
    payload = buffer.getvalue()
    events_manifest = FileManifest(
        schema_version=1,
        source="yahoo",
        market="adjustments",
        asset_class="equity",
        symbol="AAPL",
        frequency="events",
        pull_run_id="pull-test",
        pulled_at="2026-08-21T00:00:00+00:00",
        loader_version="0.3.0+test",
        source_urls=["yfinance://AAPL"],
        source_sha256s=["b" * 64],
        artifact_path="raw/yahoo/adjustments/A/AAPL.parquet",
        artifact_sha256=sha256_bytes(payload),
        row_count=events.num_rows,
        first_ts=DAY,
        last_ts=2 * DAY,
        gap_count=0,
        universe_hash="u" * 12,
    )
    store.put(events_manifest.artifact_path, payload, events_manifest)
    manifests.append(events_manifest)
    return store, manifests


def registry_of(manifests: list[FileManifest]):
    return build_from_manifests(manifests).table


def run_over(store: LocalRawStore, manifests: list[FileManifest], cfg, **kwargs):
    refs = bar_artifacts(registry_of(manifests))
    return clean_corpus(refs, lambda ref: store.get(ref.artifact_path), cfg, **kwargs)


# --- selection --------------------------------------------------------------------------


def test_the_yahoo_event_series_is_not_a_bar_artifact(tier) -> None:
    """It has no time grid. Cleaning it would partition a list of corporate actions by weekday."""
    _, manifests = tier
    paths = {ref.artifact_path for ref in bar_artifacts(registry_of(manifests))}
    assert "raw/yahoo/adjustments/A/AAPL.parquet" not in paths
    assert len(paths) == 3


def test_session_id_comes_from_the_file_not_from_a_guess(tier) -> None:
    store, manifests = tier
    for ref in bar_artifacts(registry_of(manifests)):
        table = pq.read_table(io.BytesIO(store.get(ref.artifact_path)))
        expected = {"BTCUSDT": "24x7", "EURUSD": "24x5", "AAPL": "XNYS-regular"}[ref.symbol]
        assert session_id_for(table, ref) == expected


def test_an_unknown_source_with_no_declared_session_refuses(tier) -> None:
    store, manifests = tier
    ref = bar_artifacts(registry_of(manifests))[0]
    stripped = pq.read_table(io.BytesIO(store.get(ref.artifact_path))).replace_schema_metadata({})
    ref.source = "mystery"
    with pytest.raises(ValueError, match="no fallback"):
        session_id_for(stripped, ref)


# --- the run ----------------------------------------------------------------------------


def test_a_full_run_produces_three_files_that_read_back(tier) -> None:
    store, manifests = tier
    cfg = config()
    run = run_over(store, manifests, cfg)
    assert run.failed == 0
    assert run.ok == 3
    assert run.total_bars == 400 + 400 + 400

    outputs = write_outputs(run)
    paths = clean_paths(cfg.clean_version)
    assert set(outputs) == {paths["segments"], paths["dropstats"], paths["manifest"]}

    segments = pq.read_table(io.BytesIO(outputs[paths["segments"]]))
    assert segments.num_rows == len(run.segments)
    assert set(segments["clean_config_hash"].to_pylist()) == {cfg.config_hash}
    # Every segment is bound to the bytes it came from.
    by_path = dict(
        zip(
            segments["artifact_path"].to_pylist(),
            segments["raw_artifact_sha256"].to_pylist(),
            strict=True,
        )
    )
    for manifest in manifests:
        if manifest.source != "yahoo":
            assert by_path[manifest.artifact_path] == manifest.artifact_sha256

    payload = json.loads(outputs[paths["manifest"]])
    assert payload["clean_config_hash"] == cfg.config_hash
    assert payload["series_ok"] == 3
    assert {row["source"] for row in payload["coverage"]} == {
        "binance_vision",
        "dukascopy",
        "stooq",
    }


def test_a_second_run_is_byte_identical(tier) -> None:
    """Determinism over the assembled tables, not just over one series."""
    store, manifests = tier
    cfg = config()
    first = write_outputs(run_over(store, manifests, cfg))
    second = write_outputs(run_over(store, manifests, cfg))
    paths = clean_paths(cfg.clean_version)
    assert first[paths["segments"]] == second[paths["segments"]]
    assert first[paths["dropstats"]] == second[paths["dropstats"]]


def test_an_unreadable_artifact_is_reported_not_skipped(tier) -> None:
    store, manifests = tier
    cfg = config()
    refs = bar_artifacts(registry_of(manifests))
    run = clean_corpus(
        refs,
        lambda ref: None if ref.symbol == "EURUSD" else store.get(ref.artifact_path),
        cfg,
    )
    assert run.ok == 2
    assert run.failed == 1
    assert "EURUSD" in run.failures[0]["artifact_path"]


# --- staleness --------------------------------------------------------------------------


def test_nothing_is_stale_when_nothing_changed(tier) -> None:
    store, manifests = tier
    cfg = config()
    segments = pq.read_table(
        io.BytesIO(write_outputs(run_over(store, manifests, cfg))[clean_paths(1)["segments"]])
    )
    assert stale_artifacts(bar_artifacts(registry_of(manifests)), segments) == set()


def test_a_changed_raw_file_is_detected_and_recleaned(tier) -> None:
    """The whole point of binding a segment to `raw_artifact_sha256`."""
    store, manifests = tier
    cfg = config()
    outputs = write_outputs(run_over(store, manifests, cfg))
    segments = pq.read_table(io.BytesIO(outputs[clean_paths(1)["segments"]]))
    drops = pq.read_table(io.BytesIO(outputs[clean_paths(1)["dropstats"]]))

    # The Stooq file gets restated by a later corporate action; its hash moves.
    touched = [
        m.model_copy(update={"artifact_sha256": "f" * 64}) if m.symbol == "AAPL" else m
        for m in manifests
    ]
    refs = bar_artifacts(registry_of(touched))
    assert stale_artifacts(refs, segments) == {"raw/stooq/us/1d/AAPL.parquet"}

    rerun = clean_corpus(
        refs,
        lambda ref: store.get(ref.artifact_path),
        cfg,
        existing=segments,
        existing_dropstats=drops,
        incremental=True,
    )
    assert rerun.ok == 1, "only the stale artifact was re-cleaned"
    assert rerun.reused_artifacts == 2
    # The carried-forward rows are still there, so the tables still cover the whole corpus.
    assert {row["artifact_path"] for row in rerun.segments} == {r.artifact_path for r in refs}
    assert {row["artifact_path"] for row in rerun.dropstats} == {r.artifact_path for r in refs}
    write_outputs(rerun)


def test_incremental_refuses_across_a_config_change(tier) -> None:
    store, manifests = tier
    segments = pq.read_table(
        io.BytesIO(write_outputs(run_over(store, manifests, config()))[clean_paths(1)["segments"]])
    )
    with pytest.raises(ConfigHashChanged, match="never trusted across a config change"):
        clean_corpus(
            bar_artifacts(registry_of(manifests)),
            lambda ref: store.get(ref.artifact_path),
            config(min_bars=5),
            existing=segments,
            incremental=True,
        )


# --- reports ----------------------------------------------------------------------------


def test_post_clean_views_render(tier) -> None:
    store, manifests = tier
    cfg = config()
    outputs = write_outputs(run_over(store, manifests, cfg))
    segments = pq.read_table(io.BytesIO(outputs[clean_paths(1)["segments"]]))
    dropstats = pq.read_table(io.BytesIO(outputs[clean_paths(1)["dropstats"]]))

    usable = usable_bars(segments)
    assert {g["source"] for g in usable} == {"binance_vision", "dukascopy", "stooq"}
    assert all(g["bars"] >= g["windows_512"] for g in usable)
    # No synthetic segment reaches 512 bars, so the corpus holds zero usable windows -- which is
    # exactly the distinction the report exists to make visible.
    assert sum(g["windows_512"] for g in usable) == 0

    rates = drop_rates(dropstats)
    assert {r["rule"] for r in rates} == {"gap", "jump", "illiquid", "stagnant", "min_length"}
    stagnant = [
        r for r in rates if r["source"] == "stooq" and r["rule"] in ("stagnant", "illiquid")
    ]
    assert sum(r["bars_dropped"] for r in stagnant) > 0, "the limit lock was not excised"

    top = most_cut_series(dropstats)
    assert top[0]["symbol"] == "AAPL"
    assert top[0]["pct_dropped"] > 0

    markdown = clean_summary_markdown(segments, dropstats, clean_config_hash=cfg.config_hash)
    assert "Usable corpus" in markdown and "Red flags" in markdown


def test_red_flags_fire_on_a_major_that_loses_too_much(tier) -> None:
    store, manifests = tier
    dropstats = pq.read_table(
        io.BytesIO(write_outputs(run_over(store, manifests, config()))[clean_paths(1)["dropstats"]])
    )
    flags = red_flags(dropstats, majors={"AAPL"})
    assert any(f["check"] == "major_series_loss" for f in flags)
    assert red_flags(dropstats, majors={"NOBODY"}, slice_limit_pct=100.0) == []


# --- the derived total-return tier ------------------------------------------------------


def test_identity_verdict_writes_a_manifest_and_no_bar_files(tier) -> None:
    """ADR-0019: `tr_close == close` here, and twelve thousand copies of `close` is not a tier."""
    store, manifests = tier
    run = derive_tr(store, registry_of(manifests).to_pylist(), verdict=POLICY_SPLIT_AND_DIVIDEND)
    assert run.materialized is False
    assert run.available == 1 and len(run.tickers) == 1
    assert run.coverage_pct() == 100.0
    assert run.tickers[0].materialized is False
    assert run.tickers[0].events_captured is True
    assert run.tickers[0].event_rows == 2
    # Counting dividends means downloading the event file, and the identity branch
    # has no reason to, so it does not.
    assert run.tickers[0].dividend_events is None

    payload = json.loads(run.to_json())
    assert payload["verdict"] == POLICY_SPLIT_AND_DIVIDEND
    assert payload["tickers"][0]["tr_available"] is True
    assert not list(Path(store.root).glob("derived/tr_close/**/*.parquet"))
    assert TR_MANIFEST_PATH.endswith("manifest.json")


def test_accumulation_verdict_materializes_the_tier(tier) -> None:
    """The branch a re-audit would switch on. It writes real files, and they read back."""
    store, manifests = tier
    run = derive_tr(store, registry_of(manifests).to_pylist(), verdict=POLICY_SPLIT_ONLY)
    assert run.materialized is True
    assert run.tickers[0].materialized is True
    assert run.tickers[0].dividend_events == 1

    written = sorted(Path(store.root).glob("derived/tr_close/**/*.parquet"))
    assert len(written) == 1
    table = pq.read_table(written[0])
    assert table.column_names == ["ts", "tr_close"]
    assert table.num_rows == 400
    # The dividend really did compound onto the path, so tr diverges from close.
    bars = pq.read_table(io.BytesIO(store.get("raw/stooq/us/1d/AAPL.parquet")))
    assert table["tr_close"].to_pylist() != bars["close"].to_pylist()
