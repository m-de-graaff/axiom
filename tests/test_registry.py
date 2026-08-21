"""The registry, built over a synthetic sidecar tree.

The registry is a cache with no authority, so the properties worth testing are the ones that make
a cache trustworthy: it reproduces exactly from unchanged inputs, it orders deterministically, and
it never quietly loses a row it could not read.
"""

from __future__ import annotations

import json
import os

import pytest

from axiom.provenance.manifest import FileManifest
from axiom.registry import (
    REGISTRY_SCHEMA,
    BadSidecar,
    build_from_manifests,
    coverage_matrix,
    gappiest,
    m0_verdict,
    read_registry,
    staleness,
    storage_by_source,
    summary_markdown,
    write_registry_parquet,
)
from axiom.registry.build import registry_metadata

DAY_MS = 86_400_000
JAN_2024 = 1_704_067_200_000


def manifest(
    *,
    source: str = "binance_vision",
    market: str = "spot",
    asset_class: str = "crypto",
    symbol: str = "BTCUSDT",
    frequency: str = "1h",
    rows: int = 1000,
    gaps: int = 0,
    days: int = 365,
    **extra,
) -> FileManifest:
    path = f"raw/{source}/{market}/{frequency}/{symbol}.parquet"
    return FileManifest(
        schema_version=1,
        source=source,
        market=market,
        asset_class=asset_class,
        symbol=symbol,
        frequency=frequency,
        pull_run_id="pull-1",
        pulled_at="2026-08-20T00:00:00+00:00",
        loader_version="0.2.0+abc1234",
        source_urls=["https://example/a.zip", "https://example/b.zip"],
        source_sha256s=["a" * 64, "b" * 64],
        artifact_path=path,
        artifact_sha256="c" * 64,
        row_count=rows,
        first_ts=JAN_2024,
        last_ts=JAN_2024 + days * DAY_MS,
        gap_count=gaps,
        off_grid_count=0,
        universe_hash="0123456789ab",
        **extra,
    )


def corpus() -> list[FileManifest]:
    """One artifact per M0 slice, plus a couple of extras to group over."""
    return [
        manifest(symbol="BTCUSDT", frequency="1h", rows=8760),
        manifest(symbol="BTCUSDT", frequency="1d", rows=365),
        manifest(symbol="ETHUSDT", frequency="1h", rows=8760, gaps=12),
        manifest(market="um", symbol="BTCUSDT", frequency="1h", rows=8000),
        manifest(market="um", symbol="BTCUSDT", frequency="1d", rows=360),
        manifest(
            source="dukascopy",
            market="fx",
            asset_class="fx",
            symbol="EURUSD",
            frequency="1h",
            rows=6000,
            gaps=52,
            price_side="bid",
            volume_convention="dukascopy_tick_volume",
            amount_synthesized=True,
            source_symbol="EUR/USD",
        ),
        manifest(
            source="dukascopy",
            market="fx",
            asset_class="fx",
            symbol="EURUSD",
            frequency="1d",
            rows=260,
            gaps=52,
            price_side="bid",
            source_symbol="EUR/USD",
        ),
        manifest(
            source="dukascopy",
            market="commodity",
            asset_class="commodity",
            symbol="XAUUSD",
            frequency="1h",
            rows=6000,
        ),
        manifest(
            source="dukascopy",
            market="commodity",
            asset_class="commodity",
            symbol="XAUUSD",
            frequency="1d",
            rows=260,
        ),
        manifest(
            source="stooq",
            market="us",
            asset_class="equity",
            symbol="AAPL",
            frequency="1d",
            rows=5000,
            volume_convention="shares",
            amount_synthesized=True,
            adjustment_policy="vendor_adjusted_unverified",
            source_symbol="aapl.us",
        ),
    ]


SIZES = {m.artifact_path: 1_000 * m.row_count for m in corpus()}


# --- build ---------------------------------------------------------------------------------


def test_every_sidecar_becomes_one_row():
    build = build_from_manifests(corpus(), sizes=SIZES)
    assert build.table.num_rows == len(corpus())
    assert build.table.schema == REGISTRY_SCHEMA


def test_a_rebuild_of_an_unchanged_tier_reproduces_the_hash():
    """Idempotence is the whole claim a cache with no authority has to make."""
    first = build_from_manifests(corpus(), sizes=SIZES)
    second = build_from_manifests(list(reversed(corpus())), sizes=SIZES)
    assert first.registry_hash == second.registry_hash
    assert first.table.equals(second.table)


def test_ordering_is_deterministic_regardless_of_listing_order():
    build = build_from_manifests(list(reversed(corpus())), sizes=SIZES)
    paths = build.table["artifact_path"].to_pylist()
    keys = list(
        zip(
            build.table["source"].to_pylist(),
            build.table["asset_class"].to_pylist(),
            build.table["market"].to_pylist(),
            build.table["frequency"].to_pylist(),
            build.table["symbol"].to_pylist(),
            strict=True,
        )
    )
    assert keys == sorted(keys)
    assert len(set(paths)) == len(paths)


def test_list_fields_become_a_count():
    """Provenance detail stays in the sidecar; the table keeps what group-bys can use."""
    build = build_from_manifests(corpus(), sizes=SIZES)
    assert "source_urls" not in build.table.column_names
    assert build.table["source_file_count"].to_pylist()[0] == 2


def test_source_symbol_defaults_to_the_symbol():
    build = build_from_manifests([manifest(symbol="BTCUSDT")], sizes={})
    assert build.table["source_symbol"].to_pylist() == ["BTCUSDT"]


def test_v02_fields_survive_into_the_table():
    build = build_from_manifests(corpus(), sizes=SIZES)
    rows = {r["artifact_path"]: r for r in build.table.to_pylist()}
    fx = rows["raw/dukascopy/fx/1h/EURUSD.parquet"]
    assert fx["price_side"] == "bid"
    assert fx["volume_convention"] == "dukascopy_tick_volume"
    assert fx["amount_synthesized"] is True
    assert fx["source_symbol"] == "EUR/USD"


def test_an_unreadable_sidecar_is_reported_never_dropped():
    """A registry that silently omits what it could not parse is worse than no registry."""
    build = build_from_manifests(
        corpus(), sizes=SIZES, bad=[BadSidecar("raw/x/y.manifest.json", "ValueError: edited")]
    )
    assert not build.ok
    assert build.bad[0].path == "raw/x/y.manifest.json"


def test_the_parquet_carries_its_own_hash():
    build = build_from_manifests(corpus(), sizes=SIZES)
    data = write_registry_parquet(build.table, registry_hash_value=build.registry_hash)
    meta = registry_metadata(read_registry(data))
    assert meta["axiom_registry_hash"] == build.registry_hash
    assert meta["axiom_registry_rows"] == str(build.table.num_rows)


def test_a_registry_round_trips_through_parquet():
    build = build_from_manifests(corpus(), sizes=SIZES)
    data = write_registry_parquet(build.table, registry_hash_value=build.registry_hash)
    assert read_registry(data).drop_columns([]).to_pylist() == build.table.to_pylist()


# --- canned reports -------------------------------------------------------------------------


def table():
    return build_from_manifests(corpus(), sizes=SIZES).table


def test_the_coverage_matrix_groups_the_corpus():
    rows = {(r["source"], r["market"], r["frequency"]): r for r in coverage_matrix(table())}
    assert rows[("binance_vision", "spot", "1h")]["series"] == 2
    assert rows[("binance_vision", "spot", "1h")]["bars"] == 8760 * 2
    assert rows[("stooq", "us", "1d")]["series"] == 1


def test_the_coverage_matrix_is_a_golden_shape():
    """Pinned so a schema change that silently reshapes the corpus view fails here first."""
    rows = coverage_matrix(table())
    assert [(r["source"], r["asset_class"], r["market"], r["frequency"]) for r in rows] == [
        ("binance_vision", "crypto", "spot", "1d"),
        ("binance_vision", "crypto", "spot", "1h"),
        ("binance_vision", "crypto", "um", "1d"),
        ("binance_vision", "crypto", "um", "1h"),
        ("dukascopy", "commodity", "commodity", "1d"),
        ("dukascopy", "commodity", "commodity", "1h"),
        ("dukascopy", "fx", "fx", "1d"),
        ("dukascopy", "fx", "fx", "1h"),
        ("stooq", "equity", "us", "1d"),
    ]


def test_storage_by_source_totals_the_bytes():
    entries = {e["source"]: e for e in storage_by_source(table())}
    assert entries["stooq"]["bytes"] == 5_000_000
    assert entries["dukascopy"]["series"] == 4


def test_gappiest_ranks_and_skips_the_clean_ones():
    entries = gappiest(table())
    assert entries[0]["gap_count"] == 52
    assert all(e["gap_count"] > 0 for e in entries)


def test_staleness_is_measured_against_a_pinned_now():
    now = JAN_2024 + 400 * DAY_MS
    entries = staleness(table(), now_ms=now)
    assert entries[0]["stale_days"] == 35.0  # every fixture series ends 365 days after Jan 2024
    assert entries == sorted(entries, key=lambda e: -e["stale_days"])


def test_m0_reports_a_complete_corpus_as_complete():
    verdict = {(v["asset_class"], v["market"]): v for v in m0_verdict(table())}
    assert all(v["ok"] for v in verdict.values())


def test_m0_names_the_missing_slice_rather_than_just_failing():
    """'M0 is not assembled' is useless next to 'commodities are missing at 1h'."""
    without_commodities = [m for m in corpus() if m.asset_class != "commodity"]
    verdict = {
        (v["asset_class"], v["market"]): v
        for v in m0_verdict(build_from_manifests(without_commodities, sizes=SIZES).table)
    }
    assert verdict[("commodity", "commodity")]["missing"] == ["1h", "1d"]
    assert verdict[("crypto", "spot")]["ok"]


# --- summary --------------------------------------------------------------------------------


def test_the_summary_answers_what_from_where_and_when():
    build = build_from_manifests(corpus(), sizes=SIZES)
    text = summary_markdown(
        build.table, registry_hash=build.registry_hash, now_ms=JAN_2024 + 400 * DAY_MS
    )
    assert build.registry_hash in text
    assert "## Coverage" in text
    assert "## Corpus M0" in text
    assert "dukascopy" in text and "stooq" in text and "binance_vision" in text


def test_the_summary_says_when_rows_are_missing():
    """Every number becomes a lower bound, and the reader has to be told."""
    build = build_from_manifests(corpus(), sizes=SIZES, bad=[BadSidecar("x", "boom")])
    text = summary_markdown(build.table, registry_hash=build.registry_hash, bad_count=1)
    assert "could not be read" in text
    assert "lower bound" in text


def test_bad_sidecars_serialize_for_the_companion_file():
    from axiom.registry.build import bad_sidecars_json

    payload = json.loads(bad_sidecars_json([BadSidecar("a/b.manifest.json", "ValueError: x")]))
    assert payload == [{"path": "a/b.manifest.json", "error": "ValueError: x"}]


@pytest.mark.parametrize("report", [coverage_matrix, storage_by_source, gappiest, staleness])
def test_reports_survive_an_empty_registry(report):
    empty = build_from_manifests([], sizes={}).table
    assert report(empty) == []


def test_hub_timeouts_are_widened_but_never_overridden(monkeypatch) -> None:
    """Ten seconds is fine for a handful of files and not for thirteen thousand.

    A registry build lost 19 sidecars to `ReadTimeout` and then failed outright when every retry
    round died early on one slow read. But a caller who has already set a value means it.
    """
    from axiom.raw.store import HF_TIMEOUT_ENV, set_hub_timeouts

    for name in HF_TIMEOUT_ENV:
        monkeypatch.delenv(name, raising=False)
    set_hub_timeouts()
    assert all(os.environ[name] == value for name, value in HF_TIMEOUT_ENV.items())

    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    set_hub_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "5"
