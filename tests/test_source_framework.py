"""The generic half of a pull: the parts no source owns.

`test_pull_binance.py` still drives the whole path end to end through the Binance source and is
deliberately unedited -- that it passes against the refactored driver is the refactor's acceptance
test. What is here is the machinery that exists only because there is now more than one source.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

from axiom.provenance.manifest import FileManifest
from axiom.schema.bars import BARS_SCHEMA_V1, closed_window_bars, validate_bars, weekday_utc
from axiom.sources.base import SourcePlan, WorkItem, bucket_counts, shard_dir

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
# 2024-06-03 was a Monday. Every weekday reference below is offset from it.
MONDAY = 1_717_372_800_000


def bars(ts: list[int]) -> pa.Table:
    """A minimal valid table at the given timestamps."""
    n = len(ts)
    columns = {
        "ts": pa.array(ts, pa.int64()),
        "open": pa.array([1.0] * n),
        "high": pa.array([1.0] * n),
        "low": pa.array([1.0] * n),
        "close": pa.array([1.0] * n),
        "volume": pa.array([1.0] * n),
        "amount": pa.array([1.0] * n),
        "n_trades": pa.array([None] * n, pa.int64()),
        "taker_buy_volume": pa.array([None] * n, pa.float64()),
        "taker_buy_quote_volume": pa.array([None] * n, pa.float64()),
    }
    return pa.table(columns, schema=BARS_SCHEMA_V1)


# --- work items and plans ----------------------------------------------------------------


def test_vendor_symbol_falls_back_to_our_own():
    assert WorkItem("spot", "BTCUSDT", "1h").vendor_symbol == "BTCUSDT"
    assert WorkItem("fx", "EURUSD", "1h", source_symbol="EUR/USD").vendor_symbol == "EUR/USD"


def test_a_plan_with_mismatched_lists_is_refused():
    """The two lists are the resume mechanism; a mismatch means one of them is wrong."""
    with pytest.raises(ValueError, match="parallel lists"):
        SourcePlan(["a", "b"], ["digest-a"])


def test_an_empty_plan_is_falsey():
    assert not SourcePlan([], [])
    assert SourcePlan(["a"], ["d"])


# --- folder sharding ---------------------------------------------------------------------


def test_shard_dir_buckets_on_the_first_character():
    assert shard_dir("AAPL") == "A"
    assert shard_dir("aapl") == "A"
    assert shard_dir("3M") == "3"


def test_shard_dir_sends_punctuation_to_one_bucket():
    """Tickers carry dots and dashes; neither should become a folder name."""
    assert shard_dir(".SPX") == "_"
    assert shard_dir("-X") == "_"


def test_shard_dir_refuses_an_empty_symbol():
    with pytest.raises(ValueError, match="empty symbol"):
        shard_dir("")


def test_the_folder_guard_passes_a_realistic_equity_plan():
    """The Hub degrades past ~10 k files in a folder; ADR-0016 keeps every bucket under 9 000."""
    symbols = [f"{chr(65 + i % 26)}{i:05d}" for i in range(18_000)]
    counts = bucket_counts(symbols)
    # Two files per series -- the Parquet and its sidecar -- which is what actually lands.
    assert max(counts.values()) * 2 < 9_000
    assert sum(counts.values()) == 18_000


def test_the_folder_guard_can_actually_fail():
    """A guard that cannot fire is decoration. One letter, twelve thousand tickers."""
    counts = bucket_counts([f"A{i:05d}" for i in range(12_000)])
    assert max(counts.values()) * 2 > 9_000


# --- session-aware validation --------------------------------------------------------------


def test_weekday_utc_puts_monday_at_zero():
    assert weekday_utc(np.array([MONDAY])).tolist() == [0]
    assert weekday_utc(np.array([MONDAY + 5 * DAY_MS])).tolist() == [5]  # Saturday


def test_a_weekend_bar_fails_a_24x5_intraday_series():
    saturday_noon = MONDAY + 5 * DAY_MS + 12 * HOUR_MS
    report = validate_bars(bars([MONDAY, saturday_noon]), "1h", session_id="24x5")
    assert "bars_in_weekend_close" in report.violations
    assert not report.ok


def test_the_sunday_reopen_is_allowed():
    """The week opens 21:00 or 22:00 UTC depending on European DST; 20:00 is the safe edge."""
    sunday = MONDAY + 6 * DAY_MS
    assert closed_window_bars(np.array([sunday + 21 * HOUR_MS]), "24x5", "1h").tolist() == [False]
    assert closed_window_bars(np.array([sunday + 19 * HOUR_MS]), "24x5", "1h").tolist() == [True]


def test_a_24x5_daily_series_may_carry_a_sunday_bar():
    """Daily bars are stamped at 00:00 UTC of the calendar date, so the hour test cannot apply."""
    sunday = MONDAY + 6 * DAY_MS
    report = validate_bars(bars([MONDAY, sunday]), "1d", session_id="24x5")
    assert report.ok, report.summary()


def test_crypto_keeps_off_grid_as_a_warning():
    """Binance really does publish phase-shifted bars after a restart (ADR-0010)."""
    report = validate_bars(bars([MONDAY, MONDAY + HOUR_MS + 137]), "1h", session_id="24x7")
    assert "ts_off_grid" in report.warnings
    assert report.ok


def test_equities_daily_makes_off_grid_a_violation():
    """For a vendor's daily dump, alignment is a parse guarantee rather than a market fact."""
    report = validate_bars(bars([MONDAY, MONDAY + DAY_MS + 1]), "1d", session_id="XNYS-regular")
    assert "ts_off_grid" in report.violations
    assert not report.ok


def test_a_weekday_gap_is_never_a_failure():
    """A holiday is absence of information, and the raw tier records absence rather than filling."""
    report = validate_bars(
        bars([MONDAY, MONDAY + DAY_MS, MONDAY + 4 * DAY_MS]), "1d", session_id="XNYS-regular"
    )
    assert report.ok, report.summary()


def test_an_unknown_session_is_refused():
    with pytest.raises(ValueError, match="unknown session_id"):
        validate_bars(bars([MONDAY]), "1d", session_id="NYSE")


# --- manifest backward compatibility --------------------------------------------------------

V01_SIDECAR = {
    "schema_version": 1,
    "source": "binance_vision",
    "market": "spot",
    "asset_class": "crypto",
    "symbol": "BTCUSDT",
    "frequency": "1h",
    "pull_run_id": "pull-20260801T000000Z",
    "pulled_at": "2026-08-01T00:00:00+00:00",
    "loader_version": "0.1.0+abc1234",
    "source_urls": ["https://data.binance.vision/x-1h-2024-01.zip"],
    "source_sha256s": ["a" * 64],
    "artifact_path": "raw/binance/spot/1h/BTCUSDT.parquet",
    "artifact_sha256": "b" * 64,
    "row_count": 744,
    "first_ts": 1_704_067_200_000,
    "last_ts": 1_706_745_600_000,
    "gap_count": 0,
    "off_grid_count": 0,
    "volume_convention": "base+quote_native",
    "amount_synthesized": False,
    "adjustment_policy": "none",
    "universe_hash": "0123456789ab",
}


def v01_sidecar_json() -> str:
    """A sidecar exactly as v0.1 wrote it, hash included, with none of v0.2's fields."""
    manifest = FileManifest.model_validate(V01_SIDECAR)
    payload = dict(V01_SIDECAR)
    payload["manifest_sha256"] = manifest.manifest_sha256
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def test_a_v01_sidecar_still_verifies_under_v02():
    """The whole point of defaulting the new fields out of the hash.

    Without it, adding an optional field would recompute a different digest for every sidecar
    already in `axiom-raw` and they would all be reported as edited.
    """
    manifest = FileManifest.from_json(v01_sidecar_json())
    assert manifest.price_side == "trade"
    assert manifest.redistribution_class == "loader_manifest_private_cache"


def test_the_defaults_are_absent_from_the_identity_payload():
    manifest = FileManifest.model_validate(V01_SIDECAR)
    payload = manifest.identity_payload()
    assert "price_side" not in payload
    assert "redistribution_class" not in payload
    assert "source_symbol" not in payload


def test_a_source_symbol_equal_to_the_symbol_does_not_change_the_hash():
    """Binance writes source_symbol=BTCUSDT, which says nothing the symbol did not."""
    bare = FileManifest.model_validate(V01_SIDECAR)
    echoed = bare.model_copy(update={"source_symbol": "BTCUSDT"})
    assert echoed.manifest_sha256 == bare.manifest_sha256


def test_a_real_source_symbol_does_change_the_hash():
    """It is identity: the same rows fetched under a different vendor name are a different file."""
    bare = FileManifest.model_validate(V01_SIDECAR)
    renamed = bare.model_copy(update={"source_symbol": "EUR/USD"})
    assert renamed.manifest_sha256 != bare.manifest_sha256


def test_a_non_default_price_side_changes_the_hash():
    bare = FileManifest.model_validate(V01_SIDECAR)
    bid = bare.model_copy(update={"price_side": "bid"})
    assert bid.manifest_sha256 != bare.manifest_sha256
