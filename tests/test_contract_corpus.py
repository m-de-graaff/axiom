"""The two corpus passes, end to end on a synthetic tier.

The cloud jobs are thin: download bytes, call these functions, upload what comes back. So the
things that can actually be wrong — the firewall truncation, the sketch's quantiles, the
determinism of the emitted YAML, the anchor-bar correction to the window count — are all testable
here, offline, with no market data anywhere near the machine.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from axiom.contract import load_spec
from axiom.contract.corpus import (
    SegmentRef,
    constants_tables,
    constants_yaml,
    dryrun_artifact,
    fit_artifact,
    fit_corpus,
    group_by_artifact,
    pick_audit_segments,
    quantile_rows,
    slice_segment,
    usable_windows,
)
from axiom.contract.spec import SCHEMA_VERSION, ContractConstants
from axiom.contract.stats import Sketch, SketchSet
from axiom.testing import synth
from axiom.testing.contract import constants

SPECS = [load_spec("contract_geo_v1"), load_spec("contract_ret_v1")]
FIREWALL = 1_735_689_600_000


def make_ref(**overrides) -> SegmentRef:
    defaults = dict(
        segment_id="seg-1",
        artifact_path="raw/crypto/1h/BTCUSDT.parquet",
        source="binance_vision",
        market="spot",
        symbol="BTCUSDT",
        asset_class="crypto",
        frequency="1h",
        start_ts=0,
        end_ts=2**62,
        n_bars=800,
    )
    return SegmentRef(**{**defaults, **overrides})


def artifact_bytes(n_bars: int = 800, start_ms: int = 1_600_000_000_000) -> bytes:
    """A synthetic artifact with real gaps in it.

    `synth.walk` opens every bar exactly at the previous close, which makes the gap feature
    identically zero — a distribution with no spread, which the sketch refuses on purpose. Real
    bars gap; the fixture has to as well or it tests the refusal instead of the fit.
    """
    series = synth.walk("1h", n_bars)
    rng = np.random.default_rng(11)
    opens = series.column("open") * np.exp(rng.normal(0.0, 0.001, n_bars))
    highs = np.maximum(series.column("high"), np.maximum(opens, series.column("close")))
    lows = np.minimum(series.column("low"), np.minimum(opens, series.column("close")))
    series = synth._rebuild(series, open_=opens, high=highs, low=lows)
    table = series.table
    ts = start_ms + np.arange(n_bars, dtype=np.int64) * 3_600_000
    table = table.set_column(table.schema.get_field_index("ts"), "ts", [ts])
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


# --- segment slicing and the firewall ----------------------------------------------------


def test_a_segment_is_sliced_by_timestamp_not_by_row_index() -> None:
    data = artifact_bytes(100, start_ms=1_600_000_000_000)
    bars = pq.read_table(io.BytesIO(data))
    ref = make_ref(
        start_ts=1_600_000_000_000 + 10 * 3_600_000, end_ts=1_600_000_000_000 + 19 * 3_600_000
    )

    segment = slice_segment(bars, ref)

    assert segment.num_rows == 10


def test_the_firewall_excludes_the_bar_that_lands_exactly_on_it() -> None:
    """Half-open on the right. A bar at `firewall_ts` is post-firewall, not the last pre-."""
    data = artifact_bytes(100, start_ms=FIREWALL - 50 * 3_600_000)
    bars = pq.read_table(io.BytesIO(data))

    segment = slice_segment(bars, make_ref(), before_ts=FIREWALL)

    assert segment.num_rows == 50
    assert int(segment["ts"][-1].as_py()) == FIREWALL - 3_600_000


def test_the_fit_consumes_no_bar_at_or_after_the_firewall() -> None:
    """The claim the constants manifest makes, checked rather than trusted."""
    data = artifact_bytes(400, start_ms=FIREWALL - 200 * 3_600_000)

    sketches, _ = fit_artifact(data, [make_ref(n_bars=400)], SPECS, FIREWALL)

    assert sketches.max_ts < FIREWALL
    assert sketches.bars == 200


def test_a_segment_wholly_after_the_firewall_is_skipped_not_failed() -> None:
    data = artifact_bytes(50, start_ms=FIREWALL)

    sketches, skipped = fit_artifact(data, [make_ref(n_bars=50)], SPECS, FIREWALL)

    assert skipped == ["seg-1"]
    assert sketches.bars == 0


# --- the sketch --------------------------------------------------------------------------


def test_the_sketch_finds_the_median_of_a_known_distribution() -> None:
    sketch = Sketch()
    sketch.add(np.linspace(-1.0, 1.0, 100_001))

    assert sketch.quantile(0.5) == pytest.approx(0.0, abs=1e-3)
    assert sketch.quantile(0.25) == pytest.approx(-0.5, abs=1e-3)


def test_two_sketches_merge_to_the_sketch_of_both() -> None:
    """Mergeability is why this is a histogram and not a sample."""
    values = np.random.default_rng(0).normal(0, 1, 20_000)
    whole = Sketch()
    whole.add(values)
    halves = Sketch()
    halves.add(values[:10_000])
    other = Sketch()
    other.add(values[10_000:])

    halves.merge(other)

    np.testing.assert_array_equal(halves.counts, whole.counts)


def test_values_outside_the_support_land_in_the_overflow_buckets() -> None:
    sketch = Sketch()

    sketch.add(np.array([-1e6, 0.0, 1e6]))

    assert (sketch.under, sketch.over) == (1, 1)
    assert sketch.total == 3


def test_center_and_scale_are_the_median_and_a_normal_consistent_sigma() -> None:
    sketch = Sketch()
    sketch.add(np.random.default_rng(1).normal(2.0, 0.5, 400_000))

    center, scale = sketch.center_scale()

    assert center == pytest.approx(2.0, abs=0.01)
    assert scale == pytest.approx(0.5, abs=0.01)


def test_a_distribution_with_no_spread_at_all_is_refused() -> None:
    """Half an asset class on one value is a finding, not something to divide by epsilon."""
    sketch = Sketch()
    sketch.add(np.zeros(1000))

    with pytest.raises(ValueError, match="degenerate distribution"):
        sketch.center_scale()


def test_a_sketch_survives_the_round_trip_through_its_transport_form() -> None:
    sketch = Sketch()
    sketch.add(np.random.default_rng(2).normal(0, 1, 5_000))

    restored = Sketch.from_dict(sketch.to_dict())

    np.testing.assert_array_equal(restored.counts, sketch.counts)
    assert restored.total == sketch.total


# --- the constants file ------------------------------------------------------------------


def _manifest(**overrides) -> dict:
    base = {
        "generated_utc": "2026-08-22T00:00:00Z",
        "git_commit": "abc",
        "registry_hash": "reg",
        "clean_config_hash": "cln",
        "firewall_ts": FIREWALL,
        "firewall_config_sha256": "0" * 64,
        "firewall_respected": True,
        "segments_consumed": 1,
        "bars_consumed": 100,
        "partial": False,
    }
    return {**base, **overrides}


def test_two_fits_over_the_same_bars_emit_byte_identical_yaml() -> None:
    """A constants file that differs run to run cannot be a frozen part of a contract."""
    data = artifact_bytes(600)
    grouped = {"a.parquet": [make_ref(n_bars=600)]}

    first = fit_corpus(grouped, lambda _: data, SPECS, 2**62, concurrency=2)
    second = fit_corpus(grouped, lambda _: data, SPECS, 2**62, concurrency=4)

    assert constants_yaml(
        constants_tables(first.sketches, SPECS), _manifest(), SCHEMA_VERSION
    ) == constants_yaml(constants_tables(second.sketches, SPECS), _manifest(), SCHEMA_VERSION)


def test_the_emitted_yaml_loads_back_as_a_constants_table() -> None:
    import yaml

    data = artifact_bytes(600)
    run = fit_corpus({"a.parquet": [make_ref(n_bars=600)]}, lambda _: data, SPECS, 2**62)

    payload = constants_yaml(constants_tables(run.sketches, SPECS), _manifest(), SCHEMA_VERSION)

    table = ContractConstants.model_validate(yaml.safe_load(payload))
    scaling = table.scaling_for(SPECS[0], "crypto", "1h")
    assert len(scaling) == 6
    assert all(s.scale > 0 for s in scaling)


def test_the_geometry_wicks_come_out_one_sided() -> None:
    """`upper` centers above zero and `lower` below it, or the columns got swapped."""
    data = artifact_bytes(2000)
    run = fit_corpus({"a.parquet": [make_ref(n_bars=2000)]}, lambda _: data, SPECS, 2**62)

    tables = constants_tables(run.sketches, SPECS)["geo-v1"]["crypto"]["1h"]

    assert tables["upper"]["center"] > 0
    assert tables["lower"]["center"] < 0


# --- the dryrun --------------------------------------------------------------------------


def test_the_dryrun_audits_the_split_points_it_was_given() -> None:
    data = artifact_bytes(800)
    ref = make_ref(n_bars=800)

    result, _ = dryrun_artifact(
        data, [ref], SPECS, constants(SPECS), audit_splits={ref.segment_id: [5, 300, 700]}
    )

    assert result.audits_run == 6  # three splits, two specs
    assert result.audits_passed == 6


def test_the_dryrun_hashes_exactly_one_segment_per_pinned_series() -> None:
    data = artifact_bytes(800)
    refs = [
        make_ref(
            segment_id="short", start_ts=0, end_ts=1_600_000_000_000 + 99 * 3_600_000, n_bars=100
        ),
        make_ref(segment_id="long", n_bars=800),
    ]

    _, hashes = dryrun_artifact(
        data, refs, SPECS, constants(SPECS), snapshot_series=("BTCUSDT-1h",)
    )

    assert sorted(hashes) == ["BTCUSDT-1h/geo-v1", "BTCUSDT-1h/ret-v1"]


def test_the_dryrun_reports_a_quantile_row_per_spec_class_frequency_feature() -> None:
    data = artifact_bytes(400)
    result, _ = dryrun_artifact(data, [make_ref(n_bars=400)], SPECS, constants(SPECS))

    rows = quantile_rows(result)

    assert len(rows) == 12
    assert {r["spec_id"] for r in rows} == {"geo-v1", "ret-v1"}
    assert all(0.0 <= r["clip_rate"] <= 1.0 for r in rows)


def test_a_failing_artifact_is_recorded_rather_than_taking_the_run_down() -> None:
    run = fit_corpus(
        {"good.parquet": [make_ref()], "missing.parquet": [make_ref()]},
        lambda path: artifact_bytes(600) if path == "good.parquet" else None,
        SPECS,
        2**62,
    )

    assert run.ok == 1
    assert run.failed == 1


# --- the anchor-bar correction ------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_bars", "expected"),
    [(511, 0), (512, 0), (513, 1), (1000, 488), (10, 0)],
)
def test_usable_windows_counts_feature_rows_not_bars(n_bars: int, expected: int) -> None:
    """v0.3 published `max(0, n_bars - 511)`. Bar 0 is the anchor, so every segment loses one."""
    assert usable_windows(n_bars) == expected


# --- audit selection ----------------------------------------------------------------------


def test_the_audit_picks_the_same_segments_every_time() -> None:
    refs = [make_ref(segment_id=f"s{i}", n_bars=700 + i) for i in range(200)]

    first = pick_audit_segments(refs)
    second = pick_audit_segments(refs)

    assert first == second
    assert len(first) == 50
    assert all(len(splits) == 3 for splits in first.values())


def test_the_audit_skips_segments_too_short_to_reach_the_rolling_window() -> None:
    refs = [make_ref(segment_id=f"s{i}", n_bars=100) for i in range(200)]

    assert pick_audit_segments(refs) == {}


# --- grouping ------------------------------------------------------------------------------


def test_segments_are_grouped_so_each_artifact_downloads_once() -> None:
    refs = [
        make_ref(segment_id="a", artifact_path="x.parquet", start_ts=200),
        make_ref(segment_id="b", artifact_path="x.parquet", start_ts=100),
        make_ref(segment_id="c", artifact_path="y.parquet"),
    ]

    grouped = group_by_artifact(refs)

    assert sorted(grouped) == ["x.parquet", "y.parquet"]
    assert [r.segment_id for r in grouped["x.parquet"]] == ["b", "a"]


def test_a_sketch_set_merges_its_clip_counts_too() -> None:
    key = ("geo-v1", "crypto", "1h", "gap")
    left = SketchSet(clipped={key: 3}, bars=10)
    right = SketchSet(clipped={key: 4}, bars=5)

    left.merge(right)

    assert left.clipped[key] == 7
    assert left.bars == 15


def test_the_snapshot_file_is_json_a_human_can_read() -> None:
    data = artifact_bytes(600)
    _, hashes = dryrun_artifact(
        data, [make_ref(n_bars=600)], SPECS, constants(SPECS), snapshot_series=("BTCUSDT-1h",)
    )

    text = json.dumps(hashes, indent=1, sort_keys=True)

    assert json.loads(text) == hashes
    assert all(len(digest) == 64 for digest in hashes.values())


# --- fan-out bounding ----------------------------------------------------------------------


def test_the_fan_out_does_not_queue_the_whole_corpus_before_yielding_anything() -> None:
    """The regression test for a run the runner killed.

    Each result carries about 6 MB of quantile sketches. Submitting ten thousand artifacts up
    front keeps every completed result alive inside its `Future` until the consumer reaches it,
    and the consumer never catches up — so the first corpus-wide fit climbed until it was killed,
    with nothing in the log but a cancellation.
    """
    from axiom.contract.corpus import _fan_out

    started: list[str] = []
    grouped = {f"a{i}.parquet": [make_ref()] for i in range(500)}

    generator = _fan_out(grouped, lambda path: started.append(path) or b"", lambda *_: None, 2)
    next(generator)

    assert len(started) < len(grouped)
    generator.close()


def test_the_fan_out_still_visits_every_artifact() -> None:
    from axiom.contract.corpus import _fan_out

    grouped = {f"a{i}.parquet": [make_ref()] for i in range(50)}

    seen = [path for path, _, _ in _fan_out(grouped, lambda _: b"", lambda *_: None, 4)]

    assert sorted(seen) == sorted(grouped)
