"""Bar schema v1: the invariants, the timestamp-unit detection, the Parquet round trip.

Every fixture here is synthetic. No test in this repo may touch a market-data byte -- that is a
hard constraint of the roadmap, not a style preference, and it also happens to make the suite
runnable offline in CI.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom.schema.bars import (
    BARS_SCHEMA_V1,
    FREQUENCIES,
    ROW_GROUP_SIZE,
    bars_metadata,
    count_gaps,
    count_off_grid,
    grid_step_ms,
    normalize_ts_ms,
    validate_bars,
)

#: 2024-01-01T00:00:00Z, a UTC midnight, so it is aligned to both supported grids.
EPOCH = 1_704_067_200_000


def make_bars(n: int = 10, frequency: str = "1h", start: int = EPOCH, **overrides) -> pa.Table:
    """A schema-valid table of ``n`` bars. Overrides replace a column wholesale."""
    step = grid_step_ms(frequency)
    ts = np.arange(n, dtype=np.int64) * step + start
    open_ = np.linspace(100.0, 110.0, n)
    close = open_ + 0.5
    columns = {
        "ts": ts,
        "open": open_,
        "high": np.maximum(open_, close) + 1.0,
        "low": np.minimum(open_, close) - 1.0,
        "close": close,
        "volume": np.full(n, 3.0),
        "amount": np.full(n, 330.0),
        "n_trades": np.arange(n, dtype=np.int64) + 7,
        "taker_buy_volume": np.full(n, 1.5),
        "taker_buy_quote_volume": np.full(n, 165.0),
    }
    columns.update(overrides)
    return pa.table(
        [pa.array(columns[f.name], type=f.type) for f in BARS_SCHEMA_V1],
        schema=BARS_SCHEMA_V1,
    )


def mutate(table: pa.Table, column: str, index: int, value) -> pa.Table:
    """Return ``table`` with one cell replaced, keeping the schema."""
    values = table[column].to_pylist()
    values[index] = value
    field = table.schema.field(column)
    return table.set_column(
        table.schema.get_field_index(column), field, pa.array(values, type=field.type)
    )


# --- the golden fixture ------------------------------------------------------------------


def test_golden_fixture_passes():
    report = validate_bars(make_bars(), "1h")
    assert report.ok, report.summary()
    assert report.row_count == 10
    assert "all invariants hold" in report.summary()


@pytest.mark.parametrize("frequency", sorted(FREQUENCIES))
def test_golden_fixture_passes_at_every_frequency(frequency):
    assert validate_bars(make_bars(frequency=frequency), frequency).ok


# --- one mutation per invariant ----------------------------------------------------------

#: (column, index, bad value, the violation code it must raise). One row per ADR-0010 invariant.
MUTATIONS = [
    ("ts", 5, EPOCH, "ts_not_increasing"),  # duplicate of row 0's timestamp, so also out of order
    ("high", 3, 0.0, "high_below_open_or_close"),
    ("low", 3, 1e9, "low_above_open_or_close"),
    ("volume", 4, -1.0, "volume_negative"),
    ("amount", 4, -1.0, "amount_negative"),
    ("close", 2, None, "null_close"),
    ("open", 2, float("nan"), "null_open"),
    ("volume", 6, None, "null_volume"),
    ("amount", 7, float("nan"), "null_amount"),
]


@pytest.mark.parametrize(("column", "index", "value", "code"), MUTATIONS)
def test_each_invariant_violation_is_caught(column, index, value, code):
    report = validate_bars(mutate(make_bars(), column, index, value), "1h")
    assert not report.ok
    assert code in report.violations, report.summary()
    assert report.violations[code].count >= 1


def test_high_below_low_is_reported():
    # Swapping high and low breaks three invariants at once; the point is that the specific
    # high < low check fires rather than being masked by the other two.
    table = make_bars()
    swapped = table.set_column(
        table.schema.get_field_index("high"), table.schema.field("high"), table["low"]
    )
    swapped = swapped.set_column(
        table.schema.get_field_index("low"), table.schema.field("low"), table["high"]
    )
    report = validate_bars(swapped, "1h")
    assert "high_below_low" in report.violations


def test_first_offending_row_is_reported():
    report = validate_bars(mutate(make_bars(), "volume", 4, -1.0), "1h")
    assert report.violations["volume_negative"].first_row == 4


def test_raise_on_error_names_the_violation():
    with pytest.raises(ValueError, match="volume_negative"):
        validate_bars(mutate(make_bars(), "volume", 4, -1.0), "1h", raise_on_error=True)


def test_empty_table_fails_rather_than_passing_vacuously():
    report = validate_bars(make_bars(0), "1h")
    assert not report.ok
    assert "empty" in report.violations


def test_missing_column_is_a_structural_error_not_a_violation():
    table = make_bars().drop_columns(["amount"])
    with pytest.raises(ValueError, match="missing schema columns"):
        validate_bars(table, "1h")


def test_wrong_column_type_is_rejected():
    table = make_bars()
    idx = table.schema.get_field_index("close")
    as_float32 = table.set_column(
        idx, pa.field("close", pa.float32()), table["close"].cast(pa.float32())
    )
    with pytest.raises(ValueError, match="expected double"):
        validate_bars(as_float32, "1h")


def test_unsupported_frequency_is_refused():
    with pytest.raises(ValueError, match="unsupported frequency"):
        validate_bars(make_bars(), "5m")


# --- gaps are recorded, never a violation ------------------------------------------------


def test_off_grid_bars_are_a_warning_not_a_violation():
    # Real bars from an exchange restart: still hourly-spaced, on a shifted phase. Rejecting them
    # would have thrown away BTCUSDT spot 1h over 43 rows out of 78 829.
    shifted = np.arange(10, dtype=np.int64) * 3_600_000 + EPOCH + 1_694_789
    table = make_bars(10, ts=shifted)
    report = validate_bars(table, "1h")
    assert report.ok
    assert "ts_off_grid" in report.warnings
    assert report.warnings["ts_off_grid"].count == 10
    assert count_off_grid(table["ts"], "1h") == 10
    assert "warnings" in report.summary()


def test_a_warning_does_not_trip_raise_on_error():
    shifted = np.arange(4, dtype=np.int64) * 3_600_000 + EPOCH + 1_694_789
    validate_bars(make_bars(4, ts=shifted), "1h", raise_on_error=True)


def test_an_on_grid_series_has_no_off_grid_bars():
    assert count_off_grid(make_bars(10)["ts"], "1h") == 0


def test_a_shifted_bar_still_has_to_be_increasing():
    table = make_bars(10)
    broken = mutate(table, "ts", 5, EPOCH - 1)
    report = validate_bars(broken, "1h")
    assert not report.ok
    assert "ts_not_increasing" in report.violations


def test_gap_passes_validation_and_is_counted():
    table = make_bars(10)
    with_hole = table.take([0, 1, 2, 6, 7, 8, 9])
    assert validate_bars(with_hole, "1h").ok
    assert count_gaps(with_hole["ts"], "1h") == 3


def test_no_gaps_in_a_contiguous_series():
    assert count_gaps(make_bars(10)["ts"], "1h") == 0
    assert count_gaps(make_bars(1)["ts"], "1h") == 0


# --- microsecond detection ---------------------------------------------------------------


def test_microseconds_normalize_to_the_same_series_as_milliseconds():
    ms = np.arange(24, dtype=np.int64) * 3_600_000 + EPOCH
    assert np.array_equal(normalize_ts_ms(ms), ms)
    assert np.array_equal(normalize_ts_ms(ms * 1000), ms)


def test_microsecond_detection_survives_an_arrow_column():
    table = make_bars(5)
    micros = pa.chunked_array([pa.array(table["ts"].to_numpy() * 1000, type=pa.int64())])
    assert np.array_equal(normalize_ts_ms(micros), table["ts"].to_numpy())


def test_empty_timestamps_normalize_without_guessing():
    assert normalize_ts_ms(np.array([], dtype=np.int64)).size == 0


# --- property tests ----------------------------------------------------------------------


@st.composite
def valid_bars(draw):
    """Random bar tables that satisfy every invariant by construction."""
    frequency = draw(st.sampled_from(sorted(FREQUENCIES)))
    step = grid_step_ms(frequency)
    n = draw(st.integers(min_value=1, max_value=60))
    # Gaps are legal, so the grid positions are a sorted sample rather than a run.
    slots = sorted(draw(st.sets(st.integers(0, 500), min_size=n, max_size=n)))
    price = st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False)
    opens = np.array(draw(st.lists(price, min_size=n, max_size=n)))
    closes = np.array(draw(st.lists(price, min_size=n, max_size=n)))
    spread = np.array(draw(st.lists(st.floats(0.0, 1e3, allow_nan=False), min_size=n, max_size=n)))
    return frequency, make_bars(
        n=n,
        frequency=frequency,
        ts=np.array(slots, dtype=np.int64) * step + EPOCH,
        open=opens,
        close=closes,
        high=np.maximum(opens, closes) + spread,
        low=np.minimum(opens, closes) - np.minimum(spread, np.minimum(opens, closes) * 0.5),
        volume=np.abs(spread),
        amount=np.abs(spread) * 10.0,
    )


@given(valid_bars())
@settings(max_examples=100, deadline=None)
def test_constructed_valid_tables_always_pass(case):
    frequency, table = case
    assert validate_bars(table, frequency).ok


@given(
    valid_bars(),
    st.sampled_from(["ts", "high", "low", "volume", "amount"]),
    st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_single_field_corruption_is_always_caught(case, column, magnitude):
    frequency, table = case
    if table.num_rows < 2:
        return
    index = table.num_rows // 2
    if column == "ts":
        # Below its predecessor: breaks monotonicity whatever the grid says.
        bad = int(table["ts"][0].as_py()) - grid_step_ms(frequency)
    elif column in ("volume", "amount"):
        bad = -magnitude
    elif column == "high":
        bad = min(table["open"][index].as_py(), table["close"][index].as_py()) - magnitude
    else:
        bad = max(table["open"][index].as_py(), table["close"][index].as_py()) + magnitude
    assert not validate_bars(mutate(table, column, index, bad), frequency).ok


# --- Parquet round trip ------------------------------------------------------------------


def test_parquet_round_trip_preserves_values_schema_and_metadata(tmp_path):
    table = make_bars(200)
    metadata = bars_metadata(
        source="binance_vision",
        asset_class="crypto",
        market="spot",
        symbol="BTCUSDT",
        frequency="1h",
        manifest_sha256="0" * 64,
    )
    written = table.replace_schema_metadata(metadata)
    path = tmp_path / "BTCUSDT.parquet"
    pq.write_table(written, path, compression="zstd", row_group_size=ROW_GROUP_SIZE)

    read = pq.read_table(path)
    assert read.schema.remove_metadata() == BARS_SCHEMA_V1
    assert read.equals(written)
    assert read.schema.metadata[b"symbol"] == b"BTCUSDT"
    assert read.schema.metadata[b"axiom_schema_version"] == b"1"
    assert read.schema.metadata[b"session_id"] == b"24x7"
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
