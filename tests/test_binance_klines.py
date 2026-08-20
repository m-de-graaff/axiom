"""Parse layer: header sniffing, timestamp units, ordering, and the monthly/daily seam."""

from __future__ import annotations

import pytest

from axiom.schema.bars import BARS_SCHEMA_V1, validate_bars
from axiom.sources.binance_klines import extract_csv, merge, parse_archive, read_csv_bytes, to_bars
from tests.fakes import DAY_MS, EPOCH, HOUR_MS, csv_bytes, kline_rows, kline_zip, make_zip


def table_from(rows, **kwargs):
    return to_bars(read_csv_bytes(csv_bytes(rows, **kwargs)))


# --- header sniffing ---------------------------------------------------------------------


def test_headerless_and_header_bearing_files_parse_identically():
    rows = kline_rows(5)
    assert table_from(rows).equals(table_from(rows, header=True))


def test_a_header_row_is_not_parsed_as_data():
    assert table_from(kline_rows(5), header=True).num_rows == 5


def test_quoted_header_is_still_detected():
    raw = b'"open_time",open\n' + csv_bytes(kline_rows(2))
    assert read_csv_bytes(raw).num_rows == 2


# --- column mapping ----------------------------------------------------------------------


def test_columns_map_onto_schema_v1():
    table = table_from(kline_rows(3))
    assert table.schema == BARS_SCHEMA_V1
    assert "close_time" not in table.column_names
    assert "ignore" not in table.column_names


def test_amount_is_the_native_quote_volume():
    # kline_rows writes quote_asset_volume as volume * open, so this checks the mapping went to
    # `amount` and not to something that merely looks plausible.
    table = table_from(kline_rows(1, price=100.0, volume=3.0))
    assert table["amount"][0].as_py() == pytest.approx(300.0)
    assert table["volume"][0].as_py() == pytest.approx(3.0)


def test_optional_columns_are_retained():
    table = table_from(kline_rows(2))
    assert table["n_trades"].to_pylist() == [10, 11]
    assert table["taker_buy_volume"][0].as_py() == pytest.approx(1.5)


def test_scientific_notation_parses():
    rows = kline_rows(2)
    rows[0][5] = "1.5E-8"
    rows[0][7] = "3.0e+3"
    table = table_from(rows)
    assert table["volume"][0].as_py() == pytest.approx(1.5e-8)
    assert table["amount"][0].as_py() == pytest.approx(3000.0)


# --- timestamp units ---------------------------------------------------------------------


def test_microsecond_open_times_normalize_to_milliseconds():
    ms = table_from(kline_rows(6, unit="ms"))
    us = table_from(kline_rows(6, unit="us"))
    assert ms["ts"].to_pylist() == us["ts"].to_pylist()
    assert ms["ts"][0].as_py() == EPOCH


# --- failure modes -----------------------------------------------------------------------


def test_an_empty_csv_fails_cleanly():
    with pytest.raises(ValueError, match="empty kline CSV"):
        read_csv_bytes(b"")


def test_a_header_only_file_fails_cleanly():
    with pytest.raises(ValueError, match="zero rows"):
        to_bars(read_csv_bytes(csv_bytes([], header=True)))


def test_an_archive_with_no_csv_is_refused():
    with pytest.raises(ValueError, match="exactly one CSV"):
        extract_csv(make_zip(b"nothing here", member="readme.txt"))


def test_the_csv_member_is_found_whatever_it_is_called():
    assert extract_csv(make_zip(b"a,b\n", member="BTCUSDT-1h-2024-01.csv")) == b"a,b\n"


# --- merge: ordering and the seam --------------------------------------------------------


def test_out_of_order_rows_are_sorted():
    rows = kline_rows(5)
    rows.reverse()
    table = merge([table_from(rows)])
    assert table["ts"].to_pylist() == sorted(table["ts"].to_pylist())
    assert validate_bars(table, "1h").ok


def test_archives_are_concatenated_in_period_order():
    january = table_from(kline_rows(24, start=EPOCH))
    february = table_from(kline_rows(24, start=EPOCH + 24 * HOUR_MS))
    merged = merge([january, february])
    assert merged.num_rows == 48
    assert validate_bars(merged, "1h").ok


def test_an_identical_overlap_is_deduplicated():
    monthly = table_from(kline_rows(24, start=EPOCH))
    # The daily tail republishes the last three bars of the month, identically.
    daily = table_from(kline_rows(6, start=EPOCH + 21 * HOUR_MS))
    merged = merge([monthly, daily])
    assert merged.num_rows == 27
    assert validate_bars(merged, "1h").ok


def test_a_conflicting_overlap_is_a_failure_not_a_merge():
    monthly = table_from(kline_rows(24, start=EPOCH))
    conflicting = kline_rows(6, start=EPOCH + 21 * HOUR_MS)
    conflicting[0][4] = "999999.0"  # a different close for a bar that already exists
    with pytest.raises(ValueError, match="disagree at ts="):
        merge([monthly, table_from(conflicting)], context="spot/1h/BTCUSDT")


def test_the_conflict_message_names_the_columns():
    monthly = table_from(kline_rows(4, start=EPOCH))
    conflicting = kline_rows(2, start=EPOCH + 3 * HOUR_MS)
    conflicting[0][5] = "42.0"
    with pytest.raises(ValueError, match="volume"):
        merge([monthly, table_from(conflicting)])


def test_a_three_way_duplicate_collapses_to_one():
    one = table_from(kline_rows(3, start=EPOCH))
    merged = merge([one, one, one])
    assert merged.num_rows == 3


def test_merging_nothing_is_an_error():
    with pytest.raises(ValueError, match="nothing to merge"):
        merge([])


# --- through the archive layer -----------------------------------------------------------


def test_parse_archive_round_trips_a_zip():
    table = parse_archive(kline_zip(48, step=HOUR_MS))
    assert table.num_rows == 48
    assert validate_bars(table, "1h").ok


def test_daily_bars_land_on_the_daily_grid():
    table = parse_archive(kline_zip(31, step=DAY_MS))
    assert validate_bars(table, "1d").ok
