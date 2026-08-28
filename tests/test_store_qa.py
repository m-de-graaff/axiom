"""Parquet round-trip, month-boundary merge, and the QA tripwires."""

import pandas as pd
import pytest
from axiom_data import qa, store
from conftest import bars


def test_round_trip_and_partition_layout(tmp_path):
    df = bars("2020-01-01", 24 * 40)
    paths = store.write_months(df, tmp_path, "binance", "AAAUSDT", "1h")
    assert len(paths) == 2
    assert paths[0] == store.month_path(tmp_path, "binance", "AAAUSDT", "1h", 2020, 1)

    back = store.read("AAAUSDT", "1h", root=tmp_path)
    pd.testing.assert_frame_equal(back, df.reset_index(drop=True))
    assert store.available_symbols(tmp_path, "binance", "1h") == ["AAAUSDT"]


def test_read_range_is_inclusive(tmp_path):
    store.write_months(bars("2020-01-01", 24 * 40), tmp_path, "binance", "AAAUSDT", "1h")
    got = store.read(
        "AAAUSDT", "1h", root=tmp_path, start="2020-01-10", end="2020-01-10 05:00"
    )
    assert got.ts.iloc[0] == pd.Timestamp("2020-01-10")
    assert got.ts.iloc[-1] == pd.Timestamp("2020-01-10 05:00")


def test_month_boundary_bar_survives_reingest(tmp_path):
    """A monthly source file spills one bar into the next month; merging keeps it."""
    january = bars("2020-01-01 01:00", 24 * 31)  # last bar lands on 2020-02-01 00:00
    february = bars("2020-02-01 01:00", 24 * 28)
    store.write_months(january, tmp_path, "binance", "AAAUSDT", "1h")
    store.write_months(february, tmp_path, "binance", "AAAUSDT", "1h")

    back = store.read("AAAUSDT", "1h", root=tmp_path)
    assert pd.Timestamp("2020-02-01 00:00") in set(back.ts)
    assert len(back) == len(january) + len(february)


def test_qa_flags_a_broken_bar():
    df = bars("2020-01-01", 100)
    df.loc[3, "high"] = df.loc[3, "low"] / 2
    row = qa.check_frame(df, "AAAUSDT", "1h")
    assert row["ohlc_violations"] == 1
    assert qa.violations(pd.DataFrame([row]))


def test_qa_counts_gaps():
    df = bars("2020-01-01", 100).drop(index=range(10, 20)).reset_index(drop=True)
    row = qa.check_frame(df, "AAAUSDT", "1h")
    assert row["missing_bars"] == 10
    assert not qa.violations(pd.DataFrame([row]), {"max_missing_bar_pct": 50})
    assert qa.violations(pd.DataFrame([row]), {"max_missing_bar_pct": 1})


def test_qa_clean_corpus_passes():
    row = qa.check_frame(bars("2020-01-01", 100), "AAAUSDT", "1h")
    assert qa.violations(pd.DataFrame([row])) == []


@pytest.mark.parametrize("check", qa.HARD_CHECKS)
def test_every_hard_check_is_reported(check):
    row = {c: 0 for c in qa.HARD_CHECKS} | {
        "symbol": "X", "tf": "1h", "bars": 10_000,
        "missing_bar_pct": 0.0, "zero_volume_pct": 0.0, check: 1,
    }
    assert qa.violations(pd.DataFrame([row])) == [f"X/1h: {check}=1"]
