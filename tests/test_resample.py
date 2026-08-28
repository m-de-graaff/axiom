"""Resampling is right-closed and right-labeled, or bars leak. See CLAUDE.md."""

import pandas as pd
import pytest
from axiom_data.resample import resample_ohlcv, timeframe_delta


def minute_bars(n: int, start: str = "2024-01-01 00:01") -> pd.DataFrame:
    """`n` close-labeled 1m bars whose close price is just the bar number."""
    ts = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": range(n),
            "high": [i + 0.5 for i in range(n)],
            "low": [i - 0.5 for i in range(n)],
            "close": range(n),
            "volume": [1.0] * n,
            "amount": [2.0] * n,
        }
    )


def test_labels_are_bar_close():
    # 60 bars labeled 00:01..01:00 are exactly the hour (00:00, 01:00].
    out = resample_ohlcv(minute_bars(60), "1h")
    assert len(out) == 1
    assert out.ts.iloc[0] == pd.Timestamp("2024-01-01 01:00")


def test_boundary_bar_belongs_to_the_earlier_bin():
    """The 01:00 bar closes the first hour; the 01:01 bar opens the second."""
    out = resample_ohlcv(minute_bars(61), "1h")
    assert list(out.ts) == [pd.Timestamp("2024-01-01 01:00"), pd.Timestamp("2024-01-01 02:00")]
    assert out.close.iloc[0] == 59  # last minute of hour one
    assert out.open.iloc[1] == 60  # first minute of hour two


def test_aggregation():
    out = resample_ohlcv(minute_bars(15), "15m")
    assert out.open.iloc[0] == 0
    assert out.close.iloc[0] == 14
    assert out.high.iloc[0] == 14.5
    assert out.low.iloc[0] == -0.5
    assert out.volume.iloc[0] == 15.0
    assert out.amount.iloc[0] == 30.0


def test_gaps_are_dropped_not_filled():
    """A market outage must not become a fabricated bar."""
    bars = pd.concat([minute_bars(60), minute_bars(60, start="2024-01-01 03:01")])
    out = resample_ohlcv(bars, "1h")
    assert list(out.ts) == [pd.Timestamp("2024-01-01 01:00"), pd.Timestamp("2024-01-01 04:00")]


def test_unsorted_input_is_sorted():
    out = resample_ohlcv(minute_bars(60).sample(frac=1, random_state=0), "1h")
    assert out.close.iloc[0] == 59


def test_unknown_timeframe_rejected():
    with pytest.raises(ValueError, match="unknown timeframe"):
        resample_ohlcv(minute_bars(10), "7m")


def test_timeframe_delta():
    assert timeframe_delta("4h") == pd.Timedelta(hours=4)
