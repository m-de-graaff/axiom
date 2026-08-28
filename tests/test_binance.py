"""Kline zip parsing. Offline: the fixtures are built here, no network in CI."""

import io
import zipfile

import pandas as pd
from axiom_data import binance


def make_zip(path, rows, header=False):
    csv = ",".join(binance.KLINE_COLS) + "\n" if header else ""
    csv += "\n".join(",".join(str(v) for v in r) for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(path.stem + ".csv", csv)
    path.write_bytes(buf.getvalue())
    return path


def kline_row(open_time, close=10.0):
    return [open_time, 9.0, 11.0, 8.0, close, 1.5, open_time + 59_999, 15.0, 3, 0.5, 5.0, 0]


def test_open_time_is_shifted_to_bar_close(tmp_path):
    ms = int(pd.Timestamp("2024-01-01 00:00").value // 1_000_000)
    path = make_zip(tmp_path / "X-1m-2024-01.zip", [kline_row(ms), kline_row(ms + 60_000)])
    df = binance.klines_to_df(path, "1m")
    assert list(df.ts) == [pd.Timestamp("2024-01-01 00:01"), pd.Timestamp("2024-01-01 00:02")]


def test_microsecond_timestamps(tmp_path):
    """Binance switched kline timestamps to microseconds from 2025-01."""
    us = int(pd.Timestamp("2025-03-01 00:00").value // 1_000)
    path = make_zip(tmp_path / "X-1m-2025-03.zip", [kline_row(us)], header=True)
    assert binance.klines_to_df(path, "1m").ts.iloc[0] == pd.Timestamp("2025-03-01 00:01")


def test_amount_is_quote_volume(tmp_path):
    ms = int(pd.Timestamp("2024-01-01 00:00").value // 1_000_000)
    df = binance.klines_to_df(make_zip(tmp_path / "X-1m-2024-01.zip", [kline_row(ms)]), "1m")
    assert df.amount.iloc[0] == 15.0
    assert df.volume.iloc[0] == 1.5


def test_duplicate_rows_collapse(tmp_path):
    ms = int(pd.Timestamp("2024-01-01 00:00").value // 1_000_000)
    path = make_zip(tmp_path / "X-1m-2024-01.zip", [kline_row(ms), kline_row(ms)])
    assert len(binance.klines_to_df(path, "1m")) == 1


def test_month_of():
    key = "data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03.zip"
    assert binance.month_of(key) == "2024-03"
    assert binance.month_of("x/BTCUSDT-fundingRate-2021-11.zip") == "2021-11"
