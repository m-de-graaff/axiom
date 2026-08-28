"""Right-closed, right-labeled OHLCV resampling — the single implementation.

Timestamp convention for the whole project: **`ts` is the instant the bar closes.**
A 1h bar labeled `10:00` covers `(09:00, 10:00]`. This is deliberate: a bar labeled
with its close time is complete at that label, so "context up to and including `ts`"
is always information available at `ts`. Open-time labels (what Binance and ccxt
return) make the last bar incomplete at its own label, which is the classic
off-by-one leak. Converters live at the edges — `axiom_data.binance` shifts on
ingest, and any live feed must do the same.
"""

from __future__ import annotations

import pandas as pd

# tf -> pandas offset alias. 1m is the stored source; the rest are derived.
TIMEFRAMES: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
}


def timeframe_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(TIMEFRAMES[tf])


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample close-labeled OHLCV bars to `tf`, right-closed and right-labeled.

    `df` must have a `ts` column (or DatetimeIndex) of bar-close timestamps and the
    OHLCV(+amount) columns. Empty bins (market gaps) are dropped rather than
    forward-filled — a synthetic bar is a fabricated observation.
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}; known: {sorted(TIMEFRAMES)}")

    out = df.set_index("ts") if "ts" in df.columns else df
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("resample_ohlcv needs a 'ts' column or a DatetimeIndex")
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    cols = [c for c in AGG if c in out.columns]
    res = out.resample(TIMEFRAMES[tf], label="right", closed="right").agg(
        {c: AGG[c] for c in cols}
    )
    return res.dropna(subset=["open"]).reset_index()


__all__ = ["AGG", "TIMEFRAMES", "resample_ohlcv", "timeframe_delta"]
